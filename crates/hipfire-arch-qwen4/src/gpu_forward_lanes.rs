// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact lane step (Flash-Next lane batching, fn-batch S2 AR rows, S5 MTP
//! verify rows).
//!
//! One trunk step over several admitted requests ("lanes"), each with its
//! own [`Qwen4State`]. Per lane, the committed state and logits must be
//! byte-identical to that request run alone on the singleton route the lane
//! takes: an AR row is `forward_token_or_argmax`, a verify block
//! `[seed, drafts..]` is the native MTP window's `verify_block` (armed GDN
//! ring, wide-hidden capture). The step is built from the singleton's own
//! typed layer operations with one rule:
//!
//! - **Stateful stages** (PLE convolution, GDN conv/recurrence/gate, QSA
//!   prologue/select/attend) always run on each lane's own state with the
//!   lane's own row count; the dispatch interpreter launches consecutive GDN
//!   and QSA Core steps of several lanes (AR rows and ragged verify windows)
//!   as one multi-lane launch per kernel (`HIPFIRE_QWEN4_GDN_LANES`,
//!   `HIPFIRE_QWEN4_GDN_LANES_ROWS`, `HIPFIRE_QWEN4_QSA_LANES`). A verify lane whose GDN
//!   in-projection is its own launch over exactly that lane's rows writes the lane's capture
//!   ring `inputs` in place and its Core skips the projection copy
//!   (`HIPFIRE_QWEN4_GDN_CAPTURE_DIRECT`).
//! - **Row-local stages** run either per lane ([`StageShare::PerLane`], the
//!   singleton's arm by construction) or once over every lane's rows
//!   ([`StageShare::OnceAllRows`], in groups of at most [`LANE_MAX_ROWS`];
//!   under `HIPFIRE_QWEN4_LANE_HC_ALLROWS` the HC stages of a >8-row AR /
//!   verify step take one all-rows segment instead, and under
//!   `HIPFIRE_QWEN4_LANE_ALLROWS` so does every other non-MoE once-over-all-rows
//!   stage: only the routed MoE keeps its groups, unless
//!   `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` (default on) makes it one EP call over every row on a
//!   second all-rows segment; a wide-class step (`HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY`)
//!   is one group already, so its MoE is that one call through its group).
//!   A stage moves to `OnceAllRows` only on byte evidence from the G0 oracle
//!   (`examples/fn_lane_oracle.rs --g0`): the S0 kill experiment found the
//!   few-row forward NOT bitwise equal to single rows at every N ≥ 2, so
//!   nothing is assumed row-count independent. `PerClass` executes as
//!   `PerLane`: a class-A (one-row) lane's singleton arm is the one-row
//!   kernel, and no existing kernel runs that arm over several rows.
//!
//! The GDN and QSA mixers split ([`MixerPhase`]) into their row-local input
//! projections, the per-lane state core, and the row-local output
//! projection, so the projections can share one launch while the core stays
//! per lane. Consecutive stages with the same share are emitted
//! segment-major (all of a lane's per-lane ops back to back, or all of a row
//! group's ops), so every lane sees the singleton's operation sequence, and
//! the step interpreter's fusions apply exactly where the singleton's do.
//!
//! The PLE convolution splits the same way: its stateless key / value projections are the
//! row-local stage [`StageName::PleProject`] (under `HIPFIRE_QWEN4_LANE_PLE_PROJECT`, exact
//! gfx1201, BF16 weights and a total row count one row-exact launch covers: ONE
//! `gemv_bf16_xf32_x4_rows_exact` launch each over every lane's rows, written in place into
//! the lanes' own key / value slices), and only the gate / norm / convolution tail
//! ([`StageName::PleConv`], `GroupedDepthwiseOp::projected`) runs per lane on the lane's own
//! convolution state.
//!
//! Lane steps run eager HIP (no retained tape) and never touch the resident
//! `bundle.state` / `bundle.mtp`; QSA arenas must already cover the step
//! (mapping happens in the executor's `provision_step`, never here).
//!
//! # Prefill wave
//!
//! The same step also runs a **prefill wave** ([`forward_wave`]): every lane
//! carries at least one row ([`LaneRowKind::Prefill`]; a step is all AR or all
//! prefill), and a once-over-all-rows launch covers EVERY row of the wave in
//! one segment (not [`LANE_MAX_ROWS`] groups). Stateful stages still run per
//! lane on the lane's own rows and state, so under the invariant prefill
//! profile (`Gpu::prefill_invariant_scope`, which routes every row-count
//! decision as the canonical chunk) a lane's committed state and head bytes
//! equal that request's singleton prefill. The wave head is per lane (the
//! singleton's `execute_lm_head` forms), never grouped: [`WaveHead::Final`]
//! writes one logits row (+ top-1) per lane from the lane's last row,
//! [`WaveHead::All`] writes every row of every lane. The share policy's
//! `Head` / `Argmax` entries are not consulted by a wave.
//!
//! ## Prefill wave under expert parallelism
//!
//! On an EP rank ([`forward_wave_ep`]; design in [`crate::gpu_forward::ep_wave`])
//! the wave is the EXACT joint prefill of the lanes' serial tiles: the MoE runs ONCE
//! per layer over every lane's rows on the singleton's own FULL-TABLE route (the
//! sub-512-row kernels, selected by capping the routing rows of the call at 511, with
//! real launch extents), so the layer's expert weights are read once instead of once
//! per lane tile; the row-local dense stages (embed, PLE rows, HC, the mixer in / out
//! projections) run once over all rows too (`HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT`, default
//! on; `0` runs them per lane, each launch the lane tile's by construction) and the
//! stateful cores run per lane on the lane's own rows and state. The caller's policy is replaced
//! by the EP wave policy ([`crate::gpu_forward::ep_wave::ep_wave_policy`]). A lane's
//! committed state and head are byte-equal to its serial tile because every kernel of
//! the wave's MoE call (router, shared expert, grouped expert GEMMs, combine) computes
//! each row independently of the other rows of its launch; the card oracle proves it.
//!
//! # Pipeline-parallel bands
//!
//! Under `topology.pp` each band runs the same step over ITS trunk layers
//! ([`Qwen4BandScope`](crate::weights::Qwen4BandScope)), on lane states
//! scoped to the same band: the layer program covers `scope.band` with GDN /
//! QSA slots counted locally (the state map's compact slots). Only the
//! input band embeds tokens and runs the PLE row fetch and convolution; every
//! other band starts from HC streams the caller placed in
//! `scratch.streams` rows `[0, n)` (lane order). Only the output band runs
//! the wide-hidden capture, final hyper, LM head and top-1; every other band
//! stops after its last HC write and leaves its streams in
//! `scratch.streams` rows `[0, n)` ([`StepOutputs::Edge`]). The whole-model
//! scope emits exactly the operations it always did.

use crate::bundle::Qwen4Bundle;
use crate::config::LayerType;
use crate::gpu_forward::ep_wave::{
    ep_wave_expected_moe_calls, ep_wave_joint_enabled, ep_wave_joint_for_lanes, ep_wave_layer_moe,
    ep_wave_partial_mask, ep_wave_policy_for, ep_wave_refusal, ep_wave_route, ep_wave_scope_rows,
    ep_wave_tile_exchange, EpWaveRoute,
};
use crate::gpu_forward::{
    bind_ep_lane_gather, ep_lane, ep_lane_buffer, ep_lane_check_invocations, ep_lane_moe_rows,
    lane_layer_moe,
};
use crate::gpu_forward::{
    decode_copy, dense_ref, hyper_read_desc, invalid, layer_desc, matrix_view, ple_desc,
    program_dims, qwen4_ple_rows_f16, router_columns, view, Qwen4GpuForward, Qwen4GpuForwardError,
    Qwen4GpuForwardScratch, ScratchField, SpViews, DECODE_Q8_PER_LAYER, EPSILON,
};
use crate::mesh_lane_protocol::pack_classed_row_groups;
use crate::ple_stage::ple_async_upload;
use crate::program::{
    execute_final_hyper, execute_lm_head, seal_moe_for_layer, validate_final_hyper,
    validate_lm_head, Qwen4AttentionWeights, Qwen4FfnBinding, Qwen4LayerDescription,
    Qwen4LayerScratch, Qwen4ProgramDims,
};
use crate::state::Qwen4State;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, execute_validated_steps_with_moe_hooks,
    validate_steps, ClearOp, EmbeddingOp, GatedDeltaNetOp, GroupedDepthwiseOp, HyperReadOp,
    HyperWriteOp, IndexedAttentionMode, IndexedAttentionOp, IndexedAttentionState, MixerPhase,
    MoeStepHooks, ProjectLadder, ProjectOp, RowGatherOp, Step,
};
use hipfire_runtime::external_rows::RowFetch;
use rdna_compute::prefill_invariant::QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP;
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::{DType, Gpu, GpuTensor};

/// A stage of the lane step that the share policy places.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageName {
    /// Token embedding and the HC stream broadcast.
    Embed,
    /// PLE row conversion (the host fetch and upload are one per step).
    PleRows,
    /// Every layer's attention and MLP hyper-connection reads.
    HcRead,
    /// Every layer's attention and MLP hyper-connection writes.
    HcWrite,
    /// PLE grouped convolution (stateful).
    PleConv,
    /// GDN qkv / a / b / z projections.
    GdnInProj,
    /// GDN convolution, recurrence and gate (stateful).
    GdnCore,
    /// GDN output projection.
    GdnOutProj,
    /// QSA indexer / q / k / v projections.
    QsaInProj,
    /// QSA prologue, pooling, selection and attention (stateful).
    QsaCore,
    /// QSA output projection.
    QsaOutProj,
    /// Router, routed experts and shared expert (one sealed call).
    Moe,
    /// Final hyper-connection read.
    FinalHyper,
    /// Language-model head.
    Head,
    /// Greedy top-1.
    Argmax,
    /// PLE key / value projections of every lane's rows, once per step
    /// (stateless: the grouped gate / norm / convolution tail that consumes
    /// them is [`StageName::PleConv`], per lane). Placed on its own all-rows
    /// segment when the BF16 projections run one row-exact launch over every row
    /// (`HIPFIRE_QWEN4_LANE_PLE_PROJECT`).
    PleProject,
}

impl StageName {
    pub const ALL: [StageName; 16] = [
        StageName::Embed,
        StageName::PleRows,
        StageName::HcRead,
        StageName::HcWrite,
        StageName::PleConv,
        StageName::GdnInProj,
        StageName::GdnCore,
        StageName::GdnOutProj,
        StageName::QsaInProj,
        StageName::QsaCore,
        StageName::QsaOutProj,
        StageName::Moe,
        StageName::FinalHyper,
        StageName::Head,
        StageName::Argmax,
        StageName::PleProject,
    ];

    pub fn key(self) -> &'static str {
        match self {
            StageName::Embed => "embed",
            StageName::PleRows => "ple_rows",
            StageName::HcRead => "hc_read",
            StageName::HcWrite => "hc_write",
            StageName::PleConv => "ple_conv",
            StageName::GdnInProj => "gdn_in",
            StageName::GdnCore => "gdn_core",
            StageName::GdnOutProj => "gdn_out",
            StageName::QsaInProj => "qsa_in",
            StageName::QsaCore => "qsa_core",
            StageName::QsaOutProj => "qsa_out",
            StageName::Moe => "moe",
            StageName::FinalHyper => "final_hyper",
            StageName::Head => "head",
            StageName::Argmax => "argmax",
            StageName::PleProject => "ple_project",
        }
    }

    pub fn parse(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|stage| stage.key() == key)
    }

    /// Stages that read or write request state: always per lane.
    pub fn stateful(self) -> bool {
        matches!(
            self,
            StageName::PleConv | StageName::GdnCore | StageName::QsaCore
        )
    }

    pub(crate) fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|stage| *stage == self)
            .expect("every stage is listed in ALL")
    }
}

/// Where a stage's launches run (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageShare {
    /// One launch over every lane's rows, in groups of at most
    /// [`LANE_MAX_ROWS`] rows (HC stages: one all-rows segment under
    /// `HIPFIRE_QWEN4_LANE_HC_ALLROWS`; every non-MoE stage under
    /// `HIPFIRE_QWEN4_LANE_ALLROWS`). Admitted only with G0 byte evidence.
    OnceAllRows,
    /// One launch per lane class; executes as [`StageShare::PerLane`] with
    /// the existing kernels.
    PerClass,
    /// One launch per lane with the lane's own row count.
    PerLane,
}

impl StageShare {
    fn key(self) -> &'static str {
        match self {
            StageShare::OnceAllRows => "once",
            StageShare::PerClass => "class",
            StageShare::PerLane => "lane",
        }
    }

    fn parse(key: &str) -> Option<Self> {
        match key {
            "once" => Some(StageShare::OnceAllRows),
            "class" => Some(StageShare::PerClass),
            "lane" => Some(StageShare::PerLane),
            _ => None,
        }
    }

    fn per_lane(self) -> bool {
        self != StageShare::OnceAllRows
    }
}

/// Rows a once-over-all-rows launch covers at most: the few-row kernel arms
/// (MQ6 / Q8 / BF16 row GEMVs, the Q8 decode copies) switch above 8 rows.
pub const LANE_MAX_ROWS: usize = 8;

/// Production share policy, from the G0 oracle on gfx1151 ([`GFX1151_LANE_RECEIPT`]):
/// every row-local stage — embedding, PLE rows, HC read/write, the GDN/QSA
/// projections, the sealed MoE (the grouped few-row route over ≤ 8 AR rows is
/// byte-identical per row to the one-row indexed decode), final HC read, head
/// and argmax — produced the per-lane bytes once over all rows; the stateful
/// cores stay per lane. S0's chunk-vs-singles mismatch therefore lives in the
/// multi-row GDN/QSA cores of consecutive positions, which a lane never runs.
pub const STAGE_SHARE: [(StageName, StageShare); 16] = [
    (StageName::Embed, StageShare::OnceAllRows),
    (StageName::PleRows, StageShare::OnceAllRows),
    (StageName::HcRead, StageShare::OnceAllRows),
    (StageName::HcWrite, StageShare::OnceAllRows),
    (StageName::PleConv, StageShare::PerLane),
    (StageName::GdnInProj, StageShare::OnceAllRows),
    (StageName::GdnCore, StageShare::PerLane),
    (StageName::GdnOutProj, StageShare::OnceAllRows),
    (StageName::QsaInProj, StageShare::OnceAllRows),
    (StageName::QsaCore, StageShare::PerLane),
    (StageName::QsaOutProj, StageShare::OnceAllRows),
    (StageName::Moe, StageShare::OnceAllRows),
    (StageName::FinalHyper, StageShare::OnceAllRows),
    (StageName::Head, StageShare::OnceAllRows),
    (StageName::Argmax, StageShare::OnceAllRows),
    (StageName::PleProject, StageShare::OnceAllRows),
];

/// G0 oracle receipt behind [`STAGE_SHARE`] (`fn-batch/S3-RESULT.md`): each
/// row-local stage flipped alone at k=2 and k=8, then the combined policy at
/// k=1/2/8, contexts 512/2048, per-step logits, ids and full state bytes
/// identical to the singleton.
pub const GFX1151_LANE_RECEIPT: &str = "g0 fn-batch-s3/j2-g0 2026-10-09 gfx1151";

/// G0 oracle receipt behind the gfx1201 pipeline-parallel resident-MQ2-slab
/// row-twin proofs (`fn_mesh_lane_oracle`, PP2 mq2-xts, 2x R9700): with the
/// full shared-stage policy and with every stage per lane, every lane's ids
/// and every logits row were byte-identical to the serialized PP singleton at
/// K=1..8, 8 prompts x 64 tokens, under adaptive, 1 and 4 decode micro-batches
/// and with micro-batch overlap off.
pub const GFX1201_PP2_MQ2_RECEIPT: &str =
    "g0 mcb-notes/runs/thin-pp-a1bab39d7-1 j1574 2026-10-18 gfx1201 pp2 mq2-xts";

/// G0 oracle receipt behind the gfx1201 pipeline-parallel resident-slab
/// MQ4-XTS + MQ3-I4 MoE row-twin proof (`fn_mesh_lane_oracle`, PP2 mq3-xts
/// 25/23 at prefill chunk 1024, 2x R9700): with every stage per lane, with
/// only the MoE stage shared and with the full shared-stage policy, every
/// lane's ids and every logits row were byte-identical to the serialized PP
/// singleton at K=1..8, 8 prompts x 64 tokens.
pub const GFX1201_PP2_MQ3_RECEIPT: &str =
    "g0 mgmtp/pponce-runs/proof-pp2-mq3 2026-10-30 gfx1201 pp2 mq3-xts 25/23 k1-8";

/// G0 evidence behind the gfx1201 expert-parallel (EP2, mq4-xts) row-twin
/// proofs (hold j1570, production PASS): every lane's ids and logits bytes
/// were identical to the singleton at K=1, 2, 4 and 8 over 64 steps. The
/// artifact's non-MoE tensors are Q8_0 embed (qt3), MQ6G256V2 head, GDN
/// in/out and QSA in/out (qt47), and BF16 PLE rows, hyper-connection and
/// router/shared experts; its routed experts are the packed QT44/QT53 class
/// of [`LaneExpertRoute::EpPackedStaticOrBalanced`].
pub const GFX1201_EP2_MQ4_RECEIPT: &str =
    "g0 mcb-notes/runs/ep-diag-8e148906a-1 j1570 2026-10-18 gfx1201 ep2 mq4-xts k1,2,4,8";

/// G0 oracle receipt behind the gfx1201 Q8F16 QSA row-twin proofs
/// (`QsaIn([Q8_0; 4])`, `Projection(Q8_0)` for qsa_out; EP domain, the measured
/// Q8F16 trunk-overlay scope): `lane_q8_rows_oracle` (then named `lane_q8_qsa_rows_oracle`) over the 60 real
/// overlay tensors (12 QSA layers x indexer/q/k/v/o), rows 1..=8, the
/// dispatch's few-row Q8 route (`project_weights`, `gemv_q8_0_rows_exact`) vs
/// the rows == 1 `gemv_q8_0` singleton per row: 0 mismatching bytes
/// (17748 cells, 2,774,679,552 bytes compared, fabric j6621, card A).
pub const GFX1201_Q8_QSA_ROWS_RECEIPT: &str =
    "g0 pmloop/q8lanes-ev/oracle1 j6621 2026-11-02 gfx1201 q8f16 qsa in/out rows1-8";

/// G0 oracle receipt behind the gfx1201 Q8F16 GDN in/out and head row-twin
/// proofs (`GdnIn([Q8_0; 4])`, `Projection(Q8_0)` for gdn_out and the head; EP
/// domain, the full Q8F16 trunk overlay scope): `lane_q8_rows_oracle` over all
/// 241 real overlay tensors (60 QSA, 36 x GDN qkv/z/a/b and out_proj, lm_head),
/// rows 1..=8 in both row orders, the dispatch's few-row Q8 route
/// (`project_weights`, `gemv_q8_0_rows_exact`) vs the rows == 1 `gemv_q8_0`
/// singleton per row: 0 mismatching bytes (71,307 cells, 15,172,718,592 bytes
/// compared, fabric j6678, card A).
pub const GFX1201_Q8_TRUNK_ROWS_RECEIPT: &str =
    "g0 mgmtp/q8trunk/runs/oracle1 j6678 2026-11-02 gfx1201 q8f16 gdn in/out head rows1-8";

/// G0 oracle receipt behind the gfx1201 HC8 (`.hc8` sidecar, `DType::HC8E4M3Row`)
/// row-twin proofs (`HyperRead([BF16, HC8E4M3Row, HC8E4M3Row])` for `HcRead` and
/// `FinalHyper`; EP domain; `HcWrite` keeps its BF16 `block_inject` row):
/// `lane_hc8_rows_oracle` over the real sidecar (200 sha256-verified payloads, 100
/// connections: 48 layers x attn/mlp, the MTP layer's, the final mixer), the
/// dispatch's shared all-rows launch (`execute_hyper_read`, `execute_final_hyper`, and
/// the fused write -> read chain of `execute_validated_steps` over 47 consecutive layer
/// pairs, under the all-rows wide scope for R > 8) vs R separate rows == 1 launches per
/// row, R in 1..=64 in both row orders, 2 seeds of BF16-exact stream rows plus
/// real-magnitude edge rows, over normalized/low/up/mixed (and chain streams/gates/clear):
/// 0 mismatching bytes (38,656 cells, 137,418,045,440 bytes compared, fabric j8251,
/// card B).
pub const GFX1201_HC8_ROWS_RECEIPT: &str =
    "g0 hc8/proof/oracle1 j8251 2026-11-06 gfx1201 hc8 e4m3 hc read rows1-64";

/// Byte-oracle receipt behind the gfx1201 PLE key / value projection row-twin proofs
/// (`PleProject([BF16, BF16])`; PP and EP domains): the lane-wide projection is ONE
/// `gemv_bf16_xf32_x4_rows_exact` launch over every lane's rows (all-rows arm above eight
/// rows), each output word bitwise the per-lane incumbent's (CHANGELOG.md:224, the allrows
/// header): `bf16_allrows_oracle --reread` on the real layer-1 key / value weights, lanes
/// 2 / 4 / 8 x 2 / 6 rows, incumbent per-lane chunks and the one launch both 0 mismatching
/// words against the per-row singleton.
pub const GFX1201_PLE_PROJECT_RECEIPT: &str =
    "g0 reread/runs/oracle1 j8313 2026-11-06 gfx1201 bf16 ple key/value lanes2-8 rows4-48";

/// G0 oracle receipt behind the gfx942 single-device stock-FP8 row-twin proofs
/// (`fn_lane_oracle --g0 --controls`, one MI300X, `qwen3.8-flash-next.stock-fp8.hfq`
/// sha256 `650e7f43…`, contexts 512/2048, 64 steps): the combined production
/// policy (every row-local stage once over all rows) at k=1/2/4/8 gave every
/// lane's logits rows, ids and state checkpoints byte-identical to the
/// singleton, and the negative controls were detected. The per-stage probes did
/// NOT all agree: sharing HC read, HC write, the GDN in or out projection, or
/// the MoE alone (the rest per lane) diverged, so this receipt proves the
/// combined policy only ([`COMBINED_ONLY_RECEIPTS`]). The trunk / head
/// projections are BF16 on the few-row MFMA entry (`gemm_bf16_mfma_gfx942_rows16`),
/// the routed experts the FP8 grouped MFMA route ([`LaneExpertRoute::ResidentFp8`]).
pub const GFX942_STOCK_FP8_RECEIPT: &str =
    "g0 mi300x/headline/oracle/g0 2026-11-10 gfx942 stock-fp8 single, combined policy only";

/// Receipts that evidence their rows' stages only together (the G0 oracle
/// passed the combined policy, not every single-stage probe):
/// [`resolve_with_proofs`] admits all of a receipt's stages or none.
const COMBINED_ONLY_RECEIPTS: &[&str] = &[GFX942_STOCK_FP8_RECEIPT];

/// Device topology a lane step runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneTopology {
    Single,
    Pp,
    Ep,
}

/// How the step's routed experts are bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneExpertRoute {
    ResidentMq2Slab,
    /// Resident expert-slab arenas of complete layers of MQ4-XTS plus MQ3-I4
    /// experts (at least one MQ3-I4, no E8P, W2, absent or host-mapped
    /// expert), the mq3-xts Flash-Next family.
    ResidentMq3Slab,
    ResidentPacked,
    /// Stock-FP8 routed experts (`Qwen4TierRecipe::StockFp8`): every owned
    /// MoE layer's gate_up / down aggregates resident qt=55 `FP8E4M3B128`
    /// tables (no arena, no host mapping), run by the gfx942 FP8 grouped
    /// MFMA route (`gemm_fp8b128_moe_grouped_bf16mfma`, fixed ascending K,
    /// 1..16-row tiles, the same kernels at every row count).
    ResidentFp8,
    HostMapped,
    CpuDeferred,
    Split,
    /// Expert-parallel packed QT44/QT53 routed experts: every owned MoE layer
    /// is either a fully resident static gathered decode route or a wholly
    /// host-mapped balanced-live gathered runtime route, gather installed, no
    /// CPU-deferred, tier, slab, D1/D2 or pruned experts. Any per-layer mix
    /// (all resident, all mapped, or both) is the class; the measured EP2
    /// scope carried both kinds. A per-expert unit layer (D1) joins it as the
    /// static units kind (static owners) or, with pinned owners on, as the
    /// balanced units kind (the pinned owner-assign kernel). Set only from
    /// bound EP placement and the installed gather, never from arch, env or
    /// layer counts. On the trunk-shard partial-FFN route the balanced kind
    /// (`HIPFIRE_EP_TRUNK_BALANCED_OWNERS=1`, default off) gathers every slot
    /// and folds under the fixed `e % ranks` reduction partition, so a row's
    /// partial does not depend on the other rows of its call.
    EpPackedStaticOrBalanced,
    Unclassified,
}

/// The route one stage instance runs, as its owned weights and kernels bind
/// it. A lane registers one entry per owned instance of a stage (layer, head,
/// ...); [`StageRoute::Unclassified`] is an instance the lane could not
/// positively recognise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageRoute {
    /// Proof row only: admits every instance of the stage, classified or not.
    Any,
    Unclassified,
    Embed(DType),
    PleRows(DType),
    /// PLE key / value projection weight dtypes, in that order.
    PleProject([DType; 2]),
    /// Hyper-connection read operand dtypes.
    HyperRead([DType; 3]),
    HyperWrite(DType),
    /// GDN input projections in qkv, z, a, b order.
    GdnIn([DType; 4]),
    /// A single output projection (GDN out, QSA out, LM head).
    Projection(DType),
    /// QSA input projections in indexer_qk, q, k, v order.
    QsaIn([DType; 4]),
    Moe {
        experts: LaneExpertRoute,
        dense: [DType; 5],
        decode_admitted: bool,
    },
    Argmax(DType),
}

/// The route instances a lane owns, indexed by [`StageName::index`].
pub type LaneStageRoutes = [Vec<StageRoute>; 16];

/// One G0 row-twin proof: on `arch`, over every topology of `domain`, a
/// shared launch of `stage` over instances of `route` was byte-identical to
/// the per-lane (singleton) arm, as evidenced by `receipt`. Only an `Any`
/// route matches an unclassified instance; an equality route never does.
#[derive(Clone, Copy, Debug)]
pub struct RowTwinProof {
    pub arch: &'static str,
    pub stage: StageName,
    pub route: StageRoute,
    pub domain: &'static [LaneTopology],
    pub receipt: &'static str,
}

const fn proof(
    arch: &'static str,
    stage: StageName,
    route: StageRoute,
    domain: &'static [LaneTopology],
    receipt: &'static str,
) -> RowTwinProof {
    RowTwinProof {
        arch,
        stage,
        route,
        domain,
        receipt,
    }
}

const LANE_ALL_TOPOLOGIES: &[LaneTopology] =
    &[LaneTopology::Single, LaneTopology::Pp, LaneTopology::Ep];
const LANE_PP_ONLY: &[LaneTopology] = &[LaneTopology::Pp];
const LANE_EP_ONLY: &[LaneTopology] = &[LaneTopology::Ep];
const LANE_SINGLE_ONLY: &[LaneTopology] = &[LaneTopology::Single];

/// Row-twin proofs, one row per shared stage per arch and topology domain
/// (gfx1201 carries a PP-only domain and a disjoint EP-only domain whose MoE
/// row is the packed static-or-balanced expert class; gfx942 a single-device
/// domain whose rows are the stock-FP8 BF16 trunk and FP8 expert class). A
/// stage is shared only when every instance a lane owns matches a row of its
/// arch, stage and topology; any other (arch, topology, route) keeps that
/// stage per lane until its own G0 run (other cards bind other kernels, e.g.
/// no Q8_0 decode copies on gfx1201).
///
/// The gfx1151 rows are `Any` over every topology on purpose, for
/// compatibility: the proof predates route classification and was applied to
/// every gfx1151 route.
pub static LANE_ROW_TWIN_PROOFS: &[RowTwinProof] = &[
    proof(
        "gfx1151",
        StageName::Embed,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::PleRows,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::HcRead,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::HcWrite,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::GdnInProj,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::GdnOutProj,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::QsaInProj,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::QsaOutProj,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::Moe,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::FinalHyper,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::Head,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1151",
        StageName::Argmax,
        StageRoute::Any,
        LANE_ALL_TOPOLOGIES,
        GFX1151_LANE_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Embed,
        StageRoute::Embed(DType::Q8_0),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::PleRows,
        StageRoute::PleRows(DType::BF16),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::HcRead,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::HcWrite,
        StageRoute::HyperWrite(DType::BF16),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::GdnInProj,
        StageRoute::GdnIn([DType::MQ6G256V2; 4]),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::GdnOutProj,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::QsaInProj,
        StageRoute::QsaIn([DType::MQ6G256V2; 4]),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::QsaOutProj,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Moe,
        StageRoute::Moe {
            experts: LaneExpertRoute::ResidentMq2Slab,
            dense: [DType::BF16; 5],
            decode_admitted: true,
        },
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Moe,
        StageRoute::Moe {
            experts: LaneExpertRoute::ResidentMq3Slab,
            dense: [DType::BF16; 5],
            decode_admitted: true,
        },
        LANE_PP_ONLY,
        GFX1201_PP2_MQ3_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::FinalHyper,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Head,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Argmax,
        StageRoute::Argmax(DType::F32),
        LANE_PP_ONLY,
        GFX1201_PP2_MQ2_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Embed,
        StageRoute::Embed(DType::Q8_0),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::PleRows,
        StageRoute::PleRows(DType::BF16),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::HcRead,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    // HC8 sidecar (`.hc8`, E4M3 mixers): the all-rows read is bitwise the rows == 1
    // singleton per row, oracle `lane_hc8_rows_oracle`. `HcWrite` stays BF16.
    proof(
        "gfx1201",
        StageName::HcRead,
        StageRoute::HyperRead([DType::BF16, DType::HC8E4M3Row, DType::HC8E4M3Row]),
        LANE_EP_ONLY,
        GFX1201_HC8_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::HcWrite,
        StageRoute::HyperWrite(DType::BF16),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::GdnInProj,
        StageRoute::GdnIn([DType::MQ6G256V2; 4]),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::GdnOutProj,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::QsaInProj,
        StageRoute::QsaIn([DType::MQ6G256V2; 4]),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::QsaOutProj,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    // Q8F16 (qt3) QSA trunk overlay (q/k/v/o + indexer): the few-row launches are
    // `gemv_q8_0_rows_exact`, bitwise the rows == 1 `gemv_q8_0` singleton per row.
    proof(
        "gfx1201",
        StageName::QsaInProj,
        StageRoute::QsaIn([DType::Q8_0; 4]),
        LANE_EP_ONLY,
        GFX1201_Q8_QSA_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::QsaOutProj,
        StageRoute::Projection(DType::Q8_0),
        LANE_EP_ONLY,
        GFX1201_Q8_QSA_ROWS_RECEIPT,
    ),
    // Q8F16 (qt3) GDN in/out and head (the full-trunk overlay): same few-row
    // route, per-tensor-set oracle `lane_q8_rows_oracle` (receipt pending a card).
    proof(
        "gfx1201",
        StageName::GdnInProj,
        StageRoute::GdnIn([DType::Q8_0; 4]),
        LANE_EP_ONLY,
        GFX1201_Q8_TRUNK_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::GdnOutProj,
        StageRoute::Projection(DType::Q8_0),
        LANE_EP_ONLY,
        GFX1201_Q8_TRUNK_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::FinalHyper,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::FinalHyper,
        StageRoute::HyperRead([DType::BF16, DType::HC8E4M3Row, DType::HC8E4M3Row]),
        LANE_EP_ONLY,
        GFX1201_HC8_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Head,
        StageRoute::Projection(DType::MQ6G256V2),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Head,
        StageRoute::Projection(DType::Q8_0),
        LANE_EP_ONLY,
        GFX1201_Q8_TRUNK_ROWS_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Argmax,
        StageRoute::Argmax(DType::F32),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::Moe,
        StageRoute::Moe {
            experts: LaneExpertRoute::EpPackedStaticOrBalanced,
            dense: [DType::BF16; 5],
            decode_admitted: true,
        },
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::PleProject,
        StageRoute::PleProject([DType::BF16; 2]),
        LANE_PP_ONLY,
        GFX1201_PLE_PROJECT_RECEIPT,
    ),
    proof(
        "gfx1201",
        StageName::PleProject,
        StageRoute::PleProject([DType::BF16; 2]),
        LANE_EP_ONLY,
        GFX1201_PLE_PROJECT_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::Embed,
        StageRoute::Embed(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::PleRows,
        StageRoute::PleRows(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::HcRead,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::HcWrite,
        StageRoute::HyperWrite(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::GdnInProj,
        StageRoute::GdnIn([DType::BF16; 4]),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::GdnOutProj,
        StageRoute::Projection(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::QsaInProj,
        StageRoute::QsaIn([DType::BF16; 4]),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::QsaOutProj,
        StageRoute::Projection(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::Moe,
        StageRoute::Moe {
            experts: LaneExpertRoute::ResidentFp8,
            dense: [DType::BF16; 5],
            decode_admitted: false,
        },
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::FinalHyper,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::Head,
        StageRoute::Projection(DType::BF16),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::Argmax,
        StageRoute::Argmax(DType::F32),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
    proof(
        "gfx942",
        StageName::PleProject,
        StageRoute::PleProject([DType::BF16; 2]),
        LANE_SINGLE_ONLY,
        GFX942_STOCK_FP8_RECEIPT,
    ),
];

/// A resolved lane policy: the policy, the receipts of the proofs behind
/// every shared stage (deduplicated, first-use order), and each stage that
/// runs per lane with the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanePolicyResolution {
    pub policy: LaneStagePolicy,
    pub receipts: Vec<&'static str>,
    pub per_lane: Vec<(StageName, String)>,
}

/// Resolve [`STAGE_SHARE`] against `proofs`. A shared candidate stage stays
/// shared only when a proof row exists for the arch, stage and topology and
/// EVERY owned route instance matches a row (an `Any` row, or an equal
/// non-`Unclassified` route). A stage that owns no instance is admitted
/// vacuously once such a row exists (a band without that stage must not
/// demote the rest), and cites no receipt. Stateful stages are always per
/// lane. A [`COMBINED_ONLY_RECEIPTS`] receipt admits its stages only
/// together: when any stage it has a row for (on this arch and topology)
/// stays per lane, every stage admitted under it goes back per lane.
fn resolve_with_proofs(
    arch: &str,
    topology: LaneTopology,
    routes: &LaneStageRoutes,
    proofs: &[RowTwinProof],
) -> LanePolicyResolution {
    let mut policy = LaneStagePolicy::all_per_lane();
    let mut per_lane: Vec<(StageName, String)> = Vec::new();
    let mut admitted: Vec<(StageName, Vec<&'static str>)> = Vec::new();
    for (stage, share) in STAGE_SHARE {
        if stage.stateful() {
            per_lane.push((stage, "stateful".to_string()));
            continue;
        }
        if share != StageShare::OnceAllRows {
            policy.shares[stage.index()] = share;
            continue;
        }
        let rows: Vec<&RowTwinProof> = proofs
            .iter()
            .filter(|row| row.arch == arch && row.stage == stage && row.domain.contains(&topology))
            .collect();
        if rows.is_empty() {
            per_lane.push((
                stage,
                format!(
                    "arch {arch:?} topology {topology:?} has no row-twin proof for stage {}",
                    stage.key()
                ),
            ));
            continue;
        }
        let mut stage_receipts: Vec<&'static str> = Vec::new();
        let mut refusal: Option<String> = None;
        for route in &routes[stage.index()] {
            let matched = rows.iter().find(|row| {
                row.route == StageRoute::Any
                    || (*route != StageRoute::Unclassified && row.route == *route)
            });
            match matched {
                Some(row) => {
                    if !stage_receipts.contains(&row.receipt) {
                        stage_receipts.push(row.receipt);
                    }
                }
                None => {
                    refusal = Some(if *route == StageRoute::Unclassified {
                        format!(
                            "arch {arch:?} topology {topology:?} stage {}: unclassified route has no row-twin proof",
                            stage.key()
                        )
                    } else {
                        format!(
                            "arch {arch:?} topology {topology:?} stage {}: route {route:?} has no row-twin proof",
                            stage.key()
                        )
                    });
                    break;
                }
            }
        }
        match refusal {
            Some(reason) => per_lane.push((stage, reason)),
            None => {
                policy.shares[stage.index()] = share;
                admitted.push((stage, stage_receipts));
            }
        }
    }
    for &combined in COMBINED_ONLY_RECEIPTS {
        let covers = |stage: StageName| {
            proofs.iter().any(|row| {
                row.receipt == combined
                    && row.arch == arch
                    && row.stage == stage
                    && row.domain.contains(&topology)
            })
        };
        let Some(missing) = StageName::ALL
            .into_iter()
            .find(|&stage| covers(stage) && policy.share(stage) != StageShare::OnceAllRows)
        else {
            continue;
        };
        for (stage, stage_receipts) in &admitted {
            if stage_receipts.contains(&combined) || (stage_receipts.is_empty() && covers(*stage)) {
                policy.shares[stage.index()] = StageShare::PerLane;
                per_lane.push((
                    *stage,
                    format!(
                        "arch {arch:?} topology {topology:?} stage {}: receipt {combined:?} proves only the combined policy and stage {} stays per lane",
                        stage.key(),
                        missing.key()
                    ),
                ));
            }
        }
    }
    let mut receipts: Vec<&'static str> = Vec::new();
    for (stage, stage_receipts) in &admitted {
        if policy.share(*stage) != StageShare::OnceAllRows {
            continue;
        }
        for receipt in stage_receipts {
            if !receipts.contains(receipt) {
                receipts.push(receipt);
            }
        }
    }
    LanePolicyResolution {
        policy,
        receipts,
        per_lane,
    }
}

/// One share per [`StageName`], indexed in [`StageName::ALL`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneStagePolicy {
    shares: [StageShare; 16],
}

impl LaneStagePolicy {
    /// The policy for the route instances a lane store owns on `arch` /
    /// `topology`, resolved stage by stage against [`LANE_ROW_TWIN_PROOFS`].
    pub fn for_bound_routes(
        arch: &str,
        topology: LaneTopology,
        routes: &LaneStageRoutes,
    ) -> LanePolicyResolution {
        resolve_with_proofs(arch, topology, routes, LANE_ROW_TWIN_PROOFS)
    }

    /// The G0-evidenced share table: every row-local stage shared. It is not
    /// route-bound; [`Self::for_bound_routes`] resolves it per stage.
    pub fn production() -> Self {
        let mut shares = [StageShare::PerLane; 16];
        for (stage, share) in STAGE_SHARE {
            shares[stage.index()] = share;
        }
        Self { shares }
    }

    pub fn all_per_lane() -> Self {
        Self {
            shares: [StageShare::PerLane; 16],
        }
    }

    pub fn share(&self, stage: StageName) -> StageShare {
        self.shares[stage.index()]
    }

    pub fn with(mut self, stage: StageName, share: StageShare) -> Result<Self, String> {
        if stage.stateful() && share != StageShare::PerLane {
            return Err(format!(
                "stage {} reads request state and always runs per lane",
                stage.key()
            ));
        }
        self.shares[stage.index()] = share;
        Ok(self)
    }

    /// `production` | `per-lane` | `<base>:<stage>=once|class|lane,...`
    /// with `<base>` one of the two named policies.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (base, overrides) = match spec.split_once(':') {
            Some((base, rest)) => (base, Some(rest)),
            None => (spec, None),
        };
        let mut policy = match base.trim() {
            "production" => Self::production(),
            "per-lane" => Self::all_per_lane(),
            other => return Err(format!("unknown lane policy base {other:?}")),
        };
        for item in overrides
            .into_iter()
            .flat_map(|rest| rest.split(','))
            .map(str::trim)
            .filter(|item| !item.is_empty())
        {
            let (key, value) = item
                .split_once('=')
                .ok_or_else(|| format!("lane policy item {item:?} is not stage=share"))?;
            let stage = StageName::parse(key.trim())
                .ok_or_else(|| format!("unknown lane stage {key:?}"))?;
            let share = StageShare::parse(value.trim())
                .ok_or_else(|| format!("unknown lane share {value:?}"))?;
            policy = policy.with(stage, share)?;
        }
        Ok(policy)
    }

    pub fn describe(&self) -> String {
        StageName::ALL
            .iter()
            .map(|stage| format!("{}={}", stage.key(), self.share(*stage).key()))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Stateful stages must stay per lane (a policy built through
    /// [`Self::with`] / [`Self::parse`] always satisfies this).
    pub fn validate(&self) -> Result<(), String> {
        for stage in StageName::ALL {
            if stage.stateful() && self.share(stage) != StageShare::PerLane {
                return Err(format!("stage {} must run per lane", stage.key()));
            }
        }
        Ok(())
    }
}

/// What a lane's rows do this step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRowKind {
    /// One AR row (the lane's pending seed); no GDN row capture.
    Ar,
    /// An MTP verify block `[seed, drafts..]` of `1 + draft_len` rows
    /// (`2..=LANE_MAX_ROWS`), run under the lane's armed GDN ring exactly as
    /// the singleton `verify_block` runs it.
    Verify {
        draft_len: usize,
    },
    Prefill,
}

/// One lane's rows in a step.
pub struct LaneRows<'a> {
    pub state: &'a mut Qwen4State,
    pub tokens: &'a [u32],
    pub kind: LaneRowKind,
    /// The lane verifies in the wide pass (see [`lane_is_wide`]); `false` for AR-only
    /// steps' exact rows and every prefill lane.
    pub wide: bool,
    /// The lane's request is in the row-exact class (`HIPFIRE_QWEN4_LANE_ROW_EXACT`: opt-in, default off: greedy, or
    /// with `seeded` also sampled with an explicit seed): an all-rows step carrying it runs under the
    /// row-exact scope, so its rows are bitwise its <=8-row rows whatever the batch composition.
    /// `false` for every prefill lane.
    pub exact: bool,
}

/// A vocabulary shard of the step's LM head (`HIPFIRE_QWEN4_EP_VOCAB_SHARD`): the rank projects
/// and scans only the head rows `[base, base + width)`. The step's logits are PACKED
/// `[rows][width]` in the first `rows * width` elements of [`LaneOutputs::logits`] and the top-1
/// ids are LOCAL to the shard (`0..width`); the caller adds `base` where it merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocabShardView {
    /// Global id of the shard's first head row.
    pub base: usize,
    /// Head rows in the shard (the logits row width).
    pub width: usize,
}

/// The step's outputs; row `r` belongs to the lanes concatenated in order.
pub struct LaneOutputs<'a> {
    /// F32, at least `total_rows * vocab` elements.
    pub logits: &'a GpuTensor,
    /// Raw, at least `total_rows * 4` bytes (one i32 per row).
    pub top1: &'a GpuTensor,
    /// F32, at least `total_rows * hc_count * hidden` elements: every row's
    /// final HC streams (the trunk output before the final hyper read),
    /// exactly what the singleton's wide-hidden capture writes.
    pub wide_hidden: Option<&'a GpuTensor>,
    /// The vocabulary shard this rank's head runs ([`VocabShardView`]): `logits` then hold
    /// `total_rows * width` packed elements and `top1` holds shard-local ids. `None`: the whole
    /// head, today's layout. Only an AR / verify step of the output band takes a shard (refused
    /// for prefill waves, recorded or captured steps).
    pub vocab_shard: Option<VocabShardView>,
}

/// Which logits a prefill wave writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveHead {
    /// One logits row (+ top-1) per lane, in lane order, from the lane's last
    /// row (the singleton's `Qwen4OutputRows::Final`).
    Final,
    /// Every row of every lane, in concatenated order (lab only; small totals).
    All,
}

/// One lane's rows in a prefill wave.
pub struct WaveLane<'a> {
    pub state: &'a mut Qwen4State,
    pub tokens: &'a [u32],
}

/// A prefill wave's outputs: F32 `lanes * vocab` ([`WaveHead::Final`]) or
/// `total_rows * vocab` ([`WaveHead::All`]) logits, a Raw i32 per output
/// row for the optional top-1, and the optional wide-hidden capture of every
/// row (F32, `total_rows * hc_count * hidden`, lanes concatenated: what the
/// singleton's `wide_hidden_capture` writes, the native-MTP fill's input).
pub(crate) struct WaveOutputs<'a> {
    head: WaveHead,
    logits: &'a GpuTensor,
    top1: Option<&'a GpuTensor>,
    wide_hidden: Option<&'a GpuTensor>,
}

/// Outputs of a lane step: one logits row and top-1 per AR row, a prefill
/// wave's head, or (a band that does not own the output) nothing: the band's
/// HC streams stay in `scratch.streams` rows `[0, n)` for the next band.
pub(crate) enum StepOutputs<'a> {
    Ar(&'a LaneOutputs<'a>),
    Wave(WaveOutputs<'a>),
    /// A non-output band's step; `prefill` is a prefill wave (else AR/verify
    /// rows).
    Edge {
        prefill: bool,
    },
}

impl<'a> StepOutputs<'a> {
    /// The lane step's wide-hidden capture (verify lanes, a prefill wave's
    /// optional capture); a non-output band has none.
    fn wide_hidden(&self) -> Option<&'a GpuTensor> {
        match self {
            Self::Ar(out) => out.wide_hidden,
            Self::Wave(wave) => wave.wide_hidden,
            Self::Edge { .. } => None,
        }
    }

    /// Whether the step is a prefill wave.
    fn prefill(&self) -> bool {
        matches!(self, Self::Wave(_) | Self::Edge { prefill: true })
    }
}

/// The head rows `[shard.base, shard.base + shard.width)` of the resident LM head as a
/// non-owning byte sub-tensor of the same allocation (rows are contiguous, `row_stride` bytes
/// each). Refuses any head whose allocation is not exactly `m * row_stride` bytes (a padded or
/// sidecar-carrying layout has no per-row slice) and any slice that is not 256-byte aligned.
fn lm_head_row_slice(
    full: &WeightRef<'_>,
    shard: VocabShardView,
    vocab: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    let overflow = || invalid("vocabulary shard byte range overflows");
    if full.m != vocab {
        return Err(invalid(format!(
            "vocabulary shard: the LM head has {} rows, the vocabulary is {vocab}",
            full.m
        )));
    }
    if full.rotation.is_some() || full.awq_scale.is_some() {
        return Err(invalid(
            "vocabulary shard: the LM head carries a rotation / scale sidecar",
        ));
    }
    let row_bytes = full.row_stride;
    let total = full.m.checked_mul(row_bytes).ok_or_else(overflow)?;
    if row_bytes == 0 || full.buf.buf.size() != total {
        return Err(invalid(format!(
            "vocabulary shard: the LM head allocation is {} bytes, {} rows of {row_bytes} bytes \
             are {total}",
            full.buf.buf.size(),
            full.m
        )));
    }
    let offset = shard.base.checked_mul(row_bytes).ok_or_else(overflow)?;
    let len = shard.width.checked_mul(row_bytes).ok_or_else(overflow)?;
    if shard
        .base
        .checked_add(shard.width)
        .map_or(true, |end| end > vocab)
        || offset % 256 != 0
    {
        return Err(invalid(format!(
            "vocabulary shard [{}, +{}) is not a 256-byte aligned slice of the {vocab}-row head",
            shard.base, shard.width
        )));
    }
    Ok(GpuTensor {
        buf: full.buf.buf.byte_view(offset, len),
        shape: vec![len / full.dtype.size().max(1)],
        dtype: full.dtype,
    })
}

/// A contiguous row range of the step: one lane, or a group of lanes a
/// once-over-all-rows launch covers.
#[derive(Clone, Copy, Debug)]
struct Segment {
    begin: usize,
    rows: usize,
    /// Lane whose state a projection-only mixer op names (it never reads
    /// it; the op type requires one).
    first_lane: usize,
}

fn lane_segments(rows: &[usize]) -> Vec<Segment> {
    let mut begin = 0;
    rows.iter()
        .enumerate()
        .map(|(lane, &rows)| {
            let segment = Segment {
                begin,
                rows,
                first_lane: lane,
            };
            begin += rows;
            segment
        })
        .collect()
}

/// Consecutive lanes packed into groups of at most `max_rows` rows (a lane
/// wider than that is its own group). `max_rows` is [`LANE_MAX_ROWS`], or
/// the CPU-splice window when a layer's routed experts run on the CPU: a group
/// wider than that would take the GPU grouped read, a different arithmetic
/// from the singleton's CPU decode.
fn group_segments(rows: &[usize], max_rows: usize) -> Vec<Segment> {
    let mut groups: Vec<Segment> = Vec::new();
    let mut begin = 0;
    for (lane, &lane_rows) in rows.iter().enumerate() {
        match groups.last_mut() {
            Some(group) if group.rows + lane_rows <= max_rows => group.rows += lane_rows,
            _ => groups.push(Segment {
                begin,
                rows: lane_rows,
                first_lane: lane,
            }),
        }
        begin += lane_rows;
    }
    groups
}

/// The once-over-all-rows groups a step of decode lanes with these per-lane
/// row counts and wide-verify classes runs ([`StageShare::OnceAllRows`] stages
/// launch once per group). The mesh protocol's `seal_verify` packs with the same
/// rule ([`pack_classed_row_groups`]), so the sealed ledger and the executed group
/// count agree.
pub(crate) fn row_group_count(rows: &[usize], wide: &[bool]) -> usize {
    pack_classed_row_groups(rows, wide).len()
}

/// Whether a lane verifies in the wide pass: `HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY=1`
/// and the lane is an unseeded sampled lane (`plan_wide`, from the request), or the
/// lab all-lanes override is set.
pub fn lane_is_wide(plan_wide: bool) -> bool {
    #[cfg(test)]
    if let Some((flag, all_lanes)) = TEST_WIDE_VERIFY.with(|cell| cell.get()) {
        return lane_is_wide_for(flag, all_lanes, plan_wide);
    }
    lane_is_wide_for(
        hipfire_config::qwen4_ep_sampled_wide_verify(),
        hipfire_config::qwen4_ep_wide_verify_all_lanes(),
        plan_wide,
    )
}

#[cfg(test)]
thread_local! {
    /// Test-only `(flag, all_lanes)` override of the process flags [`lane_is_wide`] reads.
    pub(crate) static TEST_WIDE_VERIFY: std::cell::Cell<Option<(bool, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// [`lane_is_wide`] with the flag values given.
pub const fn lane_is_wide_for(flag: bool, all_lanes: bool, plan_wide: bool) -> bool {
    flag && (plan_wide || all_lanes)
}

/// The decode groups of a step as GPU segments: [`pack_classed_row_groups`] with each
/// group's first row and first lane.
fn classed_segments(rows: &[usize], wide: &[bool]) -> Vec<Segment> {
    segments_of_groups(pack_classed_row_groups(rows, wide))
}

/// Packed row groups as GPU segments (first row, rows, first lane of each).
fn segments_of_groups(groups: Vec<crate::mesh_lane_protocol::RowGroup>) -> Vec<Segment> {
    let mut begin = 0;
    groups
        .into_iter()
        .map(|group| {
            let segment = Segment {
                begin,
                rows: group.rows,
                first_lane: group.first_lane,
            };
            begin += group.rows;
            segment
        })
        .collect()
}

/// Widest all-rows HC op: the wide-verify row bound every HC kernel arm admits
/// ([`rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS`]).
const HC_ALL_ROWS_MAX: usize = rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS;

/// Whether an AR / verify lane step of `group_rows` (the once-over-all-rows
/// groups of at most [`LANE_MAX_ROWS`] rows each) runs its HC stages once over
/// every row (`HIPFIRE_QWEN4_LANE_HC_ALLROWS`) instead of once per group.
/// `flag` is the exact-gfx1201 flag, `ar_step` an AR / verify step (never a
/// prefill wave or a band edge), `eager` no replay recording and no graph
/// capture (the wide few-row limit the HC kernels' all-rows arms run under
/// exists only eagerly), `cpu_splice` a layer with CPU routed experts (its
/// groups are CPU-window sized). The HC read needs `HcRead` and `HcWrite`
/// both [`StageShare::OnceAllRows`].
fn hc_all_rows_applies(
    flag: bool,
    ar_step: bool,
    eager: bool,
    cpu_splice: bool,
    group_rows: &[usize],
    policy: &LaneStagePolicy,
) -> bool {
    let n: usize = group_rows.iter().sum();
    flag && ar_step
        && eager
        && !cpu_splice
        && group_rows.len() > 1
        && group_rows
            .iter()
            .all(|&rows| (1..=LANE_MAX_ROWS).contains(&rows))
        && (LANE_MAX_ROWS + 1..=HC_ALL_ROWS_MAX).contains(&n)
        && policy.share(StageName::HcRead) == StageShare::OnceAllRows
        && policy.share(StageName::HcWrite) == StageShare::OnceAllRows
}

/// Whether an AR / verify lane step also runs every non-MoE row-local stage once over all
/// rows (`HIPFIRE_QWEN4_LANE_ALLROWS`), on top of the all-rows HC segment
/// ([`hc_all_rows_applies`], which it refines: it is only asked for a step that segment
/// admitted). Refused (`false`) for anything the wide 9..=64-row routes of the projections
/// cannot serve exactly as their G x 8 groups' arms do:
///
/// - `wide_scope_rows != 0`: a wide-verify scope is in force, so the projections would take
///   its row-exact twins instead of the wide arms (the lane step runs with no scope);
/// - `gdn_qkv_bf16`: `HIPFIRE_QWEN4_GDN_QKV_BF16` makes the GDN input projection store its
///   qkv rows as BF16 by a row-keyed decision (`mq6_i8_route`) the per-lane GDN core, which
///   routes at the lane's own rows, would not repeat;
/// - `streams_agree` false: the HC stream storage the step's `n` rows route to differs from
///   the singleton's one-row storage the lanes' own stateful stages run on.
fn all_rows_stages_apply(
    flag: bool,
    segment_applies: bool,
    wide_scope_rows: usize,
    gdn_qkv_bf16: bool,
    streams_agree: bool,
) -> bool {
    flag && segment_applies && wide_scope_rows == 0 && !gdn_qkv_bf16 && streams_agree
}

/// `HIPFIRE_QWEN4_LANE_STEP_TRACE=1` (lab, default off): one `[lane-step-trace]` stderr line per
/// lane step with its shape -- rows, groups, where the HC / non-MoE stages and the routed MoE
/// run, MoE calls per layer and the launches emitted per stage -- so a timing job can prove it
/// measured the intended step. Read once.
fn lane_step_trace_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_bool("HIPFIRE_QWEN4_LANE_STEP_TRACE", false)
    });
    *ON
}

/// Why a lane step that asked for `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` keeps its routed MoE on
/// the <= 8-row groups (`Some(reason)`), or `None` when the one all-rows MoE call applies. The
/// step the flag refines is a 9..=64-row AR / verify step the all-rows stage segment
/// admitted (`stages_apply`: flag, eager, no CPU splice, no wide scope, ...); the call is the
/// wide-group MoE over every row, so it is refused wherever a wide-class group of `n` rows
/// would be:
///
/// - `!moe_once`: the policy places the MoE per lane;
/// - `gather_cap` `None`: no EP row gather is installed (the call is the EP gathered forward);
/// - `any_wide_lane`: a wide-class lane (`HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY`) already packs
///   into its own classed group, and classes never share a group;
/// - `n > gather_cap`: `ep_lane_moe_rows`' gather capacity (a rank sizes it to 48 rows by default
///   on exact gfx1201, `HIPFIRE_QWEN4_EP_WIDE_VERIFY=0` restores 8 rows, an explicit `=1` the wide group cap);
/// - `n` above the tightest wide stage limit ([`crate::gpu_forward::ep_wide_stage_limits`]:
///   the 512-slot owner assignment, the 16-bit crossing-tag slot field, the verify window);
/// - `ar_write` (the trunk shard's fused all-reduce HC write carries the step) and `n` above
///   its `AR_HC_ROW_CAP` rows per launch;
/// - an odd per-step MoE call count (`moe_calls`, one per owned layer): the gather buffer
///   alternates per call and a step must end on the buffer it started on
///   ([`ep_lane_check_invocations`]).
fn all_rows_moe_refusal(
    stages_apply: bool,
    moe_once: bool,
    gather_cap: Option<usize>,
    n: usize,
    any_wide_lane: bool,
    ar_write: bool,
    moe_calls: usize,
) -> Option<String> {
    if !stages_apply {
        return Some(
            "the step takes no all-rows stage segment (LANE_ALLROWS / LANE_HC_ALLROWS off, not an eager 9..=64-row multi-group AR / verify step, or a CPU splice / wide scope / GDN qkv BF16 is in force)".into(),
        );
    }
    if !moe_once {
        return Some("the stage policy places the MoE per lane".into());
    }
    let Some(cap) = gather_cap else {
        return Some("no EP row gather is installed (not an EP lane step)".into());
    };
    if any_wide_lane {
        return Some("a wide-class lane keeps its classed group".into());
    }
    if n > cap {
        return Some(format!(
            "{n} rows exceed the {cap}-row EP gather capacity (48 rows by default on exact gfx1201 EP; HIPFIRE_QWEN4_EP_WIDE_VERIFY=0 restores 8, HIPFIRE_QWEN4_EP_WIDE_VERIFY_ROWS sizes it)"
        ));
    }
    let (limit, binding) =
        crate::gpu_forward::ep_wide_group_cap_for(n, crate::gpu_forward::QWEN4_EP_WIDE_TOP_K);
    if limit < n {
        return Some(format!(
            "{n} rows exceed the {limit}-row limit of stage {}",
            binding.map_or("?", |b| b.stage)
        ));
    }
    if ar_write && n > rdna_compute::ar_hc_write::AR_HC_ROW_CAP {
        return Some(format!(
            "{n} rows exceed the trunk shard's {}-row all-reduce HC write launch",
            rdna_compute::ar_hc_write::AR_HC_ROW_CAP
        ));
    }
    if moe_calls == 0 || moe_calls % 2 != 0 {
        return Some(format!(
            "{moe_calls} MoE calls per step would reuse a gather buffer across consecutive steps (the count must be even)"
        ));
    }
    None
}

/// The load-constant facts that decide whether an AR / verify lane step runs its routed MoE
/// ONCE over every row (one `ep_lane_moe_rows` call per gathered layer) or once per row group,
/// and the ONE predicate both sides ask:
///
/// - the executor ([`forward_lanes_scoped`]) builds it per step from its own `Gpu` / forward
///   ([`Self::from_parts`]) and takes `hc_all` / `all_stages` / `exact_scope` / the MoE
///   placement from [`Self::decide`];
/// - the mesh root builds it ONCE from rank 0 at load ([`Self::for_forward`], carried in
///   [`crate::mesh_lane_protocol::StagePolicy::all_rows`]) and seals the ledger's
///   `target_exchanges` from [`Self::moe_all_rows`], so a plan never depends on a count the
///   step reports after the seal. Every rank checks its own rule against the root's at
///   `begin_window`.
///
/// Per-step inputs (not stored): the groups' rows (`group_rows`), whether any lane is a
/// wide-class lane, and the step kind (AR / verify vs a prefill wave or band edge). Everything
/// here is fixed once the load is attached:
///
/// - the four `HIPFIRE_QWEN4_LANE_*` / `GDN_QKV_BF16` flags (`Gpu::flags`, read once);
/// - `wide_verify_rows`: `Gpu::wide_verify_rows` is raised only inside a forward's own
///   set/restore scope, so it is 0 on entry to a lane step;
/// - `eager`: no replay recording, no graph capture, no prefill-invariant scope; nothing in the
///   mesh lane path starts one;
/// - `cpu_splice` (`Qwen4GpuForward::has_cpu_deferred_experts`, fixed at load), `gather_cap`
///   (`Qwen4ForwardEp::gather_rows`, fixed by `attach_ep`), `trunk_ar_write` (this thread's
///   [`crate::trunk_route`] writers, installed once per rank at `attach`), `band_moe_calls` (the
///   band's layer count) and the stage `policy`;
/// - `bf16_min_rows`: the smallest row count at which `Gpu::qwen4_bf16_streams` is true (the
///   F16 WMMA threshold when the arch routes it, else never), so that
///   `qwen4_bf16_streams(n) == qwen4_bf16_streams(1)` is a function of `n` and this constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneAllRowsRule {
    /// `HIPFIRE_QWEN4_LANE_HC_ALLROWS` (`Flags::qwen4_lane_hc_allrows_enabled`).
    pub hc_flag: bool,
    /// `HIPFIRE_QWEN4_LANE_ALLROWS` (`Flags::qwen4_lane_allrows_enabled`).
    pub stages_flag: bool,
    /// `HIPFIRE_QWEN4_LANE_ALLROWS_EXACT` (`Flags::qwen4_lane_allrows_exact_enabled`).
    pub exact_flag: bool,
    /// `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` (`Flags::qwen4_lane_allrows_moe_enabled`).
    pub moe_flag: bool,
    /// `HIPFIRE_QWEN4_GDN_QKV_BF16` (`Flags::qwen4_gdn_qkv_bf16_enabled`).
    pub gdn_qkv_bf16: bool,
    /// `Gpu::wide_verify_rows` on entry to the step.
    pub wide_verify_rows: usize,
    /// No replay recording, no graph capture, no prefill-invariant scope.
    pub eager: bool,
    /// A layer with CPU routed experts (its groups are CPU-window sized).
    pub cpu_splice: bool,
    /// Smallest `n` with `Gpu::qwen4_bf16_streams(n)`; `usize::MAX` when none.
    pub bf16_min_rows: usize,
    /// The EP row gather's capacity (`None` outside an EP lane step).
    pub gather_cap: Option<usize>,
    /// The trunk shard's fused all-reduce HC write carries the step.
    pub trunk_ar_write: bool,
    /// MoE calls per step the gather buffer parity is checked on (the band's layers).
    pub band_moe_calls: usize,
    /// The stage shares (HC read / write, MoE) the rule reads.
    pub policy: LaneStagePolicy,
}

/// What [`LaneAllRowsRule::decide`] decided for one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllRowsDecision {
    /// Every HC stage runs once over all rows ([`hc_all_rows_applies`]).
    pub hc_all: bool,
    /// Every other non-MoE once-over-all-rows stage does too ([`all_rows_stages_apply`]).
    pub all_stages: bool,
    /// The all-rows steps run inside the wide few-row verify scope of `n` rows (and the
    /// row-exact lane scope, `Gpu::lane_row_exact_scope`): the lab
    /// `HIPFIRE_QWEN4_LANE_ALLROWS_EXACT` flag or any row-exact-class lane in the step.
    pub exact_scope: bool,
    /// Why a step that asked for the all-rows MoE keeps its groups ([`all_rows_moe_refusal`]).
    pub moe_refusal: Option<String>,
    /// The routed MoE is ONE call per layer over every row.
    pub moe_all: bool,
}

impl LaneAllRowsRule {
    /// A rule that never takes the all-rows MoE: every non-EP path (PP, single, scripted roots)
    /// keeps today's one-MoE-call-per-group count.
    pub fn never() -> Self {
        Self {
            hc_flag: false,
            stages_flag: false,
            exact_flag: false,
            moe_flag: false,
            gdn_qkv_bf16: false,
            wide_verify_rows: 0,
            eager: false,
            cpu_splice: false,
            bf16_min_rows: usize::MAX,
            gather_cap: None,
            trunk_ar_write: false,
            band_moe_calls: 0,
            policy: LaneStagePolicy::production(),
        }
    }

    /// The rule of a `gpu` running a band of `band_moe_calls` MoE layers under `policy`, with
    /// the forward's EP row gather capacity `gather_cap` and CPU-splice state `cpu_splice`.
    pub(crate) fn from_parts(
        gpu: &Gpu,
        gather_cap: Option<usize>,
        cpu_splice: bool,
        band_moe_calls: usize,
        policy: &LaneStagePolicy,
    ) -> Self {
        Self {
            hc_flag: gpu.flags.qwen4_lane_hc_allrows_enabled(),
            stages_flag: gpu.flags.qwen4_lane_allrows_enabled(),
            exact_flag: gpu.flags.qwen4_lane_allrows_exact_enabled(),
            moe_flag: gpu.flags.qwen4_lane_allrows_moe_enabled(),
            gdn_qkv_bf16: gpu.flags.qwen4_gdn_qkv_bf16_enabled(),
            wide_verify_rows: gpu.wide_verify_rows,
            eager: !(gpu.replay.is_recording() || gpu.graphs.capture_mode)
                && !gpu.prefill_invariant_scope,
            cpu_splice,
            bf16_min_rows: if gpu.qwen4_bf16_streams(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS)
            {
                rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
            } else {
                usize::MAX
            },
            gather_cap,
            trunk_ar_write: crate::trunk_route::attn_writer().is_some()
                || crate::trunk_route::ffn_writer().is_some(),
            band_moe_calls,
            policy: *policy,
        }
    }

    /// The rule of `fwd` on `gpu` under `policy` (what the mesh root captures from rank 0 and
    /// every rank checks against it). An EP load whose row gather is not installed errors, as
    /// the lane step does.
    pub(crate) fn for_forward(
        gpu: &Gpu,
        fwd: &Qwen4GpuForward,
        policy: &LaneStagePolicy,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let gather_cap = ep_lane(fwd.ep.as_ref(), &fwd.scratch)?.map(|ep| ep.rows_cap);
        Ok(Self::from_parts(
            gpu,
            gather_cap,
            fwd.has_cpu_deferred_experts(),
            fwd.band_scope().band.len(),
            policy,
        ))
    }

    /// The placement of every all-rows piece of a step: `ar_step` (an AR / verify step, never a
    /// prefill wave or a band edge), the once-over-all-rows `group_rows` of the packed groups
    /// whether any lane is wide-class and whether any lane is row-exact class
    /// (`any_exact_lane`, which alone turns on `exact_scope` for a step that takes the all-rows
    /// stages; no other part of the decision reads it). Pure: the single predicate behind the executor's
    /// `hc_all` / `all_stages` / `exact_scope` / `moe_all` and the root's planned exchanges.
    pub fn decide(
        &self,
        ar_step: bool,
        group_rows: &[usize],
        any_wide_lane: bool,
        any_exact_lane: bool,
    ) -> AllRowsDecision {
        let n: usize = group_rows.iter().sum();
        let hc_all = hc_all_rows_applies(
            self.hc_flag,
            ar_step,
            self.eager,
            self.cpu_splice,
            group_rows,
            &self.policy,
        );
        let all_stages = all_rows_stages_apply(
            self.stages_flag,
            hc_all,
            if self.exact_flag {
                0
            } else {
                self.wide_verify_rows
            },
            self.gdn_qkv_bf16,
            // `qwen4_bf16_streams(n) == qwen4_bf16_streams(1)`.
            (n >= self.bf16_min_rows) == (1 >= self.bf16_min_rows),
        );
        let exact_scope = (self.exact_flag || any_exact_lane) && all_stages;
        let moe_refusal = if self.moe_flag {
            all_rows_moe_refusal(
                all_stages,
                self.policy.share(StageName::Moe) == StageShare::OnceAllRows,
                self.gather_cap,
                n,
                any_wide_lane,
                self.trunk_ar_write,
                self.band_moe_calls,
            )
        } else {
            None
        };
        let moe_all = self.moe_flag && moe_refusal.is_none();
        AllRowsDecision {
            hc_all,
            all_stages,
            exact_scope,
            moe_refusal,
            moe_all,
        }
    }

    /// Whether an AR / verify step of the packed groups `group_rows` runs the routed MoE once
    /// over all rows (one call per gathered layer) rather than once per group. The root's
    /// ledger plans `target_exchanges` from this; the executor's `moe_all` is the same
    /// [`Self::decide`].
    pub fn moe_all_rows(&self, group_rows: &[usize], any_wide_lane: bool) -> bool {
        self.decide(true, group_rows, any_wide_lane, false).moe_all
    }
}

/// What the all-rows segment of an AR / verify lane step takes: the HC stages
/// (`HIPFIRE_QWEN4_LANE_HC_ALLROWS`) and, with `stages`, every other non-MoE
/// once-over-all-rows stage (`HIPFIRE_QWEN4_LANE_ALLROWS`). The routed MoE keeps its
/// <= 8-row groups unless `moe` (`HIPFIRE_QWEN4_LANE_ALLROWS_MOE`): then it is ONE call per
/// layer over every row on the MoE all-rows segment (the one after the HC one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AllRowsPlan {
    /// Whether the attention (`[0]`) / MLP (`[1]`) HC write runs all-rows; a write a TP2
    /// all-reduce carries stays per group (unless `moe`, which takes them all-rows too).
    hc_writes: [bool; 2],
    /// Every other non-MoE once-over-all-rows operation runs all-rows too.
    stages: bool,
    /// The routed MoE is one call over every row (needs `stages`).
    moe: bool,
}

/// Where a piece of a once-over-all-rows run is emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// Once per [`LANE_MAX_ROWS`]-row group, segment-major (today's emission).
    Groups,
    /// One segment over every row of the step (rows `[0, n)`): the all-rows segment.
    AllRows,
    /// The routed MoE as one call over every row: the segment after the all-rows one (the
    /// same rows, bound to the descriptions of `n` rows the wide-group MoE binds).
    AllRowsMoe,
    /// The PLE key / value projections over every row of the step: the PLE project segment
    /// (rows `[0, n)`), or the groups when the step has none.
    PleProject,
}

/// Split one run of consecutive once-over-all-rows operations into pieces of
/// consecutive operations with the same [`Placement`]. `all_rows`: `None` keeps the whole
/// run in groups (today's emission); `Some(plan)` places every HC read, the HC write the
/// plan says, (`plan.stages`) every operation of a stage other than the routed MoE over all
/// rows, and (`plan.moe`) the routed MoE on its own all-rows segment. Operations keep their
/// order.
fn run_pieces(
    run: &[(LaneOp, StageName)],
    all_rows: Option<AllRowsPlan>,
) -> Vec<(std::ops::Range<usize>, Placement)> {
    let place = |op: LaneOp, stage: StageName| match (op, all_rows) {
        (LaneOp::PleProject, _) => Placement::PleProject,
        (_, None) => Placement::Groups,
        (LaneOp::HcRead { .. }, Some(_)) => Placement::AllRows,
        (LaneOp::HcWrite { mlp, .. }, Some(plan)) if plan.hc_writes[usize::from(mlp)] => {
            Placement::AllRows
        }
        (LaneOp::HcWrite { .. }, Some(_)) => Placement::Groups,
        (_, Some(plan)) if plan.moe && plan.stages && stage == StageName::Moe => {
            Placement::AllRowsMoe
        }
        (_, Some(plan)) if plan.stages && stage != StageName::Moe => Placement::AllRows,
        _ => Placement::Groups,
    };
    let mut pieces: Vec<(std::ops::Range<usize>, Placement)> = Vec::new();
    for (i, &(op, stage)) in run.iter().enumerate() {
        let placement = place(op, stage);
        match pieces.last_mut() {
            Some((range, last)) if *last == placement => range.end = i + 1,
            _ => pieces.push((i..i + 1, placement)),
        }
    }
    pieces
}

/// The `(segment, ops)` items of one same-share run, in emission order: `run`
/// starts at program index `start`, `segs` are the segments its share places
/// it over (every group, or every lane), `all_rows` is `Some((all-rows segment,
/// plan))` for a once-over-all-rows run of an all-rows step (the MoE all-rows segment, when
/// the plan has one, is `all-rows segment + 1`), `ple_seg` the PLE project segment (every
/// row of the step) when the step projects PLE once. Every piece is emitted
/// segment-major over its own segments, pieces in order; without `all_rows` that is the
/// whole run over `segs` (today's emission).
fn run_items(
    start: usize,
    run: &[(LaneOp, StageName)],
    segs: std::ops::Range<usize>,
    all_rows: Option<(usize, AllRowsPlan)>,
    ple_seg: Option<usize>,
) -> Vec<(usize, std::ops::Range<usize>)> {
    let mut items = Vec::new();
    for (piece, placement) in run_pieces(run, all_rows.map(|(_, plan)| plan)) {
        let ops = start + piece.start..start + piece.end;
        match (placement, all_rows) {
            (Placement::AllRows, Some((seg, _))) => items.push((seg, ops)),
            (Placement::AllRowsMoe, Some((seg, _))) => items.push((seg + 1, ops)),
            (Placement::PleProject, _) => match ple_seg {
                Some(seg) => items.push((seg, ops)),
                None => items.extend(segs.clone().map(|i| (i, ops.clone()))),
            },
            _ => items.extend(segs.clone().map(|i| (i, ops.clone()))),
        }
    }
    items
}

/// Opens the wide few-row scope (`Gpu::wide_verify_rows`, so `few_row_limit`
/// admits `rows`) around exactly the all-rows steps of a lane program: the
/// interpreter announces every step before dispatching it
/// ([`MoeStepHooks::before_step`]), and the scope is `rows` for an all-rows step and the
/// step's prior value for every other step, so every other stage keeps the routes of its own
/// (at most [`LANE_MAX_ROWS`]-row) groups. Without `exact` the all-rows steps are the HC
/// read / write of `rows` rows; with `exact` (`HIPFIRE_QWEN4_LANE_ALLROWS_EXACT`, or any row-exact-class
/// lane in the step, `HIPFIRE_QWEN4_LANE_ROW_EXACT`) every step
/// of `rows` rows is: the GDN / QSA mixer steps (a lane's own rows are at most
/// [`LANE_MAX_ROWS`] < `rows`, so only the all-rows projections match) and, with `moe`, the
/// all-rows MoE call.
struct HcAllRowsScope {
    rows: usize,
    prior: usize,
    exact: bool,
    moe: bool,
}

impl MoeStepHooks for HcAllRowsScope {
    fn before_moe(&mut self, _gpu: &mut Gpu) -> Result<(), hipfire_dispatch::types::DispatchError> {
        Ok(())
    }
    fn after_moe(&mut self, _gpu: &mut Gpu) -> Result<(), hipfire_dispatch::types::DispatchError> {
        Ok(())
    }
    fn before_step(
        &mut self,
        gpu: &mut Gpu,
        step: &Step<'_>,
    ) -> Result<(), hipfire_dispatch::types::DispatchError> {
        let all_rows = match step {
            Step::HyperRead(op) => op.rows == self.rows,
            Step::HyperWrite(op) => op.rows == self.rows,
            Step::GatedDeltaNet(op) => self.exact && op.rows == self.rows,
            Step::IndexedAttention(op) => self.exact && op.rows == self.rows,
            Step::Moe(_) => self.exact && self.moe,
            _ => false,
        };
        gpu.set_wide_verify_rows(if all_rows { self.rows } else { self.prior });
        Ok(())
    }
}

/// The row views one segment's operations bind: every row-indexed scratch
/// tensor offset to the segment's first row, extending to the end of its
/// allocation (exactly what a singleton forward of these rows sees from row
/// zero). Scratch an operation consumes internally stays at row zero.
struct SegViews {
    streams: GpuTensor,
    hc_normalized: GpuTensor,
    hc_low: GpuTensor,
    hc_up: GpuTensor,
    hc_mixed: GpuTensor,
    hc_gates: GpuTensor,
    projection: GpuTensor,
    projection2: GpuTensor,
    gdn_a: GpuTensor,
    gdn_b: GpuTensor,
    gdn_gate: GpuTensor,
    gdn_beta: GpuTensor,
    gdn_recurrent_output: GpuTensor,
    gdn_z: GpuTensor,
    gdn_output: GpuTensor,
    qsa_index: GpuTensor,
    qsa_qgate: GpuTensor,
    qsa_k: GpuTensor,
    qsa_v: GpuTensor,
    qsa_output: GpuTensor,
    attention_output: GpuTensor,
    ffn_output: GpuTensor,
    router_logits: GpuTensor,
    ple_rows: GpuTensor,
    ple_query: GpuTensor,
    ple_key: GpuTensor,
    ple_value: GpuTensor,
    ple_gated: GpuTensor,
    ple_normed: GpuTensor,
    ple_output: GpuTensor,
    embeddings: GpuTensor,
    embedding_rot: GpuTensor,
    token_ids: GpuTensor,
    logits: GpuTensor,
    top1: GpuTensor,
}

/// Segment views of the scratch.  Every row-banded field comes from
/// [`Qwen4GpuForwardScratch::segment_view`]: a dedicated field is addressed at
/// its logical row width, a field of an alias set (compact scratch) at the row
/// pitch of the allocation that owns its storage, so a segment occupies the
/// same rows of the owner whichever member of the set a phase uses.
fn seg_rows(
    scratch: &Qwen4GpuForwardScratch,
    field: ScratchField,
    seg: Segment,
    logical_width: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    // A field the band's scratch plan does not own is a zero-element slot
    // (GDN / QSA scratch of a band without that layer kind, token, embedding
    // and PLE scratch off the input band): an empty view. The whole-model
    // plan owns every field, so it never takes this arm.
    if scratch.field_tensor(field).numel() == 0 {
        return Ok(view(&scratch.streams, 0, 0));
    }
    scratch.segment_view(field, seg.begin, seg.rows, logical_width)
}

/// A flat (non-banded) view of `tensor`; an absent (zero-element) slot of a
/// band's scratch plan is an empty view.
fn seg_flat(
    scratch: &Qwen4GpuForwardScratch,
    tensor: &GpuTensor,
    offset: usize,
    len: usize,
) -> GpuTensor {
    if tensor.numel() == 0 {
        view(&scratch.streams, 0, 0)
    } else {
        view(tensor, offset, len)
    }
}

/// The HC stream rows of one segment when the step stores its streams as
/// BF16 bits: rows are packed at 2 B per element (the HC kernels address row
/// `i` of a view at `2 * i * wide` bytes, the wide-hidden capture and a band
/// edge read `rows * wide * 2` contiguous bytes), so segment `begin` starts at
/// byte `2 * begin * wide`. The F32-typed field would place it at
/// `4 * begin * wide`, leaving a gap before every lane but the first.
fn bf16_stream_rows(
    scratch: &Qwen4GpuForwardScratch,
    seg: Segment,
    wide: usize,
) -> Result<GpuTensor, Qwen4GpuForwardError> {
    let tensor = scratch.field_tensor(ScratchField::Streams);
    let numel = tensor.numel();
    let offset = packed_bf16_offset(seg.begin, wide, tensor.dtype.size(), numel)
        .ok_or_else(|| invalid("lane BF16 stream rows start past their scratch tensor"))?;
    Ok(tensor.sub_offset(offset, numel - offset))
}

/// Offset, in elements of a `container_bytes`-wide tensor of `numel`
/// elements, of packed BF16 row `begin` (`wide` BF16 elements per row, 2 B
/// each); `None` when it is not element aligned or starts past the tensor.
fn packed_bf16_offset(
    begin: usize,
    wide: usize,
    container_bytes: usize,
    numel: usize,
) -> Option<usize> {
    let bytes = begin.checked_mul(wide)?.checked_mul(DType::BF16.size())?;
    if container_bytes == 0 || bytes % container_bytes != 0 {
        return None;
    }
    Some(bytes / container_bytes).filter(|offset| *offset <= numel)
}

impl SegViews {
    fn new(
        scratch: &Qwen4GpuForwardScratch,
        dims: Qwen4ProgramDims,
        vocab: usize,
        out: Option<&LaneOutputs<'_>>,
        seg: Segment,
        ple_embed_dim: usize,
        bf16_streams: bool,
        ple_normed_field: ScratchField,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let b = seg.begin;
        let wide = dims.wide();
        let hidden = dims.hidden;
        let value_heads = dims.linear_num_value_heads;
        // A prefill wave runs its head per lane through its own views: its
        // segment views carry no logits (an empty view of a scratch tensor).
        let (logits, top1) = match out {
            Some(out) => (
                view(out.logits, b * vocab, seg.rows * vocab),
                view(
                    out.top1,
                    b * std::mem::size_of::<i32>(),
                    seg.rows * std::mem::size_of::<i32>(),
                ),
            ),
            None => (view(&scratch.streams, 0, 0), view(&scratch.streams, 0, 0)),
        };
        use ScratchField as F;
        let rows = |field: F, width: usize| seg_rows(scratch, field, seg, width);
        Ok(Self {
            streams: if bf16_streams {
                bf16_stream_rows(scratch, seg, wide)?
            } else {
                rows(F::Streams, wide)?
            },
            hc_normalized: rows(F::HcNormalized, wide)?,
            hc_low: rows(F::HcLow, dims.hc_lowrank)?,
            hc_up: rows(F::HcUp, wide)?,
            hc_mixed: rows(F::HcMixed, hidden)?,
            hc_gates: rows(F::HcGates, dims.hc_count)?,
            projection: rows(F::Projection, dims.gdn_qkv())?,
            projection2: rows(F::Projection2, dims.gdn_qkv())?,
            gdn_a: rows(F::GdnA, value_heads)?,
            gdn_b: rows(F::GdnB, value_heads)?,
            gdn_gate: rows(F::GdnGate, value_heads)?,
            gdn_beta: rows(F::GdnBeta, value_heads)?,
            gdn_recurrent_output: rows(F::GdnRecurrentOutput, dims.gdn_value())?,
            gdn_z: rows(F::GdnZ, dims.gdn_value())?,
            gdn_output: rows(F::GdnOutput, dims.gdn_value())?,
            qsa_index: rows(F::QsaIndex, dims.index_width())?,
            qsa_qgate: rows(F::QsaQgate, 2 * dims.q_width())?,
            qsa_k: rows(F::QsaK, dims.kv_width())?,
            qsa_v: rows(F::QsaV, dims.kv_width())?,
            qsa_output: rows(F::QsaOutput, dims.q_width())?,
            attention_output: rows(F::AttentionOutput, hidden)?,
            ffn_output: rows(F::FfnOutput, hidden)?,
            router_logits: matrix_view(&scratch.router_logits, seg.rows, router_columns(dims))?,
            ple_rows: rows(F::PleRows, ple_embed_dim)?,
            ple_query: rows(F::PleQuery, wide)?,
            ple_key: rows(F::PleKey, wide)?,
            ple_value: rows(F::PleValue, hidden)?,
            ple_gated: rows(F::PleGated, wide)?,
            ple_normed: rows(ple_normed_field, wide)?,
            ple_output: rows(F::PleOutput, wide)?,
            embeddings: seg_flat(scratch, &scratch.embeddings, b * hidden, seg.rows * hidden),
            embedding_rot: seg_flat(
                scratch,
                &scratch.embedding_rot,
                b * hidden,
                seg.rows * hidden,
            ),
            token_ids: seg_flat(
                scratch,
                &scratch.token_ids,
                b * std::mem::size_of::<i32>(),
                seg.rows * std::mem::size_of::<i32>(),
            ),
            logits,
            top1,
        })
    }

    /// The sealed-MoE / final-hyper scratch description of this segment.
    fn layer_scratch<'a>(&'a self, s: &'a Qwen4GpuForwardScratch) -> Qwen4LayerScratch<'a> {
        Qwen4LayerScratch {
            streams: &self.streams,
            rotation: &s.rotation,
            hc_normalized: &self.hc_normalized,
            hc_low: &self.hc_low,
            hc_up: &self.hc_up,
            hc_mixed: &self.hc_mixed,
            hc_gates: &self.hc_gates,
            projection: &self.projection,
            projection2: &self.projection2,
            gdn_a: &self.gdn_a,
            gdn_b: &self.gdn_b,
            gdn_gate: &self.gdn_gate,
            gdn_beta: &self.gdn_beta,
            gdn_recurrent_output: &self.gdn_recurrent_output,
            gdn_z: &self.gdn_z,
            gdn_output: &self.gdn_output,
            qsa_index: &self.qsa_index,
            qsa_qgate: &self.qsa_qgate,
            qsa_k: &self.qsa_k,
            qsa_v: &self.qsa_v,
            qsa_output: &self.qsa_output,
            attention_output: &self.attention_output,
            ffn_output: &self.ffn_output,
            moe_shared_output: &s.moe_shared_output,
            moe_router_logits: &self.router_logits,
            moe_x_rot: &s.moe_x_rot,
            moe_gate_up: &s.moe_gate_up,
            moe_scalar: &s.moe_scalar,
            moe_gate: &s.moe_gate,
            moe_up: &s.moe_up,
            moe_hidden: &s.moe_hidden,
            moe_gate_batch: &s.moe_gate_batch,
            moe_up_batch: &s.moe_up_batch,
            moe_rot_batch: &s.moe_rot_batch,
            moe_topk_indices: &s.moe_topk_indices,
            moe_topk_weights: &s.moe_topk_weights,
            moe_down_expanded: &s.moe_down_expanded,
            moe_expert_token_counts: &s.moe_expert_token_counts,
            moe_expert_offsets: &s.moe_expert_offsets,
            moe_sorted_slot_index: &s.moe_sorted_slot_index,
            moe_expert_tile_ids: &s.moe_expert_tile_ids,
            moe_inverse_perm: &s.moe_inverse_perm,
            moe_y_gate_up_grouped: &s.moe_y_gate_up_grouped,
            moe_y_down_grouped: &s.moe_y_down_grouped,
        }
    }
}

/// One operation of the layer program, placed by its stage's share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneOp {
    HcRead {
        layer: usize,
        mlp: bool,
    },
    HcWrite {
        layer: usize,
        mlp: bool,
    },
    /// Every lane's PLE key / value projections in one row-exact launch each, ahead of the
    /// per-lane [`LaneOp::PleConv`] tails that consume their rows (see [`with_ple_project`]).
    PleProject,
    PleConv,
    Gdn {
        layer: usize,
        slot: usize,
        phase: MixerPhase,
    },
    Qsa {
        layer: usize,
        slot: usize,
        phase: MixerPhase,
    },
    Moe {
        layer: usize,
    },
}

/// The layer program's operations in order over one band's `layers`
/// (`(absolute layer, kind)`, ascending), each with the stage that places
/// it. `ple_layer` is the PLE layer when the band owns it (the input band).
/// GDN / QSA slots count from zero over the band's own layers, exactly the
/// state map's compact local slots; the whole-model band is the singleton
/// program. A mixer whose projections are both per lane stays the whole op
/// (the singleton's op itself); otherwise it splits into phases.
fn lane_program(
    layers: impl ExactSizeIterator<Item = (usize, LayerType)>,
    policy: &LaneStagePolicy,
    ple_layer: Option<usize>,
) -> Vec<(LaneOp, StageName)> {
    let mut program = Vec::with_capacity(layers.len() * 8 + 1);
    let mut gdn_slot = 0usize;
    let mut qsa_slot = 0usize;
    let mixer_phases =
        |in_stage: StageName, out_stage: StageName| -> Vec<(MixerPhase, StageName)> {
            let core = match in_stage {
                StageName::GdnInProj => StageName::GdnCore,
                _ => StageName::QsaCore,
            };
            if policy.share(in_stage).per_lane() && policy.share(out_stage).per_lane() {
                vec![(MixerPhase::Whole, core)]
            } else {
                vec![
                    (MixerPhase::InProj, in_stage),
                    (MixerPhase::Core, core),
                    (MixerPhase::OutProj, out_stage),
                ]
            }
        };
    for (layer, kind) in layers {
        if ple_layer == Some(layer) {
            program.push((LaneOp::PleConv, StageName::PleConv));
        }
        program.push((LaneOp::HcRead { layer, mlp: false }, StageName::HcRead));
        match kind {
            LayerType::LinearAttention => {
                for (phase, stage) in mixer_phases(StageName::GdnInProj, StageName::GdnOutProj) {
                    program.push((
                        LaneOp::Gdn {
                            layer,
                            slot: gdn_slot,
                            phase,
                        },
                        stage,
                    ));
                }
                gdn_slot += 1;
            }
            LayerType::FullAttention => {
                for (phase, stage) in mixer_phases(StageName::QsaInProj, StageName::QsaOutProj) {
                    program.push((
                        LaneOp::Qsa {
                            layer,
                            slot: qsa_slot,
                            phase,
                        },
                        stage,
                    ));
                }
                qsa_slot += 1;
            }
        }
        program.push((LaneOp::HcWrite { layer, mlp: false }, StageName::HcWrite));
        program.push((LaneOp::HcRead { layer, mlp: true }, StageName::HcRead));
        program.push((LaneOp::Moe { layer }, StageName::Moe));
        program.push((LaneOp::HcWrite { layer, mlp: true }, StageName::HcWrite));
    }
    program
}

/// [`lane_program`] over the layers `bundle`'s weights own (its band scope).
fn layer_program(
    bundle: &Qwen4Bundle,
    policy: &LaneStagePolicy,
    ple_layer: Option<usize>,
) -> Result<Vec<(LaneOp, StageName)>, Qwen4GpuForwardError> {
    let band = bundle.weights.band_scope().band.clone();
    for layer in band.clone() {
        if bundle.weights.layer(layer).is_none() {
            return Err(invalid(format!("lane layer {layer} is not owned")));
        }
    }
    // An owned layer's kind is the config's (`build_layer_refs_scoped`).
    let kinds = &bundle.config.layer_types;
    Ok(lane_program(
        band.map(|layer| (layer, kinds[layer])),
        policy,
        ple_layer,
    ))
}

/// Split the PLE conv's key / value projections out of a lane program: a
/// [`LaneOp::PleProject`] ([`StageName::PleProject`], once over every row) goes right before
/// the program's [`LaneOp::PleConv`], which then runs only its per-lane gate / norm /
/// convolution tail (`GroupedDepthwiseOp::projected`) on the rows it projected. A program
/// without the PLE layer (a band that is not the input band) is unchanged.
fn with_ple_project(program: &mut Vec<(LaneOp, StageName)>) {
    if let Some(at) = program.iter().position(|&(op, _)| op == LaneOp::PleConv) {
        program.insert(at, (LaneOp::PleProject, StageName::PleProject));
    }
}

/// QSA lengths a lane's state commits after the step (the singleton's
/// post-forward commit): `(lane, qsa slot, full, raw, pooled, selected, position)`.
type QsaCommit = (usize, usize, usize, usize, usize, usize, usize);

/// One exact trunk step over lanes (see the module docs): AR rows and MTP
/// verify blocks. Row `r` of the outputs belongs to the lanes concatenated in
/// slice order. On success every lane's host state is published exactly as
/// the singleton forward publishes it (QSA lengths, PLE history, the armed
/// ring's live slot, `position += rows`); a verify lane's PLE convolution
/// input rows stay in the forward scratch at the lane's first row until the
/// next forward ([`Qwen4Bundle::lane_ple_normed_rows`]). On failure nothing
/// host-side is published and the caller must treat the lanes' device state
/// as untrusted.
pub(crate) fn forward_lanes(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &LaneOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    // Only the output band has a head: another band's `out` is never read or
    // written (its views may be placeholders), and it takes no capture.
    let outputs = if bundle.weights.band_scope().output {
        StepOutputs::Ar(out)
    } else if out.wide_hidden.is_some() {
        return Err(invalid(
            "only the output band captures the lane wide hidden",
        ));
    } else {
        StepOutputs::Edge { prefill: false }
    };
    forward_lanes_with(gpu, bundle, lanes, &outputs, policy)
}

/// [`forward_lanes`] for a band that does not own the output: no head
/// outputs. On success the band's lane states are published as always and
/// the band's HC streams for the step's rows are left in the forward
/// scratch's `streams` rows `[0, total_rows)` in lane order
/// ([`Qwen4Bundle::lane_edge_streams`]). Refuses the output band.
pub(crate) fn forward_lanes_edge(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    forward_lanes_with(
        gpu,
        bundle,
        lanes,
        &StepOutputs::Edge { prefill: false },
        policy,
    )
}

fn forward_lanes_with(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &StepOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let mut forward = bundle
        .execution
        .take()
        .ok_or_else(|| invalid("Qwen4 forward resources are not attached"))?;
    let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    // The lane step's row-local launchers may take the gfx1151 PM row
    // kernels (`HIPFIRE_FN_LANE_PM_ROWS`, column-identical to the 1-row
    // incumbents); every other forward keeps the incumbents.
    let rows_scope = std::mem::replace(&mut gpu.lane_rows_scope, true);
    // A wide-class group (`HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY`) runs with the
    // verify scope CLOSED: its rows take the ordinary routes of their width
    // (grouped expert-union MoE, prefill-class dense), never the row-exact twins.
    let result = forward_lanes_scoped(gpu, &mut forward, bundle, lanes, out, policy);
    gpu.lane_rows_scope = rows_scope;
    gpu.qwen4_scope = scope;
    bundle.execution = Some(forward);
    result
}

/// One exact prefill wave over `lanes` (see the module docs): the lane step
/// with prefill lanes, run under the invariant prefill profile when
/// `HIPFIRE_PREFILL_PM_INVARIANT` is set on gfx1151. QSA arenas of every
/// lane must already be mapped through its end position
/// (`Qwen4Bundle::prefill_wave` maps them). On success every lane's host
/// state is published exactly as the singleton forward publishes it; on
/// failure nothing host-side is published and the caller must treat the
/// lanes' device state as untrusted.
///
/// `wide_hidden`: the optional capture of every row's final HC streams (F32,
/// `total_rows * hc_count * hidden`, lanes concatenated), output band only.
///
/// On an EP rank the wave is the joint EP prefill ([`forward_wave_ep`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_wave(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [WaveLane<'_>],
    head: WaveHead,
    logits: &GpuTensor,
    top1: Option<&GpuTensor>,
    wide_hidden: Option<&GpuTensor>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    // Only the output band has a head (see `forward_lanes`).
    let outputs = if bundle.weights.band_scope().output {
        StepOutputs::Wave(WaveOutputs {
            head,
            logits,
            top1,
            wide_hidden,
        })
    } else if wide_hidden.is_some() {
        return Err(invalid(
            "only the output band captures the wave wide hidden",
        ));
    } else {
        StepOutputs::Edge { prefill: true }
    };
    forward_wave_with(gpu, bundle, lanes, &outputs, policy)
}

fn forward_wave_with(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [WaveLane<'_>],
    out: &StepOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    let mut forward = bundle
        .execution
        .take()
        .ok_or_else(|| invalid("Qwen4 forward resources are not attached"))?;
    let mut rows: Vec<LaneRows<'_>> = lanes
        .iter_mut()
        .map(|lane| LaneRows {
            state: &mut *lane.state,
            tokens: lane.tokens,
            kind: LaneRowKind::Prefill,
            wide: false,
            // A prefill wave never takes the all-rows segment.
            exact: false,
        })
        .collect();
    let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    let invariant =
        rdna_compute::prefill_invariant::prefill_pm_invariant_enabled() && gpu.arch == "gfx1151";
    let invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
    let result = if forward.ep.is_some() {
        forward_wave_ep(gpu, &mut forward, bundle, &mut rows, out)
    } else {
        forward_lanes_scoped(gpu, &mut forward, bundle, &mut rows, out, policy)
    };
    gpu.prefill_invariant_scope = invariant;
    gpu.qwen4_scope = scope;
    bundle.execution = Some(forward);
    result
}

/// The exact EP prefill wave ([`crate::gpu_forward::ep_wave`] has the design): the lane
/// step over prefill lanes with the MoE ONCE per layer over every lane's rows on the
/// singleton's FULL-TABLE route (sub-512 routing kernels), the dense row-local stages once over
/// all rows too unless `HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT=0` (every other stage per lane;
/// the caller's `policy` is replaced by [`ep_wave_policy`]). Admission is decided from host
/// facts before any launch, on every rank identically; a refusal is an `ep wave refused:`
/// error and nothing ran.
fn forward_wave_ep(
    gpu: &mut Gpu,
    fwd: &mut Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &StepOutputs<'_>,
) -> Result<(), Qwen4GpuForwardError> {
    let lane_rows: Vec<usize> = lanes.iter().map(|lane| lane.tokens.len()).collect();
    // Joint dense stages only when every lane has at least the prefill lane WMMA floor of rows
    // (65 = every lane wider than the few-row window, or the LAB
    // `HIPFIRE_QWEN4_PREFILL_LWMMA_WAVE_FLOOR`); lanes below it run their row-local stages per lane
    // (`ep_wave_joint_for_lanes`). A wave is a prefill forward by construction.
    let floor = gpu.flags.qwen4_prefill_lwmma_floor_knob();
    let policy = ep_wave_policy_for(ep_wave_joint_for_lanes(
        ep_wave_joint_enabled(),
        &lane_rows,
        floor,
    ));
    let total: usize = lane_rows.iter().sum();
    let facts = fwd.ep_wave_facts(gpu, &bundle.config, &lane_rows);
    if let Some(why) = ep_wave_refusal(&facts) {
        return Err(invalid(why));
    }
    // Admitted: the route every lane's own tile takes (decided by the refusal's own rule).
    let route = ep_wave_route(&facts).map_err(invalid)?;
    // `Qwen4Bundle::with_lane` clamps a lane state's F16 arenas to its committed
    // lengths before its tile; the wave's lanes enter the same way.
    for lane in lanes.iter_mut() {
        lane.state.clamp_f16_valid();
    }
    // The routing scopes are the lanes' own (the widest lane's rows), never the total's.
    let scope_rows = ep_wave_scope_rows(&lane_rows);
    fwd.ep_wave_run(gpu, total, scope_rows, &lane_rows, route, |fwd, gpu| {
        forward_lanes_scoped(gpu, fwd, bundle, lanes, out, &policy)
    })
}

fn forward_lanes_scoped(
    gpu: &mut Gpu,
    fwd: &mut Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    lanes: &mut [LaneRows<'_>],
    out: &StepOutputs<'_>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    // A decode-share load runs lane steps on the resident bank only (no prefill
    // bank is ever swapped in across a call boundary).
    fwd.share_require_resident(gpu)?;
    // Expert parallelism: an AR / verify lane step runs the singleton's gathered
    // forward over the rank's replicated trunk (the row gather must already be
    // installed; checked before any launch). A PREFILL wave under EP is the
    // exact joint full-table prefill instead (`forward_wave_ep`).
    let ep_gather_cap = ep_lane(fwd.ep.as_ref(), &fwd.scratch)?.map(|ep| ep.rows_cap);
    let ep_lane_step = ep_gather_cap.is_some();
    let ep_wave = ep_lane_step && out.prefill();
    // The route of the wave in flight (set by `ep_wave_run`); `None` outside a wave.
    let wave_route = fwd.wave_route;
    if ep_wave {
        // `forward_wave_ep` entered the wave's scopes, reserve bank and routing-row cap; a
        // direct caller would run a wide MoE on the gather route (refused below by row cap).
        if !(fwd.lane_scope
            && fwd.prefill_scope
            && wave_route.is_some()
            && gpu.qwen4_route_rows_cap == QWEN4_EXACT_WAVE_ROUTE_ROWS_CAP)
        {
            return Err(invalid(
                "an EP prefill wave runs only inside forward_wave_ep (lane and prefill scope, wave route, routing-row cap, reserve bank)",
            ));
        }
        if policy.share(StageName::Moe) != StageShare::OnceAllRows {
            return Err(invalid(
                "an EP prefill wave runs its MoE once over all rows",
            ));
        }
    }
    let band_scope = bundle.weights.band_scope();
    if fwd.band_scope() != band_scope {
        return Err(invalid(
            "the lane forward resources do not belong to this bundle's band",
        ));
    }
    // A band's step output follows its ownership: the output band writes
    // logits / top-1 (and a verify capture); every other band hands its HC
    // streams on and has no head.
    match (band_scope.output, out) {
        (true, StepOutputs::Edge { .. }) => {
            return Err(invalid(
                "the output band's lane step writes the head outputs (no edge step)",
            ));
        }
        (false, StepOutputs::Ar(_) | StepOutputs::Wave(_)) => {
            return Err(invalid(
                "a band that does not own the output has no head outputs (edge step)",
            ));
        }
        _ => {}
    }
    // Lanes are the Flash-Next route: the stage policy, row sharing and
    // decode copies are evidenced for its recipe only.
    if !bundle.config.profile.is_flash_next() {
        return Err(invalid(format!(
            "lane steps run only the Flash-Next profile, not {}",
            bundle.config.profile.id()
        )));
    }
    // An automatic-default D2 split executes only on a non-prefill lane step.
    fwd.ensure_cpu_splice_eager(gpu, !out.prefill())?;
    let cpu_splice = fwd.has_cpu_deferred_experts();
    if cpu_splice && !out.prefill() && lanes.iter().any(|lane| lane.wide) {
        return Err(invalid(
            "wide sampled verify cannot combine with the CPU routed-expert window",
        ));
    }
    policy.validate().map_err(invalid)?;
    if lanes.is_empty() {
        return Err(invalid("a lane step needs at least one lane"));
    }
    // Lane rows mix requests: the router fault row is the row inside the step.
    gpu.set_router_row_base(0);
    let config = &bundle.config;
    let dims = program_dims(config);
    let vocab = config.vocab_size;
    // The logits row width of the head stage: the vocabulary, or the rank's vocabulary shard
    // (`LaneOutputs::vocab_shard`, packed `[rows][width]`, shard-local top-1 ids). A shard is
    // an eager AR / verify step of the output band: a prefill wave, an edge step, a recorded
    // or captured step refuses it (the exchange around it is host-ordered, never replayed).
    let vocab_shard = match out {
        StepOutputs::Ar(out) => out.vocab_shard,
        StepOutputs::Wave(_) | StepOutputs::Edge { .. } => None,
    };
    if let Some(shard) = vocab_shard {
        if out.prefill() || !band_scope.output {
            return Err(invalid(
                "a vocabulary shard is an AR / verify step of the output band",
            ));
        }
        if gpu.replay.is_recording() || gpu.graphs.capture_mode {
            return Err(invalid(
                "a vocabulary shard cannot be recorded or captured (its exchange is host-ordered)",
            ));
        }
        if shard.width == 0
            || shard
                .base
                .checked_add(shard.width)
                .map_or(true, |end| end > vocab)
        {
            return Err(invalid(format!(
                "vocabulary shard [{}, +{}) is outside the {vocab}-row head",
                shard.base, shard.width
            )));
        }
    }
    let head_width = vocab_shard.map_or(vocab, |shard| shard.width);
    let rows: Vec<usize> = lanes.iter().map(|lane| lane.tokens.len()).collect();
    let n: usize = rows.iter().sum();
    let prefill = out.prefill();
    for (lane, rows) in lanes.iter().zip(&rows) {
        if prefill != matches!(lane.kind, LaneRowKind::Prefill) {
            return Err(invalid(
                "a lane step is all AR/verify lanes or all prefill lanes",
            ));
        }
        let state = &*lane.state;
        let map = state.scope_map();
        if map.band() != band_scope.band
            || map.owns_input() != band_scope.input
            || map.owns_output() != band_scope.output
        {
            return Err(invalid(format!(
                "a lane state of band {:?} (input {}, output {}) runs on band {:?} (input {}, output {})",
                map.band(),
                map.owns_input(),
                map.owns_output(),
                band_scope.band,
                band_scope.input,
                band_scope.output
            )));
        }
        match lane.kind {
            LaneRowKind::Ar if *rows != 1 => {
                return Err(invalid("an AR lane contributes exactly one row"));
            }
            LaneRowKind::Ar if state.row_capture_armed => {
                return Err(invalid("an AR lane row never arms a GDN row capture"));
            }
            LaneRowKind::Ar => {}
            LaneRowKind::Verify { draft_len } => {
                if draft_len == 0 || *rows != draft_len + 1 || *rows > LANE_MAX_ROWS {
                    return Err(invalid(format!(
                        "a verify lane carries 1 + {draft_len} rows within {LANE_MAX_ROWS}, got {rows}"
                    )));
                }
                if cpu_splice && *rows > crate::program::QWEN4_CPU_SPLICE_MAX_ROWS {
                    return Err(invalid(format!(
                        "a {rows}-row verify lane exceeds the {}-row CPU routed-expert window; \
                         wider verify would run the GPU grouped read, not the CPU decode arithmetic \
                         (set memory.offload_exec=pcie or cap the draft length)",
                        crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
                    )));
                }
                // The singleton's `verify_block` arms the ring for exactly
                // this shape; a lane verify never runs without it.
                if !state.row_capture_armed || state.row_capture_rows() < *rows {
                    return Err(invalid(format!(
                        "a {rows}-row verify lane needs its armed GDN ring (armed {}, {} rows)",
                        state.row_capture_armed,
                        state.row_capture_rows()
                    )));
                }
            }
            LaneRowKind::Prefill if *rows == 0 => {
                return Err(invalid("a prefill lane contributes at least one row"));
            }
            LaneRowKind::Prefill if state.row_capture_armed => {
                return Err(invalid("a prefill wave never arms a GDN row capture"));
            }
            LaneRowKind::Prefill => {}
        }
        let end = state
            .position
            .checked_add(*rows)
            .ok_or_else(|| invalid("lane position overflows"))?;
        if end > state.max_seq_len {
            return Err(invalid(format!(
                "lane end position {end} exceeds its context capacity {}",
                state.max_seq_len
            )));
        }
        if end > state.mapped_context_tokens() {
            return Err(invalid(format!(
                "lane QSA context is mapped through {} tokens, the step needs {end} (provision maps it)",
                state.mapped_context_tokens()
            )));
        }

        if lane.tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(invalid("lane token id is outside the embedding vocabulary"));
        }
    }
    if n > fwd.scratch.max_chunk {
        return Err(invalid(if fwd.decode_share_active() {
            format!(
                "lane step of {n} rows exceeds the decode-share resident bank ({} rows): lane \
                 steps run on the resident bank only",
                fwd.scratch.max_chunk
            )
        } else {
            format!(
                "lane step of {n} rows exceeds the forward scratch ({} rows)",
                fwd.scratch.max_chunk
            )
        }));
    }
    // An EP tail borrow poisoned by a failed close refuses every call, lane steps included
    // (a lane step runs on the resident bank, correct inside an open epoch, so nothing else
    // is needed here); a no-op on every other load.
    fwd.ep_tail_borrow_check()?;
    // An EP reserve-sharing load: a dirty borrow epoch (a lane prefill tile on
    // the borrowed bank) is restored and the resident bank checked before any
    // launch; a no-op on every other load. An EP prefill wave took its bank
    // (restore, acquire, borrowed view) in `ep_wave_run` like a singleton tile.
    if !ep_wave {
        fwd.ep_reserve_lane_step_enter(gpu, n)?;
    }
    fwd.scratch.note_forward_rows(n);
    match out {
        StepOutputs::Ar(out) => {
            if out.logits.dtype != DType::F32 || out.logits.numel() < n * head_width {
                return Err(invalid(
                    "lane logits must be F32 with total_rows * vocab (or shard width) elements",
                ));
            }
            if out.top1.dtype != DType::Raw || out.top1.buf.size() < n * std::mem::size_of::<i32>()
            {
                return Err(invalid("lane top1 must be Raw with one i32 per row"));
            }
        }
        StepOutputs::Wave(wave) => {
            let out_rows = match wave.head {
                WaveHead::Final => lanes.len(),
                WaveHead::All => n,
            };
            if wave.logits.dtype != DType::F32 || wave.logits.numel() < out_rows * vocab {
                return Err(invalid(format!(
                    "wave logits must be F32 with {out_rows} * vocab elements for {:?} output",
                    wave.head
                )));
            }
            if let Some(top1) = wave.top1 {
                if top1.dtype != DType::Raw
                    || top1.buf.size() < out_rows * std::mem::size_of::<i32>()
                {
                    return Err(invalid("wave top1 must be Raw with one i32 per output row"));
                }
            }
        }
        StepOutputs::Edge { .. } => {}
    }
    if let Some(capture) = out.wide_hidden() {
        if capture.dtype != DType::F32 || capture.numel() < n * dims.wide() {
            return Err(invalid(
                "lane wide hidden capture must be F32 with total_rows * hc_count * hidden elements",
            ));
        }
    }
    // Before anything reads an expert a failed staged prefill left in a donor.
    fwd.restore_expert_stage(gpu)?;
    // The gathered prefill attention's F16 workspace, sized by the furthest
    // lane end (the singleton sizes it by its own end) before any launch.
    if prefill {
        let widest_lane = rows.iter().copied().max().unwrap_or(1);
        if gpu.route_rows(widest_lane) >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            let furthest = lanes
                .iter()
                .map(|lane| lane.state.position + lane.tokens.len())
                .max()
                .unwrap_or(0);
            if let Some(qsa) = lanes[0]
                .state
                .qsa
                .first()
                .filter(|qsa| qsa.f16_arena.is_none())
            {
                rdna_compute::tensor_ops::ensure_qsa_gathered_wmma_workspace(
                    gpu,
                    qsa.format,
                    config.num_key_value_heads,
                    furthest,
                )?;
            }
        }
    }

    let ple_layer = config
        .ple_layer_ids
        .first()
        .and_then(|id| id.checked_sub(1))
        .filter(|&layer| layer < config.num_hidden_layers)
        .ok_or_else(|| invalid("PLE layer id missing or outside the trunk"))?;
    let ple_geometry = config
        .ple
        .as_ref()
        .ok_or_else(|| invalid("PLE geometry missing"))?;
    let (ple_head_count, ple_row_width) = (ple_geometry.head_count, ple_geometry.row_width);
    let lane_segs = lane_segments(&rows);
    let group_segs = if prefill {
        // One launch covers every wave row.
        vec![Segment {
            begin: 0,
            rows: n,
            first_lane: 0,
        }]
    } else {
        if cpu_splice {
            group_segments(&rows, crate::program::QWEN4_CPU_SPLICE_MAX_ROWS)
        } else {
            let wide: Vec<bool> = lanes.iter().map(|lane| lane.wide).collect();
            classed_segments(&rows, &wide)
        }
    };
    let widest = group_segs.iter().map(|seg| seg.rows).max().unwrap_or(1);
    let bf16_state = if prefill {
        // One value for the whole wave: every segment's routed row count must
        // pick the same HC stream storage (always true under the invariant
        // profile; without it a wave straddling the 512-row threshold refuses).
        let state = gpu.qwen4_bf16_streams(gpu.route_rows(rows[0]));
        for seg in lane_segs.iter().chain(&group_segs) {
            if gpu.qwen4_bf16_streams(gpu.route_rows(seg.rows)) != state {
                return Err(invalid(
                    "prefill wave segments disagree on the HC stream storage (needs HIPFIRE_PREFILL_PM_INVARIANT on gfx1151, or lanes on one side of the row threshold)",
                ));
            }
        }
        state
    } else {
        // The singleton's one-row stream storage; every segment here (at most
        // LANE_MAX_ROWS rows row-exact, at most the wide cap -- below the 512-row
        // F16 WMMA threshold -- when wide) must agree with it, else the step refuses.
        let state = gpu.qwen4_bf16_streams(1);
        if gpu.qwen4_bf16_streams(widest) != state {
            return Err(invalid("lane segments disagree on the HC stream storage"));
        }
        state
    };
    let segs_for = |share: StageShare| {
        if share.per_lane() {
            0..lane_segs.len()
        } else {
            lane_segs.len()..lane_segs.len() + group_segs.len()
        }
    };
    let mut all_segs: Vec<Segment> = lane_segs.iter().chain(&group_segs).copied().collect();
    // The all-rows segment (exact gfx1201). `HIPFIRE_QWEN4_LANE_HC_ALLROWS`: an AR / verify
    // step of more than LANE_MAX_ROWS rows (several groups) runs every HC stage -- each
    // layer's reads, the writes the all-reduce does not carry, the final read -- as ONE
    // segment over all `n` rows. `HIPFIRE_QWEN4_LANE_ALLROWS` (`all_stages`, a refinement of
    // the same step) puts every other non-MoE once-over-all-rows stage on that segment too:
    // Embed, PleRows, the GDN / QSA input and output projections, Head and Argmax; only the
    // routed MoE keeps its groups. The segment is appended after the groups (`hc_seg`); a
    // prefill wave, a band edge, a recorded or captured step and a CPU splice never take it.
    let group_rows: Vec<usize> = group_segs.iter().map(|seg| seg.rows).collect();
    // The one predicate the mesh root plans its ledger from ([`LaneAllRowsRule`]): `hc_all`,
    // `all_stages`, `exact_scope` and the MoE placement all come from it.
    //
    // Lab `HIPFIRE_QWEN4_LANE_ALLROWS_EXACT`: the all-rows steps run inside the wide few-row
    // verify scope of `n` rows (`HcAllRowsScope`, descriptions and decode copies bound at `n`
    // rows under that scope), so the projections take the row-exact singleton-derived 8-row
    // tiles in one launch instead of the unscoped wide routes. The scope is ours: a scope
    // already in force no longer refuses the all-rows stages. The same scope turns on for a step
    // carrying any row-exact-class lane (`LaneRows::exact`, `HIPFIRE_QWEN4_LANE_ROW_EXACT`: greedy
    // or, with `seeded`, also seeded sampled lanes; default off), and then also sets `Gpu::lane_row_exact_scope` so lane
    // WMMA, the trunk A8 and the MoE int8 gate/up / down refuse for the step: every lane's rows are bitwise its
    // <=8-row rows whatever the batch composition.
    let rule = LaneAllRowsRule::from_parts(
        gpu,
        ep_gather_cap,
        cpu_splice,
        band_scope.band.len(),
        policy,
    );
    let decision = rule.decide(
        matches!(out, StepOutputs::Ar(_)),
        &group_rows,
        lanes.iter().any(|lane| lane.wide),
        lanes.iter().any(|lane| lane.exact),
    );
    let (hc_all, all_stages, exact_scope) =
        (decision.hc_all, decision.all_stages, decision.exact_scope);
    // The few-row limit an all-rows (`n`-row) segment's descriptions are bound under.
    let exact_limit = n.max(gpu.few_row_limit());
    // `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` (default on): on top of the all-rows stages the routed MoE is ONE
    // `ep_lane_moe_rows` call per layer over every row (the wide-group MoE of `n` rows) on its
    // own all-rows segment, and the HC writes a TP2 all-reduce carries go all-rows too: no
    // family runs per <= 8-row group. A wide-class lane step already packs into one group of
    // up to the wide cap (`pack_classed_row_groups`), so it takes this same MoE call through
    // its group and never reaches here (`hc_all` needs every group <= 8 rows); a step the
    // wide call cannot take keeps its groups ([`all_rows_moe_refusal`]).
    let AllRowsDecision {
        moe_refusal,
        moe_all,
        ..
    } = decision;
    let hc_seg: Option<usize> = hc_all.then_some(all_segs.len());
    // The MoE all-rows segment follows the HC one ([`run_items`]).
    let moe_seg: Option<usize> = hc_seg.filter(|_| moe_all).map(|hc| hc + 1);
    // The segments a stage of share `share` runs over: its groups / lanes, or the all-rows
    // segment for a once-over-all-rows stage other than the routed MoE (whose groups are
    // always `segs_for`).
    let stage_segs = |share: StageShare| match hc_seg {
        Some(all) if all_stages && !share.per_lane() => all..all + 1,
        _ => segs_for(share),
    };
    // The writes a TP2 all-reduce carries (`HyperWriteOp::ar`) stay per group, except under
    // the all-rows MoE (the fused write launches up to `AR_HC_ROW_CAP` rows, checked above).
    let hc_writes = if moe_all {
        [true, true]
    } else {
        [
            crate::trunk_route::attn_writer().is_none(),
            crate::trunk_route::ffn_writer().is_none(),
        ]
    };
    let all_rows_plan = hc_seg.map(|seg| {
        (
            seg,
            AllRowsPlan {
                hc_writes,
                stages: all_stages,
                moe: moe_all,
            },
        )
    });
    let hc_final = hc_all
        && band_scope.output
        && policy.share(StageName::FinalHyper) == StageShare::OnceAllRows;
    // The head stages (final hyper, LM head, top-1) run on the output band only.
    // A prefill wave's head runs per lane after the trunk (its own views), so
    // the policy's Head / Argmax segments exist only for AR steps.
    let head_segs = |share: StageShare| {
        if band_scope.output {
            stage_segs(share)
        } else {
            0..0
        }
    };
    let ar_segs = |share: StageShare| if prefill { 0..0 } else { head_segs(share) };
    let ar_out: Option<&LaneOutputs<'_>> = match out {
        StepOutputs::Ar(out) => Some(*out),
        StepOutputs::Wave(_) | StepOutputs::Edge { .. } => None,
    };
    // The row counts each segment's per-layer descriptions are built at (the
    // decode copies and the MoE symmetric capability depend on rows): an AR
    // step has one set at `widest`; a prefill wave gives every segment the
    // description its singleton forward would get at its own routed rows
    // (one set for all segments under the invariant profile). Under EP every
    // segment is a gathered singleton forward of its own rows (decode copies,
    // route capability and the few-row limit as the singleton at those rows),
    // so each segment's set is built at its own routing rows, which must be
    // within the gather cap and equal its real rows (refused before any launch).
    let mut seg_desc_rows: Vec<usize> = if ep_lane_step && !ep_wave {
        if moe_all {
            // One call of every row: the wide-group gather binding (`rows == route rows`,
            // within the cap, checked by the plan too).
            ep_lane_moe_rows(n, n, ep_gather_cap.unwrap_or(0))?;
        } else {
            for i in segs_for(policy.share(StageName::Moe)) {
                let seg = &all_segs[i];
                let rr = if prefill {
                    gpu.route_rows(seg.rows)
                } else {
                    seg.rows
                };
                ep_lane_moe_rows(seg.rows, rr, ep_gather_cap.unwrap_or(0))?;
            }
        }
        all_segs.iter().map(|seg| seg.rows).collect()
    } else if prefill {
        all_segs
            .iter()
            .map(|seg| gpu.route_rows(seg.rows))
            .collect()
    } else {
        vec![widest; all_segs.len()]
    };
    // The all-rows segment binds the descriptions (and decode copies) of the groups.
    if hc_all {
        all_segs.push(Segment {
            begin: 0,
            rows: n,
            first_lane: 0,
        });
        seg_desc_rows.push(if exact_scope {
            n
        } else {
            seg_desc_rows[lane_segs.len()]
        });
    }
    // The MoE all-rows segment (same rows) binds the descriptions of `n` rows: exactly the set
    // a wide-class group of `n` rows (`widest == n`) binds.
    if moe_all {
        all_segs.push(Segment {
            begin: 0,
            rows: n,
            first_lane: 0,
        });
        seg_desc_rows.push(n);
    }
    // `HIPFIRE_QWEN4_LANE_PLE_PROJECT` (default on, exact gfx1201): the PLE key / value
    // projections of every lane run ONCE over all `n` rows, on their own all-rows segment (the
    // lane-wide `PleProject` stage), instead of once per lane. The PLE rows and the key / value
    // scratch are laid out row after row in lane order (`SegViews`: dedicated fields address
    // row `begin` at `begin * width`, the compact PLE arena members at a pitch equal to their
    // width), so a projection over rows `[0, n)` writes every lane's own slice in place and each
    // lane's tail reads it through its own views. Taken for 2+ lanes of a decode step whose `n`
    // rows run ONE row-exact BF16 launch (`Gpu::bf16_rows_exact_one_launch`) under a policy that
    // shares the stage; the weights' dtype and the 1-row-lane PLE batch are checked where the
    // weights are known (`ple_project`).
    let ple_seg: Option<usize> = (band_scope.input
        && !prefill
        && lanes.len() >= 2
        && policy.share(StageName::PleProject) == StageShare::OnceAllRows
        && gpu.flags.qwen4_lane_ple_project_enabled()
        && (2..=rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS).contains(&n)
        && gpu.bf16_rows_exact_one_launch(n))
    .then(|| {
        all_segs.push(Segment {
            begin: 0,
            rows: n,
            first_lane: 0,
        });
        seg_desc_rows.push(seg_desc_rows[lane_segs.len()]);
        all_segs.len() - 1
    });
    if vocab_shard.is_some() {
        // The packed `[rows][width]` layout holds only if the head stages' segments tile the
        // step's rows contiguously in order (each segment writes its rows at `begin * width`):
        // the all-rows segment, the lanes, or the packed groups. Anything else refuses.
        for stage in [StageName::Head, StageName::Argmax] {
            let mut next = 0usize;
            for i in ar_segs(policy.share(stage)) {
                let seg = &all_segs[i];
                if seg.begin != next {
                    return Err(invalid(
                        "a vocabulary shard needs head segments that tile the step's rows in order",
                    ));
                }
                next += seg.rows;
            }
            if next != n {
                return Err(invalid(
                    "a vocabulary shard needs head segments that cover every row of the step",
                ));
            }
        }
    }

    // Token ids: one upload for every row, in lane order (the input band
    // embeds them; every other band runs from received HC streams). Outside a
    // graph capture the ids go through the pinned, fenced ring: the host fill
    // waits only on the slot's own earlier copy (never on a previous group's
    // compute), the copy is enqueued on the active stream without a host wait,
    // and the bytes stay untouched until that copy's fence completes. A
    // capture keeps the blocking `host_token_bytes` copy (no null-stream work
    // inside a capture).
    let ids_async = band_scope.input && !gpu.graphs.capture_mode;
    if band_scope.input {
        let ids_len = n * std::mem::size_of::<i32>();
        let staged_ids: &mut [u8] = if ids_async {
            fwd.lane_tokens.stage(gpu, ids_len)?
        } else {
            &mut fwd.host_token_bytes[..ids_len]
        };
        let mut row = 0usize;
        for lane in lanes.iter() {
            for &token in lane.tokens {
                staged_ids[row * 4..row * 4 + 4].copy_from_slice(&(token as i32).to_ne_bytes());
                row += 1;
            }
        }
    }
    // PLE rows (input band only): one fetch of every lane's ids (each hashed
    // from its own history), started now so it overlaps the program build.
    // The lane route is Flash-Next-only, which always carries PLE.
    let mut ple = if band_scope.input {
        let ple_owner = bundle
            .ple
            .as_ref()
            .ok_or_else(|| invalid("the lane forward requires a profile with PLE"))?;
        let ple_ids: Vec<u64> = lanes
            .iter()
            .flat_map(|lane| {
                lane.state
                    .ple_history
                    .row_ids(&ple_owner.metadata, lane.tokens)
            })
            .collect();
        Some(
            RowFetch::begin_or_adopt(&ple_owner.rows, fwd.ple_ahead.take(), ple_ids)
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?,
        )
    } else {
        drop(fwd.ple_ahead.take());
        None
    };
    // The ids this forward uploads are not a whole-row mirror of a singleton
    // upload: no later singleton forward may claim them. The lane mirror
    // names them for the step's own commit fill (input band, outside capture).
    fwd.uploaded_id_rows = 0;
    fwd.lane_uploaded_ids.1.clear();
    if ids_async {
        fwd.lane_uploaded_ids.0 = fwd.scratch.token_ids.buf.as_ptr() as usize;
        fwd.lane_uploaded_ids
            .1
            .extend(lanes.iter().flat_map(|lane| lane.tokens.iter().copied()));
    }
    if prefill {
        // Like `lane_prefill_tile`: no lookahead names this wave's next rows,
        // and (as the singleton does for every multi-row forward) no retained
        // replay is eligible.
        fwd.set_ple_lookahead(&[]);
        gpu.replay.set_forward_eligible(false);
    }

    // `HIPFIRE_QWEN4_EP_SP_SUB512_WAVE`: a trunk-shard prefill wave of at
    // least the sub-512 SP floor runs its joint HC reads / writes sequence-parallel, like a wide
    // singleton chunk (`forward_chunk_scoped`'s `SpViews`): each rank reads / writes its half of the
    // wave's rows (reduce-scatter write), a row all-gather rejoins `hc_mixed` after every read and
    // the streams before the PLE conv and the final hyper read. The HC read of the wave and of both
    // halves is the pinned-split-K prefill lane WMMA entry (the routing-row cap keeps all three
    // below 512 routing rows), so a row's bytes do not depend on which half carries it.
    let wave_sp: Option<crate::trunk_route::SpRows> = if ep_wave
        && matches!(
            wave_route,
            Some(EpWaveRoute::Partial | EpWaveRoute::PartialMask)
        )
        && group_segs.len() == 1
        && !bf16_state
        && policy.share(StageName::HcRead) == StageShare::OnceAllRows
        && policy.share(StageName::HcWrite) == StageShare::OnceAllRows
        && crate::trunk_route::sp_wave_rows_in_band(n)
        && gpu.prefill_lwmma_hc_applies(n / 2)
        && gpu.prefill_lwmma_hc_applies(n)
        && !(gpu.replay.is_recording() || gpu.graphs.capture_mode)
    {
        crate::trunk_route::sp_rows(n)
    } else {
        None
    };
    // A step that runs its all-rows segment row-exact also refuses the 9..=64-row arms that are
    // not bitwise the <=8-row route (lane WMMA, trunk A8, MoE int8 gate/up / down) for its whole execution:
    // every launch of more than 8 rows here is an all-rows-segment launch. Restored on every exit
    // path of the closure below (success and error).
    let row_exact_prior =
        exact_scope.then(|| std::mem::replace(&mut gpu.lane_row_exact_scope, true));
    let attempt = (|| -> Result<Vec<QsaCommit>, Qwen4GpuForwardError> {
        let scratch = &fwd.scratch;
        let mut views = Vec::with_capacity(all_segs.len());
        for seg in &all_segs {
            views.push(SegViews::new(
                scratch,
                dims,
                head_width,
                ar_out,
                *seg,
                ple_head_count * ple_row_width,
                // PP edges require packed BF16 rows; preserve the original
                // single-scope scratch addressing without changing its ops.
                bf16_state && !(band_scope.input && band_scope.output),
                // Every segment of the step addresses the one field its
                // total row count selects.
                scratch.ple_normed_field(n),
            )?);
        }
        let seg_scratch: Vec<Qwen4LayerScratch<'_>> = views
            .iter()
            .map(|view| view.layer_scratch(scratch))
            .collect();
        // The own-row HC views of a sequence-parallel wave ([`wave_sp`]): the group segment's
        // tensors (row 0 of every scratch field) cut at this rank's first row.
        let sp_views: Option<SpViews> = wave_sp.and_then(|rows| {
            crate::trunk_route::sp_route()
                .map(|route| SpViews::new(scratch, route, rows, dims, bf16_state))
        });
        if sp_views.is_some() {
            static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "  qwen4 EP sequence-parallel prefill wave (HIPFIRE_QWEN4_EP_SP_SUB512_WAVE): the wave's joint HC reads / writes split its rows across the two ranks (reduce-scatter HC write, row all-gather)"
                );
            }
        }
        // The streams hold only this rank's rows current (an own-row HC write ran since the
        // last stream gather): the PLE conv and the final hyper read gather them first.
        let mut sp_streams_partial = false;
        // Under EP: the row gather, the two trunk gather buffers and, per
        // segment, the scratch description whose expanded-row scratch is each
        // buffer (validated before any launch). A prefill wave runs no gather:
        // its MoE is the singleton's full-table chunk MoE.
        let ep_lane = if ep_wave {
            None
        } else {
            ep_lane(fwd.ep.as_ref(), scratch)?
        };
        let ep_seg_scratch: Option<Vec<[Qwen4LayerScratch<'_>; 2]>> = ep_lane.as_ref().map(|ep| {
            seg_scratch
                .iter()
                .map(|base| {
                    std::array::from_fn(|buffer| Qwen4LayerScratch {
                        moe_down_expanded: ep.buffers[buffer],
                        ..*base
                    })
                })
                .collect()
        });
        // The wave's wide MoE segment routes as `route_rows(n)` = the cap (511) rows: its layer
        // description (declared route capability, decode copies) is the sub-512 one every lane's
        // singleton sees, over each layer's FULL-TABLE runtime (the singleton chunk's
        // `select_layer_moe` for a sub-512-row tile).
        let wave_group_rows = gpu.route_rows(n);

        // Per-layer weight descriptions, one set per distinct row count in
        // `seg_desc_rows` (AR: one set; decode copies bind for every
        // segment: none is wider than LANE_MAX_ROWS).
        let mut desc_rows: Vec<usize> = Vec::with_capacity(1);
        for &rows in &seg_desc_rows {
            if !desc_rows.contains(&rows) {
                desc_rows.push(rows);
            }
        }
        let mut desc_sets: Vec<Vec<Qwen4LayerDescription<'_>>> =
            Vec::with_capacity(desc_rows.len());
        for &rows in &desc_rows {
            let wave_set = ep_wave && rows == wave_group_rows;
            let mut descs: Vec<Qwen4LayerDescription<'_>> =
                Vec::with_capacity(config.num_hidden_layers);
            for layer_index in band_scope.band.clone() {
                let layer_moe = if wave_set {
                    ep_wave_layer_moe(
                        wave_route.unwrap_or(EpWaveRoute::FullTable),
                        fwd.ep.as_ref(),
                        &fwd.moe,
                        &fwd.moe_borrow,
                        fwd.borrow_runtimes_active,
                        fwd.borrow_view,
                        layer_index,
                    )
                } else {
                    lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer_index)
                };
                descs.push(layer_desc(
                    &bundle.weights,
                    bundle
                        .weights
                        .layer(layer_index)
                        .ok_or_else(|| invalid(format!("lane layer {layer_index} is not owned")))?,
                    layer_moe,
                    config,
                    &fwd.decode_q8,
                    rows,
                    if exact_scope && rows == n {
                        exact_limit
                    } else {
                        gpu.few_row_limit()
                    },
                    // The wave's MoE segment declares what the lanes' own singleton tiles do at
                    // the floor in scope (Tile route: the lowered floor, symmetric capability;
                    // full-table route: pinned 512); every lane step keeps the default.
                    if wave_set {
                        gpu.qwen4_moe_sym_iu4_floor
                    } else {
                        rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS
                    },
                    None,
                )?);
            }
            desc_sets.push(descs);
        }
        let seg_desc_set: Vec<usize> = seg_desc_rows
            .iter()
            .map(|rows| desc_rows.iter().position(|k| k == rows).unwrap_or(0))
            .collect();
        // PLE weights and the F16 row staging decision: the input band only.
        let ple_w = if band_scope.input {
            let ple_weights = bundle
                .weights
                .layer(ple_layer)
                .ok_or_else(|| invalid("lane PLE layer is not owned"))?
                .ple
                .as_ref()
                .ok_or_else(|| invalid("PLE weights missing"))?;
            Some(ple_desc(&bundle.weights, ple_weights)?)
        } else {
            None
        };
        let ple_rows_f16_at = |gpu: &Gpu, rows: usize| {
            ple_w.as_ref().is_some_and(|ple_w| {
                qwen4_ple_rows_f16(
                    gpu,
                    bf16_state,
                    dims.hc_count,
                    config.hidden_size,
                    ple_w,
                    rows,
                )
            })
        };
        let ple_rows_f16 = if prefill {
            // One value for the whole wave (the PLE conv runs per lane on the
            // lane's own rows, like the singleton's decision at its row count).
            let first = ple_rows_f16_at(&*gpu, gpu.route_rows(rows[0]));
            for &lane_rows in &rows {
                if ple_rows_f16_at(&*gpu, gpu.route_rows(lane_rows)) != first {
                    return Err(invalid(
                        "prefill wave lanes disagree on F16 PLE row staging (needs HIPFIRE_PREFILL_PM_INVARIANT on gfx1151, or lanes on one side of the row threshold)",
                    ));
                }
            }
            first
        } else {
            if ple_rows_f16_at(&*gpu, 1) {
                return Err(invalid("lane steps do not stage F16 PLE rows"));
            }
            false
        };
        // The head descriptions exist on the output band only.
        // The vocabulary shard's rows of the resident head ([`LaneOutputs::vocab_shard`]): a byte
        // sub-tensor of the same allocation, `width` rows long.
        let lm_full = if band_scope.output {
            Some(dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?)
        } else {
            None
        };
        let lm_slice: Option<GpuTensor> = match (vocab_shard, lm_full.as_ref()) {
            (Some(shard), Some(full)) => Some(lm_head_row_slice(full, shard, vocab)?),
            _ => None,
        };
        let head = match lm_full {
            Some(full) => {
                let lm_head = match lm_slice.as_ref() {
                    Some(slice) => WeightRef {
                        buf: slice,
                        m: head_width,
                        ..full
                    },
                    None => full,
                };
                Some((
                    hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?,
                    lm_head,
                ))
            }
            None => None,
        };
        let head_ref = || {
            head.as_ref()
                .ok_or_else(|| invalid("a lane head stage needs the output band"))
        };
        let states: Vec<&Qwen4State> = lanes.iter().map(|lane| &*lane.state).collect();

        // Preflight the head of every segment before any launch.
        let final_preflight: Vec<usize> = match hc_seg {
            Some(hc) if hc_final => vec![hc],
            _ => head_segs(policy.share(StageName::FinalHyper)).collect(),
        };
        for i in final_preflight {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (final_hyper, _) = head_ref()?;
            validate_final_hyper(
                dims,
                &final_hyper,
                &v.streams,
                &v.layer_scratch(scratch),
                seg.rows,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("validate lane final hyper: {error:?}"))
            })?;
        }
        for i in ar_segs(policy.share(StageName::Head)) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (_, lm_head) = head_ref()?;
            let hidden = view(&v.hc_mixed, 0, seg.rows * dims.hidden);
            validate_lm_head(&lm_head, &hidden, &v.logits, seg.rows, seg.rows).map_err(
                |error| Qwen4GpuForwardError::Dispatch(format!("validate lane LM head: {error:?}")),
            )?;
        }
        // A prefill wave's per-lane head outputs and row requests: Final = the
        // lane's last row only (the singleton's `requested_rows == 1`), All =
        // every row of the lane.
        let wave_head: Vec<(GpuTensor, Option<GpuTensor>, usize)> = match out {
            StepOutputs::Wave(wave) => lane_segs
                .iter()
                .enumerate()
                .map(|(lane, seg)| match wave.head {
                    WaveHead::Final => (
                        view(wave.logits, lane * vocab, vocab),
                        wave.top1.map(|top1| {
                            view(
                                top1,
                                lane * std::mem::size_of::<i32>(),
                                std::mem::size_of::<i32>(),
                            )
                        }),
                        1,
                    ),
                    WaveHead::All => (
                        view(wave.logits, seg.begin * vocab, seg.rows * vocab),
                        wave.top1.map(|top1| {
                            view(
                                top1,
                                seg.begin * std::mem::size_of::<i32>(),
                                seg.rows * std::mem::size_of::<i32>(),
                            )
                        }),
                        seg.rows,
                    ),
                })
                .collect(),
            StepOutputs::Ar(_) | StepOutputs::Edge { .. } => Vec::new(),
        };
        for (lane, (head_logits, _, requested)) in wave_head.iter().enumerate() {
            let (seg, v) = (&lane_segs[lane], &views[lane]);
            let (_, lm_head) = head_ref()?;
            let hidden = view(&v.hc_mixed, 0, seg.rows * dims.hidden);
            validate_lm_head(&lm_head, &hidden, head_logits, seg.rows, *requested).map_err(
                |error| Qwen4GpuForwardError::Dispatch(format!("validate wave LM head: {error:?}")),
            )?;
        }

        // The singleton threads the call's canonical routing rows through the
        // dispatch context (the sealed MoE's CPU-split and verify gates read it); an
        // EP wave seals as the chunk of its total routing rows.
        let ctx = if ep_wave {
            DispatchCtx::new(gpu).with_batch_size(wave_group_rows)
        } else {
            DispatchCtx::new(gpu)
        };
        let band_first = band_scope.band.start;
        let mut program = layer_program(bundle, policy, band_scope.input.then_some(ple_layer))?;
        // `HIPFIRE_QWEN4_LANE_PLE_BATCH` (exact gfx1201, default off): AR lane
        // steps of one row per lane run the PLE chain of up to
        // `PLE_LANES_MAX` consecutive lanes as ONE op (key / value projections
        // over the chunk's rows + one batched tail) instead of the whole chain
        // per lane. A chunk needs 2+ lanes, every row tensor the chain touches
        // laid out contiguously lane after lane, and each lane's own PLE conv
        // state; anything else keeps the per-lane chain. Declared before the
        // steps: they borrow the chunks.
        let ple_chunks: Vec<(usize, Vec<&GpuTensor>)> = match &ple_w {
            Some(ple_w)
                if band_scope.input
                    && !prefill
                    && !bf16_state
                    && lane_segs.iter().all(|seg| seg.rows == 1) =>
            {
                let width = |tensor: &GpuTensor| tensor.buf.as_ptr() as usize;
                let lane_ids: Vec<usize> = (0..lane_segs.len()).collect();
                let mut chunks = Vec::new();
                for ids in lane_ids.chunks(rdna_compute::ple_lanes::PLE_LANES_MAX) {
                    let first = ids[0];
                    let contiguous = |select: fn(&SegViews) -> &GpuTensor, row: usize| {
                        ids.iter().all(|&lane| {
                            width(select(&views[lane]))
                                == width(select(&views[first])) + (lane - first) * row * 4
                        })
                    };
                    let hidden = config.hidden_size;
                    let wide = dims.hc_count * hidden;
                    let admitted = ids.len() >= 2
                        && hipfire_dispatch::pipeline::grouped_depthwise_lanes_admitted(
                            &*gpu,
                            &ple_w.key,
                            &ple_w.value,
                            ids.len(),
                        )
                        && contiguous(|v| &v.streams, wide)
                        && contiguous(|v| &v.ple_rows, ple_head_count * ple_row_width)
                        && contiguous(|v| &v.ple_key, wide)
                        && contiguous(|v| &v.ple_value, hidden)
                        && contiguous(|v| &v.ple_normed, wide);
                    if !admitted {
                        continue;
                    }
                    let chunk: Option<Vec<&GpuTensor>> = ids
                        .iter()
                        .map(|&lane| states[lane_segs[lane].first_lane].ple_conv.as_ref())
                        .collect();
                    if let Some(chunk) = chunk {
                        chunks.push((first, chunk));
                    }
                }
                chunks
            }
            _ => Vec::new(),
        };
        // The views one batched PLE op reads: ONE segment spanning the chunk's lanes. A dedicated
        // scratch field's lane view runs to the end of its tensor, so the first lane's view held
        // every lane's row; an alias-arena member (the compact plan) is bounded to its segment's
        // `rows * pitch`, so the op needs the chunk's own view.
        let ple_batch_views: Vec<SegViews> = ple_chunks
            .iter()
            .map(|(first, chunk)| {
                SegViews::new(
                    scratch,
                    dims,
                    head_width,
                    ar_out,
                    Segment {
                        begin: lane_segs[*first].begin,
                        rows: chunk.len(),
                        first_lane: lane_segs[*first].first_lane,
                    },
                    ple_head_count * ple_row_width,
                    bf16_state && !(band_scope.input && band_scope.output),
                    scratch.ple_normed_field(n),
                )
            })
            .collect::<Result<_, _>>()?;
        // The lane-wide PLE projection (`ple_seg`) runs when the weights are the BF16 pair the
        // row-exact one-launch route covers and the 1-row-lane PLE batch took no chunk (it
        // already runs one projection over its chunk); the program then carries `PleProject`
        // ahead of the per-lane tails, which skip their own projections. Anything else keeps
        // today's per-lane chain.
        let ple_project = ple_seg.is_some()
            && ple_chunks.is_empty()
            && !ple_rows_f16
            && ple_w.as_ref().is_some_and(|ple_w| {
                ple_w.key.dtype == DType::BF16 && ple_w.value.dtype == DType::BF16
            });
        if ple_project {
            with_ple_project(&mut program);
        }
        let mut steps: Vec<Step<'_>> = Vec::with_capacity(program.len() * lanes.len() * 2);
        let mut qsa_commits: Vec<QsaCommit> = Vec::new();
        // `HIPFIRE_QWEN4_GDN_CAPTURE_DIRECT` (exact gfx1201, default on): a verify lane's GDN
        // qkv in-projection that runs as its own launch over exactly that lane's rows writes the
        // lane's capture `inputs` in place, and the lane's Core skips the
        // `copy_d2d(projection -> capture.inputs)` it otherwise issues per layer. The opt-in
        // `HIPFIRE_QWEN4_GDN_QKV_BF16` lab prefill route decides the qkv storage dtype from the
        // op's own row capture, which an in-projection op never carries, and a replay recorder
        // or graph capture launches with the scratch pointers: both keep the copy.
        let gdn_capture_direct = gpu.flags.qwen4_gdn_capture_direct_enabled()
            && !gpu.flags.qwen4_gdn_qkv_bf16_enabled()
            && !(gpu.replay.is_recording() || gpu.graphs.capture_mode);
        // `(layer, lane)` of every in-projection that wrote its lane's capture directly: the
        // InProj arm records it, the Core arm of the same layer and lane reads it, so the two
        // phases cannot disagree (an InProj always precedes its Core in emission order).
        let mut gdn_direct: Vec<(usize, usize)> = Vec::new();
        let mut ple_split: Option<usize> = None;
        // MoE invocations an EP prefill wave emitted (one exchange each).
        let mut wave_moe_calls = 0usize;
        // MoE invocations emitted so far under EP, in emission (= execution)
        // order over the whole step: the gather buffer alternates per
        // invocation, whatever the lane/layer interleaving.
        let mut ep_invocations = 0usize;
        // Launches emitted per stage (one per segment per operation): the step trace's shape.
        let mut stage_calls = [0usize; StageName::ALL.len()];
        // Runs of consecutive operations with the same share, emitted
        // segment-major.
        let mut start = 0usize;
        while start < program.len() {
            let per_lane = policy.share(program[start].1).per_lane();
            let mut end = start + 1;
            while end < program.len() && policy.share(program[end].1).per_lane() == per_lane {
                end += 1;
            }
            let share = if per_lane {
                StageShare::PerLane
            } else {
                StageShare::OnceAllRows
            };
            // The run's pieces: with the all-rows segment the HC ops (and, under
            // `HIPFIRE_QWEN4_LANE_ALLROWS`, every other non-MoE op) of a once-over-all-rows
            // run form their own all-rows pieces (the routed MoE keeps its groups); each
            // piece is emitted segment-major over its own segments, pieces in order.
            let items = run_items(
                start,
                &program[start..end],
                segs_for(share),
                all_rows_plan.filter(|_| share == StageShare::OnceAllRows),
                ple_seg.filter(|_| ple_project),
            );
            for (i, ops) in items {
                let (seg, v) = (&all_segs[i], &views[i]);
                let r = seg.rows;
                // The all-rows HC segment binds the decode copies its groups bind.
                let rr = if prefill {
                    gpu.route_rows(r)
                } else if Some(i) == hc_seg {
                    if exact_scope {
                        n
                    } else {
                        widest
                    }
                } else {
                    r
                };
                let descs = &desc_sets[seg_desc_set[i]];
                let row_limit = if exact_scope && r == n {
                    exact_limit
                } else {
                    gpu.few_row_limit()
                };
                for &(op, stage) in &program[ops] {
                    stage_calls[stage.index()] += 1;
                    match op {
                        LaneOp::HcRead { layer, mlp } => {
                            let read = if mlp {
                                &descs[layer - band_first].mlp_hyper.read
                            } else {
                                &descs[layer - band_first].attn_hyper.read
                            };
                            let copy = layer * DECODE_Q8_PER_LAYER + if mlp { 2 } else { 0 };
                            let sp = sp_views.as_ref().filter(|_| i == lane_segs.len());
                            let (streams, normalized, low, up, mixed, hc_rows) = match sp {
                                Some(sp) => (
                                    &sp.streams,
                                    &sp.normalized,
                                    &sp.low,
                                    &sp.up,
                                    &sp.hc_mixed,
                                    sp.rows.own.1,
                                ),
                                None => (
                                    &v.streams,
                                    &v.hc_normalized,
                                    &v.hc_low,
                                    &v.hc_up,
                                    &v.hc_mixed,
                                    r,
                                ),
                            };
                            steps.push(Step::HyperRead(HyperReadOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: streams,
                                norm_weight: read.norm,
                                input_mix_down: decode_copy(
                                    &fwd.decode_q8,
                                    copy,
                                    rr,
                                    row_limit,
                                    read.input_mix_down,
                                ),
                                input_mix_up: decode_copy(
                                    &fwd.decode_q8,
                                    copy + 1,
                                    rr,
                                    row_limit,
                                    read.input_mix_up,
                                ),
                                normalized,
                                low,
                                up,
                                mixed,
                                rows: hc_rows,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                                low_rank: dims.hc_lowrank,
                            }));
                            // Sequence-parallel wave: the mixer and the MoE read every row.
                            if let Some(sp) = sp {
                                steps.push(Step::RowGather(RowGatherOp {
                                    tensor: &scratch.hc_mixed,
                                    f32_bf16: true,
                                    row_elems: config.hidden_size,
                                    rows: n,
                                    exec: sp.exec,
                                }));
                            }
                        }
                        LaneOp::HcWrite { layer, mlp } => {
                            let write = if mlp {
                                &descs[layer - band_first].mlp_hyper.write
                            } else {
                                &descs[layer - band_first].attn_hyper.write
                            };
                            let sp = sp_views.as_ref().filter(|_| i == lane_segs.len());
                            let (streams, normalized, mixed, gates, hc_rows) = match sp {
                                Some(sp) => (
                                    &sp.streams,
                                    &sp.normalized,
                                    if mlp {
                                        &sp.ffn_output
                                    } else {
                                        &sp.attention_output
                                    },
                                    &sp.gates,
                                    sp.rows.own.1,
                                ),
                                None => (
                                    &v.streams,
                                    &v.hc_normalized,
                                    if mlp {
                                        &v.ffn_output
                                    } else {
                                        &v.attention_output
                                    },
                                    &v.hc_gates,
                                    r,
                                ),
                            };
                            sp_streams_partial |= sp.is_some();
                            steps.push(Step::HyperWrite(HyperWriteOp {
                                ar: if mlp {
                                    crate::trunk_route::ffn_writer()
                                } else {
                                    crate::trunk_route::attn_writer()
                                },
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: streams,
                                norm_weight: write.norm,
                                block_inject: write.block_inject,
                                normalized,
                                mixed,
                                gates,
                                output: streams,
                                rows: hc_rows,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                            }));
                        }
                        LaneOp::PleProject => {
                            let ple_w = ple_w
                                .as_ref()
                                .ok_or_else(|| invalid("PLE weights missing"))?;
                            if ple_split.is_none() {
                                ple_split = Some(steps.len());
                            }
                            // Key then value, each ONE row-exact launch over this segment's
                            // rows (`ProjectLadder::TrunkRowExact`: every output row is
                            // bitwise the per-lane projection's); the PLE rows are converted
                            // at the split, before these steps run.
                            for (weight, output) in
                                [(ple_w.key, &v.ple_key), (ple_w.value, &v.ple_value)]
                            {
                                steps.push(Step::Project(ProjectOp {
                                    weight,
                                    input: &v.ple_rows,
                                    output,
                                    rows: r,
                                    rotation: Some(&scratch.rotation),
                                    ladder: ProjectLadder::TrunkRowExact,
                                    residual: None,
                                }));
                            }
                        }
                        LaneOp::PleConv => {
                            let ple_w = ple_w
                                .as_ref()
                                .ok_or_else(|| invalid("PLE weights missing"))?;
                            // Sequence-parallel wave: the PLE conv reads and writes every row of
                            // the streams (and leaves them current on both ranks).
                            if let Some(sp) = sp_views.as_ref().filter(|_| sp_streams_partial) {
                                steps.push(Step::RowGather(RowGatherOp {
                                    tensor: &scratch.streams,
                                    f32_bf16: !bf16_state,
                                    row_elems: dims.wide(),
                                    rows: n,
                                    exec: sp.exec,
                                }));
                                sp_streams_partial = false;
                            }
                            if ple_split.is_none() {
                                ple_split = Some(steps.len());
                            }
                            if let Some((chunk_index, (first, chunk))) =
                                ple_chunks.iter().enumerate().find(|(_, (first, chunk))| {
                                    (*first..*first + chunk.len()).contains(&i)
                                })
                            {
                                // `HIPFIRE_QWEN4_LANE_PLE_BATCH`: the chunk's lanes run
                                // as one op at its first lane (lane rows are
                                // independent; each keeps its own state).
                                if *first == i {
                                    let v = &ple_batch_views[chunk_index];
                                    steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                                        rotation: &scratch.rotation,
                                        state_bf16: bf16_state,
                                        rows_f16: ple_rows_f16,
                                        key: ple_w.key,
                                        value: ple_w.value,
                                        norm_key: ple_w.norm_key,
                                        norm_query: ple_w.norm_query,
                                        norm_conv: ple_w.norm_conv,
                                        conv: ple_w.conv,
                                        state: chunk[0],
                                        streams: &v.streams,
                                        rows_tensor: &v.ple_rows,
                                        query: &v.ple_query,
                                        key_scratch: &v.ple_key,
                                        value_scratch: &v.ple_value,
                                        gated: &v.ple_gated,
                                        normed: &v.ple_normed,
                                        output: &v.ple_output,
                                        rows: chunk.len(),
                                        branches: dims.hc_count,
                                        hidden: config.hidden_size,
                                        embed_dim: ple_head_count * ple_row_width,
                                        kernel_size: dims.ple_conv_kernel_dim,
                                        dilation: config.ple_conv_dilation(),
                                        epsilon: EPSILON,
                                        lane_states: chunk.as_slice(),
                                        projected: false,
                                    }));
                                }
                            } else {
                                let state = states[seg.first_lane];
                                steps.push(Step::GroupedDepthwise(GroupedDepthwiseOp {
                                    rotation: &scratch.rotation,
                                    state_bf16: bf16_state,
                                    rows_f16: ple_rows_f16,
                                    key: ple_w.key,
                                    value: ple_w.value,
                                    norm_key: ple_w.norm_key,
                                    norm_query: ple_w.norm_query,
                                    norm_conv: ple_w.norm_conv,
                                    conv: ple_w.conv,
                                    state: state
                                        .ple_conv
                                        .as_ref()
                                        .ok_or_else(|| invalid("PLE conv state missing"))?,
                                    streams: &v.streams,
                                    rows_tensor: &v.ple_rows,
                                    query: &v.ple_query,
                                    key_scratch: &v.ple_key,
                                    value_scratch: &v.ple_value,
                                    gated: &v.ple_gated,
                                    normed: &v.ple_normed,
                                    output: &v.ple_output,
                                    rows: r,
                                    branches: dims.hc_count,
                                    hidden: config.hidden_size,
                                    embed_dim: ple_head_count * ple_row_width,
                                    kernel_size: dims.ple_conv_kernel_dim,
                                    dilation: config.ple_conv_dilation(),
                                    epsilon: EPSILON,
                                    lane_states: &[],
                                    projected: ple_project,
                                }));
                            }
                        }
                        LaneOp::Gdn { layer, slot, phase } => {
                            let weights = match &descs[layer - band_first].attention {
                                Qwen4AttentionWeights::Linear(weights) => weights,
                                _ => return Err(invalid("GDN descriptor kind mismatch")),
                            };
                            let state = states[seg.first_lane];
                            let gdn = state
                                .gdn
                                .get(slot)
                                .ok_or_else(|| invalid("GDN state slot missing"))?;
                            let stateful = matches!(phase, MixerPhase::Whole | MixerPhase::Core);
                            let capture = state.gdn_row_capture(slot, r);
                            let mut projection_target = &v.projection;
                            if let (Some(capture), true) = (capture.as_ref(), gdn_capture_direct) {
                                let lane = seg.first_lane;
                                match phase {
                                    MixerPhase::InProj => {
                                        let own = lane_segs[lane];
                                        if own.begin == seg.begin && own.rows == seg.rows {
                                            gdn_direct.push((layer, lane));
                                            projection_target = capture.inputs;
                                        }
                                    }
                                    MixerPhase::Core if gdn_direct.contains(&(layer, lane)) => {
                                        projection_target = capture.inputs;
                                    }
                                    _ => {}
                                }
                            }
                            steps.push(Step::GatedDeltaNet(GatedDeltaNetOp {
                                rotation: &scratch.rotation,
                                qkv: weights.qkv,
                                conv: weights.conv,
                                in_proj_a: weights.in_proj_a,
                                in_proj_b: weights.in_proj_b,
                                a_log: weights.a_log,
                                dt_bias: weights.dt_bias,
                                z: weights.z,
                                norm: weights.norm,
                                output: weights.output,
                                recurrent: &gdn.recurrent,
                                // The lane's own live EF companion (never shared
                                // across lanes, never in the ring/snapshot).
                                format: state.gdn_format(),
                                residual: gdn.ef_residual.as_ref(),
                                conv_state: &gdn.conv,
                                projection: projection_target,
                                projection2: &v.projection2,
                                a: &v.gdn_a,
                                b: &v.gdn_b,
                                gate: &v.gdn_gate,
                                beta: &v.gdn_beta,
                                recurrent_output: &v.gdn_recurrent_output,
                                bf16_scratch: &scratch.gdn_bf16,
                                z_output: &v.gdn_z,
                                output_scratch: &v.gdn_output,
                                input: &v.hc_mixed,
                                output_tensor: &v.attention_output,
                                rows: r,
                                start_position: state.position,
                                key_heads: dims.linear_num_key_heads,
                                value_heads: dims.linear_num_value_heads,
                                key_dim: dims.linear_key_head_dim,
                                value_dim: dims.linear_value_head_dim,
                                conv_kernel: dims.linear_conv_kernel_dim,
                                input_width: dims.hidden,
                                row_capture: if stateful { capture } else { None },
                                zba_fold: fwd.trunk_zba.get(layer).and_then(Option::as_ref),
                                trunk_a4: fwd.trunk_a4.get(layer).copied().unwrap_or(0),
                                phase,
                            }));
                        }
                        LaneOp::Qsa { layer, slot, phase } => {
                            let weights = match &descs[layer - band_first].attention {
                                Qwen4AttentionWeights::Full(weights) => weights,
                                _ => return Err(invalid("QSA descriptor kind mismatch")),
                            };
                            let state = states[seg.first_lane];
                            let qsa = state
                                .qsa
                                .get(slot)
                                .ok_or_else(|| invalid("QSA state slot missing"))?;
                            let op = IndexedAttentionOp {
                                sp_select: None,
                                rotation: &scratch.rotation,
                                mode: IndexedAttentionMode::Full,
                                indexer_qk: weights.indexer_qk,
                                indexer_q_norm: weights.indexer_q_norm,
                                indexer_k_norm: weights.indexer_k_norm,
                                q: weights.q,
                                k: weights.k,
                                v: weights.v,
                                q_norm: weights.q_norm,
                                k_norm: weights.k_norm,
                                output: weights.output,
                                state: IndexedAttentionState {
                                    format: qsa.format,
                                    full_keys: &qsa.full_keys,
                                    full_values: &qsa.full_values,
                                    raw_index_keys: &qsa.raw_index_keys,
                                    pooled_keys: &qsa.pooled_keys,
                                    selected_indices: &qsa.selected_indices,
                                    f16_arena: if prefill { qsa.f16_arena_ref() } else { None },
                                    pooled_bf16: qsa.pooled_bf16_ref(),
                                    full_capacity: qsa.full_capacity,
                                    raw_capacity: qsa.raw_capacity,
                                    pooled_capacity: qsa.pooled_capacity,
                                    selected_capacity: qsa.selected_capacity,
                                    position_capacity: qsa.position_capacity,
                                    full_len: qsa.full_len,
                                    raw_len: qsa.raw_len,
                                    pooled_len: qsa.pooled_len,
                                    selected_len: qsa.selected_len,
                                    position: qsa.position,
                                },
                                input: &v.hc_mixed,
                                index_scratch: &v.qsa_index,
                                qgate_scratch: &v.qsa_qgate,
                                k_scratch: &v.qsa_k,
                                v_scratch: &v.qsa_v,
                                qsa_output: &v.qsa_output,
                                selected_scratch: &scratch.qsa_selected,
                                attention_output: &v.attention_output,
                                bf16_scratch: &scratch.gdn_bf16,
                                rows: r,
                                index_heads: dims.indexer_n_heads,
                                index_kv_heads: dims.indexer_kv_heads,
                                index_dim: dims.indexer_head_dim,
                                budget: dims.indexer_budget,
                                compress: dims.indexer_compress_ratio,
                                heads: dims.num_attention_heads,
                                kv_heads: dims.num_key_value_heads,
                                head_dim: dims.head_dim,
                                rotary_dim: config.rotary_dim(),
                                rope: config.rope(),
                                mrope: None,
                                input_width: dims.hidden,
                                trunk_a4: fwd.trunk_a4.get(layer).copied().unwrap_or(0),
                                phase,
                            };
                            if matches!(phase, MixerPhase::Whole | MixerPhase::Core) {
                                let (full_len, raw_len, pooled_len, selected_len, position) =
                                    op.next_lengths().map_err(|error| {
                                        Qwen4GpuForwardError::Dispatch(format!(
                                            "derive lane QSA state: {error:?}"
                                        ))
                                    })?;
                                qsa_commits.push((
                                    seg.first_lane,
                                    slot,
                                    full_len,
                                    raw_len,
                                    pooled_len,
                                    selected_len,
                                    position,
                                ));
                            }
                            steps.push(Step::IndexedAttention(op));
                        }
                        LaneOp::Moe { layer } => {
                            // AR lanes route by their real rows (`rr == r`); a
                            // prefill wave seals at its routed rows so a 1-row
                            // lane takes the prefill MoE under the profile.
                            let Qwen4FfnBinding::Moe(moe) = descs[layer - band_first].ffn else {
                                return Err(invalid("lane steps run only the Flash-Next MoE FFN"));
                            };
                            // The D2 executor is bound only on a non-prefill
                            // step over a whole narrow window (see
                            // `Qwen4MoeLayerRuntime::routed_view`).
                            let mut moe = moe;
                            // The runtime this invocation selected (the static
                            // runtime outside EP; under EP the singleton's
                            // gathered-forward selection, see `lane_layer_moe`; an
                            // EP prefill wave's runtime of the bank in use, exactly the
                            // singleton chunk's emission for a sub-512-row tile of the wave's
                            // route: FULL-TABLE (no gather, no tile exchange) or TILE (the
                            // compact runtime plus the owner-routed tile exchange).
                            if ep_wave {
                                let route = wave_route.unwrap_or(EpWaveRoute::FullTable);
                                if let Some(runtime) = ep_wave_layer_moe(
                                    route,
                                    fwd.ep.as_ref(),
                                    &fwd.moe,
                                    &fwd.moe_borrow,
                                    fwd.borrow_runtimes_active,
                                    fwd.borrow_view,
                                    layer,
                                ) {
                                    moe.routed_experts =
                                        runtime.routed_view(false, r, rr, false, 8);
                                }
                                if route == EpWaveRoute::Tile {
                                    moe.tile_exchange = Some(
                                        ep_wave_tile_exchange(fwd.ep.as_ref()).ok_or_else(|| {
                                            invalid(
                                                "a tile-route EP wave needs the installed tile exchange",
                                            )
                                        })?,
                                    );
                                }
                                // The PartialMask wave masks the non-owned tiles exactly as the
                                // lanes' own partial chunks do (the trunk shard's partial tile
                                // mask); the Partial wave's lanes take no mask, nor does it.
                                if route == EpWaveRoute::PartialMask {
                                    moe.tile_exchange = Some(
                                        ep_wave_partial_mask(fwd.ep.as_ref()).ok_or_else(|| {
                                            invalid(
                                                "a partial-mask EP wave needs the installed tile route's mask",
                                            )
                                        })?,
                                    );
                                }
                                wave_moe_calls += 1;
                            } else if let Some(runtime) =
                                lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer)
                            {
                                moe.routed_experts = runtime.routed_view(!prefill, r, rr, false, 8);
                            }
                            // Under EP the invocation runs the singleton's
                            // gathered forward: the row gather, the few-row
                            // shared-down output, and the next buffer of the
                            // step's alternation as the expanded-row scratch.
                            let seal_scratch = match (&ep_lane, &ep_seg_scratch) {
                                (Some(ep), Some(per_seg)) => {
                                    ep_lane_moe_rows(r, rr, ep.rows_cap)?;
                                    bind_ep_lane_gather(&mut moe, r, ep.gather, ep.shared_down);
                                    let buffer = ep_lane_buffer(ep_invocations);
                                    ep_invocations += 1;
                                    &per_seg[i][buffer]
                                }
                                _ => &seg_scratch[i],
                            };
                            let sealed = seal_moe_for_layer(&ctx, dims, moe, seal_scratch, r, rr)
                                .map_err(|error| {
                                Qwen4GpuForwardError::Dispatch(format!("seal lane MoE: {error:?}"))
                            })?;
                            match hc_seg {
                                // The all-rows MoE: ONE zero fill of its n rows ahead of its
                                // one call (the group-wise fills' union).
                                Some(_) if Some(i) == moe_seg => steps.push(Step::Clear(ClearOp {
                                    tensor: &v.ffn_output,
                                    elements: r * config.hidden_size,
                                })),
                                // All-rows HC: ONE zero fill of every row's FFN output ahead of
                                // the first group's call (the all-rows HC write+read launch
                                // before it absorbs it), so every group's call starts from the
                                // zeros its own fill would have left.
                                Some(hc)
                                    if policy.share(StageName::Moe) == StageShare::OnceAllRows =>
                                {
                                    if i == lane_segs.len() {
                                        steps.push(Step::Clear(ClearOp {
                                            tensor: &views[hc].ffn_output,
                                            elements: n * config.hidden_size,
                                        }));
                                    }
                                }
                                _ => steps.push(Step::Clear(ClearOp {
                                    tensor: &v.ffn_output,
                                    elements: r * config.hidden_size,
                                })),
                            }
                            steps.push(Step::Moe(sealed));
                        }
                    }
                }
            }
            start = end;
        }
        // Sequence-parallel wave: the final hyper read and the wide-hidden capture read every row.
        if let Some(sp) = sp_views.as_ref().filter(|_| sp_streams_partial) {
            steps.push(Step::RowGather(RowGatherOp {
                tensor: &scratch.streams,
                f32_bf16: !bf16_state,
                row_elems: dims.wide(),
                rows: n,
                exec: sp.exec,
            }));
        }
        if ep_lane.is_some() {
            ep_lane_check_invocations(ep_invocations)?;
        }
        if ep_wave && wave_moe_calls != ep_wave_expected_moe_calls(band_scope.band.len()) {
            return Err(invalid(format!(
                "an EP prefill wave emitted {wave_moe_calls} MoE invocations for {} layers; every \
                 rank must run exactly one exchange per layer",
                band_scope.band.len()
            )));
        }
        if lane_step_trace_enabled() {
            let kind = if prefill {
                "prefill"
            } else if lanes
                .iter()
                .any(|lane| matches!(lane.kind, LaneRowKind::Verify { .. }))
            {
                "verify"
            } else {
                "ar"
            };
            let calls: Vec<String> = StageName::ALL
                .iter()
                .filter(|stage| stage_calls[stage.index()] != 0)
                .map(|stage| format!("{}={}", stage.key(), stage_calls[stage.index()]))
                .collect();
            let layers = band_scope.band.len().max(1);
            let moe_calls = stage_calls[StageName::Moe.index()];
            let (moe_place, moe_rows) = if moe_all {
                ("allrows".to_string(), vec![n])
            } else {
                let place = match &moe_refusal {
                    Some(reason) => format!("groups(refused: {reason})"),
                    None => "groups".to_string(),
                };
                let moe_rows = if policy.share(StageName::Moe).per_lane() {
                    rows.clone()
                } else {
                    group_rows.clone()
                };
                (place, moe_rows)
            };
            eprintln!(
                "[lane-step-trace] kind={kind} lanes={} rows={n} lane_rows={rows:?} groups={} \
                 group_rows={group_rows:?} hc_allrows={hc_all} stages_allrows={all_stages} \
                 exact_scope={exact_scope} \
                 moe={moe_place} moe_rows={moe_rows:?} layers={layers} moe_calls={moe_calls} \
                 moe_calls_per_layer={:.2} launches=[{}]",
                lanes.len(),
                group_segs.len(),
                moe_calls as f64 / layers as f64,
                calls.join(" ")
            );
        }
        // Exact: the all-rows steps are validated under the scope they run in.
        let preflight_prior = exact_scope.then(|| gpu.set_wide_verify_rows(n));
        let preflight = validate_steps(gpu, &steps);
        if let Some(prior) = preflight_prior {
            gpu.set_wide_verify_rows(prior);
        }
        preflight.map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight lane program: {error:?}"))
        })?;

        // Inputs. The input band: ids, embeddings, HC stream broadcast. Every
        // other band runs from the HC streams its caller placed in
        // `scratch.streams` rows `[0, n)` (lane order).
        if band_scope.input {
            if ids_async {
                fwd.lane_tokens.upload(
                    gpu,
                    &scratch.token_ids.buf,
                    n * std::mem::size_of::<i32>(),
                )?;
            } else {
                gpu.memcpy_htod_auto(
                    &scratch.token_ids.buf,
                    &fwd.host_token_bytes[..n * std::mem::size_of::<i32>()],
                )?;
            }
            let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
            let embed_share = policy.share(StageName::Embed);
            for i in stage_segs(embed_share) {
                let (seg, v) = (&all_segs[i], &views[i]);
                execute_embedding(
                    gpu,
                    &EmbeddingOp {
                        table: embedding,
                        rotated: &v.embedding_rot,
                        token_ids: &v.token_ids,
                        output: &v.embeddings,
                        rows: seg.rows,
                        dim: config.hidden_size,
                    },
                )
                .map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("lane embedding: {error:?}"))
                })?;
            }
            for i in stage_segs(embed_share) {
                let (seg, v) = (&all_segs[i], &views[i]);
                let streams = view(&v.streams, 0, seg.rows * dims.wide());
                if bf16_state {
                    gpu.hc_streams_init_from_embed_batched_bf16(
                        &v.embeddings,
                        &streams,
                        config.hidden_size as i32,
                        config.hc_count as i32,
                        seg.rows as i32,
                    )?;
                } else {
                    gpu.hc_streams_init_from_embed_batched(
                        &v.embeddings,
                        &streams,
                        config.hidden_size as i32,
                        config.hc_count as i32,
                        seg.rows as i32,
                    )?;
                }
            }
        }

        // The all-rows HC steps run under the wide few-row scope (`HcAllRowsScope`); every other
        // step keeps the step's own.
        let prior_wide = gpu.wide_verify_rows;
        let execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
            let result = if hc_seg.is_some() {
                let mut scope = HcAllRowsScope {
                    rows: n,
                    prior: prior_wide,
                    exact: exact_scope,
                    moe: moe_all,
                };
                let result = execute_validated_steps_with_moe_hooks(gpu, &ctx, steps, &mut scope);
                gpu.set_wide_verify_rows(prior_wide);
                result
            } else {
                execute_validated_steps(gpu, &ctx, steps)
            };
            result.map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane program: {error:?}"))
            })
        };
        // The PLE conv (input band) splits the program: its rows are fetched,
        // staged and uploaded while the layers before it already run on the
        // device. A band without the PLE layer runs one segment.
        let split = match (&ple, ple_split) {
            (Some(_), Some(split)) => split,
            (None, None) => steps.len(),
            _ => {
                return Err(invalid(
                    "lane program PLE placement disagrees with the band's input ownership",
                ));
            }
        };
        // Arm the sequence-parallel split for the wave's steps (own-row HC writes take the
        // reduce-scatter, `RowGather`s use its ranges); disarmed on every exit of this closure.
        struct SpArm;
        impl Drop for SpArm {
            fn drop(&mut self) {
                crate::trunk_route::sp_end();
            }
        }
        let _sp_arm = sp_views.as_ref().map(|sp| {
            crate::trunk_route::sp_begin(sp.rows);
            SpArm
        });
        execute(gpu, &steps[..split])?;

        // PLE rows: wait for the fetch, stage, upload, convert.
        if let Some(ple) = ple.as_mut() {
            let lease = ple
                .wait()
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
            let upload_len = lease
                .as_bytes()
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
                .len();
            let staged = view(&scratch.ple_staged, 0, n * ple_head_count * ple_row_width);
            let buffer = fwd.host_ple.stage_buffer(gpu, upload_len)?;
            lease
                .stage_into(buffer)
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
            if ple_async_upload(fwd.host_ple.enabled(), false, true, gpu.graphs.capture_mode) {
                fwd.host_ple.upload_async(gpu, &staged.buf, upload_len)?;
            } else {
                fwd.host_ple.upload_blocking(gpu, &staged.buf, upload_len)?;
            }
            lease
                .validate_after_upload()
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
            let rows_share = policy.share(StageName::PleRows);
            // F16 rows are compact halves at the base of their view, so they are
            // converted only through the lane views the PLE conv reads (the
            // gather is elementwise: the bytes equal the singleton's).
            let ple_rows_segs = if ple_rows_f16 {
                0..lane_segs.len()
            } else {
                stage_segs(rows_share)
            };
            for i in ple_rows_segs {
                let (seg, v) = (&all_segs[i], &views[i]);
                let row_width = ple_head_count * ple_row_width;
                let staged_rows = view(&staged, seg.begin * row_width, seg.rows * row_width);
                let dest = view(&v.ple_rows, 0, seg.rows * row_width);
                if ple_rows_f16 {
                    gpu.grouped_gather_convert_bf16_f16(
                        &staged_rows,
                        &dest,
                        seg.rows,
                        ple_head_count,
                        ple_row_width,
                    )?;
                } else {
                    gpu.grouped_gather_convert_bf16(
                        &staged_rows,
                        &dest,
                        seg.rows,
                        ple_head_count,
                        ple_row_width,
                    )?;
                }
            }
        }
        execute(gpu, &steps[split..])?;

        // Wide hidden capture: every row's final streams, lanes concatenated
        // (the singleton verify's `wide_hidden_capture` copy).
        if let Some(capture) = out.wide_hidden() {
            let wide_elements = n * dims.wide();
            let source = view(&scratch.streams, 0, wide_elements);
            let destination = view(capture, 0, wide_elements);
            if bf16_state {
                rdna_compute::tensor_ops::hc_state_bf16_to_f32(
                    gpu,
                    &source,
                    &destination,
                    wide_elements,
                )?;
            } else {
                gpu.copy_d2d(&source, &destination, destination.byte_size())?;
            }
        }

        // Final HC read, LM head, greedy top-1.
        let final_share = policy.share(StageName::FinalHyper);
        let final_segs: Vec<usize> = match hc_seg {
            Some(hc) if hc_final => vec![hc],
            _ => head_segs(final_share).collect(),
        };
        for i in final_segs {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (final_hyper, _) = head_ref()?;
            // The all-rows final read runs under the wide few-row scope, like the layers' HC.
            let all_rows = Some(i) == hc_seg;
            if all_rows {
                gpu.set_wide_verify_rows(seg.rows);
            }
            let result = execute_final_hyper(
                gpu,
                dims,
                &final_hyper,
                &v.streams,
                &seg_scratch[i],
                seg.rows,
                bf16_state,
            );
            if all_rows {
                gpu.set_wide_verify_rows(prior_wide);
            }
            result.map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane final hyper: {error:?}"))
            })?;
        }
        let head_share = policy.share(StageName::Head);
        for i in ar_segs(head_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (_, lm_head) = head_ref()?;
            let hidden = view(&v.hc_mixed, 0, seg.rows * config.hidden_size);
            let rotation = view(&scratch.rotation, 0, seg.rows * config.hidden_size);
            // Exact: the all-rows LM head runs in the wide few-row scope like the steps before it.
            let scoped = exact_scope && Some(i) == hc_seg;
            if scoped {
                gpu.set_wide_verify_rows(seg.rows);
            }
            let head = execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                &view(&v.logits, 0, seg.rows * head_width),
                seg.rows,
                seg.rows,
                Some(&rotation),
            );
            if scoped {
                gpu.set_wide_verify_rows(prior_wide);
            }
            head.map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane LM head: {error:?}"))
            })?;
        }
        let argmax_share = policy.share(StageName::Argmax);
        for i in ar_segs(argmax_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: &view(&v.logits, 0, seg.rows * head_width),
                    indices: &v.top1,
                    rows: seg.rows,
                    vocab: head_width,
                },
            )?;
        }
        // Prefill wave head: per lane, the singleton's own LM-head / argmax
        // forms on the lane's rows (Final: `requested_rows == 1`).
        // `HIPFIRE_QWEN4_WAVE_HEAD_ONE` (default on, `=0` kills; exact gfx1201): a Final wave of >= 2 lanes gathers
        // every lane's final row into the first rows of `scratch.hc_mixed` (ascending, so no
        // source row is overwritten before it is read) and runs ONE all-rows LM head and ONE
        // argmax over them under the wide few-row scope, instead of one one-row GEMV per lane.
        let wave_head_one = match out {
            StepOutputs::Wave(wave)
                if matches!(wave.head, WaveHead::Final)
                    && lane_segs.len() >= 2
                    && gpu.flags.qwen4_wave_head_one_enabled() =>
            {
                Some(wave)
            }
            _ => None,
        };
        if let Some(wave) = wave_head_one {
            let lanes_n = lane_segs.len();
            let hidden_n = config.hidden_size;
            if *crate::lane_exec::LANE_TILE_PHASES {
                eprintln!("[fn-lane-wave-head] one all-rows LM head launch for {lanes_n} lanes");
            }
            for (lane, seg) in lane_segs.iter().enumerate() {
                let v = &views[lane];
                let source = view(&v.hc_mixed, (seg.rows - 1) * hidden_n, hidden_n);
                let destination = view(&scratch.hc_mixed, lane * hidden_n, hidden_n);
                if seg.begin + seg.rows - 1 != lane {
                    gpu.copy_d2d(&source, &destination, destination.byte_size())?;
                }
            }
            let (_, lm_head) = head_ref()?;
            let hidden = view(&scratch.hc_mixed, 0, lanes_n * hidden_n);
            let rotation = view(&scratch.rotation, 0, lanes_n * hidden_n);
            let logits = view(wave.logits, 0, lanes_n * vocab);
            gpu.set_wide_verify_rows(lanes_n);
            let head = execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                &logits,
                lanes_n,
                lanes_n,
                Some(&rotation),
            );
            gpu.set_wide_verify_rows(prior_wide);
            head.map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!(
                    "execute wave one-launch LM head: {error:?}"
                ))
            })?;
            if let Some(top1) = wave.top1 {
                argmax_f32(
                    gpu,
                    &ArgmaxF32 {
                        logits: &logits,
                        indices: &view(top1, 0, lanes_n * std::mem::size_of::<i32>()),
                        rows: lanes_n,
                        vocab,
                    },
                )?;
            }
        }
        for (lane, (head_logits, head_top1, requested)) in wave_head.iter().enumerate() {
            if wave_head_one.is_some() {
                break;
            }
            let (seg, v) = (&lane_segs[lane], &views[lane]);
            let (_, lm_head) = head_ref()?;
            let hidden = view(&v.hc_mixed, 0, seg.rows * config.hidden_size);
            let rotation = view(&scratch.rotation, 0, seg.rows * config.hidden_size);
            execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                head_logits,
                seg.rows,
                *requested,
                Some(&rotation),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute wave LM head: {error:?}"))
            })?;
            if let Some(top1) = head_top1 {
                argmax_f32(
                    gpu,
                    &ArgmaxF32 {
                        logits: head_logits,
                        indices: top1,
                        rows: *requested,
                        vocab,
                    },
                )?;
            }
        }
        drop(steps);
        Ok(qsa_commits)
    })();
    if let Some(prior) = row_exact_prior {
        gpu.lane_row_exact_scope = prior;
    }

    match attempt {
        Ok(qsa_commits) => {
            if ep_wave {
                fwd.ep_wave_note(n);
            }
            if let Some(ple) = ple.as_mut() {
                ple.complete();
            }
            for (lane, slot, full_len, raw_len, pooled_len, selected_len, position) in qsa_commits {
                let state = lanes[lane]
                    .state
                    .qsa_mut(slot)
                    .ok_or_else(|| invalid("lane QSA state slot disappeared"))?;
                state.full_len = full_len;
                state.raw_len = raw_len;
                state.pooled_len = pooled_len;
                state.selected_len = selected_len;
                state.position = position;
                state.clamp_f16_valid();
            }
            for lane in lanes.iter_mut() {
                for &token in lane.tokens {
                    lane.state.ple_history.push(token);
                }
                lane.state.commit_row_capture(lane.tokens.len());
                lane.state.position = lane
                    .state
                    .position
                    .checked_add(lane.tokens.len())
                    .ok_or_else(|| invalid("lane position overflows at commit"))?;
            }
            Ok(())
        }
        Err(error) => {
            // The full-table route stages nothing (no donor is overwritten): nothing to restore.
            for lane in lanes.iter() {
                lane.state.invalidate_f16_arenas();
            }
            match ple.as_mut().map_or(Ok(0u64), |ple| ple.abort()) {
                Ok(_) => Err(error),
                Err(cleanup) => Err(Qwen4GpuForwardError::Ple(format!(
                    "lane step failed: {error}; PLE cleanup failed: {cleanup}"
                ))),
            }
        }
    }
}

/// Lab probe (A2, `ep_moe_layer_probe`): run ONE trunk MoE layer of this EP
/// rank over the F32 normalized input rows `input` (`rows * hidden`), cut into
/// consecutive `groups` of rows, each group one gathered MoE invocation exactly
/// as a lane step seals it (`ep_lane_moe_rows`, `bind_ep_lane_gather`, the
/// alternating gather buffer from `start_ordinal`), with the verify scope
/// `scope_rows` open (`0` = closed, the lane step's own scope). Returns every
/// group's `ffn_output` rows, bytes, concatenated in row order, and whether the
/// layer runs balanced ownership. Every rank of the mesh must run it with the
/// same arguments (the gather is a collective).
pub(crate) fn ep_moe_probe_rank(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    layer: usize,
    input: &[f32],
    groups: &[usize],
    start_ordinal: usize,
    scope_rows: usize,
) -> Result<(Vec<u8>, bool), Qwen4GpuForwardError> {
    let mut forward = bundle
        .execution
        .take()
        .ok_or_else(|| invalid("Qwen4 forward resources are not attached"))?;
    let prior_scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    let prior_wide = gpu.set_wide_verify_rows(scope_rows);
    let result = ep_moe_probe_scoped(
        gpu,
        &mut forward,
        bundle,
        layer,
        input,
        groups,
        start_ordinal,
    );
    gpu.set_wide_verify_rows(prior_wide);
    gpu.qwen4_scope = prior_scope;
    bundle.execution = Some(forward);
    result
}

fn ep_moe_probe_scoped(
    gpu: &mut Gpu,
    fwd: &mut Qwen4GpuForward,
    bundle: &Qwen4Bundle,
    layer: usize,
    input: &[f32],
    groups: &[usize],
    start_ordinal: usize,
) -> Result<(Vec<u8>, bool), Qwen4GpuForwardError> {
    let config = &bundle.config;
    let dims = program_dims(config);
    let hidden = config.hidden_size;
    let total: usize = groups.iter().sum();
    if groups.is_empty() || input.len() != total * hidden {
        return Err(invalid(format!(
            "EP MoE probe: {} input values for {total} rows of {hidden}",
            input.len()
        )));
    }
    let ep = ep_lane(fwd.ep.as_ref(), &fwd.scratch)?
        .ok_or_else(|| invalid("EP MoE probe needs an EP forward"))?;
    for &rows in groups {
        ep_lane_moe_rows(rows, rows, ep.rows_cap)?;
        if rows > fwd.scratch.max_chunk {
            return Err(invalid(format!(
                "EP MoE probe group of {rows} rows exceeds the forward scratch ({} rows)",
                fwd.scratch.max_chunk
            )));
        }
    }
    let balanced = fwd.ep_layer_is_balanced(layer);
    gpu.set_router_row_base(0);
    let staged = gpu.upload_f32(input, &[input.len()])?;
    let ctx = DispatchCtx::new(gpu);
    let scratch = &fwd.scratch;
    let mut out = Vec::with_capacity(input.len() * std::mem::size_of::<f32>());
    let mut begin = 0usize;
    for (index, &rows) in groups.iter().enumerate() {
        let desc = layer_desc(
            &bundle.weights,
            bundle
                .weights
                .layer(layer)
                .ok_or_else(|| invalid(format!("EP MoE probe layer {layer} is not owned")))?,
            lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer),
            config,
            &fwd.decode_q8,
            rows,
            gpu.few_row_limit(),
            rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS,
            None,
        )?;
        let Qwen4FfnBinding::Moe(moe) = desc.ffn else {
            return Err(invalid(format!(
                "EP MoE probe layer {layer} is not a MoE layer"
            )));
        };
        let mut moe = moe;
        if let Some(runtime) = lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer) {
            moe.routed_experts = runtime.routed_view(true, rows, rows, false, 8);
        }
        bind_ep_lane_gather(&mut moe, rows, ep.gather, ep.shared_down);
        let router_logits = matrix_view(&scratch.router_logits, rows, router_columns(dims))?;
        let base = crate::gpu_forward::layer_scratch(scratch, &router_logits);
        let buffer = ep_lane_buffer(start_ordinal + index);
        let seal_scratch = Qwen4LayerScratch {
            moe_down_expanded: ep.buffers[buffer],
            ..base
        };
        let sealed =
            seal_moe_for_layer(&ctx, dims, moe, &seal_scratch, rows, rows).map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("seal probe MoE: {error:?}"))
            })?;
        let source = view(&staged, begin * hidden, rows * hidden);
        let destination = view(&scratch.hc_mixed, 0, rows * hidden);
        gpu.copy_d2d(&source, &destination, destination.byte_size())?;
        let steps = [
            Step::Clear(ClearOp {
                tensor: &scratch.ffn_output,
                elements: rows * hidden,
            }),
            Step::Moe(sealed),
        ];
        validate_steps(gpu, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("preflight probe MoE: {error:?}"))
        })?;
        execute_validated_steps(gpu, &ctx, &steps).map_err(|error| {
            Qwen4GpuForwardError::Dispatch(format!("execute probe MoE: {error:?}"))
        })?;
        gpu.hip.device_synchronize()?;
        let rows_out = view(&scratch.ffn_output, 0, rows * hidden);
        out.extend_from_slice(&gpu.download_raw_bytes(&rows_out)?);
        begin += rows;
    }
    drop(ctx);
    gpu.free_tensor(staged)?;
    Ok((out, balanced))
}

/// HC storage of an AR lane step. Both devices of a PP edge must agree.
pub(crate) fn lane_stream_bf16(gpu: &Gpu) -> bool {
    gpu.qwen4_bf16_streams(1)
}

impl Qwen4Bundle {
    /// The band's lane-step HC streams: after a successful lane step on a
    /// non-output band, rows `[0, total_rows)` in lane order hold the band's
    /// output streams (`rows * hc_count * hidden` elements of
    /// [`lane_stream_bf16`] storage, contiguous from the start); before the
    /// step they are the input of a non-input band. The tensor is the whole
    /// scratch field: view the live prefix.
    pub(crate) fn lane_edge_streams(&self) -> Result<&GpuTensor, String> {
        self.execution
            .as_ref()
            .map(|forward| &forward.scratch.streams)
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Qwen4BandScope;

    #[test]
    fn production_policy_round_trips_through_describe_and_parse() {
        let production = LaneStagePolicy::production();
        let described = production.describe();
        let spec = format!("per-lane:{}", described);
        assert_eq!(LaneStagePolicy::parse(&spec).unwrap(), production);
        assert_eq!(LaneStagePolicy::parse("production").unwrap(), production);
        assert_eq!(described.split(',').count(), StageName::ALL.len());
    }

    #[test]
    fn stateful_stages_refuse_shared_launches() {
        for stage in StageName::ALL {
            let shared = LaneStagePolicy::all_per_lane().with(stage, StageShare::OnceAllRows);
            assert_eq!(shared.is_err(), stage.stateful(), "{}", stage.key());
        }
        assert!(LaneStagePolicy::parse("per-lane:gdn_core=once").is_err());
        assert!(LaneStagePolicy::parse("per-lane:moe=once").is_ok());
        assert!(LaneStagePolicy::parse("per-lane:nope=once").is_err());
        assert!(LaneStagePolicy::parse("per-lane:moe=sometimes").is_err());
        assert!(LaneStagePolicy::parse("other").is_err());
    }

    #[test]
    fn stage_keys_are_unique_and_parse_back() {
        for stage in StageName::ALL {
            assert_eq!(StageName::parse(stage.key()), Some(stage));
        }
        let mut keys: Vec<_> = StageName::ALL.iter().map(|stage| stage.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), StageName::ALL.len());
    }

    #[test]
    fn production_policy_shares_row_local_stages() {
        let production = LaneStagePolicy::production();
        let shared = StageName::ALL
            .iter()
            .any(|stage| production.share(*stage) == StageShare::OnceAllRows);
        assert!(shared);
        assert!(production.validate().is_ok());
    }

    #[test]
    fn policies_compare_by_shares() {
        let production = LaneStagePolicy::production();
        assert_eq!(
            production
                .with(StageName::Moe, StageShare::OnceAllRows)
                .unwrap(),
            production
        );
        assert_ne!(
            production
                .with(StageName::Moe, StageShare::PerLane)
                .unwrap(),
            production
        );
        assert_eq!(
            LaneStagePolicy::parse("production:moe=once").unwrap(),
            production
        );
        assert_ne!(
            LaneStagePolicy::parse("production:moe=lane").unwrap(),
            production
        );
        assert_eq!(
            LaneStagePolicy::parse("per-lane").unwrap(),
            LaneStagePolicy::all_per_lane()
        );
    }

    const ALL_TOPOLOGIES: [LaneTopology; 3] =
        [LaneTopology::Single, LaneTopology::Pp, LaneTopology::Ep];

    /// Stages the production table shares: the row-local stages whose proof
    /// rows the table carries.
    fn shared_stages() -> Vec<StageName> {
        STAGE_SHARE
            .iter()
            .filter(|(stage, share)| *share == StageShare::OnceAllRows && !stage.stateful())
            .map(|(stage, _)| *stage)
            .collect()
    }

    /// A route instance a proof row admits: `Any` rows admit every instance,
    /// represented by an unclassified one.
    fn instance_of(route: StageRoute) -> StageRoute {
        if route == StageRoute::Any {
            StageRoute::Unclassified
        } else {
            route
        }
    }

    /// The owned route instances a lane store of `arch` / `topology` would
    /// register, populated from the proof rows of that arch whose domain
    /// covers the topology (one instance per row).
    fn routes_from(proofs: &[RowTwinProof], arch: &str, topology: LaneTopology) -> LaneStageRoutes {
        let mut routes = LaneStageRoutes::default();
        for proof in proofs
            .iter()
            .filter(|proof| proof.arch == arch && proof.domain.contains(&topology))
        {
            routes[proof.stage.index()].push(instance_of(proof.route));
        }
        routes
    }

    fn table_routes(arch: &str, topology: LaneTopology) -> LaneStageRoutes {
        routes_from(LANE_ROW_TWIN_PROOFS, arch, topology)
    }

    /// The route instances a PP2 resident-mq2 lane store registers, written
    /// out from the artifact header (qt47 MQ6G256V2 GDN in/out, QSA
    /// in/out and head; qt3 Q8_0 embed; BF16 PLE rows, hyper-connection,
    /// router and shared/dense experts; F32 argmax), NOT read from the proof
    /// table, so a wrong table row fails against it.
    fn pp2_mq2_routes() -> LaneStageRoutes {
        let mq6 = DType::MQ6G256V2;
        let bf16 = DType::BF16;
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Embed.index()] = vec![StageRoute::Embed(DType::Q8_0)];
        routes[StageName::PleRows.index()] = vec![StageRoute::PleRows(bf16)];
        routes[StageName::HcRead.index()] = vec![StageRoute::HyperRead([bf16; 3])];
        routes[StageName::HcWrite.index()] = vec![StageRoute::HyperWrite(bf16)];
        routes[StageName::GdnInProj.index()] = vec![StageRoute::GdnIn([mq6; 4])];
        routes[StageName::GdnOutProj.index()] = vec![StageRoute::Projection(mq6)];
        routes[StageName::QsaInProj.index()] = vec![StageRoute::QsaIn([mq6; 4])];
        routes[StageName::QsaOutProj.index()] = vec![StageRoute::Projection(mq6)];
        routes[StageName::Moe.index()] = vec![StageRoute::Moe {
            experts: LaneExpertRoute::ResidentMq2Slab,
            dense: [bf16; 5],
            decode_admitted: true,
        }];
        routes[StageName::FinalHyper.index()] = vec![StageRoute::HyperRead([bf16; 3])];
        routes[StageName::Head.index()] = vec![StageRoute::Projection(mq6)];
        routes[StageName::Argmax.index()] = vec![StageRoute::Argmax(DType::F32)];
        routes
    }

    /// The route instances a single-device gfx942 stock-FP8 lane store
    /// registers, written out from the artifact (BF16 embed, PLE rows and
    /// projections, hyper-connection, GDN in/out, QSA in/out, router /
    /// shared experts and head; resident qt=55 routed experts; F32 argmax),
    /// NOT read from the proof table.
    fn g942_stock_fp8_routes() -> LaneStageRoutes {
        let bf16 = DType::BF16;
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Embed.index()] = vec![StageRoute::Embed(bf16)];
        routes[StageName::PleRows.index()] = vec![StageRoute::PleRows(bf16)];
        routes[StageName::PleProject.index()] = vec![StageRoute::PleProject([bf16; 2])];
        routes[StageName::HcRead.index()] = vec![StageRoute::HyperRead([bf16; 3])];
        routes[StageName::HcWrite.index()] = vec![StageRoute::HyperWrite(bf16)];
        routes[StageName::GdnInProj.index()] = vec![StageRoute::GdnIn([bf16; 4])];
        routes[StageName::GdnOutProj.index()] = vec![StageRoute::Projection(bf16)];
        routes[StageName::QsaInProj.index()] = vec![StageRoute::QsaIn([bf16; 4])];
        routes[StageName::QsaOutProj.index()] = vec![StageRoute::Projection(bf16)];
        routes[StageName::Moe.index()] = vec![StageRoute::Moe {
            experts: LaneExpertRoute::ResidentFp8,
            dense: [bf16; 5],
            decode_admitted: false,
        }];
        routes[StageName::FinalHyper.index()] = vec![StageRoute::HyperRead([bf16; 3])];
        routes[StageName::Head.index()] = vec![StageRoute::Projection(bf16)];
        routes[StageName::Argmax.index()] = vec![StageRoute::Argmax(DType::F32)];
        routes
    }

    /// Stages the resolution places per lane for a reason other than state.
    fn fallback_stages(resolution: &LanePolicyResolution) -> Vec<StageName> {
        resolution
            .per_lane
            .iter()
            .filter(|(_, reason)| reason != "stateful")
            .map(|(stage, _)| *stage)
            .collect()
    }

    fn reason_of(resolution: &LanePolicyResolution, stage: StageName) -> &str {
        resolution
            .per_lane
            .iter()
            .find(|(named, _)| *named == stage)
            .map(|(_, reason)| reason.as_str())
            .unwrap_or_else(|| panic!("no per-lane reason for {}", stage.key()))
    }

    /// Every stateful stage is per lane with the reason `stateful`, once.
    fn assert_stateful_reasons(resolution: &LanePolicyResolution) {
        for stage in StageName::ALL.into_iter().filter(|stage| stage.stateful()) {
            let hits: Vec<_> = resolution
                .per_lane
                .iter()
                .filter(|(named, _)| *named == stage)
                .collect();
            assert_eq!(hits.len(), 1, "{}", stage.key());
            assert_eq!(hits[0].1, "stateful", "{}", stage.key());
            assert_eq!(resolution.policy.share(stage), StageShare::PerLane);
        }
    }

    fn production_without(fallbacks: &[StageName]) -> LaneStagePolicy {
        fallbacks
            .iter()
            .fold(LaneStagePolicy::production(), |policy, stage| {
                policy.with(*stage, StageShare::PerLane).unwrap()
            })
    }

    #[test]
    fn row_twin_table_has_one_row_per_shared_stage_arch_and_domain() {
        let shared = shared_stages();
        assert_eq!(shared.len(), 13);
        // gfx1151: one all-topology row per shared stage. gfx1201: one PP row
        // and one EP row per shared stage, plus the second PP MoE row (the
        // mq3 slab class), the two EP Q8F16 QSA overlay rows (qsa_in x4,
        // qsa_out) and the three EP Q8F16 trunk rows (gdn_in x4, gdn_out,
        // head), and the two EP HC8 rows (hc_read, final_hyper). gfx942: one
        // single-device row per shared stage (the stock-FP8 receipt).
        // `ple_project` (the lane-wide PLE key / value projection) has no gfx1151 row: its flag is
        // exact-gfx1201-only, so gfx1151 carries one row per shared stage but that one.
        assert_eq!(
            LANE_ROW_TWIN_PROOFS.len(),
            (shared.len() - 1) + 2 * shared.len() + 1 + 2 + 3 + 2 + shared.len()
        );
        for arch in ["gfx1151", "gfx1201", "gfx942"] {
            for stage in StageName::ALL {
                for topology in ALL_TOPOLOGIES {
                    let rows: Vec<_> = LANE_ROW_TWIN_PROOFS
                        .iter()
                        .filter(|proof| {
                            proof.arch == arch
                                && proof.stage == stage
                                && proof.domain.contains(&topology)
                        })
                        .collect();
                    let expected = match (arch, topology) {
                        ("gfx1151", _) => shared.contains(&stage) && stage != StageName::PleProject,
                        ("gfx1201", LaneTopology::Pp) => shared.contains(&stage),
                        ("gfx1201", LaneTopology::Ep) => shared.contains(&stage),
                        ("gfx942", LaneTopology::Single) => shared.contains(&stage),
                        _ => false,
                    };
                    // The PP MoE stage carries the mq2 and the mq3 slab rows;
                    // the EP QSA in/out stages the MQ6 and the Q8F16 rows.
                    let count = usize::from(expected)
                        + usize::from(
                            expected
                                && arch == "gfx1201"
                                && topology == LaneTopology::Pp
                                && stage == StageName::Moe,
                        )
                        + usize::from(
                            expected
                                && arch == "gfx1201"
                                && topology == LaneTopology::Ep
                                && matches!(
                                    stage,
                                    StageName::QsaInProj
                                        | StageName::QsaOutProj
                                        | StageName::GdnInProj
                                        | StageName::GdnOutProj
                                        | StageName::Head
                                        | StageName::HcRead
                                        | StageName::FinalHyper
                                ),
                        );
                    assert_eq!(rows.len(), count, "{arch} {topology:?} {}", stage.key());
                    for proof in rows {
                        assert!(!proof.receipt.is_empty());
                        match (arch, topology) {
                            ("gfx1151", _) => {
                                assert_eq!(proof.route, StageRoute::Any);
                                assert_eq!(proof.domain, ALL_TOPOLOGIES);
                                assert_eq!(proof.receipt, GFX1151_LANE_RECEIPT);
                            }
                            ("gfx942", _) => {
                                assert_ne!(proof.route, StageRoute::Any);
                                assert_ne!(proof.route, StageRoute::Unclassified);
                                assert_eq!(proof.domain, [LaneTopology::Single]);
                                assert_eq!(proof.receipt, GFX942_STOCK_FP8_RECEIPT);
                                assert_eq!(
                                    proof.route
                                        == StageRoute::Moe {
                                            experts: LaneExpertRoute::ResidentFp8,
                                            dense: [DType::BF16; 5],
                                            decode_admitted: false,
                                        },
                                    stage == StageName::Moe
                                );
                            }
                            (_, LaneTopology::Pp) => {
                                assert_ne!(proof.route, StageRoute::Any);
                                assert_ne!(proof.route, StageRoute::Unclassified);
                                assert_eq!(proof.domain, [LaneTopology::Pp]);
                                if stage == StageName::Moe
                                    && matches!(
                                        proof.route,
                                        StageRoute::Moe {
                                            experts: LaneExpertRoute::ResidentMq3Slab,
                                            ..
                                        }
                                    )
                                {
                                    assert_eq!(proof.receipt, GFX1201_PP2_MQ3_RECEIPT);
                                } else if stage == StageName::PleProject {
                                    assert_eq!(
                                        proof.route,
                                        StageRoute::PleProject([DType::BF16; 2])
                                    );
                                    assert_eq!(proof.receipt, GFX1201_PLE_PROJECT_RECEIPT);
                                } else {
                                    assert_eq!(proof.receipt, GFX1201_PP2_MQ2_RECEIPT);
                                }
                            }
                            _ => {
                                assert_ne!(proof.route, StageRoute::Any);
                                assert_ne!(proof.route, StageRoute::Unclassified);
                                assert_eq!(
                                    matches!(proof.route, StageRoute::Moe { .. }),
                                    stage == StageName::Moe
                                );
                                if stage == StageName::Moe {
                                    assert_eq!(
                                        proof.route,
                                        StageRoute::Moe {
                                            experts: LaneExpertRoute::EpPackedStaticOrBalanced,
                                            dense: [DType::BF16; 5],
                                            decode_admitted: true,
                                        }
                                    );
                                }
                                if stage == StageName::PleProject {
                                    assert_eq!(
                                        proof.route,
                                        StageRoute::PleProject([DType::BF16; 2])
                                    );
                                }
                                assert_eq!(proof.domain, [LaneTopology::Ep]);
                                let hc8_row = proof.route
                                    == StageRoute::HyperRead([
                                        DType::BF16,
                                        DType::HC8E4M3Row,
                                        DType::HC8E4M3Row,
                                    ]);
                                let q8_row = matches!(
                                    proof.route,
                                    StageRoute::QsaIn([
                                        DType::Q8_0,
                                        DType::Q8_0,
                                        DType::Q8_0,
                                        DType::Q8_0
                                    ]) | StageRoute::GdnIn([
                                        DType::Q8_0,
                                        DType::Q8_0,
                                        DType::Q8_0,
                                        DType::Q8_0
                                    ]) | StageRoute::Projection(DType::Q8_0)
                                );
                                let want = match (q8_row, stage) {
                                    (false, StageName::PleProject) => GFX1201_PLE_PROJECT_RECEIPT,
                                    (false, _) if hc8_row => GFX1201_HC8_ROWS_RECEIPT,
                                    (false, _) => GFX1201_EP2_MQ4_RECEIPT,
                                    (true, StageName::QsaInProj | StageName::QsaOutProj) => {
                                        GFX1201_Q8_QSA_ROWS_RECEIPT
                                    }
                                    (true, _) => GFX1201_Q8_TRUNK_ROWS_RECEIPT,
                                };
                                assert_eq!(proof.receipt, want, "{}", stage.key());
                            }
                        }
                    }
                }
            }
        }
        assert!(GFX1201_PP2_MQ2_RECEIPT.contains("j1574"));
        assert!(GFX1201_EP2_MQ4_RECEIPT.contains("j1570"));
        assert!(GFX1201_EP2_MQ4_RECEIPT.contains("gfx1201 ep2 mq4-xts k1,2,4,8"));
        assert_ne!(GFX1201_EP2_MQ4_RECEIPT, GFX1201_PP2_MQ2_RECEIPT);
        assert!(GFX1201_PLE_PROJECT_RECEIPT.contains("j8313"));
    }

    #[test]
    fn gfx1151_admits_every_topology_and_unclassified_route() {
        let diverse: LaneStageRoutes = std::array::from_fn(|_| {
            vec![
                StageRoute::Unclassified,
                StageRoute::Projection(DType::F16),
                StageRoute::Embed(DType::Q4K),
                StageRoute::Moe {
                    experts: LaneExpertRoute::HostMapped,
                    dense: [DType::F16; 5],
                    decode_admitted: false,
                },
                moe_route(LaneExpertRoute::EpPackedStaticOrBalanced, true),
            ]
        });
        for topology in ALL_TOPOLOGIES {
            for (routes, populated) in [
                (LaneStageRoutes::default(), false),
                (table_routes("gfx1151", topology), true),
                (pp2_mq2_routes(), true),
                (ep_production_routes(), true),
                (diverse.clone(), true),
            ] {
                let resolution = LaneStagePolicy::for_bound_routes("gfx1151", topology, &routes);
                // No gfx1151 row for the PLE project stage (exact-gfx1201 flag): the one stage
                // that stays per lane there.
                assert_eq!(
                    resolution.policy,
                    production_without(&[StageName::PleProject]),
                    "{topology:?}"
                );
                assert_eq!(
                    fallback_stages(&resolution),
                    [StageName::PleProject],
                    "{topology:?}"
                );
                assert_stateful_reasons(&resolution);
                let expected: &[&str] = if populated {
                    &[GFX1151_LANE_RECEIPT]
                } else {
                    &[]
                };
                assert_eq!(resolution.receipts, expected, "{topology:?}");
            }
        }
    }

    #[test]
    fn gfx1201_pp_resident_mq2_is_production() {
        // The bound routes of the PP2 mq2 artifact, spelled out independently
        // of the proof table, which must carry exactly these signatures (its
        // PP MoE stage lists the mq2 slab class and then the mq3 slab class).
        let routes = pp2_mq2_routes();
        let mut table = table_routes("gfx1201", LaneTopology::Pp);
        assert_eq!(table[StageName::Moe.index()].len(), 2);
        table[StageName::Moe.index()].truncate(1);
        // The PLE project route has its own test (`ple_project_is_shared_only_for_the_bf16_pair...`):
        // this fixture carries the artifact header's other stages.
        assert_eq!(table[StageName::PleProject.index()].len(), 1);
        table[StageName::PleProject.index()].clear();
        assert_eq!(routes, table);
        assert!(matches!(
            routes[StageName::Moe.index()].as_slice(),
            [StageRoute::Moe {
                experts: LaneExpertRoute::ResidentMq2Slab,
                decode_admitted: true,
                ..
            }]
        ));
        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes);
        assert_eq!(resolution.policy, LaneStagePolicy::production());
        assert_eq!(resolution.receipts, [GFX1201_PP2_MQ2_RECEIPT]);
        assert!(fallback_stages(&resolution).is_empty());
        assert_stateful_reasons(&resolution);

        // A band that owns none of a stage's instances keeps the stage shared
        // (its table row exists) and cites no receipt.
        let empty = LaneStagePolicy::for_bound_routes(
            "gfx1201",
            LaneTopology::Pp,
            &LaneStageRoutes::default(),
        );
        assert_eq!(empty.policy, LaneStagePolicy::production());
        assert!(empty.receipts.is_empty());
        assert!(fallback_stages(&empty).is_empty());
    }

    /// The PLE key / value projections' route is the BF16 pair, on gfx1201 PP and EP only: a
    /// differing, unclassified or foreign instance demotes only `ple_project`, the receipt is
    /// the j8313 byte-oracle one and comes last, and Single / other arches have no row.
    #[test]
    fn ple_project_is_shared_only_for_the_bf16_pair_on_gfx1201_pp_and_ep() {
        let bf16 = DType::BF16;
        let pair = StageRoute::PleProject([bf16; 2]);
        for (topology, base, receipt) in [
            (LaneTopology::Pp, pp2_mq2_routes(), GFX1201_PP2_MQ2_RECEIPT),
            (
                LaneTopology::Ep,
                ep_production_routes(),
                GFX1201_EP2_MQ4_RECEIPT,
            ),
        ] {
            let mut routes = base.clone();
            routes[StageName::PleProject.index()] = vec![pair];
            let resolution = LaneStagePolicy::for_bound_routes("gfx1201", topology, &routes);
            assert_eq!(
                resolution.policy,
                LaneStagePolicy::production(),
                "{topology:?}"
            );
            assert!(fallback_stages(&resolution).is_empty(), "{topology:?}");
            assert_eq!(
                resolution.receipts,
                [receipt, GFX1201_PLE_PROJECT_RECEIPT],
                "{topology:?}"
            );
            assert_stateful_reasons(&resolution);
            for bad in [
                StageRoute::PleProject([bf16, DType::F16]),
                StageRoute::PleProject([DType::Q8_0; 2]),
                StageRoute::Unclassified,
                StageRoute::PleRows(bf16),
            ] {
                // One bad instance among proven ones refuses the stage.
                let mut routes = base.clone();
                routes[StageName::PleProject.index()] = vec![pair, bad];
                let resolution = LaneStagePolicy::for_bound_routes("gfx1201", topology, &routes);
                assert_eq!(
                    fallback_stages(&resolution),
                    [StageName::PleProject],
                    "{topology:?} {bad:?}"
                );
                assert_eq!(
                    resolution.policy,
                    production_without(&[StageName::PleProject]),
                    "{topology:?} {bad:?}"
                );
                assert!(!resolution.receipts.contains(&GFX1201_PLE_PROJECT_RECEIPT));
                let reason = reason_of(&resolution, StageName::PleProject);
                assert!(reason.contains("gfx1201"), "{reason}");
                assert!(reason.contains(StageName::PleProject.key()), "{reason}");
            }
        }
        // Single and every other arch have no PLE project row: per lane, nothing inherited.
        let mut routes = pp2_mq2_routes();
        routes[StageName::PleProject.index()] = vec![pair];
        for (arch, topology) in [
            ("gfx1201", LaneTopology::Single),
            ("gfx1151", LaneTopology::Pp),
            ("gfx1151", LaneTopology::Ep),
            ("gfx1100", LaneTopology::Pp),
        ] {
            let resolution = LaneStagePolicy::for_bound_routes(arch, topology, &routes);
            assert_eq!(
                resolution.policy.share(StageName::PleProject),
                StageShare::PerLane,
                "{arch} {topology:?}"
            );
            assert!(
                !resolution.receipts.contains(&GFX1201_PLE_PROJECT_RECEIPT),
                "{arch} {topology:?}"
            );
        }
        // The stage is stateless (only the conv tail is) and its key parses back.
        assert!(!StageName::PleProject.stateful());
        assert!(StageName::PleConv.stateful());
        assert_eq!(
            LaneStagePolicy::production().share(StageName::PleConv),
            StageShare::PerLane
        );
        assert_eq!(
            LaneStagePolicy::production().share(StageName::PleProject),
            StageShare::OnceAllRows
        );
        assert_eq!(
            LaneStagePolicy::parse("production:ple_project=lane")
                .unwrap()
                .share(StageName::PleProject),
            StageShare::PerLane
        );
    }

    #[test]
    fn gfx1201_pp_resident_mq3_slab_is_production() {
        // The bound routes of the PP2 mq3-xts artifact (MQ4-XTS + MQ3-I4 slab
        // arenas, the same non-MoE tensors as the mq2 artifact), spelled out
        // independently of the proof table.
        let mut routes = pp2_mq2_routes();
        routes[StageName::Moe.index()] = vec![StageRoute::Moe {
            experts: LaneExpertRoute::ResidentMq3Slab,
            dense: [DType::BF16; 5],
            decode_admitted: true,
        }];
        let table = table_routes("gfx1201", LaneTopology::Pp);
        assert_eq!(
            table[StageName::Moe.index()][1],
            routes[StageName::Moe.index()][0]
        );
        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes);
        assert_eq!(resolution.policy, LaneStagePolicy::production());
        assert_eq!(
            resolution.receipts,
            [GFX1201_PP2_MQ2_RECEIPT, GFX1201_PP2_MQ3_RECEIPT]
        );
        assert!(fallback_stages(&resolution).is_empty());
        assert_stateful_reasons(&resolution);

        // Only the PP domain and only the decode-admitted slab class: every
        // other shape of the mq3 class keeps just the MoE stage per lane.
        for (topology, decode_admitted) in [
            (LaneTopology::Pp, false),
            (LaneTopology::Ep, true),
            (LaneTopology::Single, true),
        ] {
            let mut other = routes.clone();
            other[StageName::Moe.index()] =
                vec![moe_route(LaneExpertRoute::ResidentMq3Slab, decode_admitted)];
            let resolution = LaneStagePolicy::for_bound_routes("gfx1201", topology, &other);
            assert!(
                fallback_stages(&resolution).contains(&StageName::Moe),
                "{topology:?} {decode_admitted}"
            );
        }
        // gfx1151 keeps its own Any rows; an unproven arch never inherits.
        let other_arch = LaneStagePolicy::for_bound_routes("gfx1100", LaneTopology::Pp, &routes);
        assert_eq!(other_arch.policy, LaneStagePolicy::all_per_lane());
    }

    #[test]
    fn gfx1201_single_runs_every_stage_per_lane_with_a_named_reason() {
        for routes in [
            pp2_mq2_routes(),
            ep2_mq4_routes(vec![StageRoute::Unclassified]),
            LaneStageRoutes::default(),
        ] {
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Single, &routes);
            assert_eq!(resolution.policy, LaneStagePolicy::all_per_lane());
            assert!(resolution.receipts.is_empty());
            assert_stateful_reasons(&resolution);
            assert_eq!(fallback_stages(&resolution), shared_stages());
            for stage in shared_stages() {
                let reason = reason_of(&resolution, stage);
                assert!(reason.contains("gfx1201"), "{reason}");
                assert!(reason.contains("Single"), "{reason}");
                assert!(reason.contains(stage.key()), "{reason}");
            }
        }
    }

    /// The route instances an EP2 mq4-xts lane store registers, written out
    /// from the parsed artifact header (qt3 Q8_0 embed; qt47 MQ6G256V2 head,
    /// all 36 GDN qkv/z/a/b/out and all 12 QSA indexer/q/k/v/out; BF16 PLE
    /// rows and hyper-connection) with `moe` as the MoE instances, NOT read
    /// from the proof table.
    fn ep2_mq4_routes(moe: Vec<StageRoute>) -> LaneStageRoutes {
        let mq6 = DType::MQ6G256V2;
        let bf16 = DType::BF16;
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Embed.index()] = vec![StageRoute::Embed(DType::Q8_0)];
        routes[StageName::PleRows.index()] = vec![StageRoute::PleRows(bf16)];
        routes[StageName::HcRead.index()] = vec![StageRoute::HyperRead([bf16; 3]); 36 + 12];
        routes[StageName::HcWrite.index()] = vec![StageRoute::HyperWrite(bf16); 36 + 12];
        routes[StageName::GdnInProj.index()] = vec![StageRoute::GdnIn([mq6; 4]); 36];
        routes[StageName::GdnOutProj.index()] = vec![StageRoute::Projection(mq6); 36];
        routes[StageName::QsaInProj.index()] = vec![StageRoute::QsaIn([mq6; 4]); 12];
        routes[StageName::QsaOutProj.index()] = vec![StageRoute::Projection(mq6); 12];
        routes[StageName::Moe.index()] = moe;
        routes[StageName::FinalHyper.index()] = vec![StageRoute::HyperRead([bf16; 3])];
        routes[StageName::Head.index()] = vec![StageRoute::Projection(mq6)];
        routes[StageName::Argmax.index()] = vec![StageRoute::Argmax(DType::F32)];
        routes
    }

    fn moe_route(experts: LaneExpertRoute, decode_admitted: bool) -> StageRoute {
        StageRoute::Moe {
            experts,
            dense: [DType::BF16; 5],
            decode_admitted,
        }
    }

    /// The EP packed static-or-balanced MoE route of one owned layer.
    fn ep_moe() -> StageRoute {
        moe_route(LaneExpertRoute::EpPackedStaticOrBalanced, true)
    }

    /// The 48 trunk MoE layers of the measured EP2 artifact.
    fn ep_production_routes() -> LaneStageRoutes {
        ep2_mq4_routes(vec![ep_moe(); 48])
    }

    #[test]
    fn gfx1201_ep_shares_every_stage_with_the_ep_moe_class_at_any_layer_count() {
        let routes = ep_production_routes();
        // The independent fixture and the proof table carry the same
        // signatures for every shared stage, MoE included.
        let table = table_routes("gfx1201", LaneTopology::Ep);
        for stage in shared_stages() {
            let rows = &table[stage.index()];
            // The QSA, GDN in/out and head stages carry the MQ6 row first and the
            // Q8F16 trunk-overlay row second; hc_read and final_hyper the BF16 row
            // first and the HC8 sidecar row second; every other stage one row.
            let two_rows = matches!(
                stage,
                StageName::QsaInProj
                    | StageName::QsaOutProj
                    | StageName::GdnInProj
                    | StageName::GdnOutProj
                    | StageName::Head
                    | StageName::HcRead
                    | StageName::FinalHyper
            );
            assert_eq!(rows.len(), 1 + usize::from(two_rows), "{}", stage.key());
            assert!(
                routes[stage.index()].iter().all(|route| *route == rows[0]),
                "{}",
                stage.key()
            );
        }
        assert_eq!(
            table[StageName::QsaInProj.index()][1],
            StageRoute::QsaIn([DType::Q8_0; 4])
        );
        assert_eq!(
            table[StageName::QsaOutProj.index()][1],
            StageRoute::Projection(DType::Q8_0)
        );
        assert_eq!(
            table[StageName::GdnInProj.index()][1],
            StageRoute::GdnIn([DType::Q8_0; 4])
        );
        assert_eq!(
            table[StageName::GdnOutProj.index()][1],
            StageRoute::Projection(DType::Q8_0)
        );
        assert_eq!(
            table[StageName::Head.index()][1],
            StageRoute::Projection(DType::Q8_0)
        );
        let hc8 = StageRoute::HyperRead([DType::BF16, DType::HC8E4M3Row, DType::HC8E4M3Row]);
        assert_eq!(table[StageName::HcRead.index()][1], hc8);
        assert_eq!(table[StageName::FinalHyper.index()][1], hc8);
        assert_eq!(table[StageName::Moe.index()], [ep_moe()]);

        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &routes);
        assert_eq!(resolution.policy, LaneStagePolicy::production());
        assert_eq!(resolution.receipts, [GFX1201_EP2_MQ4_RECEIPT]);
        assert!(fallback_stages(&resolution).is_empty());
        assert_stateful_reasons(&resolution);

        // No layer-count gate: a band of any size, 27+21 or 28+20 splits and
        // a single layer all qualify the same way.
        for layers in [1usize, 20, 21, 27, 28, 48] {
            let resolution = LaneStagePolicy::for_bound_routes(
                "gfx1201",
                LaneTopology::Ep,
                &ep2_mq4_routes(vec![ep_moe(); layers]),
            );
            assert_eq!(resolution.policy, LaneStagePolicy::production(), "{layers}");
            assert_eq!(resolution.receipts, [GFX1201_EP2_MQ4_RECEIPT], "{layers}");
            assert!(fallback_stages(&resolution).is_empty(), "{layers}");
        }

        // A rank owning no MoE instance keeps MoE shared (its row exists).
        let no_moe =
            LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &ep2_mq4_routes(vec![]));
        assert_eq!(no_moe.policy, LaneStagePolicy::production());
        assert_eq!(no_moe.receipts, [GFX1201_EP2_MQ4_RECEIPT]);

        // The table's own routes and a band owning nothing resolve likewise.
        let from_table = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &table);
        assert_eq!(from_table.policy, LaneStagePolicy::production());
        assert_eq!(
            from_table.receipts,
            [
                GFX1201_EP2_MQ4_RECEIPT,
                GFX1201_HC8_ROWS_RECEIPT,
                GFX1201_Q8_TRUNK_ROWS_RECEIPT,
                GFX1201_Q8_QSA_ROWS_RECEIPT,
                GFX1201_PLE_PROJECT_RECEIPT
            ]
        );
        let empty = LaneStagePolicy::for_bound_routes(
            "gfx1201",
            LaneTopology::Ep,
            &LaneStageRoutes::default(),
        );
        assert_eq!(empty.policy, LaneStagePolicy::production());
        assert!(empty.receipts.is_empty());
    }

    /// The EP2 mq4-xts routes with the Q8F16 trunk overlay bound on the first
    /// `q8_layers` of the 12 QSA layers (indexer index_qk, q, k, v and o all
    /// `Q8_0`), the others still MQ6G256V2, written out from the overlay's
    /// payload class (qt3 = `DType::Q8_0`), NOT read from the proof table.
    fn ep2_q8_overlay_routes(q8_layers: usize) -> LaneStageRoutes {
        let mq6 = DType::MQ6G256V2;
        let q8 = DType::Q8_0;
        let mut routes = ep2_mq4_routes(vec![ep_moe(); 48]);
        routes[StageName::QsaInProj.index()] = (0..12)
            .map(|layer| StageRoute::QsaIn([if layer < q8_layers { q8 } else { mq6 }; 4]))
            .collect();
        routes[StageName::QsaOutProj.index()] = (0..12)
            .map(|layer| StageRoute::Projection(if layer < q8_layers { q8 } else { mq6 }))
            .collect();
        routes
    }

    #[test]
    fn gfx1201_ep_q8_overlay_qsa_in_out_are_shared_with_the_q8_receipt() {
        for q8_layers in [12usize, 6, 1] {
            let resolution = LaneStagePolicy::for_bound_routes(
                "gfx1201",
                LaneTopology::Ep,
                &ep2_q8_overlay_routes(q8_layers),
            );
            assert_eq!(
                resolution.policy,
                LaneStagePolicy::production(),
                "{q8_layers}"
            );
            assert!(fallback_stages(&resolution).is_empty(), "{q8_layers}");
            assert_eq!(
                resolution.receipts,
                [GFX1201_EP2_MQ4_RECEIPT, GFX1201_Q8_QSA_ROWS_RECEIPT],
                "{q8_layers}"
            );
            assert_stateful_reasons(&resolution);
        }
        // The unmodified MQ6 artifact keeps its single receipt.
        let mq6 = LaneStagePolicy::for_bound_routes(
            "gfx1201",
            LaneTopology::Ep,
            &ep2_q8_overlay_routes(0),
        );
        assert_eq!(mq6.receipts, [GFX1201_EP2_MQ4_RECEIPT]);
        assert!(GFX1201_Q8_QSA_ROWS_RECEIPT.starts_with("g0 "));
        assert!(GFX1201_Q8_QSA_ROWS_RECEIPT.contains("gfx1201"));
    }

    #[test]
    fn gfx1201_q8_qsa_proofs_are_ep_only_and_demote_only_the_unproven_member() {
        let q8 = DType::Q8_0;
        let mq6 = DType::MQ6G256V2;
        // Other topologies have no Q8 row.
        for topology in [LaneTopology::Pp, LaneTopology::Single] {
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx1201", topology, &ep2_q8_overlay_routes(12));
            for stage in [StageName::QsaInProj, StageName::QsaOutProj] {
                assert_eq!(
                    resolution.policy.share(stage),
                    StageShare::PerLane,
                    "{topology:?} {}",
                    stage.key()
                );
            }
        }
        // One Q8 member among MQ6 ones is not the registered x4 group.
        let mut mixed = ep2_q8_overlay_routes(12);
        mixed[StageName::QsaInProj.index()].push(StageRoute::QsaIn([mq6, q8, mq6, mq6]));
        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &mixed);
        assert_eq!(fallback_stages(&resolution), [StageName::QsaInProj]);
        assert!(reason_of(&resolution, StageName::QsaInProj).contains("has no row-twin proof"));
        assert_eq!(
            resolution.policy.share(StageName::QsaOutProj),
            StageShare::OnceAllRows
        );
        // A Q8 projection on a stage with no Q8 row never rides another
        // stage's row: the final hyper-connection read has none.
        let mut other = ep2_q8_overlay_routes(12);
        other[StageName::FinalHyper.index()] = vec![StageRoute::Projection(q8)];
        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &other);
        assert_eq!(fallback_stages(&resolution), [StageName::FinalHyper]);
    }

    /// The EP2 mq4-xts routes with the full Q8F16 trunk overlay: all 36 GDN
    /// in-projection groups and out_proj, the head, and `q8_qsa_layers` of the
    /// 12 QSA layers all `Q8_0`; written from the overlay's payload class
    /// (qt3 = `DType::Q8_0`), NOT read from the proof table.
    fn ep2_q8_full_routes(gdn_layers: usize, head: bool, q8_qsa_layers: usize) -> LaneStageRoutes {
        let mq6 = DType::MQ6G256V2;
        let q8 = DType::Q8_0;
        let mut routes = ep2_q8_overlay_routes(q8_qsa_layers);
        routes[StageName::GdnInProj.index()] = (0..36)
            .map(|layer| StageRoute::GdnIn([if layer < gdn_layers { q8 } else { mq6 }; 4]))
            .collect();
        routes[StageName::GdnOutProj.index()] = (0..36)
            .map(|layer| StageRoute::Projection(if layer < gdn_layers { q8 } else { mq6 }))
            .collect();
        routes[StageName::Head.index()] = vec![StageRoute::Projection(if head { q8 } else { mq6 })];
        routes
    }

    #[test]
    fn gfx1201_ep_full_q8_trunk_overlay_shares_every_stage_with_both_receipts() {
        for (gdn, head, qsa) in [
            (36usize, true, 12usize),
            (36, true, 0),
            (36, false, 12),
            (12, false, 0),
            (0, true, 0),
        ] {
            let resolution = LaneStagePolicy::for_bound_routes(
                "gfx1201",
                LaneTopology::Ep,
                &ep2_q8_full_routes(gdn, head, qsa),
            );
            let tag = format!("gdn {gdn} head {head} qsa {qsa}");
            assert_eq!(resolution.policy, LaneStagePolicy::production(), "{tag}");
            assert!(fallback_stages(&resolution).is_empty(), "{tag}");
            assert!(
                resolution.receipts.contains(&GFX1201_EP2_MQ4_RECEIPT),
                "{tag}"
            );
            assert_eq!(
                resolution.receipts.contains(&GFX1201_Q8_TRUNK_ROWS_RECEIPT),
                gdn > 0 || head,
                "{tag}"
            );
            assert_eq!(
                resolution.receipts.contains(&GFX1201_Q8_QSA_ROWS_RECEIPT),
                qsa > 0,
                "{tag}"
            );
            assert_stateful_reasons(&resolution);
        }
        // A GDN in-projection group with one Q8 member among MQ6 ones is not
        // the registered x4 group; only that stage demotes.
        let q8 = DType::Q8_0;
        let mq6 = DType::MQ6G256V2;
        let mut mixed = ep2_q8_full_routes(36, true, 12);
        mixed[StageName::GdnInProj.index()].push(StageRoute::GdnIn([q8, mq6, mq6, mq6]));
        let resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &mixed);
        assert_eq!(fallback_stages(&resolution), [StageName::GdnInProj]);
        assert!(reason_of(&resolution, StageName::GdnInProj).contains("has no row-twin proof"));
        // Other topologies have no Q8 GDN / head row.
        for topology in [LaneTopology::Pp, LaneTopology::Single] {
            let resolution = LaneStagePolicy::for_bound_routes(
                "gfx1201",
                topology,
                &ep2_q8_full_routes(36, true, 12),
            );
            for stage in [StageName::GdnInProj, StageName::GdnOutProj, StageName::Head] {
                assert_eq!(
                    resolution.policy.share(stage),
                    StageShare::PerLane,
                    "{topology:?} {}",
                    stage.key()
                );
            }
        }
    }

    /// Land gate: the GDN in/out and head Q8F16 rows must carry a real G0 card
    /// receipt, not the staging placeholder.
    #[test]
    fn q8_trunk_receipts_are_g0_receipts() {
        for receipt in [GFX1201_Q8_QSA_ROWS_RECEIPT, GFX1201_Q8_TRUNK_ROWS_RECEIPT] {
            assert!(
                receipt.starts_with("g0 ") && receipt.contains("gfx1201"),
                "Q8 row-twin receipt is not a G0 card receipt: {receipt}"
            );
        }
    }

    #[test]
    fn gfx1201_ep_moe_other_classes_and_unclassified_demote_only_moe() {
        let class = LaneExpertRoute::EpPackedStaticOrBalanced;
        let bf16 = DType::BF16;
        let cases: Vec<Vec<StageRoute>> = vec![
            vec![StageRoute::Unclassified],
            vec![moe_route(LaneExpertRoute::ResidentMq2Slab, true)],
            vec![moe_route(LaneExpertRoute::ResidentMq3Slab, true)],
            vec![moe_route(LaneExpertRoute::ResidentPacked, true)],
            vec![moe_route(LaneExpertRoute::Split, true)],
            vec![moe_route(LaneExpertRoute::HostMapped, true)],
            vec![moe_route(LaneExpertRoute::HostMapped, false)],
            vec![moe_route(LaneExpertRoute::CpuDeferred, false)],
            vec![moe_route(LaneExpertRoute::Unclassified, true)],
            vec![moe_route(class, false)],
            vec![StageRoute::Moe {
                experts: class,
                dense: [bf16, bf16, bf16, bf16, DType::F32],
                decode_admitted: true,
            }],
            vec![StageRoute::Moe {
                experts: class,
                dense: [DType::MQ6G256V2; 5],
                decode_admitted: true,
            }],
            // One bad layer among otherwise admitted ones refuses the stage.
            vec![ep_moe(), StageRoute::Unclassified],
            vec![ep_moe(), moe_route(LaneExpertRoute::HostMapped, true)],
            vec![moe_route(LaneExpertRoute::ResidentPacked, true), ep_moe()],
        ];
        for moe in cases {
            let routes = ep2_mq4_routes(moe.clone());
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &routes);
            assert_eq!(fallback_stages(&resolution), [StageName::Moe], "{moe:?}");
            assert_eq!(
                resolution.policy,
                production_without(&[StageName::Moe]),
                "{moe:?}"
            );
            assert_eq!(resolution.receipts, [GFX1201_EP2_MQ4_RECEIPT], "{moe:?}");
            assert_stateful_reasons(&resolution);
            let reason = reason_of(&resolution, StageName::Moe);
            assert!(reason.contains("gfx1201"), "{reason}");
            assert!(reason.contains("Ep"), "{reason}");
            assert!(reason.contains(StageName::Moe.key()), "{reason}");
        }
    }

    #[test]
    fn gfx1201_ep_routes_equal_the_pp_routes_and_each_domain_binds_only_its_topology() {
        let pp = pp2_mq2_routes();
        let ep = ep_production_routes();
        for stage in shared_stages()
            .into_iter()
            .filter(|s| *s != StageName::Moe && *s != StageName::PleProject)
        {
            // EP reuses the PP row's typed route for every non-MoE stage; a
            // multi-instance EP stage repeats the PP signature.
            let pp_route = pp[stage.index()][0];
            assert!(ep[stage.index()].iter().all(|route| *route == pp_route));
            let ep_row = LANE_ROW_TWIN_PROOFS
                .iter()
                .find(|row| {
                    row.arch == "gfx1201" && row.stage == stage && row.domain == [LaneTopology::Ep]
                })
                .unwrap();
            assert_eq!(ep_row.route, pp_route, "{}", stage.key());
        }
        // MoE is the one stage whose class differs between the domains.
        assert_ne!(pp[StageName::Moe.index()][0], ep[StageName::Moe.index()][0]);

        // PP routes (MoE = resident MQ2 slab) under EP: only MoE falls back.
        let pp_under_ep = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &pp);
        assert_eq!(fallback_stages(&pp_under_ep), [StageName::Moe]);
        assert_eq!(pp_under_ep.policy, production_without(&[StageName::Moe]));
        assert_eq!(pp_under_ep.receipts, [GFX1201_EP2_MQ4_RECEIPT]);

        // EP routes under PP keep citing the PP receipt only, never EP's, and
        // the EP MoE class is not admitted there.
        let ep_under_pp = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &ep);
        assert_eq!(fallback_stages(&ep_under_pp), [StageName::Moe]);
        assert_eq!(ep_under_pp.policy, production_without(&[StageName::Moe]));
        assert_eq!(ep_under_pp.receipts, [GFX1201_PP2_MQ2_RECEIPT]);
        let pp_resolution = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &pp);
        assert_eq!(pp_resolution.policy, LaneStagePolicy::production());
        assert_eq!(pp_resolution.receipts, [GFX1201_PP2_MQ2_RECEIPT]);

        // Neither domain leaks to Single.
        let single = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Single, &ep);
        assert_eq!(single.policy, LaneStagePolicy::all_per_lane());
        assert!(single.receipts.is_empty());
    }

    #[test]
    fn gfx1201_ep_dtype_mismatch_demotes_only_its_stage() {
        let mq6 = DType::MQ6G256V2;
        let mq4 = DType::MQ4G256V2;
        let base = ep_production_routes();
        let mutations = [
            (StageName::Embed, StageRoute::Embed(DType::MQ4G256V2)),
            (StageName::Embed, StageRoute::Embed(DType::F16)),
            (StageName::PleRows, StageRoute::PleRows(DType::F32)),
            (
                StageName::HcRead,
                StageRoute::HyperRead([DType::BF16, DType::BF16, DType::F32]),
            ),
            (StageName::HcWrite, StageRoute::HyperWrite(DType::F32)),
            (
                StageName::GdnInProj,
                StageRoute::GdnIn([mq6, mq6, mq6, mq4]),
            ),
            (StageName::GdnInProj, StageRoute::GdnIn([mq4; 4])),
            (StageName::GdnOutProj, StageRoute::Projection(mq4)),
            (
                StageName::QsaInProj,
                StageRoute::QsaIn([mq4, mq6, mq6, mq6]),
            ),
            (StageName::QsaInProj, StageRoute::Unclassified),
            (StageName::QsaOutProj, StageRoute::Projection(DType::BF16)),
            (StageName::Moe, moe_route(LaneExpertRoute::HostMapped, true)),
            (
                StageName::Moe,
                moe_route(LaneExpertRoute::EpPackedStaticOrBalanced, false),
            ),
            (StageName::Moe, StageRoute::Unclassified),
            (
                StageName::FinalHyper,
                StageRoute::HyperRead([DType::F32; 3]),
            ),
            (StageName::Head, StageRoute::Projection(mq4)),
            (StageName::Argmax, StageRoute::Argmax(DType::F16)),
        ];
        for (stage, route) in mutations {
            let mut routes = base.clone();
            routes[stage.index()] = vec![route];
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &routes);
            assert_eq!(fallback_stages(&resolution), [stage], "{stage:?} {route:?}");
            assert_eq!(
                resolution.policy,
                production_without(&[stage]),
                "{stage:?} {route:?}"
            );
            assert_eq!(resolution.receipts, [GFX1201_EP2_MQ4_RECEIPT]);
            assert_stateful_reasons(&resolution);
            let reason = reason_of(&resolution, stage);
            assert!(reason.contains("gfx1201"), "{reason}");
            assert!(reason.contains("Ep"), "{reason}");
            assert!(reason.contains(stage.key()), "{reason}");
            if route == StageRoute::Unclassified {
                assert!(reason.contains("unclassified"), "{reason}");
            } else {
                assert!(reason.contains(&format!("{route:?}")), "{reason}");
            }
        }
    }

    #[test]
    fn gfx1201_ep_every_layer_instance_must_match_not_only_the_first() {
        let mq6 = DType::MQ6G256V2;
        let base = ep_production_routes();
        // One layer's GDN input (of 36), QSA output (of 12), HC write or MoE
        // (of 48) off the proven route demotes exactly that stage.
        for (stage, bad) in [
            (
                StageName::GdnInProj,
                StageRoute::GdnIn([mq6, mq6, DType::BF16, mq6]),
            ),
            (
                StageName::QsaOutProj,
                StageRoute::Projection(DType::MQ4G256V2),
            ),
            (StageName::HcWrite, StageRoute::Unclassified),
            (StageName::Moe, moe_route(LaneExpertRoute::HostMapped, true)),
            (StageName::Moe, StageRoute::Unclassified),
        ] {
            for position in [0, 7, base[stage.index()].len() - 1] {
                let mut routes = base.clone();
                routes[stage.index()][position] = bad;
                let resolution =
                    LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &routes);
                assert_eq!(
                    fallback_stages(&resolution),
                    [stage],
                    "{stage:?} instance {position}"
                );
                assert_eq!(resolution.policy, production_without(&[stage]));
            }
        }
    }

    #[test]
    fn gfx1201_ep_receipts_dedup_and_combine_across_mixed_proof_domains() {
        // The real table cites the EP receipt exactly once for all twelve
        // stages, and the PP table still cites only the PP receipt.
        let table =
            LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Ep, &ep_production_routes());
        assert_eq!(table.receipts, [GFX1201_EP2_MQ4_RECEIPT]);
        let pp = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &pp2_mq2_routes());
        assert_eq!(pp.receipts, [GFX1201_PP2_MQ2_RECEIPT]);

        // A stage admitted by rows from different proof domains cites each
        // receipt once, in first-use order; a row outside the topology's
        // domain contributes nothing.
        const SINGLE_EP: &[LaneTopology] = &[LaneTopology::Single, LaneTopology::Ep];
        const EP: &[LaneTopology] = &[LaneTopology::Ep];
        const PP: &[LaneTopology] = &[LaneTopology::Pp];
        let row = |stage, route, domain, receipt| RowTwinProof {
            arch: "gfx9996",
            stage,
            route,
            domain,
            receipt,
        };
        let proofs = [
            row(
                StageName::Embed,
                StageRoute::Embed(DType::Q8_0),
                SINGLE_EP,
                "A",
            ),
            row(
                StageName::PleRows,
                StageRoute::PleRows(DType::BF16),
                EP,
                "A",
            ),
            row(
                StageName::Head,
                StageRoute::Projection(DType::F16),
                SINGLE_EP,
                "C",
            ),
            row(
                StageName::Head,
                StageRoute::Projection(DType::BF16),
                EP,
                "B",
            ),
            row(StageName::Argmax, StageRoute::Argmax(DType::F32), PP, "P"),
        ];
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Embed.index()] = vec![StageRoute::Embed(DType::Q8_0)];
        routes[StageName::PleRows.index()] = vec![StageRoute::PleRows(DType::BF16)];
        routes[StageName::Head.index()] = vec![
            StageRoute::Projection(DType::F16),
            StageRoute::Projection(DType::BF16),
            StageRoute::Projection(DType::F16),
        ];
        routes[StageName::Argmax.index()] = vec![StageRoute::Argmax(DType::F32)];
        let resolution = resolve_with_proofs("gfx9996", LaneTopology::Ep, &routes, &proofs);
        assert_eq!(resolution.receipts, ["A", "C", "B"]);
        for stage in [StageName::Embed, StageName::PleRows, StageName::Head] {
            assert_eq!(resolution.policy.share(stage), StageShare::OnceAllRows);
        }
        // Argmax's only row is PP-domain: not admitted under EP, receipt uncited.
        assert_eq!(
            resolution.policy.share(StageName::Argmax),
            StageShare::PerLane
        );
        assert!(!resolution.receipts.contains(&"P"));
        // Under Single the Head BF16 instance has no row: Head falls back and
        // its C receipt is dropped with it.
        let single = resolve_with_proofs("gfx9996", LaneTopology::Single, &routes, &proofs);
        assert_eq!(single.policy.share(StageName::Head), StageShare::PerLane);
        assert_eq!(single.receipts, ["A"]);
    }

    #[test]
    fn one_mismatched_route_demotes_only_its_stage() {
        let base = pp2_mq2_routes();
        let mq6 = DType::MQ6G256V2;
        let mq4 = DType::MQ4G256V2;
        let moe = |experts, decode_admitted| StageRoute::Moe {
            experts,
            dense: [DType::BF16; 5],
            decode_admitted,
        };
        let mutations = [
            (StageName::Embed, StageRoute::Embed(DType::F16)),
            (StageName::PleRows, StageRoute::PleRows(DType::F32)),
            (
                StageName::HcRead,
                StageRoute::HyperRead([DType::BF16, DType::BF16, DType::F32]),
            ),
            (StageName::HcRead, StageRoute::HyperWrite(DType::BF16)),
            (StageName::HcWrite, StageRoute::HyperWrite(DType::F32)),
            (
                StageName::GdnInProj,
                StageRoute::GdnIn([mq6, mq4, mq4, mq4]),
            ),
            (StageName::GdnOutProj, StageRoute::Projection(DType::BF16)),
            (
                StageName::QsaInProj,
                StageRoute::QsaIn([mq6, mq6, mq6, mq4]),
            ),
            (StageName::QsaInProj, StageRoute::Unclassified),
            (StageName::QsaOutProj, StageRoute::Projection(mq4)),
            (StageName::Moe, moe(LaneExpertRoute::HostMapped, true)),
            (StageName::Moe, moe(LaneExpertRoute::ResidentMq2Slab, false)),
            (
                StageName::FinalHyper,
                StageRoute::HyperRead([DType::F32; 3]),
            ),
            (StageName::Head, StageRoute::Projection(DType::BF16)),
            (StageName::Argmax, StageRoute::Argmax(DType::F16)),
        ];
        for (stage, route) in mutations {
            let mut routes = base.clone();
            routes[stage.index()] = vec![route];
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes);
            assert_eq!(fallback_stages(&resolution), [stage], "{stage:?} {route:?}");
            assert_eq!(
                resolution.policy,
                production_without(&[stage]),
                "{stage:?} {route:?}"
            );
            assert_eq!(resolution.receipts, [GFX1201_PP2_MQ2_RECEIPT]);
            assert_stateful_reasons(&resolution);
            let reason = reason_of(&resolution, stage);
            assert!(reason.contains("gfx1201"), "{reason}");
            assert!(reason.contains("Pp"), "{reason}");
            assert!(reason.contains(stage.key()), "{reason}");
            if route == StageRoute::Unclassified {
                assert!(reason.contains("unclassified"), "{reason}");
            } else {
                assert!(reason.contains(&format!("{route:?}")), "{reason}");
            }
        }
    }

    #[test]
    fn equality_proofs_never_match_unclassified_routes() {
        let proofs = [RowTwinProof {
            arch: "gfx9997",
            stage: StageName::Head,
            route: StageRoute::Unclassified,
            domain: &[LaneTopology::Single],
            receipt: "T-unclassified",
        }];
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Head.index()] = vec![StageRoute::Unclassified];
        let resolution = resolve_with_proofs("gfx9997", LaneTopology::Single, &routes, &proofs);
        assert_eq!(resolution.policy, LaneStagePolicy::all_per_lane());
        assert!(resolution.receipts.is_empty());
        assert!(reason_of(&resolution, StageName::Head).contains("unclassified"));
    }

    #[test]
    fn gfx942_single_stock_fp8_shares_every_row_local_stage() {
        let resolution = LaneStagePolicy::for_bound_routes(
            "gfx942",
            LaneTopology::Single,
            &g942_stock_fp8_routes(),
        );
        assert_eq!(resolution.policy, LaneStagePolicy::production());
        assert_eq!(resolution.receipts, [GFX942_STOCK_FP8_RECEIPT]);
        assert_stateful_reasons(&resolution);
        assert!(fallback_stages(&resolution).is_empty());
    }

    #[test]
    fn gfx942_proofs_bind_only_single_device_and_the_fp8_expert_class() {
        // PP / EP topologies have no gfx942 row: every stage per lane.
        for topology in [LaneTopology::Pp, LaneTopology::Ep] {
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx942", topology, &g942_stock_fp8_routes());
            assert_eq!(
                resolution.policy,
                LaneStagePolicy::all_per_lane(),
                "{topology:?}"
            );
            assert!(resolution.receipts.is_empty());
        }
        // Any other expert class (or a decode-admitted K1 route) demotes MoE, and
        // with it every stage of the combined-only receipt.
        for moe in [
            moe_route(LaneExpertRoute::Unclassified, false),
            moe_route(LaneExpertRoute::HostMapped, false),
            moe_route(LaneExpertRoute::ResidentPacked, true),
            moe_route(LaneExpertRoute::ResidentFp8, true),
        ] {
            let mut routes = g942_stock_fp8_routes();
            routes[StageName::Moe.index()] = vec![moe];
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx942", LaneTopology::Single, &routes);
            assert_eq!(
                resolution.policy,
                LaneStagePolicy::all_per_lane(),
                "{moe:?}"
            );
            assert!(resolution.receipts.is_empty(), "{moe:?}");
            assert_eq!(fallback_stages(&resolution)[0], StageName::Moe, "{moe:?}");
            assert!(reason_of(&resolution, StageName::Head).contains("combined policy"));
        }
        // An MQ trunk on gfx942 is not the proven route: nothing is shared.
        let resolution =
            LaneStagePolicy::for_bound_routes("gfx942", LaneTopology::Single, &pp2_mq2_routes());
        assert_eq!(resolution.policy, LaneStagePolicy::all_per_lane());
        assert!(resolution.receipts.is_empty());
    }

    #[test]
    fn combined_only_receipts_admit_all_or_nothing() {
        // Any one gfx942 stage left unmatched demotes the rest, including a
        // vacuously admitted stage (no instance of it owned).
        for missing in shared_stages() {
            let mut routes = g942_stock_fp8_routes();
            routes[missing.index()] = vec![StageRoute::Unclassified];
            let resolution =
                LaneStagePolicy::for_bound_routes("gfx942", LaneTopology::Single, &routes);
            assert_eq!(
                resolution.policy,
                LaneStagePolicy::all_per_lane(),
                "{missing:?}"
            );
            assert!(resolution.receipts.is_empty());
            assert_stateful_reasons(&resolution);
        }
        let mut routes = g942_stock_fp8_routes();
        routes[StageName::PleProject.index()].clear();
        let resolution = LaneStagePolicy::for_bound_routes("gfx942", LaneTopology::Single, &routes);
        assert_eq!(resolution.policy, LaneStagePolicy::production());
        routes[StageName::Head.index()] = vec![StageRoute::Projection(DType::Q8_0)];
        let resolution = LaneStagePolicy::for_bound_routes("gfx942", LaneTopology::Single, &routes);
        assert_eq!(resolution.policy, LaneStagePolicy::all_per_lane());
    }

    #[test]
    fn unknown_arch_runs_every_stage_per_lane() {
        for arch in ["gfx1100", "gfx90a", ""] {
            for topology in ALL_TOPOLOGIES {
                for routes in [
                    pp2_mq2_routes(),
                    table_routes("gfx1151", topology),
                    LaneStageRoutes::default(),
                ] {
                    let resolution = LaneStagePolicy::for_bound_routes(arch, topology, &routes);
                    assert_eq!(
                        resolution.policy,
                        LaneStagePolicy::all_per_lane(),
                        "{arch:?} {topology:?}"
                    );
                    assert!(resolution.receipts.is_empty());
                    assert_stateful_reasons(&resolution);
                    assert_eq!(fallback_stages(&resolution), shared_stages());
                    for stage in shared_stages() {
                        let reason = reason_of(&resolution, stage);
                        assert!(reason.contains(&format!("{arch:?}")), "{reason}");
                        assert!(reason.contains(&format!("{topology:?}")), "{reason}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_new_arch_proof_row_enables_its_stage_without_code_changes() {
        let proofs = [RowTwinProof {
            arch: "gfx9999",
            stage: StageName::Head,
            route: StageRoute::Projection(DType::BF16),
            domain: &[LaneTopology::Single],
            receipt: "T-head",
        }];
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Head.index()] = vec![StageRoute::Projection(DType::BF16)];
        let resolution = resolve_with_proofs("gfx9999", LaneTopology::Single, &routes, &proofs);
        assert_eq!(
            resolution.policy,
            LaneStagePolicy::all_per_lane()
                .with(StageName::Head, StageShare::OnceAllRows)
                .unwrap()
        );
        assert_eq!(resolution.receipts, ["T-head"]);
        // Every other shared stage has no row for the arch: per lane.
        let others: Vec<_> = shared_stages()
            .into_iter()
            .filter(|stage| *stage != StageName::Head)
            .collect();
        assert_eq!(fallback_stages(&resolution), others);
        assert_stateful_reasons(&resolution);

        // The row binds its arch, domain and route only.
        let other_topology = resolve_with_proofs("gfx9999", LaneTopology::Pp, &routes, &proofs);
        assert_eq!(other_topology.policy, LaneStagePolicy::all_per_lane());
        let other_arch = resolve_with_proofs("gfx1201", LaneTopology::Single, &routes, &proofs);
        assert_eq!(other_arch.policy, LaneStagePolicy::all_per_lane());
        let mut other_dtype = LaneStageRoutes::default();
        other_dtype[StageName::Head.index()] = vec![StageRoute::Projection(DType::F16)];
        let mismatch = resolve_with_proofs("gfx9999", LaneTopology::Single, &other_dtype, &proofs);
        assert_eq!(mismatch.policy, LaneStagePolicy::all_per_lane());
        assert!(mismatch.receipts.is_empty());
        // The public table is unaffected.
        let public = LaneStagePolicy::for_bound_routes("gfx9999", LaneTopology::Single, &routes);
        assert_eq!(public.policy, LaneStagePolicy::all_per_lane());
    }

    #[test]
    fn every_owned_instance_must_match_not_only_the_first() {
        let mq6 = StageRoute::Projection(DType::MQ6G256V2);
        let bf16 = StageRoute::Projection(DType::BF16);
        let base = pp2_mq2_routes();
        let resolve = |head: Vec<StageRoute>| {
            let mut routes = base.clone();
            routes[StageName::Head.index()] = head;
            LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes)
        };
        let all = resolve(vec![mq6, mq6]);
        assert_eq!(all.policy, LaneStagePolicy::production());
        assert!(fallback_stages(&all).is_empty());
        for head in [
            vec![mq6, bf16],
            vec![bf16, mq6],
            vec![mq6, mq6, StageRoute::Unclassified],
        ] {
            let resolution = resolve(head.clone());
            assert_eq!(fallback_stages(&resolution), [StageName::Head], "{head:?}");
            assert_eq!(
                resolution.policy,
                production_without(&[StageName::Head]),
                "{head:?}"
            );
        }
        // Same rule for a multi-field route: the second instance is wrong.
        let moe = base[StageName::Moe.index()][0];
        let mut routes = base.clone();
        routes[StageName::Moe.index()] = vec![moe, moe];
        let both = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes);
        assert_eq!(both.policy, LaneStagePolicy::production());
        routes[StageName::Moe.index()] = vec![
            moe,
            StageRoute::Moe {
                experts: LaneExpertRoute::ResidentPacked,
                dense: [DType::BF16; 5],
                decode_admitted: true,
            },
        ];
        let second = LaneStagePolicy::for_bound_routes("gfx1201", LaneTopology::Pp, &routes);
        assert_eq!(fallback_stages(&second), [StageName::Moe]);
    }

    #[test]
    fn receipts_dedup_and_cover_only_admitted_stages() {
        const SINGLE: &[LaneTopology] = &[LaneTopology::Single];
        let row = |stage, route, receipt| RowTwinProof {
            arch: "gfx9998",
            stage,
            route,
            domain: SINGLE,
            receipt,
        };
        let proofs = [
            row(StageName::Embed, StageRoute::Any, "R1"),
            row(StageName::PleRows, StageRoute::Any, "R1"),
            row(StageName::Head, StageRoute::Projection(DType::BF16), "R2"),
            row(StageName::Head, StageRoute::Projection(DType::F16), "R3"),
            row(StageName::Argmax, StageRoute::Argmax(DType::F32), "R5"),
        ];
        let mut routes = LaneStageRoutes::default();
        routes[StageName::Embed.index()] = vec![StageRoute::Embed(DType::Q8_0)];
        routes[StageName::PleRows.index()] = vec![StageRoute::PleRows(DType::BF16)];
        routes[StageName::Head.index()] = vec![
            StageRoute::Projection(DType::F16),
            StageRoute::Projection(DType::BF16),
            StageRoute::Projection(DType::F16),
        ];
        // Argmax mismatches its only row, so its receipt is not cited.
        routes[StageName::Argmax.index()] = vec![StageRoute::Argmax(DType::F16)];
        let resolution = resolve_with_proofs("gfx9998", LaneTopology::Single, &routes, &proofs);
        assert_eq!(resolution.receipts, ["R1", "R3", "R2"]);
        for stage in [StageName::Embed, StageName::PleRows, StageName::Head] {
            assert_eq!(resolution.policy.share(stage), StageShare::OnceAllRows);
        }
        assert_eq!(
            resolution.policy.share(StageName::Argmax),
            StageShare::PerLane
        );
        assert!(fallback_stages(&resolution).contains(&StageName::Argmax));
        assert!(!resolution.receipts.contains(&"R5"));
    }

    /// Wide-class groups: flag off, or no wide lane, is exactly the 8-row packing; a
    /// wide run packs to 32 rows; classes never share a group; the GPU segments and
    /// the protocol's groups agree on every shape.
    #[test]
    fn classed_groups_pack_by_class_and_match_the_protocol() {
        assert!(!lane_is_wide_for(false, false, true));
        assert!(!lane_is_wide_for(true, false, false));
        assert!(lane_is_wide_for(true, false, true));
        assert!(lane_is_wide_for(true, true, false));
        assert!(!lane_is_wide_for(false, true, true));
        let shapes: Vec<Vec<usize>> = vec![
            vec![4; 8],
            vec![4; 9],
            vec![2, 3, 4, 5, 6, 7, 8, 1],
            vec![1; 8],
            vec![4, 4, 4, 4, 4, 4, 4, 4, 4],
            vec![3; 11],
            vec![8; 5],
        ];
        for rows in &shapes {
            // no wide lane: identical to the plain 8-row grouping
            let none = vec![false; rows.len()];
            let plain = group_segments(rows, LANE_MAX_ROWS);
            let classed = classed_segments(rows, &none);
            assert_eq!(plain.len(), classed.len(), "{rows:?}");
            for (a, b) in plain.iter().zip(&classed) {
                assert_eq!(
                    (a.begin, a.rows, a.first_lane),
                    (b.begin, b.rows, b.first_lane)
                );
            }
            // all wide: 32-row packing
            let all = vec![true; rows.len()];
            let wide = classed_segments(rows, &all);
            assert_eq!(wide.len(), group_segments(rows, 32).len(), "{rows:?}");
            assert_eq!(
                wide.iter().map(|g| g.rows).sum::<usize>(),
                rows.iter().sum::<usize>()
            );
            assert_eq!(row_group_count(rows, &all), wide.len());
        }
        // 8 lanes x 4 rows: four exact groups, one wide group
        assert_eq!(row_group_count(&[4; 8], &[false; 8]), 4);
        assert_eq!(row_group_count(&[4; 8], &[true; 8]), 1);
        // a mixed step: classes are separate groups, in lane order
        let rows = [4usize; 8];
        let wide = [true, true, true, true, false, false, true, true];
        let groups = crate::mesh_lane_protocol::pack_classed_row_groups(&rows, &wide);
        let shape: Vec<(usize, usize, usize, bool)> = groups
            .iter()
            .map(|g| (g.first_lane, g.lanes, g.rows, g.wide))
            .collect();
        assert_eq!(
            shape,
            vec![(0, 4, 16, true), (4, 2, 8, false), (6, 2, 8, true)]
        );
        let seg = classed_segments(&rows, &wide);
        assert_eq!(
            seg.iter()
                .map(|g| (g.begin, g.rows, g.first_lane))
                .collect::<Vec<_>>(),
            vec![(0, 16, 0), (16, 8, 4), (24, 8, 6)]
        );
    }

    /// The GPU segments of the wide groups at caps 32 / 48 / 64 are the protocol's
    /// groups (first row, rows, first lane), so the executed stage launches match the
    /// sealed ledger at every cap; 8 lanes x 6 rows is two segments at 32 and one at 48.
    #[test]
    fn wide_segments_follow_the_protocol_groups_at_every_cap() {
        use crate::mesh_lane_protocol::pack_classed_row_groups_with_cap;
        let shape = |segs: &[Segment]| {
            segs.iter()
                .map(|s| (s.begin, s.rows, s.first_lane))
                .collect::<Vec<_>>()
        };
        let rows = [6usize; 8];
        let all = [true; 8];
        assert_eq!(
            shape(&segments_of_groups(pack_classed_row_groups_with_cap(
                &rows, &all, 32
            ))),
            vec![(0, 30, 0), (30, 18, 5)]
        );
        for cap in [48, 64] {
            assert_eq!(
                shape(&segments_of_groups(pack_classed_row_groups_with_cap(
                    &rows, &all, cap
                ))),
                vec![(0, 48, 0)]
            );
        }
        // Mixed step at cap 48: wide(12) / exact(6) / wide(6) back to back in row order.
        let wide = [true, true, false, true];
        assert_eq!(
            shape(&segments_of_groups(pack_classed_row_groups_with_cap(
                &[6; 4], &wide, 48
            ))),
            vec![(0, 12, 0), (12, 6, 2), (18, 6, 3)]
        );
        // The process default (no wide flag in the test process) is the 32-row cap.
        assert_eq!(
            shape(&classed_segments(&rows, &all)),
            vec![(0, 30, 0), (30, 18, 5)]
        );
    }

    #[test]
    fn segments_cover_rows_and_cap_groups() {
        let rows = vec![1usize; 11];
        let lanes = lane_segments(&rows);
        assert_eq!(lanes.len(), 11);
        assert!(lanes
            .iter()
            .enumerate()
            .all(|(i, s)| s.begin == i && s.rows == 1 && s.first_lane == i));
        let groups = group_segments(&rows, LANE_MAX_ROWS);
        assert_eq!(groups.len(), 2);
        assert_eq!(
            (groups[0].begin, groups[0].rows, groups[0].first_lane),
            (0, 8, 0)
        );
        assert_eq!(
            (groups[1].begin, groups[1].rows, groups[1].first_lane),
            (8, 3, 8)
        );
        let wide = group_segments(&[3, 9, 2], LANE_MAX_ROWS);
        assert_eq!(wide.len(), 3);
        assert_eq!((wide[1].begin, wide[1].rows), (3, 9));
        assert_eq!((wide[2].begin, wide[2].rows), (12, 2));
        // An armed CPU-splice step never packs a group past its window.
        let narrow = group_segments(
            &[1, 1, 1, 1, 1, 1],
            crate::program::QWEN4_CPU_SPLICE_MAX_ROWS,
        );
        assert_eq!(
            narrow.iter().map(|seg| seg.rows).collect::<Vec<_>>(),
            [4, 2]
        );
    }

    fn band_kinds(
        config: &crate::config::Qwen4Config,
        band: std::ops::Range<usize>,
    ) -> impl ExactSizeIterator<Item = (usize, LayerType)> + '_ {
        band.map(|layer| (layer, config.layer_types[layer]))
    }

    fn ple_layer_of(config: &crate::config::Qwen4Config) -> usize {
        config.ple_layer_ids[0] - 1
    }

    /// The whole-model layer program exactly as it was before bands existed:
    /// the frozen reference the generalized program must reproduce.
    fn frozen_singleton_program(
        layer_types: &[LayerType],
        policy: &LaneStagePolicy,
        ple_layer: usize,
    ) -> Vec<(LaneOp, StageName)> {
        let mut program = Vec::new();
        let mut gdn_slot = 0usize;
        let mut qsa_slot = 0usize;
        let mixer_phases =
            |in_stage: StageName, out_stage: StageName| -> Vec<(MixerPhase, StageName)> {
                let core = match in_stage {
                    StageName::GdnInProj => StageName::GdnCore,
                    _ => StageName::QsaCore,
                };
                if policy.share(in_stage).per_lane() && policy.share(out_stage).per_lane() {
                    vec![(MixerPhase::Whole, core)]
                } else {
                    vec![
                        (MixerPhase::InProj, in_stage),
                        (MixerPhase::Core, core),
                        (MixerPhase::OutProj, out_stage),
                    ]
                }
            };
        for layer in 0..layer_types.len() {
            if layer == ple_layer {
                program.push((LaneOp::PleConv, StageName::PleConv));
            }
            program.push((LaneOp::HcRead { layer, mlp: false }, StageName::HcRead));
            match layer_types[layer] {
                LayerType::LinearAttention => {
                    for (phase, stage) in mixer_phases(StageName::GdnInProj, StageName::GdnOutProj)
                    {
                        program.push((
                            LaneOp::Gdn {
                                layer,
                                slot: gdn_slot,
                                phase,
                            },
                            stage,
                        ));
                    }
                    gdn_slot += 1;
                }
                LayerType::FullAttention => {
                    for (phase, stage) in mixer_phases(StageName::QsaInProj, StageName::QsaOutProj)
                    {
                        program.push((
                            LaneOp::Qsa {
                                layer,
                                slot: qsa_slot,
                                phase,
                            },
                            stage,
                        ));
                    }
                    qsa_slot += 1;
                }
            }
            program.push((LaneOp::HcWrite { layer, mlp: false }, StageName::HcWrite));
            program.push((LaneOp::HcRead { layer, mlp: true }, StageName::HcRead));
            program.push((LaneOp::Moe { layer }, StageName::Moe));
            program.push((LaneOp::HcWrite { layer, mlp: true }, StageName::HcWrite));
        }
        program
    }

    fn test_policies() -> Vec<LaneStagePolicy> {
        vec![
            LaneStagePolicy::all_per_lane(),
            LaneStagePolicy::production(),
            LaneStagePolicy::all_per_lane()
                .with(StageName::GdnInProj, StageShare::OnceAllRows)
                .unwrap(),
            LaneStagePolicy::all_per_lane()
                .with(StageName::QsaOutProj, StageShare::OnceAllRows)
                .unwrap(),
        ]
    }

    #[test]
    fn single_scope_program_is_the_frozen_singleton_program() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        for policy in test_policies() {
            let program = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                single.input.then_some(ple_layer),
            );
            assert_eq!(
                program,
                frozen_singleton_program(&config.layer_types, &policy, ple_layer),
                "{}",
                policy.describe()
            );
        }
        // Explicit shape: per-lane policy keeps every mixer whole (6 ops per
        // layer + the PLE conv before the PLE layer); production splits them.
        let per_lane = lane_program(
            band_kinds(&config, single.band.clone()),
            &LaneStagePolicy::all_per_lane(),
            Some(ple_layer),
        );
        assert_eq!(per_lane.len(), config.num_hidden_layers * 6 + 1);
        assert_eq!(
            per_lane[..6],
            [
                (
                    LaneOp::HcRead {
                        layer: 0,
                        mlp: false
                    },
                    StageName::HcRead
                ),
                (
                    LaneOp::Gdn {
                        layer: 0,
                        slot: 0,
                        phase: MixerPhase::Whole
                    },
                    StageName::GdnCore
                ),
                (
                    LaneOp::HcWrite {
                        layer: 0,
                        mlp: false
                    },
                    StageName::HcWrite
                ),
                (
                    LaneOp::HcRead {
                        layer: 0,
                        mlp: true
                    },
                    StageName::HcRead
                ),
                (LaneOp::Moe { layer: 0 }, StageName::Moe),
                (
                    LaneOp::HcWrite {
                        layer: 0,
                        mlp: true
                    },
                    StageName::HcWrite
                ),
            ]
        );
        assert_eq!(per_lane[6], (LaneOp::PleConv, StageName::PleConv));
        let production = lane_program(
            band_kinds(&config, single.band.clone()),
            &LaneStagePolicy::production(),
            Some(ple_layer),
        );
        assert_eq!(production.len(), config.num_hidden_layers * 8 + 1);
    }

    /// The EP prefill wave's program, both arms. Incumbent (`HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT=0`):
    /// every stage per lane (each mixer stays the singleton's whole op on the lane's own rows)
    /// except the MoE, the only op placed once over all rows, one per layer. Joint (default): the
    /// mixers split into in-proj / core / out-proj phases (the cores stay per lane), the HC
    /// stages and projections are placed once over all rows, and still exactly one MoE per layer.
    #[test]
    fn ep_wave_program_places_only_the_moe_once() {
        use crate::gpu_forward::ep_wave::ep_wave_policy_for;
        let config = crate::config::compact_test_config();
        let ple_layer = ple_layer_of(&config);
        let single = Qwen4BandScope::single(&config);
        // Incumbent arm.
        let policy = ep_wave_policy_for(false);
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let mut once = 0usize;
        for (op, stage) in &program {
            let share = policy.share(*stage);
            if matches!(op, LaneOp::Moe { .. }) {
                assert_eq!(share, StageShare::OnceAllRows);
                once += 1;
            } else {
                assert!(share.per_lane(), "{op:?} runs per lane");
            }
            // No mixer splits: a per-lane projection pair is the whole singleton op.
            if let LaneOp::Gdn { phase, .. } | LaneOp::Qsa { phase, .. } = op {
                assert_eq!(*phase, MixerPhase::Whole);
            }
        }
        // One exchange per layer, whatever the lane count.
        assert_eq!(once, config.num_hidden_layers);
        // Six ops per layer (the mixers stay whole) plus the PLE conv.
        assert_eq!(program.len(), config.num_hidden_layers * 6 + 1);

        // Joint arm.
        let policy = ep_wave_policy_for(true);
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let (mut moe, mut hc, mut cores, mut projections) = (0usize, 0usize, 0usize, 0usize);
        for (op, stage) in &program {
            let share = policy.share(*stage);
            match op {
                LaneOp::Moe { .. } => {
                    assert_eq!(share, StageShare::OnceAllRows);
                    moe += 1;
                }
                LaneOp::HcRead { .. } | LaneOp::HcWrite { .. } => {
                    assert_eq!(share, StageShare::OnceAllRows, "{op:?}");
                    hc += 1;
                }
                LaneOp::Gdn { phase, .. } | LaneOp::Qsa { phase, .. } => match phase {
                    MixerPhase::Whole => panic!("a joint mixer splits: {op:?}"),
                    MixerPhase::Core => {
                        assert!(share.per_lane(), "{op:?} runs per lane");
                        cores += 1;
                    }
                    MixerPhase::InProj | MixerPhase::OutProj => {
                        assert_eq!(share, StageShare::OnceAllRows, "{op:?}");
                        projections += 1;
                    }
                },
                _ => assert!(share.per_lane(), "{op:?} runs per lane"),
            }
        }
        let layers = config.num_hidden_layers;
        assert_eq!(moe, layers, "one MoE exchange per layer");
        assert_eq!(hc, layers * 4);
        assert_eq!(cores, layers);
        assert_eq!(projections, layers * 2);
        // Eight ops per layer (the mixer is three) plus the PLE conv.
        assert_eq!(program.len(), layers * 8 + 1);
    }

    fn band_scopes(config: &crate::config::Qwen4Config, counts: &[usize]) -> Vec<Qwen4BandScope> {
        assert_eq!(counts.iter().sum::<usize>(), config.num_hidden_layers);
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == config.num_hidden_layers,
                    band,
                }
            })
            .collect()
    }

    #[test]
    fn band_programs_cover_the_trunk_with_local_slots() {
        let config = crate::config::compact_test_config();
        let ple_layer = ple_layer_of(&config);
        let single = Qwen4BandScope::single(&config);
        for policy in test_policies() {
            let whole = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            for counts in [
                &[24usize, 24][..],
                &[16, 16, 16][..],
                &[2, 20, 26][..],
                &[10, 20, 18][..],
                &[9, 8, 8, 8, 8, 7][..],
            ] {
                let scopes = band_scopes(&config, counts);
                let mut covered: Vec<(LaneOp, StageName)> = Vec::new();
                for scope in &scopes {
                    scope.validate(&config).unwrap();
                    let map = crate::state::Qwen4StateMap::new(&config, scope).unwrap();
                    let program = lane_program(
                        band_kinds(&config, scope.band.clone()),
                        &policy,
                        scope.input.then_some(ple_layer),
                    );
                    // The PLE conv belongs to the input band alone.
                    assert_eq!(
                        program.iter().any(|(op, _)| *op == LaneOp::PleConv),
                        scope.input
                    );
                    let (mut gdn_slots, mut qsa_slots) = (0usize, 0usize);
                    for (op, stage) in &program {
                        // Slots are the state map's compact local slots; the
                        // whole-model program names them by global ordinal.
                        let global = match *op {
                            LaneOp::Gdn { layer, slot, phase } => {
                                assert!(scope.band.contains(&layer));
                                assert_eq!(map.gdn_local(layer), Some(slot));
                                gdn_slots = gdn_slots.max(slot + 1);
                                LaneOp::Gdn {
                                    layer,
                                    slot: map.gdn_base() + slot,
                                    phase,
                                }
                            }
                            LaneOp::Qsa { layer, slot, phase } => {
                                assert!(scope.band.contains(&layer));
                                assert_eq!(map.qsa_local(layer), Some(slot));
                                qsa_slots = qsa_slots.max(slot + 1);
                                LaneOp::Qsa {
                                    layer,
                                    slot: map.qsa_base() + slot,
                                    phase,
                                }
                            }
                            LaneOp::HcRead { layer, .. }
                            | LaneOp::HcWrite { layer, .. }
                            | LaneOp::Moe { layer } => {
                                assert!(scope.band.contains(&layer));
                                *op
                            }
                            LaneOp::PleConv | LaneOp::PleProject => *op,
                        };
                        covered.push((global, *stage));
                    }
                    assert_eq!(gdn_slots, map.gdn_count());
                    assert_eq!(qsa_slots, map.qsa_count());
                }
                // Bands in order reproduce the whole-model program exactly.
                assert_eq!(covered, whole, "{counts:?} under {}", policy.describe());
            }
        }
    }

    #[test]
    fn edge_steps_have_no_capture_and_keep_their_phase() {
        for prefill in [false, true] {
            let edge = StepOutputs::Edge { prefill };
            assert_eq!(edge.prefill(), prefill);
            assert!(edge.wide_hidden().is_none());
        }
    }

    #[test]
    fn packed_bf16_rows_start_at_two_bytes_per_element() {
        let wide = 4 * 2560;
        let f32_bytes = DType::F32.size();
        let numel = 8 * wide;
        // Lane `begin` row starts at 2 * begin * wide bytes: half the F32
        // element offset, with no gap after the previous lane's rows.
        for begin in 0..=8 {
            let offset = packed_bf16_offset(begin, wide, f32_bytes, numel).unwrap();
            assert_eq!(offset * f32_bytes, begin * wide * 2);
            assert_eq!(offset, begin * wide / 2);
        }
        // The prefix of n rows is exactly n * wide * 2 contiguous bytes.
        let (first, second) = (
            packed_bf16_offset(3, wide, f32_bytes, numel).unwrap() * f32_bytes,
            packed_bf16_offset(5, wide, f32_bytes, numel).unwrap() * f32_bytes,
        );
        assert_eq!(second - first, 2 * wide * 2);
        // Past the tensor, unaligned and overflowing starts are refused.
        assert_eq!(packed_bf16_offset(17, wide, f32_bytes, numel), None);
        assert_eq!(packed_bf16_offset(1, 3, f32_bytes, numel), None);
        assert_eq!(packed_bf16_offset(usize::MAX, 2, f32_bytes, numel), None);
        assert_eq!(packed_bf16_offset(1, wide, 0, numel), None);
    }

    /// `(segment, op)` of a whole program as `forward_lanes_scoped` walks it:
    /// `lanes` lane segments then `groups` group segments, the all-rows HC
    /// segment (when `hc_all` is `Some`) appended after them.
    fn program_items(
        program: &[(LaneOp, StageName)],
        policy: &LaneStagePolicy,
        lanes: usize,
        groups: usize,
        all_rows: Option<AllRowsPlan>,
    ) -> Vec<(usize, LaneOp)> {
        let mut emitted = Vec::new();
        let mut start = 0;
        while start < program.len() {
            let per_lane = policy.share(program[start].1).per_lane();
            let mut end = start + 1;
            while end < program.len() && policy.share(program[end].1).per_lane() == per_lane {
                end += 1;
            }
            let segs = if per_lane {
                0..lanes
            } else {
                lanes..lanes + groups
            };
            let hc = all_rows
                .filter(|_| !per_lane)
                .map(|plan| (lanes + groups, plan));
            for (seg, ops) in run_items(start, &program[start..end], segs, hc, None) {
                emitted.extend(program[ops].iter().map(|&(op, _)| (seg, op)));
            }
            start = end;
        }
        emitted
    }

    /// Today's emission, written out independently of `run_pieces`: every run
    /// segment-major over its share's segments.
    fn group_major_items(
        program: &[(LaneOp, StageName)],
        policy: &LaneStagePolicy,
        lanes: usize,
        groups: usize,
    ) -> Vec<(usize, LaneOp)> {
        let mut emitted = Vec::new();
        let mut start = 0;
        while start < program.len() {
            let per_lane = policy.share(program[start].1).per_lane();
            let mut end = start + 1;
            while end < program.len() && policy.share(program[end].1).per_lane() == per_lane {
                end += 1;
            }
            let segs = if per_lane {
                0..lanes
            } else {
                lanes..lanes + groups
            };
            for seg in segs {
                emitted.extend(program[start..end].iter().map(|&(op, _)| (seg, op)));
            }
            start = end;
        }
        emitted
    }

    fn is_hc(op: LaneOp) -> bool {
        matches!(op, LaneOp::HcRead { .. } | LaneOp::HcWrite { .. })
    }

    /// `PleProject` goes right before the program's `PleConv` and nowhere else; a band without
    /// the PLE layer keeps its program.
    #[test]
    fn with_ple_project_precedes_the_ple_conv_and_leaves_other_programs_alone() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        for policy in test_policies() {
            let base = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            let mut with = base.clone();
            with_ple_project(&mut with);
            assert_eq!(with.len(), base.len() + 1);
            let conv = with
                .iter()
                .position(|&(op, _)| op == LaneOp::PleConv)
                .unwrap();
            assert_eq!(with[conv - 1], (LaneOp::PleProject, StageName::PleProject));
            let mut without = with.clone();
            without.remove(conv - 1);
            assert_eq!(without, base, "{}", policy.describe());
            // No PLE layer (a band that is not the input band): unchanged.
            let band = lane_program(band_kinds(&config, single.band.clone()), &policy, None);
            let mut same = band.clone();
            with_ple_project(&mut same);
            assert_eq!(same, band);
        }
    }

    /// The PLE project piece is ONE item on its own segment (every row of the step) right
    /// before the per-lane conv tails (stage `ple_conv`, still one per lane, in lane order);
    /// every other `(segment, op)` item is today's emission, with and without the all-rows HC
    /// segment. Without a PLE segment the piece falls back to the groups.
    #[test]
    fn ple_project_is_one_item_on_its_own_segment_ahead_of_the_per_lane_conv_tails() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let policy = LaneStagePolicy::production();
        assert_eq!(policy.share(StageName::PleProject), StageShare::OnceAllRows);
        assert_eq!(policy.share(StageName::PleConv), StageShare::PerLane);
        let base = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let mut program = base.clone();
        with_ple_project(&mut program);
        for (lanes, groups) in [(2, 1), (8, 4), (8, 8)] {
            for all_rows in [None, plan([true, true], false), plan([true, true], true)] {
                // The PLE project segment is appended after the groups and the all-rows one.
                let ple_seg = lanes + groups + usize::from(all_rows.is_some());
                let mut emitted: Vec<(usize, LaneOp)> = Vec::new();
                let mut start = 0;
                while start < program.len() {
                    let per_lane = policy.share(program[start].1).per_lane();
                    let mut end = start + 1;
                    while end < program.len() && policy.share(program[end].1).per_lane() == per_lane
                    {
                        end += 1;
                    }
                    let segs = if per_lane {
                        0..lanes
                    } else {
                        lanes..lanes + groups
                    };
                    let hc = all_rows
                        .filter(|_| !per_lane)
                        .map(|plan| (lanes + groups, plan));
                    for (seg, ops) in
                        run_items(start, &program[start..end], segs, hc, Some(ple_seg))
                    {
                        emitted.extend(program[ops].iter().map(|&(op, _)| (seg, op)));
                    }
                    start = end;
                }
                let tag = format!(
                    "lanes {lanes} groups {groups} all_rows {}",
                    all_rows.is_some()
                );
                let projects: Vec<usize> = (0..emitted.len())
                    .filter(|&i| emitted[i].1 == LaneOp::PleProject)
                    .collect();
                assert_eq!(projects.len(), 1, "{tag}");
                assert_eq!(emitted[projects[0]].0, ple_seg, "{tag}");
                let convs: Vec<usize> = (0..emitted.len())
                    .filter(|&i| emitted[i].1 == LaneOp::PleConv)
                    .collect();
                assert_eq!(convs.len(), lanes, "{tag}");
                for (lane, &at) in convs.iter().enumerate() {
                    assert_eq!(emitted[at], (lane, LaneOp::PleConv), "{tag}");
                    assert_eq!(at, projects[0] + 1 + lane, "{tag}");
                }
                let mut rest = emitted.clone();
                rest.remove(projects[0]);
                assert_eq!(
                    rest,
                    program_items(&base, &policy, lanes, groups, all_rows),
                    "{tag}"
                );
            }
        }
        let run = [(LaneOp::PleProject, StageName::PleProject)];
        assert_eq!(run_items(0, &run, 3..5, None, None), [(3, 0..1), (4, 0..1)]);
        assert_eq!(run_items(0, &run, 3..5, None, Some(9)), [(9, 0..1)]);
    }

    #[test]
    fn hc_all_rows_flag_off_emits_todays_group_major_program() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        for policy in test_policies() {
            let program = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            for (lanes, groups) in [(1, 1), (8, 1), (8, 2), (8, 4), (16, 8)] {
                assert_eq!(
                    program_items(&program, &policy, lanes, groups, None),
                    group_major_items(&program, &policy, lanes, groups),
                    "{} lanes {lanes} groups {groups}",
                    policy.describe()
                );
            }
        }
    }

    #[test]
    fn hc_all_rows_places_each_hc_stage_once_and_keeps_every_other_stage_in_groups() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let layers = config.num_hidden_layers;
        let policy = LaneStagePolicy::production();
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        for (lanes, groups) in [(8, 2), (8, 3), (8, 4), (16, 8)] {
            let hc_seg = lanes + groups;
            let today = group_major_items(&program, &policy, lanes, groups);
            let all = program_items(&program, &policy, lanes, groups, plan([true, true], false));
            // Today: every HC op runs once per group.
            assert_eq!(
                today.iter().filter(|(_, op)| is_hc(*op)).count(),
                layers * 4 * groups
            );
            // All-rows: every HC op once, on the all-rows segment alone.
            let hc_items: Vec<_> = all.iter().filter(|(_, op)| is_hc(*op)).collect();
            assert_eq!(hc_items.len(), layers * 4);
            assert!(hc_items.iter().all(|(seg, _)| *seg == hc_seg));
            // Every other op: the very (segment, op) multiset of today's emission.
            let rest = |items: &[(usize, LaneOp)]| {
                let mut rest: Vec<_> = items
                    .iter()
                    .filter(|(_, op)| !is_hc(*op))
                    .map(|&(seg, op)| (seg, format!("{op:?}")))
                    .collect();
                rest.sort();
                rest
            };
            assert_eq!(rest(&all), rest(&today), "lanes {lanes} groups {groups}");
            // Each segment still sees its non-HC ops in today's relative order.
            for seg in 0..hc_seg {
                let seq = |items: &[(usize, LaneOp)]| -> Vec<LaneOp> {
                    items
                        .iter()
                        .filter(|(s, op)| *s == seg && !is_hc(*op))
                        .map(|&(_, op)| op)
                        .collect()
                };
                assert_eq!(seq(&all), seq(&today), "segment {seg}");
            }
            // The all-rows HC ops keep the program's order: per layer read(attn),
            // write(attn), read(mlp), write(mlp).
            let hc_ops: Vec<LaneOp> = hc_items.iter().map(|&&(_, op)| op).collect();
            let expected: Vec<LaneOp> = program
                .iter()
                .map(|&(op, _)| op)
                .filter(|op| is_hc(*op))
                .collect();
            assert_eq!(hc_ops, expected);
        }
    }

    #[test]
    fn hc_all_rows_leaves_all_reduce_writes_and_per_lane_runs_in_place() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let layers = config.num_hidden_layers;
        let policy = LaneStagePolicy::production();
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let (lanes, groups) = (8, 4);
        let hc_seg = lanes + groups;
        // The FFN write carried by the all-reduce stays per group; reads and the
        // mixer writes go all-rows.
        let items = program_items(&program, &policy, lanes, groups, plan([true, false], false));
        for (seg, op) in &items {
            match op {
                LaneOp::HcRead { .. } | LaneOp::HcWrite { mlp: false, .. } => {
                    assert_eq!(*seg, hc_seg, "{op:?}")
                }
                LaneOp::HcWrite { mlp: true, .. } => assert!(*seg >= lanes && *seg < hc_seg),
                _ => assert_ne!(*seg, hc_seg, "{op:?}"),
            }
        }
        assert_eq!(
            items
                .iter()
                .filter(|(_, op)| matches!(op, LaneOp::HcWrite { mlp: true, .. }))
                .count(),
            layers * groups
        );
        // Per-lane stages are emitted exactly as today, with and without the arm.
        let per_lane = |items: &[(usize, LaneOp)]| -> Vec<(usize, LaneOp)> {
            items
                .iter()
                .copied()
                .filter(|(seg, _)| *seg < lanes)
                .collect()
        };
        assert_eq!(
            per_lane(&items),
            per_lane(&group_major_items(&program, &policy, lanes, groups))
        );
    }

    /// The exact EP prefill wave never takes the AR step's all-rows HC placement. Incumbent arm
    /// (`HIPFIRE_EP_LANE_PREFILL_WAVE_JOINT=0`): every HC stage is PER LANE (each lane's HC read /
    /// write are its singleton tile's launches at the lane's own rows and stream kind; under the
    /// trunk shard each write is its own all-reduce at the lane's row count), refused by the step
    /// kind AND by the policy. Joint arm (default): the HC stages are one segment over every
    /// lane's rows by the wave's own policy, and that segment is not the AR step's placement:
    /// `hc_all_rows_applies` stays refused for the wave's step kind.
    #[test]
    fn the_exact_wave_never_takes_the_all_rows_hc_placement() {
        use crate::gpu_forward::ep_wave::ep_wave_policy_for;
        let incumbent = ep_wave_policy_for(false);
        assert_eq!(incumbent.share(StageName::HcRead), StageShare::PerLane);
        assert_eq!(incumbent.share(StageName::HcWrite), StageShare::PerLane);
        let joint = ep_wave_policy_for(true);
        assert_eq!(joint.share(StageName::HcRead), StageShare::OnceAllRows);
        assert_eq!(joint.share(StageName::HcWrite), StageShare::OnceAllRows);
        for groups in [&[8usize, 8][..], &[8, 8, 8, 8], &[5, 5]] {
            // The wave is a prefill step, never an AR step, whichever policy it runs.
            for wave in [&incumbent, &joint] {
                assert!(!hc_all_rows_applies(true, false, true, false, groups, wave));
            }
            // Incumbent: refused by the policy even for an AR step's shape.
            assert!(!hc_all_rows_applies(
                true, true, true, false, groups, &incumbent
            ));
        }
    }

    #[test]
    fn hc_all_rows_applies_only_to_eager_multi_group_ar_steps() {
        let production = LaneStagePolicy::production();
        let eight_by_two = [8usize, 8];
        let applies = |flag, ar, eager, cpu, groups: &[usize], policy: &LaneStagePolicy| {
            hc_all_rows_applies(flag, ar, eager, cpu, groups, policy)
        };
        assert!(applies(true, true, true, false, &eight_by_two, &production));
        // Flag off, a prefill wave / band edge (not an AR step), a recorded or
        // captured step, a CPU splice: never.
        assert!(!applies(
            false,
            true,
            true,
            false,
            &eight_by_two,
            &production
        ));
        assert!(!applies(
            true,
            false,
            true,
            false,
            &eight_by_two,
            &production
        ));
        assert!(!applies(
            true,
            true,
            false,
            false,
            &eight_by_two,
            &production
        ));
        assert!(!applies(true, true, true, true, &eight_by_two, &production));
        // One group (<= 8 rows) has nothing to share; an over-wide group, an empty one and
        // more than the 64 wide-verify rows are refused.
        assert!(!applies(true, true, true, false, &[8], &production));
        assert!(!applies(true, true, true, false, &[3, 4], &production));
        assert!(!applies(true, true, true, false, &[9, 8], &production));
        assert!(!applies(true, true, true, false, &[8, 0], &production));
        assert!(applies(true, true, true, false, &[8; 8], &production));
        assert!(!applies(true, true, true, false, &[8; 9], &production));
        // 24 and 32 rows (three / four groups of 8, or six / eight of 4).
        assert!(applies(true, true, true, false, &[8, 8, 8], &production));
        assert!(applies(true, true, true, false, &[4; 8], &production));
        // The HC read and write must both be once-over-all-rows.
        for stage in [StageName::HcRead, StageName::HcWrite] {
            let per_lane = LaneStagePolicy::production()
                .with(stage, StageShare::PerLane)
                .unwrap();
            assert!(!applies(true, true, true, false, &eight_by_two, &per_lane));
        }
        assert!(!applies(
            true,
            true,
            true,
            false,
            &eight_by_two,
            &LaneStagePolicy::all_per_lane()
        ));
    }

    fn plan(hc_writes: [bool; 2], stages: bool) -> Option<AllRowsPlan> {
        Some(AllRowsPlan {
            hc_writes,
            stages,
            moe: false,
        })
    }

    /// The `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` plan: every stage all-rows, every HC write too.
    fn plan_moe() -> Option<AllRowsPlan> {
        Some(AllRowsPlan {
            hc_writes: [true, true],
            stages: true,
            moe: true,
        })
    }

    fn stage_of(program: &[(LaneOp, StageName)], op: LaneOp) -> StageName {
        program
            .iter()
            .find(|(candidate, _)| *candidate == op)
            .map(|&(_, stage)| stage)
            .expect("every emitted op is in the program")
    }

    /// Without `stages` the all-rows segment carries the HC ops and nothing else (H0's
    /// placement), whatever the policy.
    #[test]
    fn all_rows_without_stages_is_the_hc_only_placement() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        for policy in test_policies() {
            let program = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            for (lanes, groups) in [(8, 2), (16, 8)] {
                let all_seg = lanes + groups;
                for (seg, op) in
                    program_items(&program, &policy, lanes, groups, plan([true, true], false))
                {
                    let on_all = seg == all_seg;
                    let hc_once = is_hc(op) && !policy.share(stage_of(&program, op)).per_lane();
                    assert_eq!(on_all, hc_once, "{op:?} {}", policy.describe());
                }
            }
        }
    }

    /// `HIPFIRE_QWEN4_LANE_ALLROWS`: every non-MoE once-over-all-rows op appears ONCE on the
    /// all-rows segment (one launch per op per layer), the routed MoE stays on its groups
    /// between the all-rows HC read and write of its block, and per-lane stages are emitted
    /// exactly as today.
    #[test]
    fn all_rows_stages_place_every_non_moe_once_op_once_and_keep_the_moe_in_groups() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let layers = config.num_hidden_layers;
        for policy in test_policies() {
            let program = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            let moe_once = !policy.share(StageName::Moe).per_lane();
            for (lanes, groups) in [(8, 2), (8, 3), (8, 4), (16, 8)] {
                let all_seg = lanes + groups;
                let today = group_major_items(&program, &policy, lanes, groups);
                let items =
                    program_items(&program, &policy, lanes, groups, plan([true, true], true));
                let label = format!("{} lanes {lanes} groups {groups}", policy.describe());
                for &(seg, op) in &items {
                    let stage = stage_of(&program, op);
                    if policy.share(stage).per_lane() {
                        assert!(seg < lanes, "{op:?} {label}");
                    } else if stage == StageName::Moe {
                        assert!((lanes..all_seg).contains(&seg), "{op:?} {label}");
                    } else {
                        assert_eq!(seg, all_seg, "{op:?} {label}");
                    }
                }
                // Each non-MoE once op: exactly once, in program order.
                let on_all: Vec<LaneOp> = items
                    .iter()
                    .filter(|(seg, _)| *seg == all_seg)
                    .map(|&(_, op)| op)
                    .collect();
                let expected: Vec<LaneOp> = program
                    .iter()
                    .filter(|&&(_, stage)| {
                        !policy.share(stage).per_lane() && stage != StageName::Moe
                    })
                    .map(|&(op, _)| op)
                    .collect();
                assert_eq!(on_all, expected, "{label}");
                // The MoE: once per layer per group (or lane), never on the all-rows segment.
                let moe_items = items
                    .iter()
                    .filter(|(_, op)| matches!(op, LaneOp::Moe { .. }))
                    .count();
                assert_eq!(
                    moe_items,
                    layers * if moe_once { groups } else { lanes },
                    "{label}"
                );
                // Per-lane segments see exactly today's emission.
                let per_lane = |items: &[(usize, LaneOp)]| -> Vec<(usize, LaneOp)> {
                    items
                        .iter()
                        .copied()
                        .filter(|(seg, _)| *seg < lanes)
                        .collect()
                };
                assert_eq!(per_lane(&items), per_lane(&today), "{label}");
                // Each group's MoE calls run between the all-rows HC read and write of
                // their own block.
                if moe_once && policy.share(StageName::HcRead) == StageShare::OnceAllRows {
                    let at = |wanted: LaneOp, nth: usize| {
                        items
                            .iter()
                            .enumerate()
                            .filter(|(_, (_, op))| *op == wanted)
                            .nth(nth)
                            .map(|(index, _)| index)
                    };
                    for layer in 0..layers {
                        let read = at(LaneOp::HcRead { layer, mlp: true }, 0).unwrap();
                        let write = at(LaneOp::HcWrite { layer, mlp: true }, 0).unwrap();
                        for nth in 0..groups {
                            let moe = at(LaneOp::Moe { layer }, nth).unwrap();
                            assert!(read < moe && moe < write, "layer {layer} {label}");
                        }
                    }
                }
            }
        }
    }

    /// A write a TP2 all-reduce carries stays per group even with every other non-MoE op
    /// all-rows.
    #[test]
    fn all_rows_stages_keep_all_reduce_writes_in_groups() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let layers = config.num_hidden_layers;
        let policy = LaneStagePolicy::production();
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let (lanes, groups) = (8, 4);
        let all_seg = lanes + groups;
        let items = program_items(&program, &policy, lanes, groups, plan([true, false], true));
        for &(seg, op) in &items {
            match op {
                LaneOp::HcWrite { mlp: true, .. } | LaneOp::Moe { .. } => {
                    assert!((lanes..all_seg).contains(&seg), "{op:?}")
                }
                _ if policy.share(stage_of(&program, op)).per_lane() => assert!(seg < lanes),
                _ => assert_eq!(seg, all_seg, "{op:?}"),
            }
        }
        assert_eq!(
            items
                .iter()
                .filter(|(_, op)| matches!(op, LaneOp::HcWrite { mlp: true, .. }))
                .count(),
            layers * groups
        );
    }

    /// The refinement applies only to a step the all-rows segment admitted (so never a
    /// prefill wave, band edge, recorded / captured step, CPU splice, prefill-invariant scope
    /// or a step of at most 8 rows), with the flag, no wide-verify scope in force, no
    /// row-keyed GDN qkv BF16 store and one HC stream storage.
    #[test]
    fn all_rows_stages_apply_only_where_the_segment_applies() {
        assert!(all_rows_stages_apply(true, true, 0, false, true));
        assert!(!all_rows_stages_apply(false, true, 0, false, true));
        assert!(!all_rows_stages_apply(true, false, 0, false, true));
        assert!(!all_rows_stages_apply(true, true, 8, false, true));
        assert!(!all_rows_stages_apply(true, true, 0, true, true));
        assert!(!all_rows_stages_apply(true, true, 0, false, false));
        let production = LaneStagePolicy::production();
        let segment = |ar, eager, cpu, groups: &[usize]| {
            hc_all_rows_applies(true, ar, eager, cpu, groups, &production)
        };
        let stages = |segment_applies| all_rows_stages_apply(true, segment_applies, 0, false, true);
        assert!(stages(segment(true, true, false, &[8, 8])));
        // Prefill wave / band edge, recorded or captured (or invariant) step, CPU splice.
        assert!(!stages(segment(false, true, false, &[8, 8])));
        assert!(!stages(segment(true, false, false, &[8, 8])));
        assert!(!stages(segment(true, true, true, &[8, 8])));
        // n <= 8: one group, nothing to share.
        for groups in [&[1usize][..], &[8], &[4]] {
            assert!(!stages(segment(true, true, false, groups)));
        }
    }

    /// `HIPFIRE_QWEN4_LANE_ALLROWS_MOE`: no op runs per group. Every non-MoE once op is on the
    /// all-rows segment, every HC write too (an all-reduce write included), and the routed MoE
    /// appears exactly once per layer on the segment after it, between the all-rows HC read and
    /// write of its block; per-lane stages are emitted exactly as today.
    #[test]
    fn all_rows_moe_runs_one_call_per_layer_and_no_op_per_group() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let layers = config.num_hidden_layers;
        for policy in test_policies() {
            let program = lane_program(
                band_kinds(&config, single.band.clone()),
                &policy,
                Some(ple_layer),
            );
            let moe_once = !policy.share(StageName::Moe).per_lane();
            for (lanes, groups) in [(8, 2), (8, 6), (16, 8)] {
                let hc_seg = lanes + groups;
                let moe_seg = hc_seg + 1;
                let today = group_major_items(&program, &policy, lanes, groups);
                let items = program_items(&program, &policy, lanes, groups, plan_moe());
                let label = format!("{} lanes {lanes} groups {groups}", policy.describe());
                for &(seg, op) in &items {
                    let stage = stage_of(&program, op);
                    if policy.share(stage).per_lane() {
                        assert!(seg < lanes, "{op:?} {label}");
                    } else if stage == StageName::Moe {
                        assert_eq!(seg, moe_seg, "{op:?} {label}");
                    } else {
                        assert_eq!(seg, hc_seg, "{op:?} {label}");
                    }
                }
                // No once op is emitted on a group segment.
                assert!(
                    items
                        .iter()
                        .all(|(seg, _)| *seg < lanes || *seg >= lanes + groups),
                    "{label}"
                );
                if moe_once {
                    let moe_items = items
                        .iter()
                        .filter(|(_, op)| matches!(op, LaneOp::Moe { .. }))
                        .count();
                    assert_eq!(moe_items, layers, "{label}");
                    if policy.share(StageName::HcRead) == StageShare::OnceAllRows
                        && policy.share(StageName::HcWrite) == StageShare::OnceAllRows
                    {
                        let at = |wanted: LaneOp| {
                            items.iter().position(|(_, op)| *op == wanted).unwrap()
                        };
                        for layer in 0..layers {
                            let read = at(LaneOp::HcRead { layer, mlp: true });
                            let moe = at(LaneOp::Moe { layer });
                            let write = at(LaneOp::HcWrite { layer, mlp: true });
                            assert!(read < moe && moe < write, "layer {layer} {label}");
                        }
                    }
                }
                let per_lane = |items: &[(usize, LaneOp)]| -> Vec<(usize, LaneOp)> {
                    items
                        .iter()
                        .copied()
                        .filter(|(seg, _)| *seg < lanes)
                        .collect()
                };
                assert_eq!(per_lane(&items), per_lane(&today), "{label}");
            }
        }
    }

    /// Without `moe` the plan is byte-for-byte the all-rows-stages placement: the MoE stays on
    /// the groups.
    #[test]
    fn all_rows_plan_without_moe_keeps_the_moe_on_groups() {
        let config = crate::config::compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let ple_layer = ple_layer_of(&config);
        let policy = LaneStagePolicy::production();
        let program = lane_program(
            band_kinds(&config, single.band.clone()),
            &policy,
            Some(ple_layer),
        );
        let (lanes, groups) = (8, 6);
        let items = program_items(&program, &policy, lanes, groups, plan([true, true], true));
        for &(seg, op) in &items {
            if matches!(op, LaneOp::Moe { .. }) {
                assert!((lanes..lanes + groups).contains(&seg), "{op:?}");
            }
        }
    }

    /// The plan-time refusals of the one all-rows MoE call: each keeps the groups.
    #[test]
    fn all_rows_moe_refuses_what_the_wide_call_cannot_take() {
        // 48 rows, EP gather cap 48, no shard writer, 40 owned layers: applies.
        let refusal = |stages, once, cap, n, wide, ar, calls| {
            all_rows_moe_refusal(stages, once, cap, n, wide, ar, calls)
        };
        assert_eq!(refusal(true, true, Some(48), 48, false, false, 40), None);
        assert_eq!(refusal(true, true, Some(48), 48, false, true, 40), None);
        assert_eq!(refusal(true, true, Some(51), 51, false, false, 40), None);
        // The all-rows stage segment did not apply / the MoE is per lane / no EP gather.
        assert!(refusal(false, true, Some(48), 48, false, false, 40).is_some());
        assert!(refusal(true, false, Some(48), 48, false, false, 40).is_some());
        assert!(refusal(true, true, None, 48, false, false, 40).is_some());
        // A wide-class lane keeps its classed group.
        assert!(refusal(true, true, Some(48), 48, true, false, 40).is_some());
        // Over the gather capacity (8 rows without a wide verify flag, the wide cap with one).
        assert!(refusal(true, true, Some(8), 16, false, false, 40)
            .unwrap()
            .contains("gather capacity"));
        assert!(refusal(true, true, Some(32), 48, false, false, 40)
            .unwrap()
            .contains("32-row"));
        // Over the tightest wide stage limit (51 rows at top-10), whatever the capacity.
        let over = refusal(true, true, Some(64), 52, false, false, 40).unwrap();
        assert!(over.contains("51-row limit"), "{over}");
        // An odd (or empty) MoE call count would reuse a gather buffer across steps.
        assert!(refusal(true, true, Some(48), 48, false, false, 39).is_some());
        assert!(refusal(true, true, Some(48), 48, false, false, 0).is_some());
    }

    /// A wide-class lane step packs into ONE group of up to the wide cap, so its MoE is already
    /// one call over every row and the all-rows segment (which needs several groups) stays out
    /// of it: `HIPFIRE_QWEN4_LANE_ALLROWS` composes with the wide path by not applying to it.
    #[test]
    fn wide_class_lane_step_is_one_group_and_takes_no_all_rows_segment() {
        use crate::mesh_lane_protocol::pack_classed_row_groups_with_cap;
        let rows = [6usize; 8];
        let wide_groups =
            segments_of_groups(pack_classed_row_groups_with_cap(&rows, &[true; 8], 48));
        assert_eq!(wide_groups.len(), 1);
        assert_eq!(wide_groups[0].rows, 48);
        let group_rows: Vec<usize> = wide_groups.iter().map(|seg| seg.rows).collect();
        let production = LaneStagePolicy::production();
        assert!(!hc_all_rows_applies(
            true,
            true,
            true,
            false,
            &group_rows,
            &production
        ));
        // Row-exact lanes of the same rows (6 + 6 > 8 rows) are one group each: several groups.
        let exact = segments_of_groups(pack_classed_row_groups_with_cap(&rows, &[false; 8], 48));
        assert_eq!(exact.len(), 8);
        let exact_rows: Vec<usize> = exact.iter().map(|seg| seg.rows).collect();
        assert!(hc_all_rows_applies(
            true,
            true,
            true,
            false,
            &exact_rows,
            &production
        ));
    }

    /// The V1 default-on landing never reaches a step of <= 8 rows: no composition of 1..=8 rows
    /// into once-over-all-rows groups takes the all-rows segment (so no all-rows stage and no
    /// all-rows MoE), whatever the flags say; the one MoE call is open exactly to 9..=48 rows at the
    /// default exact-gfx1201 EP gather capacity and refuses 49 and above (the groups stay).
    #[test]
    fn v1_default_never_touches_steps_of_eight_rows_or_fewer() {
        fn compositions(total: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if total == 0 {
                out.push(prefix.clone());
                return;
            }
            for first in 1..=total.min(LANE_MAX_ROWS) {
                prefix.push(first);
                compositions(total - first, prefix, out);
                prefix.pop();
            }
        }
        let production = LaneStagePolicy::production();
        let cap = crate::gpu_forward::ep_gather_rows_resolved("gfx1201", None, None, false, 32);
        assert_eq!(cap, 48);
        for n in 1..=LANE_MAX_ROWS {
            let mut groups = Vec::new();
            compositions(n, &mut Vec::new(), &mut groups);
            assert!(!groups.is_empty());
            for group_rows in groups {
                let segment =
                    hc_all_rows_applies(true, true, true, false, &group_rows, &production);
                assert!(!segment, "{group_rows:?}");
                let stages = all_rows_stages_apply(true, segment, 0, false, true);
                assert!(!stages, "{group_rows:?}");
                assert!(
                    all_rows_moe_refusal(stages, true, Some(cap), n, false, false, 40).is_some(),
                    "{group_rows:?}"
                );
            }
        }
        // Above 8 rows the one call is open to the capacity and refuses past it.
        for n in LANE_MAX_ROWS + 1..=cap {
            assert_eq!(
                all_rows_moe_refusal(true, true, Some(cap), n, false, false, 40),
                None,
                "{n}"
            );
        }
        assert!(all_rows_moe_refusal(true, true, Some(cap), cap + 1, false, false, 40).is_some());
        // The kill switch (`HIPFIRE_QWEN4_EP_WIDE_VERIFY=0`: 8-row capacity) keeps every group.
        let killed =
            crate::gpu_forward::ep_gather_rows_resolved("gfx1201", Some(false), None, false, 32);
        assert_eq!(killed, LANE_MAX_ROWS);
        for n in LANE_MAX_ROWS + 1..=cap {
            assert!(all_rows_moe_refusal(true, true, Some(killed), n, false, false, 40).is_some());
        }
    }

    /// The facts of a default exact-gfx1201 EP load: every flag on, an eager step, a 48-row
    /// gather, 48 band layers, the production stage shares.
    fn ep_rule() -> LaneAllRowsRule {
        LaneAllRowsRule {
            hc_flag: true,
            stages_flag: true,
            exact_flag: false,
            moe_flag: true,
            gdn_qkv_bf16: false,
            wide_verify_rows: 0,
            eager: true,
            cpu_splice: false,
            bf16_min_rows: 512,
            gather_cap: Some(48),
            trunk_ar_write: false,
            band_moe_calls: 48,
            policy: LaneStagePolicy::production(),
        }
    }

    /// The expectation the rule is tested against, written out from the building blocks with the
    /// executor's own arguments (what `forward_lanes_scoped` called before the rule existed).
    fn blocks_moe_all(
        rule: &LaneAllRowsRule,
        ar_step: bool,
        group_rows: &[usize],
        any_wide: bool,
        streams_agree: bool,
    ) -> (bool, bool, bool, bool) {
        let n: usize = group_rows.iter().sum();
        let hc_all = hc_all_rows_applies(
            rule.hc_flag,
            ar_step,
            rule.eager,
            rule.cpu_splice,
            group_rows,
            &rule.policy,
        );
        let all_stages = all_rows_stages_apply(
            rule.stages_flag,
            hc_all,
            if rule.exact_flag {
                0
            } else {
                rule.wide_verify_rows
            },
            rule.gdn_qkv_bf16,
            streams_agree,
        );
        let moe_all = rule.moe_flag
            && all_rows_moe_refusal(
                all_stages,
                rule.policy.share(StageName::Moe) == StageShare::OnceAllRows,
                rule.gather_cap,
                n,
                any_wide,
                rule.trunk_ar_write,
                rule.band_moe_calls,
            )
            .is_none();
        (hc_all, all_stages, rule.exact_flag && all_stages, moe_all)
    }

    /// The shared predicate IS the executor's decision: `decide` equals the building blocks the
    /// executor used to call inline, over every packing of 1..=8-row lanes up to 8 lanes and the
    /// states that change an input, and `moe_all_rows` is its AR-step `moe_all`.
    #[test]
    fn lane_all_rows_rule_decides_exactly_what_the_building_blocks_decide() {
        let mut rules = vec![ep_rule()];
        for edit in [
            (|r: &mut LaneAllRowsRule| r.moe_flag = false) as fn(&mut LaneAllRowsRule),
            |r| r.stages_flag = false,
            |r| r.hc_flag = false,
            |r| r.exact_flag = true,
            |r| r.gdn_qkv_bf16 = true,
            |r| r.wide_verify_rows = 16,
            |r| r.eager = false,
            |r| r.cpu_splice = true,
            |r| r.gather_cap = None,
            |r| r.gather_cap = Some(32),
            |r| r.trunk_ar_write = true,
            |r| r.band_moe_calls = 47,
        ] {
            let mut rule = ep_rule();
            edit(&mut rule);
            rules.push(rule);
        }
        rules.push(LaneAllRowsRule::never());
        for rule in &rules {
            for lanes in 1..=8usize {
                for rows in 1..=LANE_MAX_ROWS {
                    let lane_rows = vec![rows; lanes];
                    for any_wide in [false, true] {
                        let wide = vec![any_wide; lanes];
                        let groups = crate::mesh_lane_protocol::pack_classed_row_groups_with_cap(
                            &lane_rows, &wide, 32,
                        );
                        let group_rows: Vec<usize> = groups.iter().map(|g| g.rows).collect();
                        for ar in [true, false] {
                            let (hc_all, all_stages, exact_scope, moe_all) =
                                blocks_moe_all(rule, ar, &group_rows, any_wide, true);
                            let got = rule.decide(ar, &group_rows, any_wide, false);
                            let tag = format!("{rule:?} ar={ar} {group_rows:?} wide={any_wide}");
                            assert_eq!(got.hc_all, hc_all, "{tag}");
                            assert_eq!(got.all_stages, all_stages, "{tag}");
                            assert_eq!(got.exact_scope, exact_scope, "{tag}");
                            assert_eq!(got.moe_all, moe_all, "{tag}");
                            // The refusal reason exists exactly for a step that asked for the
                            // call and did not take it.
                            assert_eq!(
                                got.moe_refusal.is_some(),
                                rule.moe_flag && !moe_all,
                                "{tag}"
                            );
                            if ar {
                                assert_eq!(
                                    rule.moe_all_rows(&group_rows, any_wide),
                                    moe_all,
                                    "{tag}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// A row-exact-class lane in the step turns `exact_scope` on exactly where the all-rows stages
    /// apply, and changes nothing else (the root's ledger and planned exchanges read the rest).
    #[test]
    fn lane_all_rows_rule_any_exact_lane_only_sets_exact_scope() {
        for rule in [ep_rule(), LaneAllRowsRule::never()] {
            for groups in [&[8usize, 8][..], &[6; 8], &[4; 4], &[8; 8], &[4]] {
                for any_wide in [false, true] {
                    for ar in [true, false] {
                        let base = rule.decide(ar, groups, any_wide, false);
                        let exact = rule.decide(ar, groups, any_wide, true);
                        let tag = format!("{rule:?} ar={ar} {groups:?} wide={any_wide}");
                        assert_eq!(
                            exact.exact_scope,
                            base.all_stages || base.exact_scope,
                            "{tag}"
                        );
                        assert_eq!(exact.exact_scope, exact.all_stages, "{tag}");
                        assert_eq!(exact.hc_all, base.hc_all, "{tag}");
                        assert_eq!(exact.all_stages, base.all_stages, "{tag}");
                        assert_eq!(exact.moe_all, base.moe_all, "{tag}");
                        assert_eq!(exact.moe_refusal, base.moe_refusal, "{tag}");
                    }
                }
            }
        }
        let rule = ep_rule();
        assert!(rule.decide(true, &[6; 8], false, true).exact_scope);
        assert!(!rule.decide(true, &[6; 8], false, false).exact_scope);
    }

    /// `qwen4_bf16_streams(n) == qwen4_bf16_streams(1)` as a function of `n` and the load's
    /// threshold: where `n` rows would pick another HC stream storage than the singleton's one
    /// row the all-rows stages (and so the all-rows MoE) refuse.
    #[test]
    fn lane_all_rows_rule_refuses_when_the_stream_storage_differs_from_one_row() {
        let groups = [6usize; 8];
        let mut rule = ep_rule();
        assert!(rule.moe_all_rows(&groups, false));
        // A threshold at or below 48 rows (48 rows stream BF16, one row does not).
        rule.bf16_min_rows = 48;
        assert!(!rule.moe_all_rows(&groups, false));
        assert!(rule.decide(true, &groups, false, false).hc_all);
        assert!(!rule.decide(true, &groups, false, false).all_stages);
        // Above 48 rows both pick the F32 storage.
        rule.bf16_min_rows = 49;
        assert!(rule.moe_all_rows(&groups, false));
        // A load that never streams BF16 (`usize::MAX`) always agrees.
        rule.bf16_min_rows = usize::MAX;
        assert!(rule.moe_all_rows(&groups, false));
    }

    /// The rule of every non-EP root never plans the all-rows MoE; the EP rule opens exactly the
    /// rows the executor's refusal opens (9..=48 rows of several groups at a 48-row gather) and
    /// is the shared predicate for AR steps only.
    #[test]
    fn lane_all_rows_rule_never_and_the_ep_row_window() {
        let never = LaneAllRowsRule::never();
        for groups in [&[8usize, 8][..], &[6; 8], &[4; 4], &[8; 8]] {
            assert!(!never.moe_all_rows(groups, false), "{groups:?}");
            assert!(never
                .decide(true, groups, false, false)
                .moe_refusal
                .is_none());
        }
        let rule = ep_rule();
        // C1: one lane of four rows (one group); two groups of eight; C8 48 rows.
        assert!(!rule.moe_all_rows(&[4], false));
        assert!(rule.moe_all_rows(&[8, 8], false));
        assert!(rule.moe_all_rows(&[6; 8], false));
        // 64 rows exceed the 48-row gather capacity: the groups stay, with a reason.
        let refused = rule.decide(true, &[8; 8], false, false);
        assert!(!refused.moe_all);
        assert!(refused.moe_refusal.unwrap().contains("gather capacity"));
        // A wide-class lane keeps its group.
        assert!(!rule.moe_all_rows(&[6; 8], true));
        // A prefill wave / band edge (not an AR step) never takes the segment.
        assert!(!rule.decide(false, &[6; 8], false, false).moe_all);
    }
}
