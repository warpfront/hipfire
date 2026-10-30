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
//!   prologue/select/attend) always run per lane, on the lane's state, with
//!   the lane's own row count.
//! - **Row-local stages** run either per lane ([`StageShare::PerLane`], the
//!   singleton's arm by construction) or once over every lane's rows
//!   ([`StageShare::OnceAllRows`], in groups of at most [`LANE_MAX_ROWS`]).
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
use crate::gpu_forward::{
    bind_ep_lane_gather, ep_lane, ep_lane_buffer, ep_lane_check_invocations, ep_lane_moe_rows,
    lane_layer_moe,
};
use crate::gpu_forward::{
    decode_copy, dense_ref, hyper_read_desc, invalid, layer_desc, matrix_view, ple_desc,
    program_dims, qwen4_ple_rows_f16, router_columns, view, Qwen4GpuForward, Qwen4GpuForwardError,
    Qwen4GpuForwardScratch, ScratchField, DECODE_Q8_PER_LAYER, EPSILON,
};
use crate::ple_stage::ple_async_upload;
use crate::program::{
    execute_final_hyper, execute_lm_head, seal_moe_for_layer, validate_final_hyper,
    validate_lm_head, Qwen4AttentionWeights, Qwen4FfnBinding, Qwen4LayerDescription,
    Qwen4LayerScratch, Qwen4ProgramDims,
};
use crate::state::Qwen4State;
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::{
    execute_embedding, execute_validated_steps, validate_steps, ClearOp, EmbeddingOp,
    GatedDeltaNetOp, GroupedDepthwiseOp, HyperReadOp, HyperWriteOp, IndexedAttentionMode,
    IndexedAttentionOp, IndexedAttentionState, MixerPhase, Step,
};
use hipfire_runtime::external_rows::RowFetch;
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
}

impl StageName {
    pub const ALL: [StageName; 15] = [
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
    /// [`LANE_MAX_ROWS`] rows. Admitted only with G0 byte evidence.
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
pub const STAGE_SHARE: [(StageName, StageShare); 15] = [
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
    HostMapped,
    CpuDeferred,
    Split,
    /// Expert-parallel packed QT44/QT53 routed experts: every owned MoE layer
    /// is either a fully resident static gathered decode route or a wholly
    /// host-mapped balanced-live gathered runtime route, gather installed, no
    /// CPU-deferred, tier, slab, D1/D2 or pruned experts. Any per-layer mix
    /// (all resident, all mapped, or both) is the class; the measured EP2
    /// scope carried both kinds. Set only from bound EP placement and the
    /// installed gather, never from arch, env or layer counts.
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
pub type LaneStageRoutes = [Vec<StageRoute>; 15];

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

/// Row-twin proofs, one row per shared stage per arch and topology domain
/// (gfx1201 carries a PP-only domain and a disjoint EP-only domain whose MoE
/// row is the packed static-or-balanced expert class). A stage is
/// shared only when every instance a lane owns matches a row of its arch,
/// stage and topology; any other (arch, topology, route) keeps that stage per
/// lane until its own G0 run (other cards bind other kernels, e.g. no Q8_0
/// decode copies on gfx1201).
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
    proof(
        "gfx1201",
        StageName::FinalHyper,
        StageRoute::HyperRead([DType::BF16; 3]),
        LANE_EP_ONLY,
        GFX1201_EP2_MQ4_RECEIPT,
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
/// lane.
fn resolve_with_proofs(
    arch: &str,
    topology: LaneTopology,
    routes: &LaneStageRoutes,
    proofs: &[RowTwinProof],
) -> LanePolicyResolution {
    let mut policy = LaneStagePolicy::all_per_lane();
    let mut receipts: Vec<&'static str> = Vec::new();
    let mut per_lane: Vec<(StageName, String)> = Vec::new();
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
                for receipt in stage_receipts {
                    if !receipts.contains(&receipt) {
                        receipts.push(receipt);
                    }
                }
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
    shares: [StageShare; 15],
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
        let mut shares = [StageShare::PerLane; 15];
        for (stage, share) in STAGE_SHARE {
            shares[stage.index()] = share;
        }
        Self { shares }
    }

    pub fn all_per_lane() -> Self {
        Self {
            shares: [StageShare::PerLane; 15],
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
/// `total_rows * vocab` ([`WaveHead::All`]) logits, and a Raw i32 per output
/// row for the optional top-1.
pub(crate) struct WaveOutputs<'a> {
    head: WaveHead,
    logits: &'a GpuTensor,
    top1: Option<&'a GpuTensor>,
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
    /// The lane step's wide-hidden capture (verify lanes); a prefill wave and
    /// a non-output band have none.
    fn wide_hidden(&self) -> Option<&'a GpuTensor> {
        match self {
            Self::Ar(out) => out.wide_hidden,
            Self::Wave(_) | Self::Edge { .. } => None,
        }
    }

    /// Whether the step is a prefill wave.
    fn prefill(&self) -> bool {
        matches!(self, Self::Wave(_) | Self::Edge { prefill: true })
    }
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
/// row counts runs ([`StageShare::OnceAllRows`] stages launch once per
/// group).
pub(crate) fn row_group_count(rows: &[usize]) -> usize {
    group_segments(rows, LANE_MAX_ROWS).len()
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
pub(crate) fn forward_wave(
    gpu: &mut Gpu,
    bundle: &mut Qwen4Bundle,
    lanes: &mut [WaveLane<'_>],
    head: WaveHead,
    logits: &GpuTensor,
    top1: Option<&GpuTensor>,
    policy: &LaneStagePolicy,
) -> Result<(), Qwen4GpuForwardError> {
    // Only the output band has a head (see `forward_lanes`).
    let outputs = if bundle.weights.band_scope().output {
        StepOutputs::Wave(WaveOutputs { head, logits, top1 })
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
        })
        .collect();
    let scope = std::mem::replace(&mut gpu.qwen4_scope, true);
    let invariant =
        rdna_compute::prefill_invariant::prefill_pm_invariant_enabled() && gpu.arch == "gfx1151";
    let invariant = std::mem::replace(&mut gpu.prefill_invariant_scope, invariant);
    let result = forward_lanes_scoped(gpu, &mut forward, bundle, &mut rows, out, policy);
    gpu.prefill_invariant_scope = invariant;
    gpu.qwen4_scope = scope;
    bundle.execution = Some(forward);
    result
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
    // Expert parallelism: every MoE invocation runs the singleton's gathered
    // forward over the rank's replicated trunk (the row gather must already be
    // installed; checked before any launch).
    let ep_lane_step = ep_lane(fwd.ep.as_ref(), &fwd.scratch)?.is_some();
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
    policy.validate().map_err(invalid)?;
    if lanes.is_empty() {
        return Err(invalid("a lane step needs at least one lane"));
    }
    // Lane rows mix requests: the router fault row is the row inside the step.
    gpu.set_router_row_base(0);
    let config = &bundle.config;
    let dims = program_dims(config);
    let vocab = config.vocab_size;
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
    fwd.scratch.note_forward_rows(n);
    match out {
        StepOutputs::Ar(out) => {
            if out.logits.dtype != DType::F32 || out.logits.numel() < n * vocab {
                return Err(invalid(
                    "lane logits must be F32 with total_rows * vocab elements",
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
        group_segments(
            &rows,
            if cpu_splice {
                crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
            } else {
                LANE_MAX_ROWS
            },
        )
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
        // The singleton's one-row stream storage; every segment here is at
        // most LANE_MAX_ROWS rows and must agree with it.
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
    let all_segs: Vec<Segment> = lane_segs.iter().chain(&group_segs).copied().collect();
    // The head stages (final hyper, LM head, top-1) run on the output band only.
    // A prefill wave's head runs per lane after the trunk (its own views), so
    // the policy's Head / Argmax segments exist only for AR steps.
    let head_segs = |share: StageShare| {
        if band_scope.output {
            segs_for(share)
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
    let seg_desc_rows: Vec<usize> = if ep_lane_step {
        for i in segs_for(policy.share(StageName::Moe)) {
            let seg = &all_segs[i];
            let rr = if prefill {
                gpu.route_rows(seg.rows)
            } else {
                seg.rows
            };
            ep_lane_moe_rows(seg.rows, rr)?;
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
    // upload: no later singleton forward may claim them.
    fwd.uploaded_id_rows = 0;
    if prefill {
        // Like `lane_prefill_tile`: no lookahead names this wave's next rows,
        // and (as the singleton does for every multi-row forward) no retained
        // replay is eligible.
        fwd.set_ple_lookahead(&[]);
        gpu.replay.set_forward_eligible(false);
    }

    let attempt = (|| -> Result<Vec<QsaCommit>, Qwen4GpuForwardError> {
        let scratch = &fwd.scratch;
        let mut views = Vec::with_capacity(all_segs.len());
        for seg in &all_segs {
            views.push(SegViews::new(
                scratch,
                dims,
                vocab,
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
        // Under EP: the row gather, the two trunk gather buffers and, per
        // segment, the scratch description whose expanded-row scratch is each
        // buffer (validated before any launch).
        let ep_lane = ep_lane(fwd.ep.as_ref(), scratch)?;
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
            let mut descs: Vec<Qwen4LayerDescription<'_>> =
                Vec::with_capacity(config.num_hidden_layers);
            for layer_index in band_scope.band.clone() {
                descs.push(layer_desc(
                    &bundle.weights,
                    bundle
                        .weights
                        .layer(layer_index)
                        .ok_or_else(|| invalid(format!("lane layer {layer_index} is not owned")))?,
                    lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer_index),
                    config,
                    &fwd.decode_q8,
                    rows,
                    gpu.few_row_limit(),
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
        let head = if band_scope.output {
            Some((
                hyper_read_desc(&bundle.weights, &bundle.weights.root.final_hyper)?,
                dense_ref(&bundle.weights, &bundle.weights.root.lm_head)?,
            ))
        } else {
            None
        };
        let head_ref = || {
            head.as_ref()
                .ok_or_else(|| invalid("a lane head stage needs the output band"))
        };
        let states: Vec<&Qwen4State> = lanes.iter().map(|lane| &*lane.state).collect();

        // Preflight the head of every segment before any launch.
        for i in head_segs(policy.share(StageName::FinalHyper)) {
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

        let ctx = DispatchCtx::new(gpu);
        let band_first = band_scope.band.start;
        let program = layer_program(bundle, policy, band_scope.input.then_some(ple_layer))?;
        let mut steps: Vec<Step<'_>> = Vec::with_capacity(program.len() * lanes.len() * 2);
        let mut qsa_commits: Vec<QsaCommit> = Vec::new();
        let mut ple_split: Option<usize> = None;
        // MoE invocations emitted so far under EP, in emission (= execution)
        // order over the whole step: the gather buffer alternates per
        // invocation, whatever the lane/layer interleaving.
        let mut ep_invocations = 0usize;
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
            for i in segs_for(share) {
                let (seg, v) = (&all_segs[i], &views[i]);
                let r = seg.rows;
                let rr = if prefill { gpu.route_rows(r) } else { r };
                let descs = &desc_sets[seg_desc_set[i]];
                for &(op, _) in &program[start..end] {
                    match op {
                        LaneOp::HcRead { layer, mlp } => {
                            let read = if mlp {
                                &descs[layer - band_first].mlp_hyper.read
                            } else {
                                &descs[layer - band_first].attn_hyper.read
                            };
                            let copy = layer * DECODE_Q8_PER_LAYER + if mlp { 2 } else { 0 };
                            steps.push(Step::HyperRead(HyperReadOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: &v.streams,
                                norm_weight: read.norm,
                                input_mix_down: decode_copy(
                                    &fwd.decode_q8,
                                    copy,
                                    rr,
                                    gpu.few_row_limit(),
                                    read.input_mix_down,
                                ),
                                input_mix_up: decode_copy(
                                    &fwd.decode_q8,
                                    copy + 1,
                                    rr,
                                    gpu.few_row_limit(),
                                    read.input_mix_up,
                                ),
                                normalized: &v.hc_normalized,
                                low: &v.hc_low,
                                up: &v.hc_up,
                                mixed: &v.hc_mixed,
                                rows: r,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                                low_rank: dims.hc_lowrank,
                            }));
                        }
                        LaneOp::HcWrite { layer, mlp } => {
                            let write = if mlp {
                                &descs[layer - band_first].mlp_hyper.write
                            } else {
                                &descs[layer - band_first].attn_hyper.write
                            };
                            steps.push(Step::HyperWrite(HyperWriteOp {
                                rotation: &scratch.rotation,
                                state_bf16: bf16_state,
                                input: &v.streams,
                                norm_weight: write.norm,
                                block_inject: write.block_inject,
                                normalized: &v.hc_normalized,
                                mixed: if mlp {
                                    &v.ffn_output
                                } else {
                                    &v.attention_output
                                },
                                gates: &v.hc_gates,
                                output: &v.streams,
                                rows: r,
                                branches: dims.hc_count,
                                hidden: config.hidden_size,
                            }));
                        }
                        LaneOp::PleConv => {
                            let ple_w = ple_w
                                .as_ref()
                                .ok_or_else(|| invalid("PLE weights missing"))?;
                            let state = states[seg.first_lane];
                            if ple_split.is_none() {
                                ple_split = Some(steps.len());
                            }
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
                            }));
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
                                projection: &v.projection,
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
                                row_capture: if stateful {
                                    state.gdn_row_capture(slot, r)
                                } else {
                                    None
                                },
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
                            // gathered-forward selection, see
                            // `lane_layer_moe`).
                            if let Some(runtime) = lane_layer_moe(fwd.ep.as_ref(), &fwd.moe, layer)
                            {
                                moe.routed_experts = runtime.routed_view(!prefill, r, rr, false, 8);
                            }
                            // Under EP the invocation runs the singleton's
                            // gathered forward: the row gather, the few-row
                            // shared-down output, and the next buffer of the
                            // step's alternation as the expanded-row scratch.
                            let seal_scratch = match (&ep_lane, &ep_seg_scratch) {
                                (Some(ep), Some(per_seg)) => {
                                    ep_lane_moe_rows(r, rr)?;
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
                            steps.push(Step::Clear(ClearOp {
                                tensor: &v.ffn_output,
                                elements: r * config.hidden_size,
                            }));
                            steps.push(Step::Moe(sealed));
                        }
                    }
                }
            }
            start = end;
        }
        if ep_lane.is_some() {
            ep_lane_check_invocations(ep_invocations)?;
        }
        validate_steps(gpu, &steps).map_err(|error| {
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
            for i in segs_for(embed_share) {
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
            for i in segs_for(embed_share) {
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

        let execute = |gpu: &mut Gpu, steps: &[Step<'_>]| {
            execute_validated_steps(gpu, &ctx, steps).map_err(|error| {
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
                segs_for(rows_share)
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
        for i in head_segs(final_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (final_hyper, _) = head_ref()?;
            execute_final_hyper(
                gpu,
                dims,
                &final_hyper,
                &v.streams,
                &seg_scratch[i],
                seg.rows,
                bf16_state,
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane final hyper: {error:?}"))
            })?;
        }
        let head_share = policy.share(StageName::Head);
        for i in ar_segs(head_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            let (_, lm_head) = head_ref()?;
            let hidden = view(&v.hc_mixed, 0, seg.rows * config.hidden_size);
            let rotation = view(&scratch.rotation, 0, seg.rows * config.hidden_size);
            execute_lm_head(
                gpu,
                &lm_head,
                &hidden,
                &view(&v.logits, 0, seg.rows * vocab),
                seg.rows,
                seg.rows,
                Some(&rotation),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("execute lane LM head: {error:?}"))
            })?;
        }
        let argmax_share = policy.share(StageName::Argmax);
        for i in ar_segs(argmax_share) {
            let (seg, v) = (&all_segs[i], &views[i]);
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: &view(&v.logits, 0, seg.rows * vocab),
                    indices: &v.top1,
                    rows: seg.rows,
                    vocab,
                },
            )?;
        }
        // Prefill wave head: per lane, the singleton's own LM-head / argmax
        // forms on the lane's rows (Final: `requested_rows == 1`).
        for (lane, (head_logits, head_top1, requested)) in wave_head.iter().enumerate() {
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

    match attempt {
        Ok(qsa_commits) => {
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
        assert_eq!(shared.len(), 12);
        // gfx1151: one all-topology row per shared stage. gfx1201: one PP row
        // and one EP row per shared stage, plus the second PP MoE row (the
        // mq3 slab class).
        assert_eq!(LANE_ROW_TWIN_PROOFS.len(), 3 * shared.len() + 1);
        for arch in ["gfx1151", "gfx1201"] {
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
                        ("gfx1151", _) => shared.contains(&stage),
                        ("gfx1201", LaneTopology::Pp) => shared.contains(&stage),
                        ("gfx1201", LaneTopology::Ep) => shared.contains(&stage),
                        _ => false,
                    };
                    // The PP MoE stage carries the mq2 and the mq3 slab rows.
                    let count = usize::from(expected)
                        + usize::from(
                            expected
                                && arch == "gfx1201"
                                && topology == LaneTopology::Pp
                                && stage == StageName::Moe,
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
                                assert_eq!(proof.domain, [LaneTopology::Ep]);
                                assert_eq!(proof.receipt, GFX1201_EP2_MQ4_RECEIPT);
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
                assert_eq!(
                    resolution.policy,
                    LaneStagePolicy::production(),
                    "{topology:?}"
                );
                assert!(fallback_stages(&resolution).is_empty(), "{topology:?}");
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
            assert_eq!(rows.len(), 1, "{}", stage.key());
            assert!(
                routes[stage.index()].iter().all(|route| *route == rows[0]),
                "{}",
                stage.key()
            );
        }
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
        assert_eq!(from_table.receipts, [GFX1201_EP2_MQ4_RECEIPT]);
        let empty = LaneStagePolicy::for_bound_routes(
            "gfx1201",
            LaneTopology::Ep,
            &LaneStageRoutes::default(),
        );
        assert_eq!(empty.policy, LaneStagePolicy::production());
        assert!(empty.receipts.is_empty());
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
        for stage in shared_stages().into_iter().filter(|s| *s != StageName::Moe) {
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
    fn unknown_arch_runs_every_stage_per_lane() {
        for arch in ["gfx1100", "gfx942", ""] {
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
                            LaneOp::PleConv => *op,
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
}
