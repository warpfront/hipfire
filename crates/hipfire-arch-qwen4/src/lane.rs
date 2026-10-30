// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact lane store of the Flash-Next batch route.
//!
//! A lane is one admitted request with its own [`Qwen4State`]; nothing is
//! shared between lanes but the weights. A store staged with a
//! [`LaneSpecConfig`] makes every lane an MTP lane: it also owns the lane's
//! MTP head state, draft request state, pending hidden, GDN capture ring and
//! window policy ([`LaneSpec`]), and the store shares the step buffers MTP
//! windows need (wide-hidden capture, the batched draft's top-1 slots, the
//! prompt-fill append scratch, the step cost model). [`Qwen4LaneStore`] owns
//! the lanes and the step output buffers; [`crate::lane_exec::Qwen4LaneExecutor`]
//! runs plans over it. The host-side lane table ([`LaneTable`]) is generic
//! over its device payload so its admission/release/poison rules are
//! covered by CPU tests.
//!
//! Lane device states are allocated on a slot's first admission and kept
//! across releases: a later admission into that slot resets the state in
//! place instead of allocating again.
//!
//! A store staged `sampled` (`serve.batch_sampled`) also admits sampled
//! requests: [`LaneSampling`] set on a lane after admission makes every token
//! the lane commits one host draw of the AR singleton's sampler
//! (`llama::sample_top_k_p` with neutral penalties) from the lane's own F32
//! logits row, against the lane's own xorshift32 state (the stream
//! `reset_cpu_sampler_rng(seed)` starts), in emission order. An MTP lane
//! verifies a sampled window with the naive rule (drafts stay the head's
//! argmax; each verify row is one AR draw; a draft is accepted iff it equals
//! its row's draw), so every emitted token is exactly one AR draw. Lanes
//! without a [`LaneSampling`] run the greedy device-argmax path unchanged
//! (no extra D2H, no extra host work).
//!
//! Under `q8_ef` each lane's [`Qwen4State`] also owns its live F16 GDN
//! residual companions (`GdnGpuState::ef_residual`): allocated and freed
//! with the lane state, zeroed by the state's own reset/restore/rollback
//! (so slot reuse in `admit` and a window's partial rollback reset the
//! lane's residual, and an all-accept commit keeps it), never shared across
//! lanes and never part of the capture ring or any snapshot. Lane forwards
//! pass each lane's own residual and the state's actual `gdn_format()` to
//! the GDN step.

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config};
use crate::expert_arena::{ExpertArenaLayer, ExpertArenaPlan};
use crate::expert_slabs::ExpertFormat;
use crate::gpu_forward::{EpLaneExpertKind, Qwen4GpuForward, DECODE_Q8_PER_LAYER};
use crate::gpu_forward_lanes::{
    LaneExpertRoute, LaneStagePolicy, LaneStageRoutes, LaneTopology, StageName, StageRoute,
    LANE_MAX_ROWS,
};
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::lane_exec::LanePhase;
use crate::lane_rank::{DecodeAdmission, LaneProto};
use crate::mesh_rank_copy::HeadLagStore;
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuState, MTP_FILL_ROWS};
use crate::mtp_lanes::PendingWindow;
use crate::mtp_spec::{
    mtp_fill_scratch_requested, mtp_fill_scratch_wanted, native_mtp_row_capture,
    Qwen4PromotionParts,
};
use crate::state::{Qwen4State, Qwen4StateFormat};
use crate::weights::{
    HyperConnectionWeights, MoeWeights, Qwen4BandScope, Qwen4FfnWeights, Qwen4LayerWeights,
    Qwen4TierRecipe, Qwen4Weights, TensorRef,
};
use hipfire_dispatch::pipeline::DraftHeadRequestState;
use hipfire_runtime::ngram_mod::NgramModConfig;
use hipfire_runtime::slot_batch::RequestEpoch;
use rdna_compute::expert_capabilities::{self, ExpertQt, Route};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

pub mod census;
pub(crate) mod groups;
use groups::GroupPlanner;

/// Most lanes one store holds.
pub(crate) const MAX_LANES: usize = 64;

/// Topology of THIS rank's bundle. Expert parallelism is read from the
/// rank's own weights (peers hold no root `bundle.ep`); a band that is not the
/// whole model, or that carries pipeline metadata, is pipeline parallel.
fn lane_topology(bundle: &Qwen4Bundle) -> LaneTopology {
    if bundle.weights.ep_placement().is_some() {
        LaneTopology::Ep
    } else if *bundle.weights.band_scope() != Qwen4BandScope::single(&bundle.config)
        || bundle.pp_band.is_some()
    {
        LaneTopology::Pp
    } else {
        LaneTopology::Single
    }
}

/// Expert route of THIS owned band, from the executing forward and the
/// assembled plan metadata only. Anything not positively recognised is
/// [`LaneExpertRoute::Unclassified`], which no stage proof admits.
fn lane_expert_route(bundle: &Qwen4Bundle) -> LaneExpertRoute {
    if bundle.execution.is_none() {
        LaneExpertRoute::Unclassified
    } else if !bundle.split_layers().is_empty() {
        LaneExpertRoute::Split
    } else if bundle.has_cpu_deferred_experts() {
        LaneExpertRoute::CpuDeferred
    } else if bundle.weights.expert_tiers().is_some() {
        LaneExpertRoute::Unclassified
    } else if bundle.weights.ep_placement().is_some() {
        classify_ep_owned_experts(bundle)
    } else {
        classify_owned_experts(&bundle.weights, &bundle.config)
    }
}

/// What one owned layer's experts were observed to be.
struct OwnedLayerExperts {
    host_mapped: bool,
    /// E8P experts of a slab arena (0 for a packed layer).
    e8p_experts: usize,
    /// MQ3-I4 (W3) experts of a slab arena (0 for a packed layer).
    mq3_experts: usize,
}

/// Classify every layer of the owned band. Any host-mapped expert storage
/// observed is [`LaneExpertRoute::HostMapped`]; any other layer that is not
/// positively recognised makes the whole band `Unclassified`.
fn classify_owned_experts(weights: &Qwen4Weights, config: &Qwen4Config) -> LaneExpertRoute {
    let band = weights.band_scope().band.clone();
    if band.is_empty() {
        return LaneExpertRoute::Unclassified;
    }
    let arenas = match &weights.manifest.identity.recipe {
        Qwen4TierRecipe::FlashNextSlab { arenas, .. } => Some(arenas.as_ref()),
        Qwen4TierRecipe::FlashNextPacked { .. } => None,
        _ => return LaneExpertRoute::Unclassified,
    };
    let (mut host_mapped, mut refused, mut e8p_experts, mut mq3_experts) =
        (false, false, 0usize, 0usize);
    for absolute in band {
        match classify_owned_layer(weights, config, arenas, absolute) {
            Ok(layer) => {
                host_mapped |= layer.host_mapped;
                e8p_experts += layer.e8p_experts;
                mq3_experts += layer.mq3_experts;
            }
            Err(_) => refused = true,
        }
    }
    if host_mapped {
        LaneExpertRoute::HostMapped
    } else if refused {
        LaneExpertRoute::Unclassified
    } else if arenas.is_none() {
        LaneExpertRoute::ResidentPacked
    } else if e8p_experts > 0 && mq3_experts == 0 {
        // The strict mq2 family: MQ4-XTS plus E8P experts, at least one E8P.
        LaneExpertRoute::ResidentMq2Slab
    } else if mq3_experts > 0 && e8p_experts == 0 {
        // The strict mq3 family: MQ4-XTS plus MQ3-I4 experts, at least one MQ3-I4.
        LaneExpertRoute::ResidentMq3Slab
    } else {
        LaneExpertRoute::Unclassified
    }
}

/// One owned EP layer's observed expert binding (see
/// [`classify_ep_owned_experts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EpOwnedLayer {
    /// The pair the two resident aggregates' actual dtypes name.
    qt: Option<ExpertQt>,
    gate_up_mapped: bool,
    down_mapped: bool,
    /// The forward's own kind of this layer
    /// ([`crate::gpu_forward::Qwen4GpuForward::lane_ep_expert_kind`]).
    kind: Option<EpLaneExpertKind>,
}

/// Every owned layer positively one valid kind: QT44/QT53, both aggregates
/// resident (the static runtime) or both host-mapped (the live balanced
/// runtime), and the forward's kind agreeing with that residence. Any mix of
/// the two kinds across layers qualifies; there is no layer, expert or rank
/// count in the rule. An empty band owns nothing to classify.
fn classify_ep_layers(layers: impl IntoIterator<Item = Option<EpOwnedLayer>>) -> LaneExpertRoute {
    let mut owned = false;
    for layer in layers {
        let valid = layer.is_some_and(|layer| {
            layer.qt == Some(expert_capabilities::QT_MQ4_XTS)
                && layer.gate_up_mapped == layer.down_mapped
                && matches!(
                    (layer.kind, layer.gate_up_mapped),
                    (Some(EpLaneExpertKind::StaticResidentPacked), false)
                        | (Some(EpLaneExpertKind::BalancedHostMappedPacked), true)
                )
        });
        if !valid {
            return LaneExpertRoute::Unclassified;
        }
        owned = true;
    }
    if owned {
        LaneExpertRoute::EpPackedStaticOrBalanced
    } else {
        LaneExpertRoute::Unclassified
    }
}

/// EP expert route of THIS rank's whole-model band, from its actual forward
/// and bound weights: the forward's EP placement is the weights', the recipe
/// is Flash-Next packed (no slab arena, tier, pipeline band or other recipe)
/// and every owned layer is a MoE layer of one valid kind
/// ([`classify_ep_layers`]). Anything else is `Unclassified`.
fn classify_ep_owned_experts(bundle: &Qwen4Bundle) -> LaneExpertRoute {
    let weights = &bundle.weights;
    let (Some(forward), Some(placement)) = (bundle.execution.as_ref(), weights.ep_placement())
    else {
        return LaneExpertRoute::Unclassified;
    };
    if forward.ep_placement() != Some(placement)
        || !matches!(
            weights.manifest.identity.recipe,
            Qwen4TierRecipe::FlashNextPacked { .. }
        )
        || *weights.band_scope() != Qwen4BandScope::single(&bundle.config)
        || bundle.pp_band.is_some()
        || weights.expert_tiers().is_some()
    {
        return LaneExpertRoute::Unclassified;
    }
    classify_ep_layers(
        weights
            .band_scope()
            .band
            .clone()
            .map(|absolute| ep_owned_layer(weights, forward, absolute)),
    )
}

/// What the bound weights and the forward say of one owned EP layer; `None`
/// when it is not an assembled arena-free MoE layer with resolvable aggregates.
fn ep_owned_layer(
    weights: &Qwen4Weights,
    forward: &Qwen4GpuForward,
    absolute: usize,
) -> Option<EpOwnedLayer> {
    let moe = weights.layer(absolute)?.ffn.moe()?;
    if moe.expert_arena.is_some() {
        return None;
    }
    let gate_up = weights.resident(&moe.experts_gate_up).ok()?;
    let down = weights.resident(&moe.experts_down).ok()?;
    Some(EpOwnedLayer {
        qt: ExpertQt::of_dtypes(gate_up.dtype, down.dtype),
        gate_up_mapped: gate_up.buf.is_host_mapped(),
        down_mapped: down.buf.is_host_mapped(),
        kind: forward.lane_ep_expert_kind(absolute),
    })
}

/// One owned layer under the recipe's route: a slab layer holds its single
/// arena and per-expert plan, a packed layer its two resident aggregates.
fn classify_owned_layer(
    weights: &Qwen4Weights,
    config: &Qwen4Config,
    arenas: Option<&ExpertArenaPlan>,
    absolute: usize,
) -> Result<OwnedLayerExperts, String> {
    let layer = weights
        .layer(absolute)
        .ok_or_else(|| format!("layer {absolute} is not assembled on this rank"))?;
    let moe = layer
        .ffn
        .moe()
        .ok_or_else(|| format!("layer {absolute} is not a MoE layer"))?;
    let resident = |reference| {
        weights
            .resident(reference)
            .map_err(|error| format!("layer {absolute}: {error}"))
    };
    match (arenas, moe.expert_arena.as_ref()) {
        (Some(plan), Some(arena_ref)) => {
            if resident(arena_ref)?.buf.is_host_mapped() {
                return Ok(OwnedLayerExperts {
                    host_mapped: true,
                    e8p_experts: 0,
                    mq3_experts: 0,
                });
            }
            let planned = plan
                .layer(absolute)
                .ok_or_else(|| format!("layer {absolute}: the arena plan has no such layer"))?;
            let (e8p_experts, mq3_experts) =
                slab_layer_cold_experts(planned, config.num_experts)
                    .map_err(|error| format!("layer {absolute}: {error}"))?;
            Ok(OwnedLayerExperts {
                host_mapped: false,
                e8p_experts,
                mq3_experts,
            })
        }
        (None, None) => {
            let gate_up = resident(&moe.experts_gate_up)?;
            let down = resident(&moe.experts_down)?;
            let host_mapped = gate_up.buf.is_host_mapped() || down.buf.is_host_mapped();
            if !host_mapped && (gate_up.dtype != DType::MQ4G256V2 || down.dtype != DType::MQ4G128V2)
            {
                return Err(format!(
                    "layer {absolute}: packed experts are {:?}/{:?}, not MQ4G256V2/MQ4G128V2",
                    gate_up.dtype, down.dtype
                ));
            }
            Ok(OwnedLayerExperts {
                host_mapped,
                e8p_experts: 0,
                mq3_experts: 0,
            })
        }
        _ => Err(format!(
            "layer {absolute}: the recipe and the layer disagree on an expert arena"
        )),
    }
}

/// The `(E8P, MQ3-I4)` expert counts of a slab layer whose plan is exactly
/// `num_experts` present MQ4-XTS experts plus either E8P or MQ3-I4 experts
/// (never both in one layer). An absent (pruned) expert, a W2 expert, an
/// E8P/W3 mix, or any other expert count is refused: the lane proofs cover
/// complete layers of MQ4-XTS with one cold format only.
fn slab_layer_cold_experts(
    layer: &ExpertArenaLayer,
    num_experts: usize,
) -> Result<(usize, usize), String> {
    if num_experts == 0 || layer.experts.len() != num_experts {
        return Err(format!(
            "the arena plans {} experts, the model has {num_experts}",
            layer.experts.len()
        ));
    }
    let (mut e8p, mut mq3) = (0usize, 0usize);
    for (expert, entry) in layer.experts.iter().enumerate() {
        if entry.is_absent() {
            return Err(format!("expert {expert} is absent"));
        }
        match entry.format {
            ExpertFormat::Mq4Xts => {}
            ExpertFormat::E8pI4 => e8p += 1,
            ExpertFormat::Mq3I4 => mq3 += 1,
            other => {
                return Err(format!(
                    "expert {expert} is {other:?}, not MQ4-XTS, MQ3-I4 or E8P-I4"
                ))
            }
        }
    }
    if e8p > 0 && mq3 > 0 {
        return Err("the layer mixes E8P-I4 and MQ3-I4 experts".to_string());
    }
    Ok((e8p, mq3))
}

/// Dtype one projection or tensor binds in the lane forward; `None` when its
/// reference or resident weight could not be resolved.
type Bound = Option<DType>;

/// The bound dtypes of one hyper-connection read: norm, down, up.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HyperFacts {
    norm: Bound,
    down: Bound,
    up: Bound,
}

impl HyperFacts {
    const UNRESOLVED: Self = Self {
        norm: None,
        down: None,
        up: None,
    };
}

/// One layer's token mixer: the input projections in lane order (GDN: qkv, z,
/// in_proj_a, in_proj_b; QSA: indexer_qk, q, k, v) and the output projection.
#[derive(Clone, Debug, PartialEq, Eq)]
enum MixerFacts {
    Gdn { input: [Bound; 4], output: Bound },
    Qsa { input: [Bound; 4], output: Bound },
}

/// One owned MoE layer's bound dense dtypes (router, shared gate, shared up,
/// shared down, shared-gate scalar) and its routed experts' exact quant pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MoeFacts {
    dense: [Bound; 5],
    /// Each distinct quant pair of the experts present on the layer (an absent,
    /// pruned expert names none); `None` when the layer's expert storage is not
    /// positively known (tier layout, an EP band that is not positively the
    /// packed static/balanced class, other recipe, unreadable plan).
    expert_qts: Option<Vec<ExpertQt>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FfnFacts {
    Moe(MoeFacts),
    Dense,
    /// The layer or its FFN weights could not be resolved.
    Missing,
}

/// Everything the lane forward binds for one owned layer.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LayerFacts {
    attn_hyper: HyperFacts,
    mlp_hyper: HyperFacts,
    /// `block_inject` dtype of the attention and MLP hyper writes.
    attn_inject: Bound,
    mlp_inject: Bound,
    /// `None`: the layer kind is unknown, so no mixer stage can be named.
    mixer: Option<MixerFacts>,
    ffn: FfnFacts,
}

/// What the PLE row staging of the input band delivers to the lane step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagedRows {
    /// The model has no PLE: the stage does not exist.
    Absent,
    /// The model has PLE but the rows provider is not bound.
    Unresolved,
    Known(DType),
}

/// Plain-data facts of every stage instance one owned band executes, gathered
/// from the bound weights and the executing forward
/// ([`gather_stage_facts`]) and turned into routes by
/// [`derive_stage_routes`], which needs no device.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StageFacts {
    topology: LaneTopology,
    /// The band embeds tokens and stages PLE rows.
    input: bool,
    /// The band runs the final hyper read, the LM head and the top-1.
    output: bool,
    embedding: Bound,
    ple_rows: StagedRows,
    /// Whole-band expert class ([`lane_expert_route`]); only the MoE stage
    /// falls back to it.
    experts: LaneExpertRoute,
    layers: Vec<LayerFacts>,
    final_hyper: HyperFacts,
    head: Bound,
}

/// Quant pair of the experts present on one slab layer, each distinct pair
/// once, in format order. Four presence flags: no per-expert allocation or
/// repeated capability lookups.
fn slab_layer_expert_qts(layer: &ExpertArenaLayer) -> Vec<ExpertQt> {
    const FORMATS: [ExpertFormat; 4] = [
        ExpertFormat::Mq4Xts,
        ExpertFormat::Mq3I4,
        ExpertFormat::Mq2I4,
        ExpertFormat::E8pI4,
    ];
    let mut present = [false; 4];
    for expert in layer.experts.iter().filter(|expert| !expert.is_absent()) {
        if let Some(slot) = FORMATS.iter().position(|format| *format == expert.format) {
            present[slot] = true;
        }
    }
    FORMATS
        .iter()
        .zip(present)
        .filter(|(_, present)| *present)
        .map(|(format, _)| {
            let (gate_up, down) = format.pair();
            ExpertQt::new(gate_up, down)
        })
        .collect()
}

/// Gather the stage facts of THIS rank's owned band. Every dtype is the
/// resolved `weights.resident(ref).dtype`, never a name, recipe or manifest
/// dtype; an HC down/up or shared-expert projection binds its Q8_0 decode copy
/// when the forward holds one (`decode_q8`, `DECODE_Q8_PER_LAYER` per absolute
/// layer: HC attn down/up 0/1, HC mlp down/up 2/3, shared gate/up/down 4/5/6,
/// as `decode_copy` binds it for every AR row count), else its source. A
/// reference that does not resolve (or a missing forward, which decides the
/// copies) leaves `None` for the stage it feeds only.
fn gather_stage_facts(bundle: &Qwen4Bundle) -> StageFacts {
    let weights = &bundle.weights;
    let config = &bundle.config;
    let scope = weights.band_scope();
    let topology = lane_topology(bundle);
    let copies: Option<&[GpuTensor]> = bundle
        .execution
        .as_ref()
        .map(|forward| forward.decode_q8.as_slice());
    let source =
        |reference: &TensorRef| -> Bound { weights.resident(reference).ok().map(|t| t.dtype) };
    let bound = |reference: &TensorRef, layer: usize, slot: usize| -> Bound {
        let source_dtype = source(reference)?;
        let copies = copies?;
        Some(
            copies
                .get(layer * DECODE_Q8_PER_LAYER + slot)
                .map_or(source_dtype, |copy| copy.dtype),
        )
    };
    let hyper = |hyper: &HyperConnectionWeights, layer: usize, slot: usize| HyperFacts {
        norm: source(&hyper.hc_norm),
        down: bound(&hyper.input_mix_down, layer, slot),
        up: bound(&hyper.input_mix_up, layer, slot + 1),
    };
    let recipe = &weights.manifest.identity.recipe;
    let experts = lane_expert_route(bundle);
    let expert_qts = |layer: usize, moe: &MoeWeights| -> Option<Vec<ExpertQt>> {
        // Under EP the pair is read only for a band that positively classified
        // as packed static/balanced ([`classify_ep_owned_experts`]).
        if weights.expert_tiers().is_some()
            || (topology == LaneTopology::Ep
                && experts != LaneExpertRoute::EpPackedStaticOrBalanced)
        {
            return None;
        }
        match recipe {
            Qwen4TierRecipe::FlashNextSlab { arenas, .. } => {
                source(moe.expert_arena.as_ref()?)?;
                Some(slab_layer_expert_qts(arenas.layer(layer)?))
            }
            Qwen4TierRecipe::FlashNextPacked { .. } => {
                ExpertQt::of_dtypes(source(&moe.experts_gate_up)?, source(&moe.experts_down)?)
                    .map(|qt| vec![qt])
            }
            _ => None,
        }
    };
    let layer_facts = |layer: &Qwen4LayerWeights| -> LayerFacts {
        let absolute = layer.layer;
        let mixer = match layer.kind {
            LayerType::LinearAttention => Some(layer.gdn.as_ref().map_or(
                MixerFacts::Gdn {
                    input: [None; 4],
                    output: None,
                },
                |gdn| MixerFacts::Gdn {
                    input: [
                        source(&gdn.qkv),
                        source(&gdn.z),
                        source(&gdn.in_proj_a),
                        source(&gdn.in_proj_b),
                    ],
                    output: source(&gdn.output),
                },
            )),
            LayerType::FullAttention => Some(layer.attention.as_ref().map_or(
                MixerFacts::Qsa {
                    input: [None; 4],
                    output: None,
                },
                |qsa| MixerFacts::Qsa {
                    input: [
                        source(&qsa.indexer_qk),
                        source(&qsa.q),
                        source(&qsa.k),
                        source(&qsa.v),
                    ],
                    output: source(&qsa.output),
                },
            )),
        };
        let ffn = match &layer.ffn {
            Qwen4FfnWeights::Moe(moe) => FfnFacts::Moe(MoeFacts {
                dense: [
                    source(&moe.gate),
                    bound(&moe.shared_gate, absolute, 4),
                    bound(&moe.shared_up, absolute, 5),
                    bound(&moe.shared_down, absolute, 6),
                    source(&moe.shared_gate_scalar),
                ],
                expert_qts: expert_qts(absolute, moe),
            }),
            Qwen4FfnWeights::Dense(_) => FfnFacts::Dense,
        };
        LayerFacts {
            attn_hyper: hyper(&layer.attn_hyper, absolute, 0),
            mlp_hyper: hyper(&layer.mlp_hyper, absolute, 2),
            attn_inject: source(&layer.attn_hyper.block_inject),
            mlp_inject: source(&layer.mlp_hyper.block_inject),
            mixer,
            ffn,
        }
    };
    let layers = scope
        .band
        .clone()
        .map(|absolute| match weights.layer(absolute) {
            Some(layer) => layer_facts(layer),
            None => LayerFacts {
                attn_hyper: HyperFacts::UNRESOLVED,
                mlp_hyper: HyperFacts::UNRESOLVED,
                attn_inject: None,
                mlp_inject: None,
                mixer: config.layer_types.get(absolute).map(|kind| match kind {
                    LayerType::LinearAttention => MixerFacts::Gdn {
                        input: [None; 4],
                        output: None,
                    },
                    LayerType::FullAttention => MixerFacts::Qsa {
                        input: [None; 4],
                        output: None,
                    },
                }),
                ffn: FfnFacts::Missing,
            },
        })
        .collect();
    let root = &weights.root;
    StageFacts {
        topology,
        input: scope.input,
        output: scope.output,
        embedding: source(&root.embedding),
        // The lane AR step converts BF16 staged rows (`grouped_gather_convert_bf16`
        // requires BF16) and refuses F16 staging (`qwen4_ple_rows_f16` needs
        // >= 512 routed rows), so a bound PLE row provider stages BF16.
        ple_rows: match (&bundle.ple, &config.ple) {
            (Some(_), _) => StagedRows::Known(DType::BF16),
            (None, Some(_)) => StagedRows::Unresolved,
            (None, None) => StagedRows::Absent,
        },
        experts,
        layers,
        final_hyper: HyperFacts {
            norm: source(&root.final_hyper.hc_norm),
            down: source(&root.final_hyper.input_mix_down),
            up: source(&root.final_hyper.input_mix_up),
        },
        head: source(&root.lm_head),
    }
}

fn all_bound<const N: usize>(dtypes: [Bound; N]) -> Option<[DType; N]> {
    let mut bound = [DType::F32; N];
    for (slot, dtype) in bound.iter_mut().zip(dtypes) {
        *slot = dtype?;
    }
    Some(bound)
}

fn hyper_read_route(hyper: &HyperFacts) -> StageRoute {
    all_bound([hyper.norm, hyper.down, hyper.up])
        .map_or(StageRoute::Unclassified, StageRoute::HyperRead)
}

fn add_route(routes: &mut LaneStageRoutes, stage: StageName, route: StageRoute) {
    let list = &mut routes[stage.index()];
    if !list.contains(&route) {
        list.push(route);
    }
}

/// One owned MoE layer's route: its dense dtypes, the band's expert class
/// (under EP only [`LaneExpertRoute::EpPackedStaticOrBalanced`] survives) and
/// whether the decode K1 route admits EVERY
/// expert quant pair present on the layer. Capability alone never certifies
/// residence: a host-mapped, CPU-deferred, split or unclassified band keeps
/// its own `experts` class. `verdicts` memoises one capability lookup per
/// distinct quant pair across the band's layers.
fn moe_route(
    arch: &str,
    facts: &StageFacts,
    moe: &MoeFacts,
    verdicts: &mut Vec<(ExpertQt, bool)>,
) -> StageRoute {
    let Some(dense) = all_bound(moe.dense) else {
        return StageRoute::Unclassified;
    };
    // Only the positively classified EP band keeps its class under EP, and
    // that class never labels another topology; every other class is an
    // unproven EP route.
    let experts = match (facts.topology, facts.experts) {
        (LaneTopology::Ep, LaneExpertRoute::EpPackedStaticOrBalanced) => facts.experts,
        (LaneTopology::Ep, _) | (_, LaneExpertRoute::EpPackedStaticOrBalanced) => {
            LaneExpertRoute::Unclassified
        }
        (_, experts) => experts,
    };
    let decode_admitted = moe.expert_qts.as_deref().is_some_and(|qts| {
        !qts.is_empty()
            && qts.iter().all(|&qt| {
                if let Some((_, admitted)) = verdicts.iter().find(|(known, _)| *known == qt) {
                    return *admitted;
                }
                let admitted =
                    expert_capabilities::capability(arch, qt, Route::DecodeGemv).admitted();
                verdicts.push((qt, admitted));
                admitted
            })
    });
    StageRoute::Moe {
        experts,
        dense,
        decode_admitted,
    }
}

/// Per-stage routes of every stage instance the owned band executes. A stage
/// the band does not execute stays empty (stateful stages are always per lane
/// and carry no route); an executed instance whose weight, reference or owner
/// is missing yields [`StageRoute::Unclassified`] for its own stage only.
fn derive_stage_routes(arch: &str, facts: &StageFacts) -> LaneStageRoutes {
    let mut routes: LaneStageRoutes = std::array::from_fn(|_| Vec::new());
    let mut verdicts: Vec<(ExpertQt, bool)> = Vec::new();
    if facts.input {
        add_route(
            &mut routes,
            StageName::Embed,
            facts
                .embedding
                .map_or(StageRoute::Unclassified, StageRoute::Embed),
        );
        match facts.ple_rows {
            StagedRows::Absent => {}
            StagedRows::Unresolved => {
                add_route(&mut routes, StageName::PleRows, StageRoute::Unclassified)
            }
            StagedRows::Known(dtype) => {
                add_route(&mut routes, StageName::PleRows, StageRoute::PleRows(dtype))
            }
        }
    }
    for layer in &facts.layers {
        for (hyper, inject) in [
            (&layer.attn_hyper, layer.attn_inject),
            (&layer.mlp_hyper, layer.mlp_inject),
        ] {
            add_route(&mut routes, StageName::HcRead, hyper_read_route(hyper));
            add_route(
                &mut routes,
                StageName::HcWrite,
                inject.map_or(StageRoute::Unclassified, StageRoute::HyperWrite),
            );
        }
        let projection =
            |dtype: Bound| dtype.map_or(StageRoute::Unclassified, StageRoute::Projection);
        match &layer.mixer {
            Some(MixerFacts::Gdn { input, output }) => {
                add_route(
                    &mut routes,
                    StageName::GdnInProj,
                    all_bound(*input).map_or(StageRoute::Unclassified, StageRoute::GdnIn),
                );
                add_route(&mut routes, StageName::GdnOutProj, projection(*output));
            }
            Some(MixerFacts::Qsa { input, output }) => {
                add_route(
                    &mut routes,
                    StageName::QsaInProj,
                    all_bound(*input).map_or(StageRoute::Unclassified, StageRoute::QsaIn),
                );
                add_route(&mut routes, StageName::QsaOutProj, projection(*output));
            }
            None => {
                for stage in [
                    StageName::GdnInProj,
                    StageName::GdnOutProj,
                    StageName::QsaInProj,
                    StageName::QsaOutProj,
                ] {
                    add_route(&mut routes, stage, StageRoute::Unclassified);
                }
            }
        }
        let moe = match &layer.ffn {
            FfnFacts::Moe(moe) => moe_route(arch, facts, moe, &mut verdicts),
            FfnFacts::Dense | FfnFacts::Missing => StageRoute::Unclassified,
        };
        add_route(&mut routes, StageName::Moe, moe);
    }
    if facts.output {
        add_route(
            &mut routes,
            StageName::FinalHyper,
            hyper_read_route(&facts.final_hyper),
        );
        add_route(
            &mut routes,
            StageName::Head,
            facts
                .head
                .map_or(StageRoute::Unclassified, StageRoute::Projection),
        );
        // The step logits are an F32 scratch (`Qwen4LaneStore::logits`).
        add_route(
            &mut routes,
            StageName::Argmax,
            StageRoute::Argmax(DType::F32),
        );
    }
    routes
}

/// The store's one stage-policy log line: proof receipts and the reason each
/// stage runs per lane, `none` when empty.
fn stage_policy_log(
    policy: &LaneStagePolicy,
    evidence: &str,
    per_lane: &[(StageName, String)],
) -> String {
    let proofs = if evidence.is_empty() {
        "none"
    } else {
        evidence
    };
    let reasons = per_lane
        .iter()
        .map(|(stage, reason)| format!("{}={reason}", stage.key()))
        .collect::<Vec<_>>()
        .join(",");
    let reasons = if reasons.is_empty() {
        "none"
    } else {
        reasons.as_str()
    };
    format!(
        "  qwen4 lanes: stage policy {} (proofs {proofs}; per-lane: {reasons})",
        policy.describe()
    )
}

/// What a staged lane store reports (load ack, oracle receipts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneStoreReceipt {
    pub kv_backend: &'static str,
    pub qsa_format: &'static str,
    pub gdn_format: &'static str,
    /// GDN capture rows armed per lane (0: AR lanes arm no ring; MTP lanes
    /// arm `k + 1`).
    pub ring_rows: usize,
    pub max_lanes: usize,
    pub row_budget: usize,
    /// Device bytes the lanes' QSA context arenas commit right now.
    pub mapped_bytes: u64,
    pub stage_policy: String,
    pub stage_evidence: String,
    /// `"mtp"` when every lane is an MTP lane, else `"off"`.
    pub spec: &'static str,
    /// Most drafts one window carries (0 without spec).
    pub spec_k: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    /// The prompt is being forwarded tile by tile.
    Prefilling,
    /// The prompt is done; one AR row per step.
    Decoding,
    /// A step failed with this lane in it: it runs nothing and commits
    /// nothing until released.
    Poisoned,
}

/// A sampled lane's request sampler: the AR singleton's `sample_cpu` policy
/// with neutral penalties (`llama::sample_top_k_p`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneSampling {
    /// `> 0` and finite (greedy lanes carry no `LaneSampling`).
    pub temperature: f32,
    pub top_p: f32,
    /// Passed through as sent, exactly as `sample_cpu` receives it.
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    /// The singleton's request seed (`request_seed_for`); the lane RNG starts
    /// where `reset_cpu_sampler_rng(seed)` leaves the sampler state (0 -> 1).
    pub seed: u32,
}

impl LaneSampling {
    /// The lane RNG's initial xorshift32 state (never 0).
    pub(crate) fn initial_rng(&self) -> u32 {
        if self.seed == 0 {
            1
        } else {
            self.seed
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.temperature.is_finite() && self.temperature > 0.0 {
            Ok(())
        } else {
            Err(format!(
                "lane sampling: temperature {} is not positive and finite",
                self.temperature
            ))
        }
    }
}

/// Host bookkeeping of one admitted request.
#[derive(Clone, Debug)]
pub struct LaneHost {
    pub epoch: RequestEpoch,
    /// Stable lane index.
    pub slot: usize,
    pub prompt: Vec<u32>,
    /// Prompt tokens forwarded so far.
    pub prefilled: usize,
    /// Picks committed so far (the first is the prompt-final pick).
    pub generated: Vec<u32>,
    /// The next AR row's token.
    pub pending_seed: Option<u32>,
    pub status: LaneStatus,
    /// Admission order across the store's lifetime.
    pub stamp: u64,
    /// `Some`: every token this lane commits is a host draw ([`LaneSampling`]);
    /// `None`: greedy device argmax. Cleared by admission and release.
    pub sampling: Option<LaneSampling>,
    /// The lane's xorshift32 sampler state (`0` while `sampling` is `None`).
    pub rng: u32,
}

/// Staging of the lanes' native MTP windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSpecConfig {
    /// Most drafts per window, `1..=7` (`k + 1` verify rows fit
    /// [`LANE_MAX_ROWS`] and the lane's `k + 1`-row GDN ring).
    pub k: usize,
    /// The end-of-turn token a window stops at; `None` uses the model's EOS.
    pub end_of_turn: Option<u32>,
}

/// The opt-in copy lane of a store (lab/oracle only: the default store stages
/// none). Arming it widens every lane's verify rows and GDN ring to
/// `1 + max(k, take cap)` rows, gives each spec lane `head_lag_rows` owed
/// head rows of storage and arms the root's copy lane
/// ([`crate::mesh_lane_protocol::MeshRoot::with_copy_lane`]) for every spec
/// request. [`crate::mesh_rank_copy::copy_budget`] is the one place that
/// derives the budget the allocator and the census both read.
#[derive(Clone, Copy, Debug)]
pub struct LaneCopyConfig {
    /// The copy context's n-gram config (`mtp_spec::qwen4_ngram_mod_config`).
    pub ngram: NgramModConfig,
    /// Owed head rows one retired lane may hold before the root flushes them
    /// (`WindowBudgets.head_lag_rows`; at least 1).
    pub head_lag_rows: usize,
}

/// A store's resolved copy lane: the request and what its budget resolved to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoreCopy {
    pub config: LaneCopyConfig,
    /// Copy candidates one window verifies at most.
    pub take_cap: usize,
    /// Verify rows one lane holds: `1 + max(k, take_cap)`.
    pub verify_rows: usize,
}

/// Counters of one MTP lane's windows (read after the request, or at any
/// step boundary).
#[derive(Clone, Copy, Debug, Default)]
pub struct LaneSpecStats {
    /// Native windows: batched windows and `ArSynced` AR rows.
    pub windows: u64,
    /// Head-free AR rows after retirement.
    pub ar_windows: u64,
    /// Drafts verified.
    pub drafts: u64,
    /// Drafts accepted.
    pub accepted: u64,
    /// The lane's floor retired it to head-free AR.
    pub retired: bool,
    /// Copy windows (final source `Copy`, after any confirmation).
    pub copy_windows: u64,
    /// Copy candidates verified.
    pub copy_drafts: u64,
    /// Copy candidates accepted.
    pub copy_accepted: u64,
    /// Confirmed first candidates / missed confirmations of copy plans.
    pub copy_confirmed: u64,
    pub copy_missed: u64,
    /// Forced advances executed.
    pub forced_windows: u64,
}

/// Where a lane's head stands against its target and the root's record
/// (lab: the oracle's head self-consistency check).
#[cfg(feature = "lab")]
#[derive(Clone, Debug)]
pub struct LabHeadFacts {
    /// The lane's head position (rows it holds).
    pub head_position: usize,
    /// The lane's target position.
    pub target_position: usize,
    /// Owed head rows the rank holds, and where they start.
    pub owed_rows: usize,
    pub owed_start: usize,
    /// The root's record of the head: its state and the owed tokens.
    pub root_head: crate::mesh_lane_protocol::HeadState,
    pub root_owed: Vec<u32>,
}

/// Lab route modes of the store's window router.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabRoutes {
    /// The lane floor (production).
    #[default]
    Floor,
    /// `Batched{K}`, `Batched{1}`, `ArSynced`, `Batched{2}` three times,
    /// then `ArRetired`.
    Cycle,
    /// Every window drafts its whole budget.
    Batched,
}

/// One MTP lane's device state next to its trunk state.
pub struct LaneSpec {
    /// The lane's MTP head state (QSA arenas, selection, wide hidden).
    pub head: MtpGpuState,
    /// The draft head's request-local policy (full-vocabulary hold, margin).
    pub draft: DraftHeadRequestState,
    /// `[hc_count * hidden]` F32: the next window's row-0 head hidden.
    pub pending_hidden: GpuTensor,
    /// The routed window between `draft_step` and `commit_step`.
    pub(crate) window: Option<PendingWindow>,
    /// Owed head rows of a retired lane with a copy future (`None`: the
    /// store stages no copy lane, a retired lane is head-free).
    pub(crate) lag: Option<HeadLagStore>,
    pub stats: LaneSpecStats,
}

impl LaneSpec {
    /// A fresh spec state: the head is built on the resident head's backend
    /// and context bound. The trunk state's verify ring is the payload's
    /// ([`new_payload`]), owned by every band that owns target layers.
    /// `lag_rows > 0` also allocates the owed-row storage
    /// ([`crate::mesh_rank_copy::lag_allocation_sizes`], what the census
    /// charges).
    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        spec: LaneSpecConfig,
        head_max_seq: usize,
        backend: Qwen4KvBackend,
        lag_rows: usize,
    ) -> Result<Self, String> {
        let config = &bundle.config;
        let head = MtpGpuState::new_with_backend(gpu, config, head_max_seq, backend)
            .map_err(|e| format!("lane head state: {e}"))?;
        let width = config
            .hc_count
            .checked_mul(config.hidden_size)
            .ok_or_else(|| "lane pending hidden width overflow".to_string())?;
        let pending_hidden = match gpu.zeros(&[width], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = head.free_gpu(gpu);
                return Err(format!("lane pending hidden: {error}"));
            }
        };
        let lag = if lag_rows > 0 {
            match HeadLagStore::new(gpu, config, lag_rows) {
                Ok(lag) => Some(lag),
                Err(error) => {
                    let _ = head.free_gpu(gpu);
                    let _ = gpu.free_tensor(pending_hidden);
                    return Err(error);
                }
            }
        } else {
            None
        };
        Ok(Self {
            head,
            draft: DraftHeadRequestState::default(),
            pending_hidden,
            window: None,
            lag,
            stats: LaneSpecStats::default(),
        })
    }

    /// Back to a request's start: empty head, zero pending hidden, no
    /// window, no owed rows, fresh counters (the window router lives at the
    /// root).
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.head
            .reset(gpu)
            .map_err(|e| format!("lane head reset: {e}"))?;
        self.draft = DraftHeadRequestState::default();
        gpu.hip
            .memset(&self.pending_hidden.buf, 0, self.pending_hidden.buf.size())
            .map_err(|e| format!("lane pending hidden reset: {e}"))?;
        self.window = None;
        if let Some(lag) = self.lag.as_mut() {
            lag.clear();
        }
        self.stats = LaneSpecStats::default();
        Ok(())
    }

    /// Start this (kept or fresh) lane spec from a promoted singleton's
    /// `parts`: its pending hidden copied in, no window, no owed rows, fresh
    /// counters. The head and draft request state are the singleton's own and
    /// swap in at commit; its agreement and retirement seed the root's lane
    /// policy.
    fn promote(&mut self, gpu: &mut Gpu, parts: &Qwen4PromotionParts<'_>) -> Result<(), String> {
        let bytes = self.pending_hidden.byte_size();
        if parts.pending_hidden.byte_size() != bytes {
            return Err(format!(
                "pending hidden is {} bytes, the lane's is {bytes}",
                parts.pending_hidden.byte_size()
            ));
        }
        gpu.copy_d2d(parts.pending_hidden, &self.pending_hidden, bytes)
            .map_err(|e| format!("pending hidden copy: {e}"))?;
        self.draft = DraftHeadRequestState::default();
        self.window = None;
        if let Some(lag) = self.lag.as_mut() {
            lag.clear();
        }
        self.stats = LaneSpecStats::default();
        Ok(())
    }
}

/// What a slot's spec payload is built from: the lane config, the resident
/// head's context bound and backend, and the owed head rows of storage
/// (`0`: none, the store stages no copy lane).
type SpecAlloc = (LaneSpecConfig, usize, Qwen4KvBackend, usize);

/// A fresh slot's payload: a lane trunk state with its `ring_rows`-row GDN
/// capture ring (every band that owns target layers; `0`: AR lanes arm no
/// ring) and, with `spec_cfg`, its MTP head state (a failure frees what was
/// allocated).
fn new_payload(
    gpu: &mut Gpu,
    bundle: &Qwen4Bundle,
    slot: usize,
    ring_rows: usize,
    spec_cfg: Option<SpecAlloc>,
) -> Result<(Qwen4State, Option<LaneSpec>), String> {
    let mut state = bundle
        .new_lane_state(gpu)
        .map_err(|e| format!("admit: allocate lane {slot}: {e}"))?;
    if ring_rows > 0 {
        if let Err(error) =
            state.ensure_row_capture(gpu, ring_rows, bundle.config.linear_conv_kernel_dim - 1)
        {
            let _ = state.free_gpu(gpu);
            return Err(format!("admit: allocate lane {slot}: lane ring: {error}"));
        }
    }
    let spec = match spec_cfg {
        None => None,
        Some((config, head_max_seq, backend, lag_rows)) => {
            match LaneSpec::new(gpu, bundle, config, head_max_seq, backend, lag_rows) {
                Ok(spec) => Some(spec),
                Err(error) => {
                    let _ = state.free_gpu(gpu);
                    return Err(format!("admit: allocate lane {slot}: {error}"));
                }
            }
        }
    };
    Ok((state, spec))
}

/// One admitted request's whole device state.
pub struct Qwen4LaneState {
    pub host: LaneHost,
    pub state: Qwen4State,
    /// `Some` on every lane of a store staged with spec, `None` on AR lanes.
    pub spec: Option<LaneSpec>,
}

/// A lane-table cell: host bookkeeping plus a device payload that outlives
/// the request (the table keeps it when the lane is released).
trait LaneCell {
    type Payload;
    fn assemble(host: LaneHost, payload: Self::Payload) -> Self;
    fn host(&self) -> &LaneHost;
    fn host_mut(&mut self) -> &mut LaneHost;
}

impl LaneCell for Qwen4LaneState {
    type Payload = (Qwen4State, Option<LaneSpec>);

    fn assemble(host: LaneHost, (state, spec): Self::Payload) -> Self {
        Self { host, state, spec }
    }

    fn host(&self) -> &LaneHost {
        &self.host
    }

    fn host_mut(&mut self) -> &mut LaneHost {
        &mut self.host
    }
}

/// Slot table of the lanes: which slots hold a live request and the cells
/// (host + payload) allocated so far. Epoch lookups match both the request
/// tag and the owner generation, so a stale generation never matches.
struct LaneTable<C: LaneCell> {
    cells: Vec<Option<C>>,
    live: Vec<bool>,
    next_stamp: u64,
}

impl<C: LaneCell> LaneTable<C> {
    fn new(max_lanes: usize) -> Self {
        Self {
            cells: (0..max_lanes).map(|_| None).collect(),
            live: vec![false; max_lanes],
            next_stamp: 0,
        }
    }

    fn max_lanes(&self) -> usize {
        self.cells.len()
    }

    fn slot_of(&self, epoch: &RequestEpoch) -> Option<usize> {
        self.cells.iter().enumerate().find_map(|(slot, cell)| {
            let cell = cell.as_ref()?;
            (self.live[slot] && cell.host().epoch == *epoch).then_some(slot)
        })
    }

    fn get(&self, epoch: &RequestEpoch) -> Option<&C> {
        self.slot_of(epoch)
            .and_then(|slot| self.cells[slot].as_ref())
    }

    fn get_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut C> {
        let slot = self.slot_of(epoch)?;
        self.cells[slot].as_mut()
    }

    fn active(&self) -> usize {
        self.live.iter().filter(|live| **live).count()
    }

    /// Live epochs in slot order.
    fn epochs(&self) -> Vec<RequestEpoch> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(slot, _)| self.live[*slot])
            .filter_map(|(_, cell)| cell.as_ref().map(|c| c.host().epoch))
            .collect()
    }

    /// Admit `epoch` into the lowest free slot as a request about to prefill.
    /// `prepare(slot, existing)` is called exactly once: with the slot's kept
    /// payload cell (it must reset it in place and return `Ok(None)`), or
    /// with `None` when the slot has never held one (it must return the new
    /// payload). The new host entry is published only after `prepare`
    /// succeeded.
    fn admit(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        self.admit_host(epoch, prompt, None, prepare)
    }

    /// Admit `epoch` as an already-decoding request (a promoted singleton):
    /// the whole `prompt` is prefilled, `generated` picks are committed and
    /// `pending_seed` is the next AR row's token. Same `prepare` contract and
    /// refusals as [`Self::admit`].
    fn admit_decoding(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        generated: Vec<u32>,
        pending_seed: u32,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        self.admit_host(epoch, prompt, Some((generated, pending_seed)), prepare)
    }

    fn admit_host(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        decoding: Option<(Vec<u32>, u32)>,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        if !epoch.is_admitted() {
            return Err(format!("admit: generation 0 is never admitted ({epoch:?})"));
        }
        if self.slot_of(&epoch).is_some() {
            return Err(format!("admit: epoch {epoch:?} is already live"));
        }
        if prompt.is_empty() {
            return Err("admit: empty prompt".to_string());
        }
        let Some(slot) = self.live.iter().position(|live| !*live) else {
            return Err(format!("admit: all {} lanes are busy", self.max_lanes()));
        };
        let (prefilled, generated, pending_seed, status) = match decoding {
            Some((generated, seed)) => (prompt.len(), generated, Some(seed), LaneStatus::Decoding),
            None => (0, Vec::new(), None, LaneStatus::Prefilling),
        };
        let host = LaneHost {
            epoch,
            slot,
            prompt,
            prefilled,
            generated,
            pending_seed,
            status,
            stamp: self.next_stamp,
            sampling: None,
            rng: 0,
        };
        match self.cells[slot].as_mut() {
            Some(cell) => {
                if prepare(slot, Some(&mut *cell))?.is_some() {
                    return Err("admit: prepare returned a payload for a kept cell".to_string());
                }
                *cell.host_mut() = host;
            }
            None => {
                let payload = prepare(slot, None)?.ok_or_else(|| {
                    "admit: prepare returned no payload for a new cell".to_string()
                })?;
                self.cells[slot] = Some(C::assemble(host, payload));
            }
        }
        self.live[slot] = true;
        self.next_stamp += 1;
        Ok(slot)
    }

    /// Free `epoch`'s slot; its payload stays for the next admission.
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        let slot = self
            .slot_of(epoch)
            .ok_or_else(|| format!("release: unknown or stale epoch {epoch:?}"))?;
        self.live[slot] = false;
        let host = self.cells[slot]
            .as_mut()
            .expect("live slot has a cell")
            .host_mut();
        host.prompt = Vec::new();
        host.generated = Vec::new();
        host.pending_seed = None;
        host.sampling = None;
        host.rng = 0;
        Ok(())
    }

    /// Mark `epoch` untrusted; unknown epochs are ignored.
    fn poison(&mut self, epoch: &RequestEpoch) {
        if let Some(cell) = self.get_mut(epoch) {
            cell.host_mut().status = LaneStatus::Poisoned;
        }
    }

    /// Replace a Decoding lane's next AR token.
    fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        let host = self
            .get_mut(epoch)
            .ok_or_else(|| format!("force_seed: unknown or stale epoch {epoch:?}"))?
            .host_mut();
        if host.status != LaneStatus::Decoding {
            return Err(format!(
                "force_seed: lane {:?} is {:?}, not Decoding",
                epoch, host.status
            ));
        }
        host.pending_seed = Some(token);
        Ok(())
    }

    /// Disjoint `&mut` cells of the live lanes `epochs`, in that order.
    fn pick_mut(&mut self, epochs: &[RequestEpoch]) -> Result<Vec<&mut C>, String> {
        let slots = epochs
            .iter()
            .map(|epoch| {
                self.slot_of(epoch)
                    .ok_or_else(|| format!("unknown or stale epoch {epoch:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut refs: Vec<Option<&mut C>> = self.cells.iter_mut().map(|c| c.as_mut()).collect();
        slots
            .iter()
            .map(|&slot| {
                refs[slot]
                    .take()
                    .ok_or_else(|| format!("lane slot {slot} named twice"))
            })
            .collect()
    }

    fn into_cells(self) -> Vec<C> {
        self.cells.into_iter().flatten().collect()
    }
}

/// Download `rows` consecutive F32 logits rows of `logits` starting at row
/// `first_row` into the front of `scratch` (grown only if shorter than
/// `rows * vocab`, never shrunk): one synchronous D2H.
pub(crate) fn download_lane_rows(
    gpu: &Gpu,
    logits: &GpuTensor,
    first_row: usize,
    rows: usize,
    vocab: usize,
    scratch: &mut Vec<f32>,
) -> Result<(), String> {
    let n = rows
        .checked_mul(vocab)
        .ok_or_else(|| "lane logits rows overflow".to_string())?;
    let begin = first_row
        .checked_mul(vocab)
        .ok_or_else(|| "lane logits row offset overflow".to_string())?;
    if n == 0
        || begin
            .checked_add(n)
            .map_or(true, |end| end > logits.numel())
    {
        return Err(format!(
            "lane logits rows {first_row}..{} are outside the step logits",
            first_row.saturating_add(rows)
        ));
    }
    if scratch.len() < n {
        scratch.resize(n, 0.0);
    }
    let view = logits.sub_offset(begin, n);
    // SAFETY: `scratch` holds at least `n` initialized f32s; any byte pattern
    // is a valid f32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, n * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &view.buf)
        .map_err(|e| format!("lane logits download: {e}"))?;
    gpu.take_device_wait_error()
        .map_err(|e| format!("lane logits download: {e}"))
}

/// One draw of a sampled lane: the AR singleton's `sample_cpu` (neutral
/// penalties) over `row`, advancing the lane's `rng`.
pub(crate) fn lane_draw(s: &LaneSampling, rng: &mut u32, row: &[f32]) -> u32 {
    hipfire_runtime::llama::sample_top_k_p_with_state(
        row,
        s.temperature,
        s.top_p,
        s.top_k,
        s.min_p,
        rng,
    )
}

/// Naive (SpecInfer) acceptance of one sampled verify window: `rows` holds
/// the window's `drafts.len() + 1` consecutive F32 rows of `vocab` logits.
/// Rows are drawn in order and only up to the verdict row
/// ([`hipfire_runtime::spec_sampling::accept_naive_prefix`]), so each emitted
/// token is exactly one draw of the AR sampler, in AR order.
pub(crate) fn lane_sampled_commit(
    drafts: &[u32],
    rows: &[f32],
    vocab: usize,
    s: &LaneSampling,
    rng: &mut u32,
    eos: u32,
) -> Result<hipfire_runtime::spec::GreedyAccept, String> {
    if vocab == 0 || rows.len() != (drafts.len() + 1) * vocab {
        return Err(format!(
            "lane sampled commit: {} logits for {} drafts of vocab {vocab}",
            rows.len(),
            drafts.len()
        ));
    }
    hipfire_runtime::spec_sampling::accept_naive_prefix(drafts, Some(eos), |row| {
        Ok(lane_draw(s, rng, &rows[row * vocab..(row + 1) * vocab]))
    })
}

/// What [`lane_sample_step`] did to a request's picks.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LaneSampleOutcome {
    /// The lane has no sampler: `picks` are untouched, nothing was fetched.
    Greedy,
    /// An AR row: `picks[0]` is now the lane's draw.
    Ar,
    /// A verify window: `picks` now make the greedy rule commit exactly the
    /// naive window (`picks[..committed.len()] == committed`); the naive
    /// acceptance is returned for the caller to cross-check.
    Naive(hipfire_runtime::spec::GreedyAccept),
}

/// The sampler step of one decode request. `picks` are the request's device
/// argmax picks (one per row); `drafts` is `None` for an AR row and the
/// window's drafts for a verify. A lane with no [`LaneSampling`] returns
/// [`LaneSampleOutcome::Greedy`] without touching `picks` or calling `fetch`.
/// Otherwise `fetch(scratch)` must fill the front of `scratch` with the
/// request's `picks.len()` consecutive logits rows (one D2H); the lane's rows
/// are then drawn in emission order against `host.rng`.
pub(crate) fn lane_sample_step(
    host: &mut LaneHost,
    drafts: Option<&[u32]>,
    picks: &mut [u32],
    eos: u32,
    vocab: usize,
    scratch: &mut Vec<f32>,
    fetch: impl FnOnce(&mut Vec<f32>) -> Result<(), String>,
) -> Result<LaneSampleOutcome, String> {
    let Some(s) = host.sampling else {
        return Ok(LaneSampleOutcome::Greedy);
    };
    debug_assert_ne!(host.rng, 0, "a sampled lane's xorshift state is never 0");
    let rows = picks.len();
    let want = match drafts {
        None => 1,
        Some(drafts) => drafts.len() + 1,
    };
    if rows != want || vocab == 0 {
        return Err(format!(
            "lane sample step: {rows} picks for a {want}-row request (vocab {vocab})"
        ));
    }
    fetch(&mut *scratch)?;
    if scratch.len() < rows * vocab {
        return Err(format!(
            "lane sample step: fetched {} logits, need {}",
            scratch.len(),
            rows * vocab
        ));
    }
    let logits = &scratch[..rows * vocab];
    match drafts {
        None => {
            picks[0] = lane_draw(&s, &mut host.rng, logits);
            Ok(LaneSampleOutcome::Ar)
        }
        Some(drafts) => {
            let naive = lane_sampled_commit(drafts, logits, vocab, &s, &mut host.rng, eos)?;
            picks[..naive.committed.len()].copy_from_slice(&naive.committed);
            Ok(LaneSampleOutcome::Naive(naive))
        }
    }
}

/// Buffers and policy shared by the MTP lanes of one store.
pub(crate) struct SpecStore {
    pub config: LaneSpecConfig,
    /// The resident head's context bound: every lane head is built with it.
    pub head_max_seq: usize,
    pub backend: Qwen4KvBackend,
    /// Owed head rows of storage each lane holds (`0`: no copy lane).
    pub lag_rows: usize,
    /// F32, `rows_cap * hc_count * hidden`: the wide hidden of every decode
    /// row of a step (the lane forward's capture).
    pub hidden: GpuTensor,
    /// Raw, `max_lanes * Gpu::TOPK8_OUT_BYTES`: one slot per still-drafting
    /// lane per draft step.
    pub draft_top1: GpuTensor,
    /// Host mirror of `draft_top1`.
    pub draft_top1_host: Vec<u8>,
    /// Shared prompt-fill append scratch (`None` when neither batched fill nor
    /// the DET prompt fill wants it, or the head does not support it).
    pub append_scratch: Option<MtpAppendScratch>,
    pub routes: LabRoutes,
    /// Lab negative control: a rejected verify keeps its full rows.
    pub skip_rollback: bool,
}

impl SpecStore {
    fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        config: LaneSpecConfig,
        max_lanes: usize,
        rows_cap: usize,
        lag_rows: usize,
    ) -> Result<Self, String> {
        let mtp = bundle
            .mtp
            .as_ref()
            .ok_or_else(|| "lane store: MTP lanes need the attached MTP head".to_string())?;
        let (head_max_seq, backend) = (mtp.max_seq, mtp.state.backend());
        let wide = bundle
            .config
            .hc_count
            .checked_mul(bundle.config.hidden_size)
            .ok_or_else(|| "lane store: hidden width overflow".to_string())?;
        let hidden_elements = rows_cap
            .checked_mul(wide)
            .ok_or_else(|| "lane store: hidden capture extent overflow".to_string())?;
        let hidden = gpu
            .zeros(&[hidden_elements], DType::F32)
            .map_err(|e| format!("lane store: hidden capture: {e}"))?;
        let top1_bytes = max_lanes * Gpu::TOPK8_OUT_BYTES;
        let draft_top1 = match gpu.zeros(&[top1_bytes], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(hidden);
                return Err(format!("lane store: draft top-1: {error}"));
            }
        };
        let append_scratch =
            if mtp_fill_scratch_wanted(gpu) && bundle.mtp_append_rows_supported(gpu) {
                let rows = MTP_FILL_ROWS.min(bundle.spec_chunk_rows().unwrap_or(1).max(1));
                // Prompt-id rows = append rows: a tile of at most `rows` rows
                // uploads its ids once; a longer one uploads per sub-chunk
                // (`append_rows`' own upload, the same bytes).
                match MtpAppendScratch::new(gpu, &bundle.config, rows, rows) {
                    Ok(scratch) => Some(scratch),
                    Err(error) => {
                        let _ = gpu.free_tensor(hidden);
                        let _ = gpu.free_tensor(draft_top1);
                        return Err(format!("lane store: append scratch: {error}"));
                    }
                }
            } else {
                None
            };
        Ok(Self {
            config,
            head_max_seq,
            backend,
            lag_rows,
            hidden,
            draft_top1,
            draft_top1_host: vec![0; top1_bytes],
            append_scratch,
            routes: LabRoutes::Floor,
            skip_rollback: false,
        })
    }
}

/// Disjoint borrows one step needs out of the store.
pub(crate) struct StepParts<'a> {
    /// The planned lanes, in plan order.
    pub lanes: Vec<&'a mut Qwen4LaneState>,
    /// F32, `(rows_cap + max_lanes) * vocab`.
    pub logits: &'a GpuTensor,
    /// Raw, `(rows_cap + max_lanes) * 4` bytes.
    pub top1: &'a GpuTensor,
    pub policy: LaneStagePolicy,
    /// Host mirror of `top1`.
    pub top1_host: &'a mut Vec<u8>,
    /// Decode rows the step buffers hold: `max_lanes * (k + 1)` with spec,
    /// else `max_lanes`.
    pub rows_cap: usize,
    /// Reused host F32 scratch for the logits rows of sampled lanes (sized
    /// once at [`Qwen4LaneStore::set_sampled`], grown only if a request needs
    /// more rows; never allocated per step).
    pub sample_host: &'a mut Vec<f32>,
    pub spec: Option<&'a mut SpecStore>,
}

/// The lane owner: lanes, step output buffers, stage policy and the step
/// phase machine. See the module docs of [`crate::lane_exec`].
pub struct Qwen4LaneStore {
    table: LaneTable<Qwen4LaneState>,
    row_budget: usize,
    ring_rows: usize,
    format: Qwen4StateFormat,
    backend: Qwen4KvBackend,
    policy: LaneStagePolicy,
    /// Receipts of the stage proofs that bound `policy` (comma-joined), frozen
    /// at construction; empty for an env override or an explicit
    /// [`Self::set_stage_policy`].
    evidence: String,
    /// Micro-batch group count policy of the PP decode wavefront (consulted
    /// only on the output-band root; see [`groups`]).
    groups: GroupPlanner,
    /// Decode rows the step buffers hold (`max_lanes * (k + 1)` with spec,
    /// else `max_lanes`).
    rows_cap: usize,
    /// Rows `[0, rows_cap)`: the decode rows of a step (the planned decode
    /// requests' rows concatenated in plan order); `[rows_cap, rows_cap +
    /// max_lanes)`: the prompt-completing prefill rows. `None` on a band
    /// that does not run the LM head (only the single scope and the output
    /// band hold step outputs).
    logits: Option<GpuTensor>,
    top1: Option<GpuTensor>,
    /// Host mirror of `top1` (empty without step outputs).
    top1_host: Vec<u8>,
    spec: Option<SpecStore>,
    /// `serve.batch_sampled`: lanes may carry a [`LaneSampling`].
    sampled: bool,
    /// Host scratch the sampled lanes' downloaded logits rows reuse (empty
    /// unless staged `sampled` on a band with step outputs).
    sample_host: Vec<f32>,
    /// The band this store's lane states are scoped to.
    scope: Qwen4BandScope,
    pub(crate) phase: LanePhase,
    pub(crate) next_step_id: u64,
    /// Start of the step's wall clock: set by `draft_step` (or provision when
    /// no draft ran), taken by the forward.
    pub(crate) step_started: Option<Instant>,
    /// The lane protocol's root, rank session and in-flight step; taken out
    /// for the duration of a step entry (see [`crate::lane_rank`]).
    proto: Option<LaneProto>,
    /// The copy lane this store was staged with (`None`: the default store).
    copy: Option<StoreCopy>,
}

impl Qwen4LaneStore {
    /// An empty store for `max_lanes` lanes (states are allocated on first
    /// admission into a slot) and its step output buffers. With `spec`
    /// every lane is an MTP lane.
    pub(crate) fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_lanes: usize,
        row_budget: usize,
        spec: Option<LaneSpecConfig>,
        copy: Option<LaneCopyConfig>,
    ) -> Result<Self, String> {
        if !bundle.config.profile.is_flash_next() {
            return Err(format!(
                "lane store: lanes run only the Flash-Next profile, not {}",
                bundle.config.profile.id()
            ));
        }
        if max_lanes == 0 || max_lanes > MAX_LANES {
            return Err(format!(
                "lane store: max_lanes {max_lanes} outside 1..={MAX_LANES}"
            ));
        }
        if row_budget == 0 {
            return Err("lane store: row_budget must be at least 1".to_string());
        }
        // Developer overrides, refused before any device allocation.
        let group_override =
            groups::groups_override_from_env().map_err(|e| format!("lane store: {e}"))?;
        let policy_override =
            groups::policy_override_from_env().map_err(|e| format!("lane store: {e}"))?;
        if let Some(requested) = group_override {
            eprintln!(
                "  qwen4 lanes: {} = {requested} fixes the PP decode micro-batch count (clamped to min(lanes, 2 * stages))",
                groups::GROUPS_ENV
            );
        }
        let (policy, evidence) = match policy_override {
            Some(policy) => {
                eprintln!(
                    "  qwen4 lanes: {} selects stage policy {}",
                    groups::POLICY_ENV,
                    policy.describe()
                );
                (policy, String::new())
            }
            // Shared stages only where every stage instance this band runs
            // matches a proof: routes come from the bound weights.
            None => {
                let facts = gather_stage_facts(bundle);
                let routes = derive_stage_routes(&gpu.arch, &facts);
                let resolution =
                    LaneStagePolicy::for_bound_routes(&gpu.arch, facts.topology, &routes);
                let evidence = resolution.receipts.join(",");
                eprintln!(
                    "{}",
                    stage_policy_log(&resolution.policy, &evidence, &resolution.per_lane)
                );
                (resolution.policy, evidence)
            }
        };
        // This band's ownership: the single scope allocates the historical
        // whole-model store.
        let scope = bundle.weights.band_scope().clone();
        // A head owner (the single scope, every EP rank, the PP output band)
        // holds the MTP head; any other band that owns target layers holds
        // only its lanes' verify rings (target-only ownership).
        let proto = LaneProto::new(copy.map(|c| c.ngram))?;
        if let Some(spec) = spec {
            let verify_rows = if bundle.has_cpu_deferred_experts() {
                crate::program::QWEN4_CPU_SPLICE_MAX_ROWS
            } else {
                LANE_MAX_ROWS
            };
            if spec.k == 0 || spec.k >= verify_rows {
                return Err(format!(
                    "lane store: MTP k {} outside 1..={} for this execution mode",
                    spec.k,
                    verify_rows - 1
                ));
            }
            if scope.output && bundle.mtp.is_none() {
                return Err("lane store: MTP lanes need the attached MTP head".to_string());
            }
            if !native_mtp_row_capture(gpu, &bundle.config) {
                return Err(
                    "lane store: MTP lanes need the GDN row capture this GPU lacks".to_string(),
                );
            }
        }
        // The copy lane (opt-in) widens every lane's verify rows and ring to
        // `1 + max(k, take cap)` and adds the owed-row storage; without it the
        // store keeps its `k + 1`-row shape exactly.
        let (ring_rows, rows_cap, lag_rows, copy_arming) = match (spec, copy) {
            (Some(spec), Some(copy)) => {
                if !scope.output {
                    return Err(
                        "lane store: the copy lane needs a head-owning band (single card)"
                            .to_string(),
                    );
                }
                let row_exact = bundle.row_exact_verify_rows(gpu);
                // Only the non-context fields are read here; the census
                // reads the same function with the real context.
                let budget = crate::mesh_rank_copy::copy_budget(
                    max_lanes,
                    spec.k,
                    &copy,
                    row_exact,
                    Qwen4ContextCommit::legacy(1),
                    1,
                )
                .map_err(|e| format!("lane store: {e}"))?;
                (
                    budget.capture_rows_per_lane,
                    budget.step_rows(),
                    budget.head_lag_rows,
                    Some(StoreCopy {
                        config: copy,
                        take_cap: budget.copy_take_cap,
                        verify_rows: budget.verify_rows_per_lane,
                    }),
                )
            }
            (None, Some(_)) => {
                return Err("lane store: the copy lane needs MTP lanes (spec)".to_string());
            }
            (Some(spec), None) => (spec.k + 1, max_lanes * (spec.k + 1), 0, None),
            (None, None) => (0, max_lanes, 0, None),
        };
        let rows = rows_cap + max_lanes;
        let vocab = bundle.lane_vocab();
        // Step logits and top-1 exist only where the LM head runs: the
        // single scope or the output band.
        let (logits, top1) = if scope.output {
            let logit_elements = rows
                .checked_mul(vocab)
                .ok_or_else(|| "lane store: logits extent overflow".to_string())?;
            let logits = gpu
                .zeros(&[logit_elements], DType::F32)
                .map_err(|e| format!("lane store: logits: {e}"))?;
            let top1 = match gpu.zeros(&[rows * std::mem::size_of::<i32>()], DType::Raw) {
                Ok(tensor) => tensor,
                Err(error) => {
                    let _ = gpu.free_tensor(logits);
                    return Err(format!("lane store: top1: {error}"));
                }
            };
            (Some(logits), Some(top1))
        } else {
            (None, None)
        };
        let spec_store = match spec {
            Some(config) if scope.output => {
                match SpecStore::new(gpu, bundle, config, max_lanes, rows_cap, lag_rows) {
                    Ok(store) => Some(store),
                    Err(error) => {
                        for tensor in [logits, top1].into_iter().flatten() {
                            let _ = gpu.free_tensor(tensor);
                        }
                        return Err(error);
                    }
                }
            }
            _ => None,
        };
        Ok(Self {
            table: LaneTable::new(max_lanes),
            row_budget: row_budget.min(max_lanes),
            ring_rows,
            format: bundle.state_format(),
            backend: bundle.state.qsa_backend(),
            policy,
            evidence,
            groups: GroupPlanner::new(group_override),
            rows_cap,
            logits,
            top1,
            top1_host: if scope.output {
                vec![0; rows * std::mem::size_of::<i32>()]
            } else {
                Vec::new()
            },
            spec: spec_store,
            sampled: false,
            sample_host: Vec::new(),
            scope,
            phase: LanePhase::Idle,
            next_step_id: 0,
            step_started: None,
            proto: Some(proto),
            copy: copy_arming,
        })
    }

    /// Device bytes the store commits at load: `max_lanes` lane states
    /// ([`Qwen4State::device_bytes`]) plus the step logits and top-1
    /// buffers; with `spec_k` each lane also holds its MTP head state, its
    /// `k + 1`-row GDN ring and pending hidden, and the store the wide-hidden
    /// capture, the draft top-1 slots and the prompt-fill append scratch
    /// (counted at its `MTP_FILL_ROWS` bound: the real scratch is
    /// `min(MTP_FILL_ROWS, forward max_chunk)` rows). `_row_budget` is
    /// accepted for the frozen signature: decode rows share the
    /// `max_lanes * (k + 1)` output rows, so the budget adds nothing.
    /// The whole model: [`Self::device_bytes_scoped`] over
    /// [`Qwen4BandScope::single`].
    pub fn device_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        max_lanes: usize,
        row_budget: usize,
        spec_k: Option<usize>,
    ) -> Option<u64> {
        Self::device_bytes_scoped(
            config,
            format,
            context,
            max_lanes,
            row_budget,
            spec_k,
            &Qwen4BandScope::single(config),
        )
    }

    /// [`Self::device_bytes`] of the store one pipeline band holds: the
    /// census of [`census::ar_lane_census`] / [`census::lane_spec_census`]
    /// under the legacy budget ([`census::LaneSpecBudget::legacy`]: copy not
    /// armed, a `k + 1`-row ring). The lane states and rings cover only the
    /// band's layers; the step logits/top-1 buffers exist only on a band
    /// that runs the LM head (`scope.output`) and every MTP head charge only
    /// on the head owner (the output band). A non-output band with `spec_k`
    /// holds its lanes' target rings alone. `None` on overflow or a `spec_k`
    /// outside the lane budget, which [`Self::new`] refuses.
    pub fn device_bytes_scoped(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        max_lanes: usize,
        _row_budget: usize,
        spec_k: Option<usize>,
        scope: &Qwen4BandScope,
    ) -> Option<u64> {
        // The legacy wrapper has no rank: rank 0 owns the head iff its band
        // runs the LM head (the single scope and the output band do).
        let (rank, owner_mask) = if scope.output { (0, 1) } else { (0, 2) };
        let append_scratch = (mtp_fill_scratch_requested()
            && MtpAppendScratch::allocation_sizes(config, MTP_FILL_ROWS, MTP_FILL_ROWS).is_some())
        .then_some((MTP_FILL_ROWS, MTP_FILL_ROWS));
        let inputs = census::CensusInputs {
            config,
            format,
            scope,
            rank,
            append_scratch,
        };
        let census = match spec_k {
            None => census::ar_lane_census(max_lanes, context, &inputs).ok()?,
            Some(k) => {
                let budget =
                    census::LaneSpecBudget::new(max_lanes, k, 0, 0, *context, owner_mask).ok()?;
                census::lane_spec_census(&budget, &inputs).ok()?
            }
        };
        census.total()
    }

    /// [`Self::device_bytes`] of a copy-armed store: the census of the
    /// widened budget ([`crate::mesh_rank_copy::copy_budget`]: `1 + max(k,
    /// take cap)` verify rows and ring rows per lane, `head_lag_rows` owed
    /// rows per lane) that [`Self::new`] allocates, so census == allocation.
    /// `row_exact_rows` is the arch's row-exact verify width.
    pub fn device_bytes_with_copy(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        max_lanes: usize,
        spec_k: usize,
        row_exact_rows: usize,
        copy: &LaneCopyConfig,
    ) -> Option<u64> {
        let scope = Qwen4BandScope::single(config);
        let append_scratch = (mtp_fill_scratch_requested()
            && MtpAppendScratch::allocation_sizes(config, MTP_FILL_ROWS, MTP_FILL_ROWS).is_some())
        .then_some((MTP_FILL_ROWS, MTP_FILL_ROWS));
        let inputs = census::CensusInputs {
            config,
            format,
            scope: &scope,
            rank: 0,
            append_scratch,
        };
        let budget = crate::mesh_rank_copy::copy_budget(
            max_lanes,
            spec_k,
            copy,
            row_exact_rows,
            *context,
            1,
        )
        .ok()?;
        census::lane_spec_census(&budget, &inputs).ok()?.total()
    }

    /// Admit `epoch` with its canonical `prompt` into the lowest free lane
    /// and return its slot. A slot that held a request before resets its
    /// state in place (an MTP lane also its head, draft policy, pending
    /// hidden, window policy and counters); a fresh slot allocates one.
    pub fn admit(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, String> {
        let ring_rows = self.ring_rows;
        let spec_cfg = self
            .spec
            .as_ref()
            .map(|s| (s.config, s.head_max_seq, s.backend, s.lag_rows));
        self.table
            .admit(epoch, prompt, |slot, existing| match existing {
                Some(cell) => {
                    cell.state
                        .reset(gpu)
                        .map_err(|e| format!("admit: reset lane {slot}: {e}"))?;
                    cell.state.row_capture_armed = false;
                    match (spec_cfg, cell.spec.as_mut()) {
                        (Some(_), Some(spec)) => spec
                            .reset(gpu)
                            .map_err(|e| format!("admit: reset lane {slot}: {e}"))?,
                        (None, None) => {}
                        _ => {
                            return Err(format!(
                                "admit: lane {slot} spec state does not match the store"
                            ));
                        }
                    }
                    Ok(None)
                }
                None => new_payload(gpu, bundle, slot, ring_rows, spec_cfg).map(Some),
            })
    }

    /// Admit `epoch` straight into Decoding (a promoted singleton): `prompt`
    /// is fully prefilled, `generated` picks are committed, `pending_seed`
    /// is the next AR row. The lane's trunk and head states are not touched
    /// here (the bundle swaps the singleton's in at commit); a fresh slot
    /// allocates them like [`Self::admit`]. An MTP store takes the singleton's
    /// pending hidden (one device copy) and the lane policy started from
    /// `parts`; an AR store takes none. Refused while a step is in flight.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admit_promoted(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        generated: Vec<u32>,
        pending_seed: u32,
        position: usize,
        parts: Option<Qwen4PromotionParts<'_>>,
    ) -> Result<usize, String> {
        if self.phase != LanePhase::Idle {
            return Err("admit_promoted: a step is in flight".to_string());
        }
        let ring_rows = self.ring_rows;
        let spec_cfg = self
            .spec
            .as_ref()
            .map(|s| (s.config, s.head_max_seq, s.backend, s.lag_rows));
        match (spec_cfg.is_some(), parts.is_some()) {
            (true, false) => {
                return Err(
                    "admit_promoted: an MTP lane store needs the drafter's parts".to_string(),
                );
            }
            (false, true) => {
                return Err("admit_promoted: an AR lane store takes no drafter parts".to_string());
            }
            _ => {}
        }
        let root_prompt = prompt.clone();
        let slot = self.table.admit_decoding(
            epoch,
            prompt,
            generated,
            pending_seed,
            |slot, existing| match existing {
                Some(cell) => {
                    match (spec_cfg, cell.spec.as_mut(), parts.as_ref()) {
                        (Some(_), Some(spec), Some(parts)) => spec
                            .promote(gpu, parts)
                            .map_err(|e| format!("admit_promoted: lane {slot}: {e}"))?,
                        (None, None, None) => {}
                        _ => {
                            return Err(format!(
                                "admit_promoted: lane {slot} spec state does not match the store"
                            ));
                        }
                    }
                    cell.state.row_capture_armed = false;
                    Ok(None)
                }
                None => {
                    let (state, mut spec) = new_payload(gpu, bundle, slot, ring_rows, spec_cfg)?;
                    let seeded = match (spec_cfg, spec.as_mut(), parts.as_ref()) {
                        (Some(_), Some(spec), Some(parts)) => spec.promote(gpu, parts),
                        _ => Ok(()),
                    };
                    if let Err(error) = seeded {
                        let _ = state.free_gpu(gpu);
                        if let Some(spec) = spec {
                            let _ = spec.head.free_gpu(gpu);
                            let _ = gpu.free_tensor(spec.pending_hidden);
                            if let Some(lag) = spec.lag {
                                let _ = lag.free_gpu(gpu);
                            }
                        }
                        return Err(format!("admit_promoted: lane {slot}: {error}"));
                    }
                    Ok(Some((state, spec)))
                }
            },
        )?;
        // The root learns the promoted request: the singleton's agreement
        // seeds the lane policy and its retirement is kept (the floor is
        // never reset by a promotion).
        let k = spec_cfg.map(|(config, _, _, _)| config.k);
        let eos = spec_cfg
            .and_then(|(config, _, _, _)| config.end_of_turn)
            .unwrap_or(bundle.config.eos_token_id);
        let promoted = parts.as_ref().map(|p| (p.agreement, p.retired));
        let admitted = self.proto_mut()?.admit_decoding(DecodeAdmission {
            epoch,
            position,
            seed: pending_seed,
            eos,
            rng_seed: 0,
            prompt: &root_prompt,
            spec_k: k,
            promoted,
        });
        if let Err(error) = admitted {
            let _ = self.table.release(&epoch);
            return Err(format!("admit_promoted: {error}"));
        }
        Ok(slot)
    }

    /// Free `epoch`'s lane (its device state stays allocated for reuse).
    pub fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        self.table.release(epoch)?;
        if let Some(proto) = self.proto.as_mut() {
            proto.release(epoch);
        }
        Ok(())
    }

    pub fn poison(&mut self, epoch: &RequestEpoch) {
        if let Some(lane) = self.table.get_mut(epoch) {
            // A half-run window's tickets die with it: the next admission's
            // reset invalidates them.
            lane.state.row_capture_armed = false;
            if let Some(spec) = lane.spec.as_mut() {
                spec.window = None;
            }
        }
        self.table.poison(epoch);
    }

    /// Replace a Decoding lane's next AR token (the singleton's think-budget
    /// forced close tokens); `generated` is untouched.
    pub fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        self.table.force_seed(epoch, token)?;
        // A PP band's lane is driven by the PP executor and never reaches
        // the single-card root; its host mirror is its whole state.
        let proto = self.proto_mut()?;
        if proto.root.has_lane(*epoch) {
            proto.force_seed(epoch, token)?;
        }
        Ok(())
    }

    /// The lab route mode of the window router (the floor on AR stores).
    pub(crate) fn routes(&self) -> LabRoutes {
        self.spec.as_ref().map_or(LabRoutes::Floor, |s| s.routes)
    }

    /// The copy lane this store was staged with (`None`: the default store,
    /// copy off on every lane).
    pub(crate) fn copy(&self) -> Option<StoreCopy> {
        self.copy
    }

    /// Whether the store runs the copy lane.
    pub fn copy_armed(&self) -> bool {
        self.copy.is_some()
    }

    /// Verify rows one lane holds at most: `1 + max(k, take cap)` with the
    /// copy lane, else the row-exact guard ([`LANE_MAX_ROWS`], the plan's
    /// historical bound).
    pub(crate) fn verify_rows_per_lane(&self) -> usize {
        self.copy.map_or(LANE_MAX_ROWS, |c| c.verify_rows)
    }

    /// Whether this store was staged to admit sampled lanes
    /// (`serve.batch_sampled`).
    pub fn sampled(&self) -> bool {
        self.sampled
    }

    /// Record `serve.batch_sampled`. Turning it on pre-sizes the host logits
    /// scratch (bands with step outputs only) for the widest request, one AR
    /// row or `k + 1` verify rows, once; turning it off clears every lane's
    /// sampler. Staging-time call: no step may be in flight.
    pub(crate) fn set_sampled(&mut self, on: bool, vocab: usize) {
        self.sampled = on;
        if on {
            if self.logits.is_some() {
                let rows = self.spec.as_ref().map_or(1, |spec| spec.config.k + 1);
                let need = rows.saturating_mul(vocab);
                if self.sample_host.len() < need {
                    self.sample_host.resize(need, 0.0);
                }
            }
        } else {
            for cell in self.table.cells.iter_mut().flatten() {
                cell.host.sampling = None;
                cell.host.rng = 0;
            }
            self.sample_host = Vec::new();
        }
    }

    /// What [`Self::set_lane_sampling`] checks, changing nothing (a PP mesh
    /// checks every band before it sets any).
    pub(crate) fn check_lane_sampling(
        &self,
        epoch: &RequestEpoch,
        sampling: &LaneSampling,
    ) -> Result<(), String> {
        if !self.sampled {
            return Err(
                "set_lane_sampling: the lane store is not staged sampled (serve.batch_sampled)"
                    .to_string(),
            );
        }
        if self.phase != LanePhase::Idle {
            return Err("set_lane_sampling: a step is in flight".to_string());
        }
        sampling.validate()?;
        let host = &self
            .table
            .get(epoch)
            .ok_or_else(|| format!("set_lane_sampling: unknown or stale epoch {epoch:?}"))?
            .host;
        if host.status != LaneStatus::Prefilling
            || host.prefilled != 0
            || !host.generated.is_empty()
        {
            return Err(format!(
                "set_lane_sampling: lane {epoch:?} is {:?} with {} prefilled and {} committed; \
                 sampling is set before the first step",
                host.status,
                host.prefilled,
                host.generated.len()
            ));
        }
        Ok(())
    }

    /// Make `epoch`'s lane a sampled lane: host only, after admission and
    /// before the lane's first step. Refused unless the store is staged
    /// sampled, no step is in flight, and the lane is Prefilling with nothing
    /// prefilled and committed. The lane RNG starts at
    /// [`LaneSampling::initial_rng`].
    pub fn set_lane_sampling(
        &mut self,
        epoch: &RequestEpoch,
        sampling: LaneSampling,
    ) -> Result<(), String> {
        self.check_lane_sampling(epoch, &sampling)?;
        let host = &mut self
            .table
            .get_mut(epoch)
            .expect("checked by check_lane_sampling")
            .host;
        host.rng = sampling.initial_rng();
        host.sampling = Some(sampling);
        Ok(())
    }

    pub fn lane(&self, epoch: &RequestEpoch) -> Option<&Qwen4LaneState> {
        self.table.get(epoch)
    }

    pub(crate) fn lane_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut Qwen4LaneState> {
        self.table.get_mut(epoch)
    }

    /// Live epochs in slot order.
    pub fn epochs(&self) -> Vec<RequestEpoch> {
        self.table.epochs()
    }

    /// The step logits (F32, `(rows_cap + max_lanes)` rows of `vocab`):
    /// row `r < rows_cap` is the `r`-th decode row of the plan (the planned
    /// decode requests' rows concatenated in plan order: an AR request is
    /// one row, a Verify request `1 + draft_len` rows); the i-th Prefill
    /// request is row `rows_cap + i` (written on a prompt-completing tile of
    /// an AR lane; an MTP lane's prompt-completing pick is read by the
    /// singleton chunk path and leaves no logits row). `rows_cap` is
    /// `max_lanes` without spec, so an AR-only plan is the S3 layout.
    /// Valid until the next forward; read it before `commit_step`. `None`
    /// on a store without step outputs ([`Self::has_step_outputs`]).
    pub fn logits(&self) -> Option<&GpuTensor> {
        self.logits.as_ref()
    }

    /// First prompt-completing logits row: `rows_cap` (`max_lanes * (k + 1)`
    /// with spec, else `max_lanes`).
    pub fn prefill_logits_row(&self) -> usize {
        self.rows_cap
    }

    /// The MTP staging (`None`: AR lanes).
    pub fn spec(&self) -> Option<LaneSpecConfig> {
        self.spec.as_ref().map(|s| s.config)
    }

    /// `epoch`'s window counters (`None`: unknown lane or an AR lane); the
    /// retirement is the root's.
    pub fn lane_spec_stats(&self, epoch: &RequestEpoch) -> Option<LaneSpecStats> {
        let spec = self.lane(epoch)?.spec.as_ref()?;
        Some(LaneSpecStats {
            retired: self
                .proto
                .as_ref()
                .and_then(|p| p.root.lane_retired(*epoch))
                .unwrap_or(false),
            ..spec.stats
        })
    }

    /// The lane's next AR row token as the root holds it (`LaneHost::
    /// pending_seed` mirrors it). `None`: no such lane at the root.
    pub fn root_seed(&self, epoch: &RequestEpoch) -> Option<u32> {
        self.proto
            .as_ref()?
            .root
            .lane_position(*epoch)
            .map(|(_, seed)| seed)
    }

    pub(crate) fn proto_mut(&mut self) -> Result<&mut LaneProto, String> {
        self.proto
            .as_mut()
            .ok_or_else(|| "lane protocol state is checked out".to_string())
    }

    /// Check the protocol state out for one step entry (the entry puts it
    /// back with [`Self::put_proto`]).
    pub(crate) fn take_proto(&mut self) -> Result<LaneProto, String> {
        self.proto
            .take()
            .ok_or_else(|| "lane protocol state is checked out".to_string())
    }

    pub(crate) fn put_proto(&mut self, proto: LaneProto) {
        self.proto = Some(proto);
    }

    /// Lab negative control: a rejected verify keeps its full rows.
    pub(crate) fn skip_rollback(&self) -> bool {
        self.spec.as_ref().is_some_and(|s| s.skip_rollback)
    }

    /// Choose the window router (lab): the floor (default), a fixed cycle or
    /// full-budget batched windows. No effect on AR lanes.
    #[cfg(feature = "lab")]
    pub fn lab_set_routes(&mut self, routes: LabRoutes) {
        if let Some(spec) = self.spec.as_mut() {
            spec.routes = routes;
        }
    }

    /// Negative control (lab): a rejected verify keeps its full rows.
    #[cfg(feature = "lab")]
    pub fn lab_set_skip_rollback(&mut self, on: bool) {
        if let Some(spec) = self.spec.as_mut() {
            spec.skip_rollback = on;
        }
    }

    /// Negative control (lab): zero `epoch`'s pending hidden (the next
    /// window's row-0 head hidden), as if a promotion had lost the copy.
    #[cfg(feature = "lab")]
    pub fn lab_zero_pending_hidden(
        &mut self,
        gpu: &mut Gpu,
        epoch: &RequestEpoch,
    ) -> Result<(), String> {
        let spec = self
            .lane(epoch)
            .ok_or_else(|| format!("lab_zero_pending_hidden: unknown or stale epoch {epoch:?}"))?
            .spec
            .as_ref()
            .ok_or_else(|| "lab_zero_pending_hidden: AR lane".to_string())?;
        gpu.hip
            .memset(&spec.pending_hidden.buf, 0, spec.pending_hidden.buf.size())
            .map_err(|e| format!("lab_zero_pending_hidden: {e}"))
    }

    /// Every byte family of `epoch`'s MTP head state
    /// ([`crate::mtp_gpu::lab_head_items`]), then `draft_request_state`
    /// (`full_steps` u32 LE, `margin` f32 LE) and `pending_hidden` (F32 LE).
    #[cfg(feature = "lab")]
    pub fn lab_lane_items(
        &self,
        gpu: &Gpu,
        epoch: &RequestEpoch,
    ) -> Result<Vec<(String, Vec<u8>)>, String> {
        let spec = self
            .lane(epoch)
            .ok_or_else(|| format!("lab_lane_items: unknown or stale epoch {epoch:?}"))?
            .spec
            .as_ref()
            .ok_or_else(|| "lab_lane_items: AR lane".to_string())?;
        let mut items = crate::mtp_gpu::lab_head_items(gpu, &spec.head)?;
        let mut request = spec.draft.full_steps.to_le_bytes().to_vec();
        request.extend_from_slice(&spec.draft.margin.to_le_bytes());
        items.push(("draft_request_state".to_string(), request));
        let pending = gpu
            .download_f32(&spec.pending_hidden)
            .map_err(|e| format!("lab_lane_items: pending hidden: {e}"))?;
        items.push((
            "pending_hidden".to_string(),
            pending.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ));
        Ok(items)
    }

    /// Where `epoch`'s head stands against its target and the root's record
    /// (the oracle's head self-consistency check): `None` for an AR lane or an
    /// unknown epoch.
    #[cfg(feature = "lab")]
    pub fn lab_head_facts(&self, epoch: &RequestEpoch) -> Option<LabHeadFacts> {
        let (head_position, target_position, owed_rows, owed_start) =
            crate::mesh_rank_copy::head_facts(self.lane(epoch)?)?;
        let (root_head, root_owed) = self.proto.as_ref()?.root.head_state(*epoch)?;
        Some(LabHeadFacts {
            head_position,
            target_position,
            owed_rows,
            owed_start,
            root_head,
            root_owed: root_owed.to_vec(),
        })
    }

    /// Disjoint borrows of the `epochs` lanes (plan order) and the step
    /// output buffers. Only a store with output buffers
    /// ([`Self::has_step_outputs`]: the single scope or the output band)
    /// has a step; a non-output band reaches its lanes through
    /// [`Self::band_lanes_mut`].
    pub(crate) fn step_parts(&mut self, epochs: &[RequestEpoch]) -> Result<StepParts<'_>, String> {
        let Self {
            table,
            policy,
            logits,
            top1,
            top1_host,
            rows_cap,
            sample_host,
            spec,
            scope,
            ..
        } = self;
        let (Some(logits), Some(top1)) = (logits.as_ref(), top1.as_ref()) else {
            return Err(format!(
                "lane store: band {:?} is not the output band and has no step logits",
                scope.band
            ));
        };
        Ok(StepParts {
            lanes: table.pick_mut(epochs)?,
            logits,
            top1,
            policy: *policy,
            top1_host,
            rows_cap: *rows_cap,
            sample_host,
            spec: spec.as_mut(),
        })
    }

    /// Disjoint borrows of the `epochs` lanes (plan order) of any band, with
    /// no step buffers: what a non-output band's lane step runs on.
    pub(crate) fn band_lanes_mut(
        &mut self,
        epochs: &[RequestEpoch],
    ) -> Result<Vec<&mut Qwen4LaneState>, String> {
        self.table.pick_mut(epochs)
    }

    /// The band this store's lane states are scoped to.
    pub fn band_scope(&self) -> &Qwen4BandScope {
        &self.scope
    }

    /// The store holds the step logits and top-1 buffers: the single scope
    /// or the output band.
    pub fn has_step_outputs(&self) -> bool {
        self.logits.is_some()
    }

    /// Disjoint borrows of the `epochs` lanes (request order) and the MTP
    /// buffers, for the draft phase.
    pub(crate) fn draft_parts(
        &mut self,
        epochs: &[RequestEpoch],
    ) -> Result<(Vec<&mut Qwen4LaneState>, Option<&mut SpecStore>), String> {
        let Self { table, spec, .. } = self;
        Ok((table.pick_mut(epochs)?, spec.as_mut()))
    }

    pub fn receipt(&self, gpu: &Gpu) -> LaneStoreReceipt {
        let mapped: usize = self
            .table
            .cells
            .iter()
            .flatten()
            .map(|lane| {
                lane.state.mapped_context_bytes(gpu).unwrap_or(0)
                    + lane
                        .spec
                        .as_ref()
                        .map_or(0, |spec| spec.head.mapped_context_bytes(gpu).unwrap_or(0))
            })
            .sum();
        LaneStoreReceipt {
            kv_backend: self.backend.name(),
            qsa_format: self.format.qsa.name(),
            gdn_format: self.format.gdn.name(),
            ring_rows: self.ring_rows,
            max_lanes: self.max_lanes(),
            row_budget: self.row_budget,
            mapped_bytes: mapped as u64,
            stage_policy: self.policy.describe(),
            stage_evidence: self.evidence.clone(),
            spec: if self.spec.is_some() { "mtp" } else { "off" },
            spec_k: self.spec.as_ref().map_or(0, |s| s.config.k),
        }
    }

    pub fn stage_policy(&self) -> LaneStagePolicy {
        self.policy
    }

    /// Receipts of the stage proofs behind the current policy; empty when it
    /// is an override.
    pub fn evidence(&self) -> &str {
        &self.evidence
    }

    /// An explicit policy is never proof-bound, even when it equals the
    /// resolved one: the evidence is cleared.
    pub fn set_stage_policy(&mut self, policy: LaneStagePolicy) {
        if self.policy != policy {
            // Walls measured under one policy say nothing about another.
            self.groups.clear();
        }
        self.policy = policy;
        self.evidence.clear();
    }

    /// Micro-batch group count for a PP decode step over `rows` (rows per
    /// lane, plan order); see [`groups::GroupPlanner::select`].
    pub(crate) fn select_decode_groups<I>(&mut self, stages: usize, rows: I) -> usize
    where
        I: ExactSizeIterator<Item = usize> + Clone,
    {
        self.groups.select(&self.policy, stages, rows)
    }

    /// Record one complete successful PP decode step's wall (microseconds).
    pub(crate) fn observe_decode_groups<I>(
        &mut self,
        stages: usize,
        rows: I,
        groups: usize,
        wall_us: f64,
    ) -> bool
    where
        I: Iterator<Item = usize> + Clone,
    {
        self.groups
            .observe(&self.policy, stages, rows, groups, wall_us)
    }

    pub fn max_lanes(&self) -> usize {
        self.table.max_lanes()
    }

    /// Most AR rows one step may carry.
    pub fn row_budget(&self) -> usize {
        self.row_budget
    }

    /// Live lanes.
    pub fn active(&self) -> usize {
        self.table.active()
    }

    /// Free every lane state and the step buffers; the first error wins.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            table,
            logits,
            top1,
            spec,
            ..
        } = self;
        let mut first: Option<String> = None;
        for lane in table.into_cells() {
            let Qwen4LaneState { state, spec, .. } = lane;
            if let Err(error) = state.free_gpu(gpu) {
                first.get_or_insert(format!("lane state: {error}"));
            }
            if let Some(LaneSpec {
                head,
                pending_hidden,
                lag,
                ..
            }) = spec
            {
                if let Some(error) = head.free_gpu(gpu) {
                    first.get_or_insert(format!("lane mtp state: {error}"));
                }
                if let Err(error) = gpu.free_tensor(pending_hidden) {
                    first.get_or_insert(format!("lane pending hidden: {error}"));
                }
                if let Some(lag) = lag {
                    if let Err(error) = lag.free_gpu(gpu) {
                        first.get_or_insert(error);
                    }
                }
            }
        }
        if let Some(SpecStore {
            hidden,
            draft_top1,
            append_scratch,
            ..
        }) = spec
        {
            for (name, tensor) in [("hidden capture", hidden), ("draft top1", draft_top1)] {
                if let Err(error) = gpu.free_tensor(tensor) {
                    first.get_or_insert(format!("lane {name}: {error}"));
                }
            }
            if let Some(scratch) = append_scratch {
                if let Err(error) = scratch.free_gpu(gpu) {
                    first.get_or_insert(format!("lane append scratch: {error}"));
                }
            }
        }
        for (name, tensor) in [("logits", logits), ("top1", top1)] {
            let Some(tensor) = tensor else { continue };
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("lane {name}: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_arena::ArenaExpert;

    struct TestCell {
        host: LaneHost,
        payload: u32,
    }

    impl LaneCell for TestCell {
        type Payload = u32;

        fn assemble(host: LaneHost, payload: u32) -> Self {
            Self { host, payload }
        }

        fn host(&self) -> &LaneHost {
            &self.host
        }

        fn host_mut(&mut self) -> &mut LaneHost {
            &mut self.host
        }
    }

    fn epoch(tag: u64, generation: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: generation,
        }
    }

    /// Admission with a counting payload factory.
    fn admit(
        table: &mut LaneTable<TestCell>,
        epoch: RequestEpoch,
        prompt: &[u32],
        made: &mut usize,
    ) -> Result<usize, String> {
        table.admit(epoch, prompt.to_vec(), |slot, existing| match existing {
            Some(cell) => {
                cell.payload += 100;
                Ok(None)
            }
            None => {
                *made += 1;
                Ok(Some(slot as u32))
            }
        })
    }

    #[test]
    fn admission_takes_the_lowest_free_slot() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        assert_eq!(admit(&mut table, epoch(1, 1), &[7], &mut made), Ok(0));
        assert_eq!(admit(&mut table, epoch(2, 1), &[7], &mut made), Ok(1));
        assert_eq!(admit(&mut table, epoch(3, 1), &[7], &mut made), Ok(2));
        table.release(&epoch(2, 1)).unwrap();
        assert_eq!(admit(&mut table, epoch(4, 1), &[7], &mut made), Ok(1));
        assert_eq!(table.active(), 3);
        let host = &table.get(&epoch(4, 1)).unwrap().host;
        assert_eq!(host.status, LaneStatus::Prefilling);
        assert_eq!(
            (host.prefilled, host.generated.len(), host.pending_seed),
            (0, 0, None)
        );
        assert_eq!(host.stamp, 3);
        assert_eq!(table.epochs(), vec![epoch(1, 1), epoch(4, 1), epoch(3, 1)]);
    }

    #[test]
    fn admission_refusals() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(admit(&mut table, epoch(1, 0), &[7], &mut made)
            .unwrap_err()
            .contains("generation 0"));
        assert!(admit(&mut table, epoch(1, 1), &[], &mut made)
            .unwrap_err()
            .contains("empty prompt"));
        assert_eq!(made, 0);
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(1, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(3, 1), &[7], &mut made)
            .unwrap_err()
            .contains("busy"));
        assert_eq!(made, 2);
        assert_eq!(table.active(), 2);
    }

    /// Promoted admission with a counting payload factory.
    fn admit_decoding(
        table: &mut LaneTable<TestCell>,
        epoch: RequestEpoch,
        prompt: &[u32],
        made: &mut usize,
    ) -> Result<usize, String> {
        table.admit_decoding(
            epoch,
            prompt.to_vec(),
            vec![4, 5],
            6,
            |slot, existing| match existing {
                Some(cell) => {
                    cell.payload += 100;
                    Ok(None)
                }
                None => {
                    *made += 1;
                    Ok(Some(slot as u32))
                }
            },
        )
    }

    #[test]
    fn promoted_admission_publishes_a_decoding_lane() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert_eq!(
            admit_decoding(&mut table, epoch(1, 1), &[7, 8, 9], &mut made),
            Ok(0)
        );
        let host = &table.get(&epoch(1, 1)).unwrap().host;
        assert_eq!(host.status, LaneStatus::Decoding);
        assert_eq!(host.prompt, vec![7, 8, 9]);
        assert_eq!(host.prefilled, 3);
        assert_eq!(host.generated, vec![4, 5]);
        assert_eq!(host.pending_seed, Some(6));
        assert_eq!((host.slot, host.stamp), (0, 0));
        assert_eq!(table.active(), 1);
        // A plain admission still starts Prefilling next to it, in order.
        assert_eq!(admit(&mut table, epoch(2, 1), &[7], &mut made), Ok(1));
        assert_eq!(
            table.get(&epoch(2, 1)).unwrap().host.status,
            LaneStatus::Prefilling
        );
        assert_eq!(table.get(&epoch(2, 1)).unwrap().host.stamp, 1);
        // The slot's payload is kept across a release and reused by a
        // promoted admission without remaking it.
        table.release(&epoch(1, 1)).unwrap();
        assert_eq!(
            admit_decoding(&mut table, epoch(3, 1), &[9, 9], &mut made),
            Ok(0)
        );
        assert_eq!(made, 2);
        let cell = table.get(&epoch(3, 1)).unwrap();
        assert_eq!(cell.payload, 100);
        assert_eq!(cell.host.status, LaneStatus::Decoding);
        assert_eq!(cell.host.prefilled, 2);
        assert_eq!(cell.host.stamp, 2);
    }

    #[test]
    fn promoted_admission_refusals() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(admit_decoding(&mut table, epoch(1, 0), &[7], &mut made)
            .unwrap_err()
            .contains("generation 0"));
        assert!(admit_decoding(&mut table, epoch(1, 1), &[], &mut made)
            .unwrap_err()
            .contains("empty prompt"));
        assert_eq!((made, table.active(), table.next_stamp), (0, 0, 0));
        admit_decoding(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        assert!(admit_decoding(&mut table, epoch(1, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        // A live plain admission is a duplicate too.
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        assert!(admit_decoding(&mut table, epoch(2, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        assert!(admit_decoding(&mut table, epoch(3, 1), &[7], &mut made)
            .unwrap_err()
            .contains("busy"));
        assert_eq!((made, table.active(), table.next_stamp), (2, 2, 2));
        // A failed prepare leaves no half-published lane.
        table.release(&epoch(2, 1)).unwrap();
        let error = table
            .admit_decoding(epoch(4, 1), vec![7], vec![1], 2, |_, _| {
                Err("no memory".to_string())
            })
            .unwrap_err();
        assert_eq!(error, "no memory");
        assert!(table.get(&epoch(4, 1)).is_none());
        assert_eq!((table.active(), table.next_stamp), (1, 2));
    }

    #[test]
    fn a_failed_prepare_leaves_the_table_untouched() {
        let mut table = LaneTable::<TestCell>::new(1);
        let error = table
            .admit(epoch(1, 1), vec![7], |_, _| Err("no memory".to_string()))
            .unwrap_err();
        assert_eq!(error, "no memory");
        assert_eq!(table.active(), 0);
        assert_eq!(table.next_stamp, 0);
        assert!(table.get(&epoch(1, 1)).is_none());
    }

    #[test]
    fn release_keeps_the_payload_and_reuse_does_not_remake_it() {
        let mut table = LaneTable::<TestCell>::new(1);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7, 8], &mut made).unwrap();
        assert_eq!(table.get(&epoch(1, 1)).unwrap().payload, 0);
        table.release(&epoch(1, 1)).unwrap();
        assert_eq!(table.active(), 0);
        assert!(table.get(&epoch(1, 1)).is_none());
        admit(&mut table, epoch(2, 1), &[9], &mut made).unwrap();
        assert_eq!(made, 1);
        let cell = table.get(&epoch(2, 1)).unwrap();
        assert_eq!(cell.payload, 100);
        assert_eq!(cell.host.prompt, vec![9]);
        assert_eq!(cell.host.stamp, 1);
    }

    #[test]
    fn release_of_an_unknown_epoch_fails() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(table.release(&epoch(9, 1)).is_err());
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        table.release(&epoch(1, 1)).unwrap();
        assert!(table.release(&epoch(1, 1)).is_err());
    }

    #[test]
    fn a_new_generation_of_the_same_tag_is_admitted_after_release() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(5, 1), &[7], &mut made).unwrap();
        // Same tag, other generation: a different owner, not a duplicate.
        assert!(admit(&mut table, epoch(5, 2), &[7], &mut made).is_ok());
        table.release(&epoch(5, 1)).unwrap();
        // The stale generation matches nothing, the live one still does.
        assert!(table.get(&epoch(5, 1)).is_none());
        assert!(table.get(&epoch(5, 2)).is_some());
        assert!(table.release(&epoch(5, 1)).is_err());
        admit(&mut table, epoch(5, 3), &[7], &mut made).unwrap();
        assert!(table.get(&epoch(5, 3)).is_some());
        assert!(table.get(&epoch(5, 1)).is_none());
    }

    #[test]
    fn poison_marks_only_live_matching_lanes() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        table.poison(&epoch(1, 2));
        table.poison(&epoch(9, 1));
        assert_eq!(
            table.get(&epoch(1, 1)).unwrap().host.status,
            LaneStatus::Prefilling
        );
        table.poison(&epoch(1, 1));
        assert_eq!(
            table.get(&epoch(1, 1)).unwrap().host.status,
            LaneStatus::Poisoned
        );
        assert_eq!(
            table.get(&epoch(2, 1)).unwrap().host.status,
            LaneStatus::Prefilling
        );
    }

    #[test]
    fn force_seed_needs_a_live_decoding_lane() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        // Prefilling lane: refused.
        assert!(table.force_seed(&epoch(1, 1), 5).is_err());
        table.get_mut(&epoch(1, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.get_mut(&epoch(1, 1)).unwrap().host.generated.push(4);
        table.force_seed(&epoch(1, 1), 5).unwrap();
        let host = &table.get(&epoch(1, 1)).unwrap().host;
        assert_eq!(host.pending_seed, Some(5));
        assert_eq!(host.generated, vec![4]);
        // Poisoned lane and unknown epoch: refused.
        table.get_mut(&epoch(2, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.poison(&epoch(2, 1));
        assert!(table.force_seed(&epoch(2, 1), 5).is_err());
        assert!(table.force_seed(&epoch(9, 1), 5).is_err());
        assert!(table.force_seed(&epoch(1, 2), 5).is_err());
    }

    #[test]
    fn pick_mut_returns_disjoint_lanes_in_request_order() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        let picked = table.pick_mut(&[epoch(3, 1), epoch(1, 1)]).unwrap();
        assert_eq!(
            picked.iter().map(|c| c.host.slot).collect::<Vec<_>>(),
            vec![2, 0]
        );
        assert!(table.pick_mut(&[epoch(1, 1), epoch(1, 1)]).is_err());
        assert!(table.pick_mut(&[epoch(1, 9)]).is_err());
    }

    fn arena_layer(formats: &[ExpertFormat]) -> ExpertArenaLayer {
        let mut cursor = 0usize;
        let experts = formats
            .iter()
            .map(|&format| {
                let entry = ArenaExpert {
                    format,
                    gate_up_offset: cursor,
                    down_offset: cursor + format.gate_up_bytes(),
                };
                cursor += format.slab_bytes();
                entry
            })
            .collect();
        ExpertArenaLayer {
            file_offset: 0,
            bytes: cursor,
            experts,
        }
    }

    #[test]
    fn slab_layer_counts_e8p_in_a_mixed_mq4_e8p_group() {
        let mut formats = vec![ExpertFormat::Mq4Xts; 6];
        formats[1] = ExpertFormat::E8pI4;
        formats[4] = ExpertFormat::E8pI4;
        assert_eq!(
            slab_layer_cold_experts(&arena_layer(&formats), 6),
            Ok((2, 0))
        );
        assert_eq!(
            slab_layer_cold_experts(&arena_layer(&[ExpertFormat::Mq4Xts; 3]), 3),
            Ok((0, 0))
        );
        assert_eq!(
            slab_layer_cold_experts(&arena_layer(&[ExpertFormat::E8pI4; 3]), 3),
            Ok((3, 0))
        );
    }

    #[test]
    fn slab_layer_counts_mq3_in_a_mixed_mq4_mq3_group() {
        let mut formats = vec![ExpertFormat::Mq4Xts; 6];
        formats[0] = ExpertFormat::Mq3I4;
        formats[3] = ExpertFormat::Mq3I4;
        formats[5] = ExpertFormat::Mq3I4;
        assert_eq!(
            slab_layer_cold_experts(&arena_layer(&formats), 6),
            Ok((0, 3))
        );
        assert_eq!(
            slab_layer_cold_experts(&arena_layer(&[ExpertFormat::Mq3I4; 3]), 3),
            Ok((0, 3))
        );
    }

    #[test]
    fn slab_layer_refuses_w2_and_mixed_cold_groups() {
        let mut formats = vec![ExpertFormat::Mq4Xts; 4];
        formats[2] = ExpertFormat::Mq2I4;
        let error = slab_layer_cold_experts(&arena_layer(&formats), 4).unwrap_err();
        assert!(error.contains("expert 2"), "{error}");
        assert!(error.contains("Mq2I4"), "{error}");
        // E8P with a W3 expert in the same layer is refused too.
        let mixed = [
            ExpertFormat::E8pI4,
            ExpertFormat::Mq3I4,
            ExpertFormat::Mq4Xts,
        ];
        assert!(slab_layer_cold_experts(&arena_layer(&mixed), 3).is_err());
    }

    #[test]
    fn slab_layer_refuses_absent_experts_and_a_wrong_expert_count() {
        let mut layer = arena_layer(&[ExpertFormat::E8pI4; 4]);
        layer.experts[3].down_offset = layer.experts[3].gate_up_offset;
        let error = slab_layer_cold_experts(&layer, 4).unwrap_err();
        assert!(error.contains("expert 3 is absent"), "{error}");
        let layer = arena_layer(&[ExpertFormat::E8pI4; 4]);
        assert!(slab_layer_cold_experts(&layer, 5).is_err());
        assert!(slab_layer_cold_experts(&arena_layer(&[]), 0).is_err());
    }

    const MQ6: DType = DType::MQ6G256V2;

    fn hyper_facts(down: DType) -> HyperFacts {
        HyperFacts {
            norm: Some(DType::BF16),
            down: Some(down),
            up: Some(down),
        }
    }

    fn moe_ffn(qts: Option<Vec<ExpertQt>>) -> FfnFacts {
        FfnFacts::Moe(MoeFacts {
            dense: [Some(DType::BF16); 5],
            expert_qts: qts,
        })
    }

    fn layer_facts(mixer: MixerFacts, ffn: FfnFacts) -> LayerFacts {
        LayerFacts {
            attn_hyper: hyper_facts(DType::BF16),
            mlp_hyper: hyper_facts(DType::BF16),
            attn_inject: Some(DType::BF16),
            mlp_inject: Some(DType::BF16),
            mixer: Some(mixer),
            ffn,
        }
    }

    fn gdn() -> MixerFacts {
        MixerFacts::Gdn {
            input: [Some(MQ6); 4],
            output: Some(MQ6),
        }
    }

    fn qsa() -> MixerFacts {
        MixerFacts::Qsa {
            input: [Some(MQ6); 4],
            output: Some(MQ6),
        }
    }

    /// One GDN layer then one QSA layer on a whole-model band over resident
    /// packed MQ4-XTS experts.
    fn stage_facts() -> StageFacts {
        let qts = || Some(vec![expert_capabilities::QT_MQ4_XTS]);
        StageFacts {
            topology: LaneTopology::Single,
            input: true,
            output: true,
            embedding: Some(DType::Q8_0),
            ple_rows: StagedRows::Known(DType::BF16),
            experts: LaneExpertRoute::ResidentPacked,
            layers: vec![
                layer_facts(gdn(), moe_ffn(qts())),
                layer_facts(qsa(), moe_ffn(qts())),
            ],
            final_hyper: hyper_facts(DType::BF16),
            head: Some(MQ6),
        }
    }

    const ARCH: &str = "gfx1151";

    fn moe_stage(facts: &StageFacts) -> Vec<StageRoute> {
        derive_stage_routes(ARCH, facts)[StageName::Moe.index()].clone()
    }

    #[test]
    fn every_executed_stage_instance_yields_its_actual_route() {
        let routes = derive_stage_routes(ARCH, &stage_facts());
        let bf16 = DType::BF16;
        let stage = |name: StageName| routes[name.index()].clone();
        assert_eq!(
            stage(StageName::Embed),
            vec![StageRoute::Embed(DType::Q8_0)]
        );
        assert_eq!(stage(StageName::PleRows), vec![StageRoute::PleRows(bf16)]);
        assert_eq!(
            stage(StageName::HcRead),
            vec![StageRoute::HyperRead([bf16; 3])]
        );
        assert_eq!(
            stage(StageName::HcWrite),
            vec![StageRoute::HyperWrite(bf16)]
        );
        assert_eq!(
            stage(StageName::GdnInProj),
            vec![StageRoute::GdnIn([MQ6; 4])]
        );
        assert_eq!(
            stage(StageName::GdnOutProj),
            vec![StageRoute::Projection(MQ6)]
        );
        assert_eq!(
            stage(StageName::QsaInProj),
            vec![StageRoute::QsaIn([MQ6; 4])]
        );
        assert_eq!(
            stage(StageName::QsaOutProj),
            vec![StageRoute::Projection(MQ6)]
        );
        assert_eq!(
            stage(StageName::Moe),
            vec![StageRoute::Moe {
                experts: LaneExpertRoute::ResidentPacked,
                dense: [bf16; 5],
                decode_admitted: true,
            }]
        );
        assert_eq!(
            stage(StageName::FinalHyper),
            vec![StageRoute::HyperRead([bf16; 3])]
        );
        assert_eq!(stage(StageName::Head), vec![StageRoute::Projection(MQ6)]);
        assert_eq!(
            stage(StageName::Argmax),
            vec![StageRoute::Argmax(DType::F32)]
        );
        // Stateful stages carry no route.
        for name in StageName::ALL.into_iter().filter(|name| name.stateful()) {
            assert!(stage(name).is_empty(), "{name:?}");
        }
    }

    #[test]
    fn a_differing_dtype_isolates_only_its_own_stage() {
        let base = derive_stage_routes(ARCH, &stage_facts());
        let mut facts = stage_facts();
        let Some(MixerFacts::Qsa { input, .. }) = facts.layers[1].mixer.as_mut() else {
            panic!("layer 1 is QSA");
        };
        input[1] = Some(DType::Q8_0);
        let routes = derive_stage_routes(ARCH, &facts);
        for name in StageName::ALL {
            if name == StageName::QsaInProj {
                assert_eq!(
                    routes[name.index()],
                    vec![StageRoute::QsaIn([MQ6, DType::Q8_0, MQ6, MQ6])]
                );
            } else {
                assert_eq!(routes[name.index()], base[name.index()], "{name:?}");
            }
        }
        // A second layer with another dtype keeps both routes of the stage.
        let mut facts = stage_facts();
        let mut extra = layer_facts(gdn(), moe_ffn(Some(vec![expert_capabilities::QT_MQ4_XTS])));
        extra.attn_hyper = hyper_facts(DType::Q8_0);
        facts.layers.push(extra);
        let routes = derive_stage_routes(ARCH, &facts);
        assert_eq!(
            routes[StageName::HcRead.index()],
            vec![
                StageRoute::HyperRead([DType::BF16; 3]),
                StageRoute::HyperRead([DType::BF16, DType::Q8_0, DType::Q8_0]),
            ]
        );
        assert_eq!(
            routes[StageName::GdnInProj.index()],
            base[StageName::GdnInProj.index()]
        );
    }

    #[test]
    fn a_missing_weight_is_unclassified_for_its_stage_only() {
        let base = derive_stage_routes(ARCH, &stage_facts());
        type Mutation = Box<dyn Fn(&mut StageFacts)>;
        let cases: Vec<(StageName, Mutation)> = vec![
            (
                StageName::Embed,
                Box::new(|facts: &mut StageFacts| facts.embedding = None),
            ),
            (
                StageName::Head,
                Box::new(|facts: &mut StageFacts| facts.head = None),
            ),
            (
                StageName::GdnOutProj,
                Box::new(|facts: &mut StageFacts| {
                    if let Some(MixerFacts::Gdn { output, .. }) = facts.layers[0].mixer.as_mut() {
                        *output = None;
                    }
                }),
            ),
            (
                StageName::HcWrite,
                Box::new(|facts: &mut StageFacts| facts.layers[1].mlp_inject = None),
            ),
        ];
        for (stage, mutate) in cases {
            let mut facts = stage_facts();
            mutate(&mut facts);
            let routes = derive_stage_routes(ARCH, &facts);
            assert!(
                routes[stage.index()].contains(&StageRoute::Unclassified),
                "{stage:?}"
            );
            for name in StageName::ALL.into_iter().filter(|name| *name != stage) {
                assert_eq!(
                    routes[name.index()],
                    base[name.index()],
                    "{stage:?} {name:?}"
                );
            }
        }
        // An unresolved layer degrades the stages that layer feeds, not the
        // band's root stages.
        let mut facts = stage_facts();
        facts.layers[0] = LayerFacts {
            attn_hyper: HyperFacts::UNRESOLVED,
            mlp_hyper: HyperFacts::UNRESOLVED,
            attn_inject: None,
            mlp_inject: None,
            mixer: Some(MixerFacts::Gdn {
                input: [None; 4],
                output: None,
            }),
            ffn: FfnFacts::Missing,
        };
        let routes = derive_stage_routes(ARCH, &facts);
        for name in [
            StageName::HcRead,
            StageName::HcWrite,
            StageName::GdnInProj,
            StageName::GdnOutProj,
            StageName::Moe,
        ] {
            assert!(
                routes[name.index()].contains(&StageRoute::Unclassified),
                "{name:?}"
            );
        }
        for name in [
            StageName::Embed,
            StageName::PleRows,
            StageName::QsaInProj,
            StageName::QsaOutProj,
            StageName::FinalHyper,
            StageName::Head,
            StageName::Argmax,
        ] {
            assert_eq!(routes[name.index()], base[name.index()], "{name:?}");
        }
    }

    #[test]
    fn a_dense_ffn_or_missing_ple_provider_is_unclassified() {
        let mut facts = stage_facts();
        facts.layers[1].ffn = FfnFacts::Dense;
        assert!(moe_stage(&facts).contains(&StageRoute::Unclassified));
        let mut facts = stage_facts();
        facts.ple_rows = StagedRows::Unresolved;
        assert_eq!(
            derive_stage_routes(ARCH, &facts)[StageName::PleRows.index()],
            vec![StageRoute::Unclassified]
        );
        facts.ple_rows = StagedRows::Absent;
        assert!(derive_stage_routes(ARCH, &facts)[StageName::PleRows.index()].is_empty());
    }

    #[test]
    fn only_the_owner_bands_run_their_root_stages() {
        let mut facts = stage_facts();
        facts.input = false;
        facts.output = false;
        let routes = derive_stage_routes(ARCH, &facts);
        for name in [
            StageName::Embed,
            StageName::PleRows,
            StageName::FinalHyper,
            StageName::Head,
            StageName::Argmax,
        ] {
            assert!(routes[name.index()].is_empty(), "{name:?}");
        }
        assert!(!routes[StageName::HcRead.index()].is_empty());
        assert!(!routes[StageName::Moe.index()].is_empty());
        // A band without a GDN layer has no GDN stage instance.
        let mut facts = stage_facts();
        facts.layers.remove(0);
        let routes = derive_stage_routes(ARCH, &facts);
        assert!(routes[StageName::GdnInProj.index()].is_empty());
        assert!(routes[StageName::GdnOutProj.index()].is_empty());
        assert!(!routes[StageName::QsaInProj.index()].is_empty());
    }

    #[test]
    fn decode_admission_needs_every_present_expert_pair() {
        use expert_capabilities::{QT_E8P, QT_MQ4_XTS, QT_W2, QT_W3};
        let admitted = |arch: &str, qts: Option<Vec<ExpertQt>>| {
            let mut facts = stage_facts();
            facts.layers[0].ffn = moe_ffn(qts);
            let routes = derive_stage_routes(arch, &facts);
            match &routes[StageName::Moe.index()][0] {
                StageRoute::Moe {
                    decode_admitted, ..
                } => *decode_admitted,
                other => panic!("not a MoE route: {other:?}"),
            }
        };
        assert!(admitted(ARCH, Some(vec![QT_MQ4_XTS])));
        assert!(admitted(ARCH, Some(vec![QT_MQ4_XTS, QT_E8P])));
        // W2 has no kernel anywhere: one missing pair refuses the layer.
        assert!(!admitted(ARCH, Some(vec![QT_MQ4_XTS, QT_W2])));
        assert!(!admitted(ARCH, Some(vec![QT_W2])));
        // No W3 kernel on gfx1100; an unknown architecture serves nothing.
        assert!(!admitted("gfx1100", Some(vec![QT_W3])));
        assert!(!admitted("gfx0000", Some(vec![QT_MQ4_XTS])));
        // Unknown or empty expert storage is never admitted.
        assert!(!admitted(ARCH, None));
        assert!(!admitted(ARCH, Some(Vec::new())));
    }

    #[test]
    fn capability_never_certifies_the_residence_route() {
        for experts in [
            LaneExpertRoute::HostMapped,
            LaneExpertRoute::CpuDeferred,
            LaneExpertRoute::Split,
            LaneExpertRoute::Unclassified,
        ] {
            let mut facts = stage_facts();
            facts.experts = experts;
            let routes = derive_stage_routes(ARCH, &facts);
            // Only the MoE stage takes the class; the dense stages do not.
            assert_eq!(
                routes[StageName::Moe.index()],
                vec![StageRoute::Moe {
                    experts,
                    dense: [DType::BF16; 5],
                    decode_admitted: true,
                }]
            );
            assert_eq!(
                routes[StageName::HcRead.index()],
                vec![StageRoute::HyperRead([DType::BF16; 3])]
            );
        }
        // EP never carries an expert class.
        let mut facts = stage_facts();
        facts.topology = LaneTopology::Ep;
        assert_eq!(
            moe_stage(&facts),
            vec![StageRoute::Moe {
                experts: LaneExpertRoute::Unclassified,
                dense: [DType::BF16; 5],
                decode_admitted: true,
            }]
        );
    }

    #[test]
    fn ep_moe_stage_keeps_only_the_positive_ep_class() {
        let ep_facts = |experts: LaneExpertRoute| {
            let mut facts = stage_facts();
            facts.topology = LaneTopology::Ep;
            facts.experts = experts;
            facts
        };
        assert_eq!(
            moe_stage(&ep_facts(LaneExpertRoute::EpPackedStaticOrBalanced)),
            vec![StageRoute::Moe {
                experts: LaneExpertRoute::EpPackedStaticOrBalanced,
                dense: [DType::BF16; 5],
                decode_admitted: true,
            }]
        );
        // Every other class stays unproven under EP.
        for experts in [
            LaneExpertRoute::ResidentMq2Slab,
            LaneExpertRoute::ResidentMq3Slab,
            LaneExpertRoute::ResidentPacked,
            LaneExpertRoute::HostMapped,
            LaneExpertRoute::CpuDeferred,
            LaneExpertRoute::Split,
            LaneExpertRoute::Unclassified,
        ] {
            assert_eq!(
                moe_stage(&ep_facts(experts)),
                vec![StageRoute::Moe {
                    experts: LaneExpertRoute::Unclassified,
                    dense: [DType::BF16; 5],
                    decode_admitted: true,
                }],
                "{experts:?}"
            );
        }
        // The EP class never labels a non-EP topology.
        for topology in [LaneTopology::Single, LaneTopology::Pp] {
            let mut facts = stage_facts();
            facts.topology = topology;
            facts.experts = LaneExpertRoute::EpPackedStaticOrBalanced;
            assert_eq!(
                moe_stage(&facts),
                vec![StageRoute::Moe {
                    experts: LaneExpertRoute::Unclassified,
                    dense: [DType::BF16; 5],
                    decode_admitted: true,
                }],
                "{topology:?}"
            );
        }
    }

    /// One owned EP layer of the given residence, bound as its kind says.
    fn ep_layer(mapped: bool) -> Option<EpOwnedLayer> {
        Some(EpOwnedLayer {
            qt: Some(expert_capabilities::QT_MQ4_XTS),
            gate_up_mapped: mapped,
            down_mapped: mapped,
            kind: Some(if mapped {
                EpLaneExpertKind::BalancedHostMappedPacked
            } else {
                EpLaneExpertKind::StaticResidentPacked
            }),
        })
    }

    fn ep_band(resident: usize, mapped: usize) -> Vec<Option<EpOwnedLayer>> {
        std::iter::repeat_n(ep_layer(false), resident)
            .chain(std::iter::repeat_n(ep_layer(true), mapped))
            .collect()
    }

    #[test]
    fn ep_class_takes_any_mix_of_static_resident_and_balanced_mapped_layers() {
        // 27/21 and 28/20 trunk splits, every resident, every mapped, one
        // layer, and an interleaved order: each layer only needs one valid kind.
        for (resident, mapped) in [(27, 21), (28, 20), (48, 0), (0, 48), (1, 0), (0, 1), (5, 3)] {
            assert_eq!(
                classify_ep_layers(ep_band(resident, mapped)),
                LaneExpertRoute::EpPackedStaticOrBalanced,
                "{resident}/{mapped}"
            );
        }
        let interleaved: Vec<_> = (0..7).map(|layer| ep_layer(layer % 3 == 0)).collect();
        assert_eq!(
            classify_ep_layers(interleaved),
            LaneExpertRoute::EpPackedStaticOrBalanced
        );
    }

    #[test]
    fn ep_class_refuses_any_layer_that_is_not_positively_one_kind() {
        let unclassified = LaneExpertRoute::Unclassified;
        // An empty band owns no layer to classify.
        assert_eq!(classify_ep_layers(Vec::new()), unclassified);
        let spoil = |mutate: &dyn Fn(&mut EpOwnedLayer)| {
            let mut band = ep_band(27, 21);
            let mut layer = band[30].unwrap();
            mutate(&mut layer);
            band[30] = Some(layer);
            classify_ep_layers(band)
        };
        // A layer that is not an assembled MoE layer.
        let mut band = ep_band(27, 21);
        band[3] = None;
        assert_eq!(classify_ep_layers(band), unclassified);
        // Not the QT44/QT53 pair (other pair, or unresolved dtypes).
        assert_eq!(
            spoil(&|layer| layer.qt = Some(expert_capabilities::QT_E8P)),
            unclassified
        );
        assert_eq!(spoil(&|layer| layer.qt = None), unclassified);
        // Mixed residence between a layer's two tensors.
        assert_eq!(spoil(&|layer| layer.down_mapped = false), unclassified);
        assert_eq!(spoil(&|layer| layer.gate_up_mapped = false), unclassified);
        // The forward's kind is missing or disagrees with the mapped bit.
        assert_eq!(spoil(&|layer| layer.kind = None), unclassified);
        assert_eq!(
            spoil(&|layer| layer.kind = Some(EpLaneExpertKind::StaticResidentPacked)),
            unclassified
        );
        let mut band = ep_band(27, 21);
        band[0] = Some(EpOwnedLayer {
            kind: Some(EpLaneExpertKind::BalancedHostMappedPacked),
            ..band[0].unwrap()
        });
        assert_eq!(classify_ep_layers(band), unclassified);
    }

    #[test]
    fn capability_lookups_are_memoised_per_quant_pair() {
        use expert_capabilities::{QT_E8P, QT_MQ4_XTS};
        let facts = stage_facts();
        let moe = MoeFacts {
            dense: [Some(DType::BF16); 5],
            expert_qts: Some(vec![QT_MQ4_XTS, QT_E8P]),
        };
        let mut verdicts = Vec::new();
        for _ in 0..4 {
            moe_route(ARCH, &facts, &moe, &mut verdicts);
        }
        assert_eq!(verdicts.len(), 2);
    }

    #[test]
    fn slab_layer_expert_pairs_list_each_present_format_once() {
        use expert_capabilities::{QT_E8P, QT_MQ4_XTS, QT_W2, QT_W3};
        let formats = [
            ExpertFormat::E8pI4,
            ExpertFormat::Mq4Xts,
            ExpertFormat::E8pI4,
            ExpertFormat::Mq4Xts,
        ];
        assert_eq!(
            slab_layer_expert_qts(&arena_layer(&formats)),
            vec![QT_MQ4_XTS, QT_E8P]
        );
        // An absent (pruned) expert is not present, whatever its format.
        let mut layer = arena_layer(&[ExpertFormat::Mq3I4, ExpertFormat::Mq4Xts]);
        layer.experts[0].down_offset = layer.experts[0].gate_up_offset;
        assert_eq!(slab_layer_expert_qts(&layer), vec![QT_MQ4_XTS]);
        let layer = arena_layer(&[ExpertFormat::Mq2I4, ExpertFormat::Mq3I4]);
        assert_eq!(slab_layer_expert_qts(&layer), vec![QT_W3, QT_W2]);
        assert!(slab_layer_expert_qts(&arena_layer(&[])).is_empty());
    }

    #[test]
    fn the_stage_policy_log_names_proofs_and_per_lane_reasons() {
        let policy = LaneStagePolicy::production();
        assert_eq!(
            stage_policy_log(&policy, "", &[]),
            format!(
                "  qwen4 lanes: stage policy {} (proofs none; per-lane: none)",
                policy.describe()
            )
        );
        let reasons = [
            (StageName::Moe, "no proof".to_string()),
            (StageName::GdnCore, "stateful".to_string()),
        ];
        assert_eq!(
            stage_policy_log(&policy, "r1,r2", &reasons),
            format!(
                "  qwen4 lanes: stage policy {} (proofs r1,r2; per-lane: moe=no proof,gdn_core=stateful)",
                policy.describe()
            )
        );
    }
}

#[cfg(test)]
mod band_accounting_tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::state::scope_layer_counts;

    fn partition(config: &Qwen4Config, counts: &[usize]) -> Vec<Qwen4BandScope> {
        assert_eq!(counts.iter().sum::<usize>(), config.num_hidden_layers);
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                let scope = Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == config.num_hidden_layers,
                    band,
                };
                scope.validate(config).expect("valid band");
                scope
            })
            .collect()
    }

    #[test]
    fn band_store_bytes_sum_to_the_single_scope_store() {
        let config = compact_test_config();
        let context = Qwen4ContextCommit::legacy(1024);
        let format = Qwen4StateFormat::F32;
        let single = Qwen4BandScope::single(&config);
        for lanes in [1usize, 4, 8] {
            let whole = Qwen4LaneStore::device_bytes_scoped(
                &config, format, &context, lanes, lanes, None, &single,
            )
            .unwrap();
            assert_eq!(
                Qwen4LaneStore::device_bytes(&config, format, &context, lanes, lanes, None),
                Some(whole)
            );
            for counts in [
                &[24usize, 24][..],
                &[20, 28][..],
                &[16, 16, 16][..],
                &[9, 17, 22][..],
            ] {
                let scopes = partition(&config, counts);
                let bands: Vec<u64> = scopes
                    .iter()
                    .map(|scope| {
                        Qwen4LaneStore::device_bytes_scoped(
                            &config, format, &context, lanes, lanes, None, scope,
                        )
                        .unwrap()
                    })
                    .collect();
                // The step logits and top-1 are charged once, on the output
                // band; every lane's state is split with no duplication.
                assert_eq!(bands.iter().sum::<u64>(), whole, "{counts:?} lanes={lanes}");
                let head = (2 * lanes * (config.vocab_size * 4 + 4)) as u64;
                let output = scopes.iter().position(|scope| scope.output).unwrap();
                let states: u64 = scopes
                    .iter()
                    .map(|scope| {
                        lanes as u64
                            * Qwen4State::device_bytes_scoped(&config, format, &context, scope)
                                .unwrap() as u64
                    })
                    .sum();
                assert_eq!(states + head, whole);
                for (index, (&bytes, scope)) in bands.iter().zip(&scopes).enumerate() {
                    let own = lanes as u64
                        * Qwen4State::device_bytes_scoped(&config, format, &context, scope).unwrap()
                            as u64;
                    assert_eq!(bytes, own + if index == output { head } else { 0 });
                }
            }
        }
    }

    #[test]
    fn non_output_band_spec_bytes_are_target_only() {
        let config = compact_test_config();
        let context = Qwen4ContextCommit::legacy(1024);
        let format = Qwen4StateFormat::F32;
        let single = Qwen4BandScope::single(&config);
        assert!(Qwen4LaneStore::device_bytes_scoped(
            &config,
            format,
            &context,
            2,
            2,
            Some(3),
            &single
        )
        .is_some());
        let scopes = partition(&config, &[24, 24]);
        let bytes = |spec_k, scope: &Qwen4BandScope| {
            Qwen4LaneStore::device_bytes_scoped(&config, format, &context, 2, 2, spec_k, scope)
                .unwrap()
        };
        // The non-output band holds its lanes' capture rings and no head,
        // logits or wide hidden: exactly the AR store plus the rings.
        let (gdn_layers, _) = scope_layer_counts(&config, &scopes[0]);
        let ring =
            Qwen4State::row_capture_bytes_scoped(&config, format.gdn, 4, gdn_layers).unwrap();
        assert_eq!(
            bytes(Some(3), &scopes[0]),
            bytes(None, &scopes[0]) + 2 * ring as u64
        );
        // The MTP head, ring and wide buffers add more on the output band.
        assert!(bytes(Some(3), &scopes[1]) > bytes(None, &scopes[1]));
    }
}

#[cfg(test)]
mod lane_sampling_tests {
    use super::*;
    use crate::mtp_spec::accept_native_greedy;
    use hipfire_runtime::llama::{
        reset_cpu_sampler_rng, sample_top_k_p, sampler_rng_restore, sampler_rng_snapshot,
    };

    /// `llama::SAMPLER_STATE` is process-global; every test here that resets
    /// or draws from it holds this lock for its whole reset-draw-assert span
    /// (no other test of this crate touches the global sampler RNG).
    static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn rng_guard() -> std::sync::MutexGuard<'static, ()> {
        RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    const VOCAB: usize = 300;

    fn xorshift(state: &mut u32) -> u32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *state
    }

    /// `n` deterministic logits rows of `VOCAB`: a spread wide enough that the
    /// 20/64-candidate pools carry real mass, every third row quantized to
    /// half-integers (ties), every fifth row with a `-inf` entry.
    fn logit_rows(seed: u32, n: usize) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761) | 1;
        let mut out = Vec::with_capacity(n * VOCAB);
        for r in 0..n {
            for _ in 0..VOCAB {
                let v = (xorshift(&mut s) % 4000) as f32 / 1000.0;
                out.push(if r % 3 == 1 {
                    (v * 2.0).round() / 2.0
                } else {
                    v
                });
            }
            if r % 5 == 2 {
                out[r * VOCAB + 7] = f32::NEG_INFINITY;
            }
        }
        out
    }

    fn sampling_for(case: u32) -> LaneSampling {
        LaneSampling {
            temperature: [0.3f32, 0.7, 1.0][case as usize % 3],
            top_p: if case % 2 == 0 { 0.95 } else { 1.0 },
            top_k: [None, Some(20u32), Some(40)][(case as usize / 3) % 3],
            min_p: [None, Some(0.05f32)][(case as usize / 9) % 2],
            // Seed 0 is the `reset_cpu_sampler_rng` 0 -> 1 mapping.
            seed: case.wrapping_mul(7919),
        }
    }

    fn host_with(sampling: Option<LaneSampling>) -> LaneHost {
        LaneHost {
            epoch: RequestEpoch {
                request_tag: 1,
                owner_generation: 1,
            },
            slot: 0,
            prompt: vec![1],
            prefilled: 0,
            generated: Vec::new(),
            pending_seed: None,
            status: LaneStatus::Prefilling,
            stamp: 0,
            sampling,
            rng: sampling.map_or(0, |s| s.initial_rng()),
        }
    }

    fn row(logits: &[f32], i: usize) -> &[f32] {
        &logits[i * VOCAB..(i + 1) * VOCAB]
    }

    /// The singleton's draws over rows `0..count`: the global RNG reset to
    /// the request seed, then sequential `sample_top_k_p` (what
    /// `sample_cpu` calls with neutral penalties). Returns the tokens and the
    /// global state after the last draw.
    fn ar_draws(logits: &[f32], count: usize, s: &LaneSampling) -> (Vec<u32>, u32) {
        reset_cpu_sampler_rng(s.seed);
        let tokens = (0..count)
            .map(|i| sample_top_k_p(row(logits, i), s.temperature, s.top_p, s.top_k, s.min_p))
            .collect();
        (tokens, sampler_rng_snapshot())
    }

    #[test]
    fn sampled_verify_windows_reproduce_the_ar_stream() {
        let _g = rng_guard();
        let saved = sampler_rng_snapshot();
        let target = 40usize;
        let (mut windows, mut rejected, mut full_accepts, mut eos_stops) = (0, 0, 0, 0);
        for case in 0..26u32 {
            for depth in 1..=4usize {
                let s = sampling_for(case + depth as u32);
                let logits = logit_rows(case * 31 + depth as u32, target + 8);
                let (ar, _) = ar_draws(&logits, target + 8, &s);
                // Odd cases end at an EOS the stream actually draws.
                let eos = if case % 2 == 1 {
                    ar[9 + depth]
                } else {
                    u32::MAX
                };
                let mut host = host_with(Some(s));
                let mut scratch: Vec<f32> = Vec::new();
                let mut stream: Vec<u32> = Vec::new();
                let mut step = 0usize;
                while stream.len() < target && stream.last() != Some(&eos) {
                    let pos = stream.len();
                    // The window's drafts follow the AR stream, every third
                    // one corrupted so rejections and bonus rows both occur.
                    let drafts: Vec<u32> = (0..depth)
                        .map(|j| {
                            let t = ar[pos + j];
                            if (case as usize + step + j) % 3 == 0 {
                                (t + 1) % VOCAB as u32
                            } else {
                                t
                            }
                        })
                        .collect();
                    // Device argmax stand-ins: distinct from any draw.
                    let mut picks = vec![u32::MAX - 1; depth + 1];
                    let window = &logits[pos * VOCAB..(pos + depth + 1) * VOCAB];
                    let outcome = lane_sample_step(
                        &mut host,
                        Some(&drafts),
                        &mut picks,
                        eos,
                        VOCAB,
                        &mut scratch,
                        |scratch| {
                            scratch.clear();
                            scratch.extend_from_slice(window);
                            Ok(())
                        },
                    )
                    .unwrap();
                    let LaneSampleOutcome::Naive(naive) = outcome else {
                        panic!("a verify window is a naive outcome: {outcome:?}");
                    };
                    // The picks make the lane's greedy accept commit exactly
                    // the naive window.
                    assert_eq!(
                        accept_native_greedy(&drafts, &picks, Some(eos)).unwrap(),
                        naive
                    );
                    assert_eq!(&picks[..naive.committed.len()], &naive.committed[..]);
                    windows += 1;
                    rejected += usize::from(naive.accepted < depth);
                    full_accepts += usize::from(naive.accepted == depth);
                    stream.extend_from_slice(&naive.committed);
                    step += 1;
                }
                assert_eq!(
                    &stream[..],
                    &ar[..stream.len()],
                    "case {case} depth {depth}"
                );
                if stream.last() == Some(&eos) {
                    eos_stops += 1;
                    let first = ar.iter().position(|&t| t == eos).unwrap();
                    assert_eq!(first + 1, stream.len(), "stops at the first EOS draw");
                } else {
                    assert!(stream.len() >= target);
                }
                // Exactly one draw per emitted token: the lane RNG sits where
                // the singleton's global RNG does after as many AR draws.
                let (_, state) = ar_draws(&logits, stream.len(), &s);
                assert_eq!(host.rng, state, "case {case} depth {depth}");
            }
        }
        assert!(windows >= 26 * 4, "{windows} windows");
        assert!(rejected > 0 && full_accepts > 0 && eos_stops > 0);
        sampler_rng_restore(saved);
    }

    #[test]
    fn sampled_ar_rows_reproduce_the_ar_stream() {
        let _g = rng_guard();
        let saved = sampler_rng_snapshot();
        for case in 0..26u32 {
            let s = sampling_for(case);
            let count = 30;
            let logits = logit_rows(case + 1000, count);
            let (ar, state) = ar_draws(&logits, count, &s);
            let mut host = host_with(Some(s));
            let mut scratch: Vec<f32> = Vec::new();
            let mut got = Vec::new();
            for i in 0..count {
                let mut picks = [u32::MAX - 1];
                let outcome = lane_sample_step(
                    &mut host,
                    None,
                    &mut picks,
                    0,
                    VOCAB,
                    &mut scratch,
                    |scratch| {
                        scratch.clear();
                        scratch.extend_from_slice(row(&logits, i));
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(outcome, LaneSampleOutcome::Ar);
                got.push(picks[0]);
            }
            assert_eq!(got, ar, "case {case}");
            assert_eq!(host.rng, state, "case {case}");
        }
        sampler_rng_restore(saved);
    }

    #[test]
    fn a_lane_without_sampling_is_untouched() {
        let mut host = host_with(None);
        let mut scratch: Vec<f32> = Vec::new();
        for drafts in [None, Some(&[4u32, 5][..])] {
            let mut picks = vec![9u32, 8, 7][..drafts.map_or(1, |d| d.len() + 1)].to_vec();
            let before = picks.clone();
            let outcome = lane_sample_step(
                &mut host,
                drafts,
                &mut picks,
                3,
                VOCAB,
                &mut scratch,
                |_| panic!("a greedy lane never fetches logits"),
            )
            .unwrap();
            assert_eq!(outcome, LaneSampleOutcome::Greedy);
            assert_eq!(picks, before);
        }
        assert_eq!((host.rng, scratch.len()), (0, 0));
    }

    #[test]
    fn sample_step_refuses_a_row_count_mismatch() {
        let s = sampling_for(1);
        let mut host = host_with(Some(s));
        let mut scratch: Vec<f32> = Vec::new();
        let mut picks = [1u32, 2];
        let error = lane_sample_step(&mut host, None, &mut picks, 0, VOCAB, &mut scratch, |_| {
            panic!("refused before the fetch")
        })
        .unwrap_err();
        assert!(error.contains("2 picks for a 1-row"), "{error}");
        // Short logits are refused too.
        let mut picks = [1u32, 2];
        let error = lane_sample_step(
            &mut host,
            Some(&[5]),
            &mut picks,
            0,
            VOCAB,
            &mut scratch,
            |scratch| {
                scratch.clear();
                scratch.resize(VOCAB, 0.0);
                Ok(())
            },
        )
        .unwrap_err();
        assert!(error.contains("need"), "{error}");
    }

    #[test]
    fn sampling_validation_and_rng_seed() {
        let ok = sampling_for(4);
        assert!(ok.validate().is_ok());
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let s = LaneSampling {
                temperature: bad,
                ..ok
            };
            assert!(s.validate().is_err(), "temperature {bad}");
        }
        assert_eq!(LaneSampling { seed: 0, ..ok }.initial_rng(), 1);
        assert_eq!(LaneSampling { seed: 77, ..ok }.initial_rng(), 77);
    }

    struct Cell {
        host: LaneHost,
    }

    impl LaneCell for Cell {
        type Payload = ();

        fn assemble(host: LaneHost, _: ()) -> Self {
            Self { host }
        }

        fn host(&self) -> &LaneHost {
            &self.host
        }

        fn host_mut(&mut self) -> &mut LaneHost {
            &mut self.host
        }
    }

    #[test]
    fn a_reused_slot_never_inherits_the_sampler() {
        let epoch = |tag| RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        };
        let mut table = LaneTable::<Cell>::new(1);
        let prepare = |_: usize, existing: Option<&mut Cell>| match existing {
            Some(_) => Ok(None),
            None => Ok(Some(())),
        };
        table.admit(epoch(1), vec![7], prepare).unwrap();
        let host = table.get_mut(&epoch(1)).unwrap().host_mut();
        assert_eq!((host.sampling, host.rng), (None, 0));
        host.sampling = Some(sampling_for(5));
        host.rng = 99;
        table.release(&epoch(1)).unwrap();
        let released = table.cells[0].as_ref().unwrap().host();
        assert_eq!((released.sampling, released.rng), (None, 0));
        // A promoted admission into the reused slot is greedy too.
        table
            .admit_decoding(epoch(2), vec![7], vec![4], 5, prepare)
            .unwrap();
        let host = table.get(&epoch(2)).unwrap().host();
        assert_eq!((host.sampling, host.rng), (None, 0));
    }
}
