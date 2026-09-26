// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed Qwen4 manifest declarations and GPU-owned assembly.
//!
//! The names in this module are the names emitted by the pinned
//! `Qwen/Qwen3.8-Flash-Next` checkpoint.  In particular, the decoder uses the
//! `model.language_model.*` namespace, QSA keeps separate q/k/v projections,
//! and the PLE table is a set of numerically suffixed external rows.  The
//! runtime's [`WeightLoadTransaction`] remains the only allocation owner until
//! this module finalizes handles into [`Qwen4Weights`].

use crate::config::{LayerType, Qwen4Config};
#[cfg(test)]
use crate::config::{Qwen4MtpConfig, RecurrentStateDType, SourceDType};
use hipfire_runtime::external_rows::RowEncoding;
use hipfire_runtime::model_source::SourceRangeDescriptor;
#[cfg(test)]
use hipfire_runtime::weight_manifest::WeightResidency;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, ExpertSourceLayout, PinTarget, PlacementHint, ShardPolicy, StateEntry,
    StateKind, WeightEntry,
};
use hipfire_runtime::weight_store::{TakenWeight, WeightHandle, WeightLoadTransaction};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeSet;
use std::fmt;

/// The routed experts' declared targets: gate/up at the aligned-K group width
/// and down at the row-local one.
///
/// The E8 lattice format is deliberately *not* used here.  It rotates and
/// encodes in 256-wide FWHT segments, so `K % 256 == 0` is a precondition of
/// both the encoder and the decode kernels (`groups_per_row = K / 256`), and
/// `experts.down_proj` reduces over `moe_intermediate_size = 640` — three
/// quarters of a segment short.  The row-local MQ4G128V2 group exists for
/// exactly this case, and both tiers cost the same 4.25 bits per weight, so the
/// expert family keeps the pair the qwen4 typed route's kernels are built on.
pub const ROUTED_GATE_UP_DTYPE: DType = DType::MQ4G256V2;
pub const ROUTED_DOWN_DTYPE: DType = DType::MQ4G128V2;
pub const PLE_SHARD_ROWS: usize = 2_500_012;
pub use crate::ple::{PLE_HEAD_COUNT, PLE_ROW_WIDTH};
pub const PLE_SHARD_COUNT: usize = 128;

/// Packed formats this family's planner recognises.
///
/// A declaration outside this set is refused by name rather than demoted to
/// BF16, so the predicate is a recognition list, not the tier policy: the
/// policy lives in [`qwen4_matrix_dtype`].
fn qwen4_quantized_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::MQ4G256V2 | DType::MQ4G128V2 | DType::MQ6G256V2 | DType::MFP4G32E8SOA | DType::Q8_0
    )
}

/// Payloads that stay source-exact BF16 even though they are rank-2 matrices.
///
/// The Qwen4 family quantizes every matrix whose consumers read a quantized
/// weight — the trunk projections, the shared expert, the router, the PLE
/// key/value projections and the embedding/head — mirroring the canonical
/// `qwen3.6-35b-a3b` layout.  The MTP connector matrices are the exception:
/// their owner (`mtp_gpu`) still binds source BF16, so declaring them
/// quantized here would produce an artifact nothing can read.
const QWEN4_SOURCE_EXACT_SUFFIXES: &[&str] = &["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"];

/// Payload classes that stay source-exact BF16 despite being rank-2 matrices.
///
/// The hyper-connection mixer is four output rows over a `hc_count * hidden`
/// reduction — the thinnest matrix in the network against the widest activation
/// — so it keeps source bytes rather than being packed for a rounding error's
/// worth of traffic.
const QWEN4_SOURCE_EXACT_MARKERS: &[&str] = &["hyper_connection"];

/// The two classes the repo's MoE recipes hold at eight bits rather than four.
///
/// Every family's embedding arm ships Q8 (never MQ4), and the quantizer's MoE
/// default promotes the whole fixed tier — attention, lm_head, embed, router —
/// to Q8F16 (qt=3, 34 bytes per 32 weights) because it is quality-critical and
/// small relative to the routed experts.  `qwen3.6-35b-a3b.mq6` ships its
/// lm_head, embed and router at qt=3 while its experts carry the six-bit tier;
/// muse_glimmer's untied 202k-vocab head was forced to Q8 after an MQ4 build
/// shipped.  For Qwen4's untied 248k-vocab head that is 0.68 GB per token
/// instead of 1.27, at a tier the repo already trusts for this exact shape.
const QWEN4_Q8_MARKERS: &[&str] = &["lm_head.weight", "embed_tokens.weight"];

fn qwen4_q8_dtype(name: &str) -> bool {
    QWEN4_Q8_MARKERS.iter().any(|marker| name.ends_with(marker))
}

fn qwen4_quantizable_matrix(name: &str, shape: &[usize]) -> bool {
    if shape.len() == 3 {
        return name.ends_with(".experts.down_proj") || name.ends_with(".experts.gate_up_proj");
    }
    if shape.len() != 2
        || QWEN4_SOURCE_EXACT_SUFFIXES
            .iter()
            .any(|s| name.ends_with(s))
        || QWEN4_SOURCE_EXACT_MARKERS
            .iter()
            .any(|marker| name.contains(marker))
    {
        return false;
    }
    // The packed set is the wide attention/GDN projections, which dominate
    // decode weight traffic.  The shared expert, the router, and the PLE
    // projections stay source-exact: the shared expert is as wide as any
    // trunk matrix but rides the sealed-MoE route, and the router decides
    // expert selection from 512 logits.
    if !name.contains(".linear_attn.") && !name.contains(".self_attn.") {
        return false;
    }
    // The eight-bit block format carries any K that is a multiple of its
    // 32-weight block, so alignment beyond the admission boundary above is not
    // a packing precondition.  A K that is not a multiple of 128 stays BF16.
    shape.last().is_some_and(|k| k % 128 == 0)
}

/// The tier each packed class ships at.
///
/// Routed experts keep the four-bit MQ4 family: their error is averaged over the
/// ten experts a token selects, they are where nearly all of the weight traffic
/// is (512 experts x two matrices per layer), and the qwen4 typed route's
/// grouped kernels are built on this exact qt44/qt53 pair.  The rank-2 trunk
/// carries eight bits instead — a dense projection writes its error straight
/// into the residual stream, and this family has already measured what a
/// lossier tier does there: a four-bit trunk opened a reasoning block the model
/// could not close.  Q8F16 costs 8.5 bpw but needs no rotation basis at all, and
/// the trunk is a rounding error next to the experts.
fn qwen4_matrix_dtype(shape: &[usize]) -> DType {
    let k = shape.last().copied().unwrap_or_default();
    if shape.len() == 3 {
        // gate/up reduces over `hidden` (256-aligned) and takes the aligned-K
        // group; down reduces over `moe_intermediate_size` and takes the
        // row-local one.
        if k % 256 == 0 {
            ROUTED_GATE_UP_DTYPE
        } else {
            ROUTED_DOWN_DTYPE
        }
    } else {
        DType::Q8_0
    }
}

fn qwen4_target_dtype(name: &str, shape: &[usize], requested: DType) -> DType {
    if qwen4_q8_dtype(name) && matches!(requested, DType::BF16 | DType::Q8_0) {
        return DType::Q8_0;
    }
    if qwen4_quantizable_matrix(name, shape)
        && (requested == DType::BF16 || qwen4_quantized_dtype(requested))
    {
        qwen4_matrix_dtype(shape)
    } else {
        requested
    }
}

/// Q8 targets accept a BF16 source (the checkpoint) or an already-converted
/// Q8 payload, and nothing else: any other packed family is a different
/// geometry and would be misread here.
fn qwen4_q8_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0])
}

/// Routed experts accept the checkpoint's BF16 or the published four-bit
/// payload at either declared geometry.
fn qwen4_quant_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, ROUTED_GATE_UP_DTYPE, ROUTED_DOWN_DTYPE])
}

/// The trunk's wide attention/GDN projections are the one packed class whose
/// tier the *artifact* chooses rather than this build.
///
/// Both rungs describe the same logical matrix and load as the same resident
/// handle: the six-bit rung carries MQ6G256V2 (qt=47) and the eight-bit rung
/// carries Q8F16 (qt=3), which is the tier this build declares.  The loader
/// reads the tier the artifact actually ships and derives every row geometry
/// from that tag, so the source contract admits either payload.  Naming only
/// the tier this build happens to write would make a published rung of the
/// other tier unloadable by the very tree that produced it.
///
/// The set stays closed beyond that pair: four-bit families are a group width
/// the trunk's decode kernels do not read, and their admission would only move
/// the failure to the artifact boundary.
fn qwen4_trunk_matrix_source() -> DTypeConstraint {
    DTypeConstraint::source_from_sources(vec![DType::BF16, DType::Q8_0, DType::MQ6G256V2])
}

/// True for the rank-2 trunk attention/GDN matrices — the class
/// [`qwen4_trunk_matrix_source`] governs.
fn qwen4_trunk_matrix(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && qwen4_quantizable_matrix(name, shape)
}

/// The packed tier the rank-2 trunk matrices ship at.
///
/// `HIPFIRE_QWEN4_TRUNK_TIER=mq6` puts the wide attention/GDN projections on
/// the aligned-K six-bit group (MQ6G256V2, qt=47, 6.25 bpw) instead of the
/// eight-bit recipe this build declares by default (Q8F16, qt=3, 8.5 bpw).
/// Any other value — an unset variable included — selects that default, so the
/// knob only ever adds the six-bit rung and every recipe and artifact written
/// so far keeps its meaning.
///
/// The producer and the loader reach the declaration through the same
/// [`Qwen4Manifest::build`], and the trunk's source contract
/// ([`qwen4_trunk_matrix_source`]) admits both rungs, so each build loads the
/// other's artifact.  The MTP namespace declares its own targets through
/// [`qwen4_target_dtype`] and never reads this.
fn qwen4_trunk_tier(value: Option<&str>) -> DType {
    match value {
        Some("mq6") => DType::MQ6G256V2,
        _ => DType::Q8_0,
    }
}

/// `HIPFIRE_QWEN4_MTP_TIER=source` declares the rank-2 MTP matrices at their
/// source's own BF16 instead of the eight-bit recipe the shared matrix
/// classifier otherwise picks for them.
///
/// Scratch measurements found no acceptance lift from this tier alone; it
/// remains an opt-in comparison against the published Q8F16 MTP projections.
/// Unset, the existing recipe is unchanged, and the producer and loader use
/// the same manifest declaration.
fn qwen4_mtp_source_tier(value: Option<&str>) -> bool {
    matches!(value, Some("source"))
}

thread_local! {
    /// Test-only override so a manifest assertion can exercise the knob without
    /// a process-global environment write.  Production always reads the env.
    static MTP_SOURCE_TIER_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// The predicate the manifest actually consults, so a test can prove the knob
/// *reaches* the five rank-2 MTP attention matrices rather than only parsing.
fn mtp_source_tier_active() -> bool {
    MTP_SOURCE_TIER_OVERRIDE
        .with(|value| value.get())
        .unwrap_or_else(|| {
            qwen4_mtp_source_tier(std::env::var("HIPFIRE_QWEN4_MTP_TIER").ok().as_deref())
        })
}

/// The declared target of one entry, with the selected trunk tier applied.
///
/// Only the rank-2 trunk attention/GDN class moves, and only when the request
/// was packed in the first place: a request that [`qwen4_target_dtype`] leaves
/// unpacked (F32, or a class this family keeps source-exact) keeps its dtype.
fn qwen4_trunk_target(name: &str, shape: &[usize], requested: DType, trunk_tier: DType) -> DType {
    let dtype = qwen4_target_dtype(name, shape, requested);
    if dtype == DType::Q8_0 && qwen4_trunk_matrix(name, shape) {
        trunk_tier
    } else {
        dtype
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum TensorRole {
    TokenEmbedding,
    LanguageHead,
    HyperConnection,
    HyperConnectionNorm,
    HyperConnectionBlockInject,
    HyperConnectionDown,
    HyperConnectionUp,
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    GdnDtBias,
    GdnALog,
    GdnNorm,
    GdnConv,
    GdnOutput,
    QsaQ,
    QsaK,
    QsaV,
    QsaOutput,
    QsaQNorm,
    QsaKNorm,
    QsaIndexerQk,
    QsaIndexerQNorm,
    QsaIndexerKNorm,
    Router,
    SharedExpertGateScalar,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    RoutedGateUp,
    RoutedDown,
    PleShard,
    PleKey,
    PleValue,
    PleNorm,
    PleConv,
    PleMetadata,
    MtpEmbeddingProjection,
    MtpHiddenProjection,
    MtpNorm,
}

/// Typed logical reference.  The runtime manifest remains authoritative for
/// placement, aliases, source dtype, and residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorRef {
    pub fn new(
        name: impl Into<String>,
        role: TensorRole,
        layer: Option<usize>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
            return Err(WeightError::InvalidShape { name, shape });
        }
        let dtype = qwen4_target_dtype(&name, &shape, dtype);
        Ok(Self {
            name,
            role,
            layer,
            shape,
            dtype,
        })
    }
}

/// I64 metadata is retained as an exact source declaration rather than being
/// coerced into a floating-point `WeightEntry`.  The HFQM writer serializes
/// these arrays under its versioned `qwen4_ple` object; they never become GPU
/// resident weight handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
}

impl MetadataTensor {
    fn i64(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            source_dtype: "I64".to_string(),
        }
    }
}

/// Pure logical Qwen4 manifest.  No GPU, source, or allocator is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4Manifest {
    pub weights: Vec<WeightEntry>,
    pub state: Vec<StateEntry>,
    pub metadata: Vec<MetadataTensor>,
}

impl Qwen4Manifest {
    /// The declaration both the producer and the loader read.
    ///
    /// The trunk's packed tier comes from `HIPFIRE_QWEN4_TRUNK_TIER`; the
    /// source-tier experiment affects only rank-2 MTP records.
    pub fn build(config: &Qwen4Config) -> Result<Self, WeightError> {
        Self::build_with_trunk_tier(
            config,
            qwen4_trunk_tier(std::env::var("HIPFIRE_QWEN4_TRUNK_TIER").ok().as_deref()),
        )
    }

    /// [`Self::build`] against an explicit trunk tier, so the selector's effect
    /// on the declaration is testable without a process-global environment.
    fn build_with_trunk_tier(config: &Qwen4Config, trunk_tier: DType) -> Result<Self, WeightError> {
        config.validate().map_err(WeightError::Config)?;
        let bf16 = DTypeConstraint::source_exact(DType::BF16);
        // Converted HFQM records may already carry either MQ4G256V2 (qt=44)
        // or MQ4G128V2 (qt=53).  The source checkpoint itself remains BF16.
        let quant_matrix = qwen4_quant_source();
        let quant_q8 = qwen4_q8_source();
        let quant_trunk = qwen4_trunk_matrix_source();
        let model = |name: &str,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::model_with_dtype_constraint(name, shape, dtype, source.clone(), policy)
        };
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     requested_dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            let dtype = qwen4_trunk_target(name, &shape, requested_dtype, trunk_tier);
            let source = if qwen4_trunk_matrix(name, &shape) {
                &quant_trunk
            } else if dtype == DType::Q8_0 {
                &quant_q8
            } else if qwen4_quantized_dtype(dtype) {
                &quant_matrix
            } else {
                source
            };
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let hidden = config.hidden_size;
        let hc_wide = config
            .hc_count
            .checked_mul(hidden)
            .ok_or_else(|| WeightError::ShapeOverflow("hyper-connection width".to_string()))?;
        let hc_rank = config.hc_lowrank;
        let q_dim = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA q width".to_string()))?;
        let kv_dim = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA kv width".to_string()))?;
        let indexer_qk_dim = (config.indexer_n_heads + config.indexer_kv_heads)
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA indexer width".to_string()))?;
        let gdn_qk = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("GDN key width".to_string()))?;
        let gdn_v = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("GDN value width".to_string()))?;
        let gdn_qkv = 2usize
            .checked_mul(gdn_qk)
            .and_then(|v| v.checked_add(gdn_v))
            .ok_or_else(|| WeightError::ShapeOverflow("GDN qkv width".to_string()))?;
        let ple_channels = config
            .ple_embed_dim
            .checked_mul(config.hc_count)
            .ok_or_else(|| WeightError::ShapeOverflow("PLE channel width".to_string()))?;

        let mut weights = Vec::with_capacity(1_330);
        weights.push(
            model(
                "model.language_model.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Embed),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        weights.push(
            model(
                "lm_head.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        // MTP aliases exactly the input embedding.  The target has no MTP
        // lm_head tensor; the distinct trunk lm_head remains the output head.
        weights.push(
            model(
                "mtp.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Tied {
                    source: "model.language_model.embed_tokens.weight".to_string(),
                },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        // Final HC mixer is the only trunk-level post-layer transform.  Its
        // hc_norm is the final normalization; no invented RMSNorm is added.
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                vec![hc_rank, hc_wide],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                vec![hc_wide, hc_rank],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );

        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "attn_hyper_connection",
                hidden,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "mlp_hyper_connection",
                hidden,
                hc_wide,
                hc_rank,
                &bf16,
            );
            match kind {
                LayerType::LinearAttention => {
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.A_log"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        layer_idx,
                        vec![gdn_qkv, 1, config.linear_conv_kernel_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        layer_idx,
                        vec![gdn_qkv, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        layer_idx,
                        vec![gdn_v, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        layer_idx,
                        vec![config.linear_value_head_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        layer_idx,
                        vec![hidden, gdn_v],
                        DType::BF16,
                        ShardPolicy::RowShard { axis: 1 },
                        &bf16,
                    ));
                }
                LayerType::FullAttention => {
                    push_qsa_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        q_dim,
                        kv_dim,
                        indexer_qk_dim,
                        config,
                        &bf16,
                    );
                }
            }
            push_moe_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                hidden,
                config,
                &bf16,
                &quant_matrix,
            );
            if layer_idx == 1 {
                push_ple_entries(
                    &mut weights,
                    &layer,
                    &prefix,
                    layer_idx,
                    hidden,
                    ple_channels,
                    config,
                    &bf16,
                );
            }
        }

        // Native MTP has one full-attention/MoE/HC layer and no PLE.  Keep the
        // source's exact names and shapes; fc_hidden is [H,H] in this
        // checkpoint even though its consumer widens the activation.
        weights.push(mtp_model(
            "mtp.fc_embedding.weight",
            vec![hidden, hidden],
            DType::BF16,
            ShardPolicy::ColumnShard { axis: 0 },
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.fc_hidden.weight",
            vec![hidden, hidden],
            DType::BF16,
            ShardPolicy::ColumnShard { axis: 0 },
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.pre_fc_norm_embedding.weight",
            vec![hidden],
            DType::BF16,
            ShardPolicy::Replicate,
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.pre_fc_norm_hidden.weight",
            vec![hc_wide],
            DType::BF16,
            ShardPolicy::Replicate,
            &bf16,
        ));
        push_mtp_hyper_connection_entries(&mut weights, hc_wide, hc_rank, &bf16);
        push_hyper_connection_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            "attn_hyper_connection",
            hidden,
            hc_wide,
            hc_rank,
            &bf16,
        );
        push_hyper_connection_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            "mlp_hyper_connection",
            hidden,
            hc_wide,
            hc_rank,
            &bf16,
        );
        push_qsa_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
            &bf16,
        );
        push_moe_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            hidden,
            config,
            &bf16,
            &quant_matrix,
        );

        // These declarations preserve the exact I64 payload identities for
        // the qwen4_ple container.  They are deliberately not WeightEntry
        // values because no GPU tensor or float dtype may represent them.
        let metadata = vec![
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                vec![3],
            ),
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                vec![PLE_HEAD_COUNT],
            ),
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
                vec![PLE_HEAD_COUNT],
            ),
        ];

        let mut state = Vec::new();
        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            state.push(match kind {
                LayerType::LinearAttention => StateEntry::new(StateKind::Recurrent, layer_idx),
                LayerType::FullAttention => StateEntry::new(
                    StateKind::Kv {
                        quant: "f32-qsa-full-raw-pooled".into(),
                    },
                    layer_idx,
                ),
            });
            if kind == LayerType::LinearAttention {
                state.push(StateEntry::new(StateKind::Conv, layer_idx));
            }
        }
        // PLE is injected at zero-based layer 1 and retains nine rows of
        // depthwise-convolution history.  StateKind has no family-local name;
        // the architecture state owner keeps this entry distinct by type.
        state.push(StateEntry::new(StateKind::Conv, 1));
        Ok(Self {
            weights,
            state,
            metadata,
        })
    }

    pub fn entry(&self, name: &str, layer: Option<usize>) -> Option<&WeightEntry> {
        self.weights
            .iter()
            .find(|entry| entry.name == name && entry.layer == layer)
    }

    pub fn external_entries(&self) -> impl Iterator<Item = &WeightEntry> {
        self.weights
            .iter()
            .filter(|entry| entry.residency.is_external())
    }

    pub fn resident_keys(&self) -> impl Iterator<Item = (&str, Option<usize>)> {
        self.weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
            .map(|entry| (entry.name.as_str(), entry.layer))
    }
    /// Number of source tensor records represented by this text manifest.
    /// The only non-source record is the tied MTP embedding alias; the three
    /// exact I64 PLE metadata arrays are counted from `metadata`.
    pub fn source_tensor_count(&self) -> usize {
        let aliases = self
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        self.weights
            .len()
            .saturating_sub(aliases)
            .saturating_add(self.metadata.len())
    }
}

fn mtp_model(
    name: &str,
    shape: Vec<usize>,
    requested_dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
) -> WeightEntry {
    // The existing source-tier experiment changes only rank-2 MTP entries;
    // the routed experts are rank-3, and their writer has no BF16 arm.
    if mtp_source_tier_active() && shape.len() == 2 {
        return WeightEntry::model_with_dtype_constraint(
            name,
            shape,
            DType::BF16,
            DTypeConstraint::source_exact(DType::BF16),
            policy,
        );
    }
    let dtype = qwen4_target_dtype(name, &shape, requested_dtype);
    // Mirror the trunk's `model` closure: the wide attention/GDN projections
    // ship at whichever tier the artifact declares (MQ6G256V2 or Q8), and a Q8
    // target otherwise takes the Q8 source set (BF16 checkpoint or an
    // already-converted Q8 payload).  `qwen4_quantized_dtype` is true for Q8_0
    // as well, so testing it first would hand every eight-bit MTP projection
    // the expert constraint (BF16 | MQ4G256V2 | MQ4G128V2) — families that are
    // a different geometry and would be misread there.
    let source = if qwen4_trunk_matrix(name, &shape) {
        qwen4_trunk_matrix_source()
    } else if dtype == DType::Q8_0 {
        qwen4_q8_source()
    } else if qwen4_quantized_dtype(dtype) {
        qwen4_quant_source()
    } else {
        source.clone()
    };
    WeightEntry::model_with_dtype_constraint(name, shape, dtype, source, policy)
}

fn mtp_layer(
    name: &str,
    _layer: usize,
    shape: Vec<usize>,
    dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
) -> WeightEntry {
    // MTP names live in a separate model namespace, not the trunk's layer
    // placement scope.  Keep their manifest identity layerless.
    mtp_model(name, shape, dtype, policy, source)
}

fn push_hyper_connection_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    name: &str,
    hidden: usize,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let prefix = format!("{prefix}.{name}");
    weights.push(layer(
        &format!("{prefix}.block_inject_weight.weight"),
        layer_idx,
        vec![4, hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.hc_norm.weight"),
        layer_idx,
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_down.weight"),
        layer_idx,
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_up.weight"),
        layer_idx,
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    let _ = hidden;
}

fn push_mtp_hyper_connection_entries(
    weights: &mut Vec<WeightEntry>,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) {
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.hc_norm.weight",
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
}

fn push_qsa_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let push =
        |weights: &mut Vec<WeightEntry>, suffix: &str, shape: Vec<usize>, policy: ShardPolicy| {
            weights.push(layer(
                &format!("{prefix}.self_attn.{suffix}"),
                layer_idx,
                shape,
                DType::BF16,
                policy,
                bf16,
            ));
        };
    push(
        weights,
        "indexer.index_qk_proj.weight",
        vec![indexer_qk_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "indexer.k_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "indexer.q_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "o_proj.weight",
        vec![hidden, q_dim],
        ShardPolicy::RowShard { axis: 1 },
    );
    push(
        weights,
        "q_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "q_proj.weight",
        vec![2 * q_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "v_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
}

fn push_moe_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
    quant_matrix: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let moe = format!("{prefix}.mlp");
    weights.push(layer(
        &format!("{moe}.experts.down_proj"),
        layer_idx,
        vec![config.num_experts, hidden, config.moe_intermediate_size],
        ROUTED_DOWN_DTYPE,
        ShardPolicy::Replicate,
        quant_matrix,
    ));
    weights.push(layer(
        &format!("{moe}.experts.gate_up_proj"),
        layer_idx,
        vec![config.num_experts, 2 * config.moe_intermediate_size, hidden],
        ROUTED_GATE_UP_DTYPE,
        ShardPolicy::Replicate,
        quant_matrix,
    ));
    weights.push(layer(
        &format!("{moe}.gate.weight"),
        layer_idx,
        vec![config.num_experts, hidden],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.down_proj.weight"),
        layer_idx,
        vec![hidden, config.shared_expert_intermediate_size],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.gate_proj.weight"),
        layer_idx,
        vec![config.shared_expert_intermediate_size, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.up_proj.weight"),
        layer_idx,
        vec![config.shared_expert_intermediate_size, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert_gate.weight"),
        layer_idx,
        vec![1, hidden],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
}

fn push_ple_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ple_channels: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let ple = format!("{prefix}.ple");
    weights.push(layer(
        &format!("{ple}.conv1d.weight"),
        layer_idx,
        vec![ple_channels, 1, config.ple_conv_kernel_size],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.key_proj.weight"),
        layer_idx,
        vec![ple_channels, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_conv.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_key.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_query.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.value_proj.weight"),
        layer_idx,
        vec![hidden, hidden],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    // The PLE row tier is declared by the artifact, not by this manifest: a
    // sealed artifact written before the Q8F16 tier carries BF16 rows and must
    // stay loadable, so both tiers are admitted and the reader derives its page
    // and read geometry from whichever one the shard declares.
    let ple_rows = DTypeConstraint::source_from_sources(vec![DType::Q8_0, DType::BF16]);
    for shard in 0..PLE_SHARD_COUNT {
        let name = format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight");
        weights.push(
            layer(
                &name,
                layer_idx,
                vec![PLE_SHARD_ROWS, PLE_ROW_WIDTH],
                DType::Q8_0,
                ShardPolicy::Replicate,
                &ple_rows,
            )
            // Q8F16 is the current writer's tier: five `[f16 scale][32 x i8]`
            // blocks per 160-value row, 170 bytes instead of the 320 a BF16 row
            // costs. The declared stride here is the default; a sealed BF16
            // shard is accepted by the same constraint and validated against its
            // own dtype-derived stride.
            .external_rows(
                RowEncoding::Q8F16.encoded_row_bytes(PLE_ROW_WIDTH),
                ple_valid_rows_for_shard(shard),
            ),
        );
    }
}

/// GPU placement requested by the typed Qwen4 assembler.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qwen4Placement {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionReadWeights {
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionWeights {
    pub block_inject: TensorRef,
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct GdnWeights {
    pub a_log: TensorRef,
    pub dt_bias: TensorRef,
    pub in_proj_a: TensorRef,
    pub in_proj_b: TensorRef,
    pub qkv: TensorRef,
    pub z: TensorRef,
    pub norm: TensorRef,
    pub conv: TensorRef,
    pub output: TensorRef,
}

#[derive(Clone, Debug)]
pub struct QsaWeights {
    pub indexer_qk: TensorRef,
    pub indexer_k_norm: TensorRef,
    pub indexer_q_norm: TensorRef,
    pub k_norm: TensorRef,
    pub k: TensorRef,
    pub output: TensorRef,
    pub q_norm: TensorRef,
    pub q: TensorRef,
    pub v: TensorRef,
}

#[derive(Clone, Debug)]
pub struct PleWeights {
    pub conv: TensorRef,
    pub key: TensorRef,
    pub norm_conv: TensorRef,
    pub norm_key: TensorRef,
    pub norm_query: TensorRef,
    pub value: TensorRef,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub experts_down: TensorRef,
    pub experts_gate_up: TensorRef,
    /// The pinned checkpoint carries one stacked gate/up source.  Gate and up
    /// remain distinct logical projection slices in the sealed runtime plan;
    /// do not rewrite the artifact merely to spell them as separate sources.
    pub expert_source_layout: ExpertSourceLayout,
    pub gate: TensorRef,
    pub shared_down: TensorRef,
    pub shared_gate: TensorRef,
    pub shared_up: TensorRef,
    pub shared_gate_scalar: TensorRef,
}

#[derive(Clone, Debug)]
pub struct Qwen4LayerWeights {
    pub layer: usize,
    pub kind: LayerType,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: Option<QsaWeights>,
    pub gdn: Option<GdnWeights>,
    pub ple: Option<PleWeights>,
    pub moe: MoeWeights,
}

#[derive(Clone, Debug)]
pub struct Qwen4MtpWeights {
    pub fc_embedding: TensorRef,
    pub fc_hidden: TensorRef,
    pub pre_fc_norm_embedding: TensorRef,
    pub pre_fc_norm_hidden: TensorRef,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: QsaWeights,
    pub moe: MoeWeights,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Typed trunk-level roots required by the native forward path.
///
/// The references are immutable logical identities.  Their resident buffers
/// are resolved through [`Qwen4Weights::resident`] so aliases remain owned by
/// the canonical load transaction and the forward path never performs source
/// I/O or an ad-hoc upload.
#[derive(Clone, Debug)]
pub struct Qwen4RootWeights {
    pub embedding: TensorRef,
    pub lm_head: TensorRef,
    pub final_hyper: HyperConnectionReadWeights,
}

/// Published resident ownership.  External PLE descriptors remain in the
/// attached transaction census; no fake handle is stored here.
pub struct Qwen4Weights {
    pub manifest: Qwen4Manifest,
    pub taken: Vec<TakenWeight>,
    pub root: Qwen4RootWeights,
    pub layer_refs: Vec<Qwen4LayerWeights>,
    pub mtp: Qwen4MtpWeights,
}

impl Qwen4Weights {
    pub fn assemble(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        placements: &[Qwen4Placement],
    ) -> Result<Self, WeightError> {
        let manifest = Qwen4Manifest::build(config)?;
        let expected = manifest
            .resident_keys()
            .map(|(name, layer)| (name.to_string(), layer))
            .collect::<BTreeSet<_>>();
        let provided = placements
            .iter()
            .map(|placement| (placement.name.clone(), placement.layer))
            .collect::<BTreeSet<_>>();
        if expected != provided {
            return Err(WeightError::PlacementSetMismatch {
                expected: expected.len(),
                actual: provided.len(),
            });
        }
        let mut assembly = tx.begin_assembly();
        for placement in placements {
            if assembly
                .take(&placement.name, placement.layer, placement.device)
                .is_none()
            {
                return Err(WeightError::MissingPlacement {
                    name: placement.name.clone(),
                    layer: placement.layer,
                    device: placement.device,
                });
            }
        }
        let taken = assembly.commit().finalize();
        let root = build_root_refs(config)?;
        let layer_refs = build_layer_refs(config)?;
        let mtp = build_mtp_refs(config)?;
        Ok(Self {
            manifest,
            taken,
            root,
            layer_refs,
            mtp,
        })
    }

    /// Resolve one immutable typed reference to its resident GPU buffer.
    ///
    /// Qwen4 is currently admitted only on the Single route, so device zero is
    /// the sole logical placement.  Tied weights are followed by their
    /// canonical source name and never trigger I/O or allocation.
    pub fn resident(&self, reference: &TensorRef) -> Result<&GpuTensor, WeightError> {
        self.resident_named(&reference.name, reference.layer, 0, 0)
    }

    fn resident_named(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
        depth: usize,
    ) -> Result<&GpuTensor, WeightError> {
        if depth > self.taken.len() {
            return Err(WeightError::DescriptorMismatch(format!(
                "{name}: resident alias cycle"
            )));
        }
        let Some(taken) = self.taken.iter().find(|taken| {
            taken.key.name == name && taken.key.layer == layer && taken.key.device == device
        }) else {
            return Err(WeightError::MissingResident {
                name: name.to_string(),
                layer,
                device,
            });
        };
        match &taken.handle {
            WeightHandle::Resident(tensor) => Ok(tensor),
            WeightHandle::Alias(source) => {
                let source = source.clone();
                self.resident_named(&source, layer, device, depth + 1)
            }
        }
    }

    pub fn external_descriptor<'a>(
        tx: &'a WeightLoadTransaction,
        placement: &Qwen4Placement,
    ) -> Option<&'a SourceRangeDescriptor> {
        tx.external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Explicit GPU teardown for handles finalized from the canonical
    /// transaction.  The attached transaction census is drained separately,
    /// after readers and these resident handles are gone.
    pub fn free_gpu(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        let mut first = None;
        for taken in self.taken {
            if let WeightHandle::Resident(tensor) = taken.handle {
                if let Err(error) = gpu.free_tensor(tensor) {
                    if first.is_none() {
                        first = Some(error);
                    }
                }
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn build_root_refs(config: &Qwen4Config) -> Result<Qwen4RootWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let embedding = TensorRef::new(
        "model.language_model.embed_tokens.weight",
        TensorRole::TokenEmbedding,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    let lm_head = TensorRef::new(
        "lm_head.weight",
        TensorRole::LanguageHead,
        None,
        vec![config.vocab_size, hidden],
        DType::BF16,
    )?;
    Ok(Qwen4RootWeights {
        embedding,
        lm_head,
        final_hyper: build_hyper_read_refs(
            "model.language_model.hyper_connection_mixer",
            None,
            hc_wide,
            config.hc_lowrank,
        )?,
    })
}

fn build_hyper_read_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionReadWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(HyperConnectionReadWeights {
        hc_norm: tr(
            "hc_norm.weight",
            TensorRole::HyperConnectionNorm,
            vec![hc_wide],
        )?,
        input_mix_down: tr(
            "input_mix_weight_down.weight",
            TensorRole::HyperConnectionDown,
            vec![hc_rank, hc_wide],
        )?,
        input_mix_up: tr(
            "input_mix_weight_up.weight",
            TensorRole::HyperConnectionUp,
            vec![hc_wide, hc_rank],
        )?,
    })
}

fn build_hyper_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionWeights, WeightError> {
    let read = build_hyper_read_refs(prefix, layer, hc_wide, hc_rank)?;
    let block_inject = TensorRef::new(
        format!("{prefix}.block_inject_weight.weight"),
        TensorRole::HyperConnectionBlockInject,
        layer,
        vec![4, hc_wide],
        DType::BF16,
    )?;
    Ok(HyperConnectionWeights {
        block_inject,
        hc_norm: read.hc_norm,
        input_mix_down: read.input_mix_down,
        input_mix_up: read.input_mix_up,
    })
}

fn build_layer_refs(config: &Qwen4Config) -> Result<Vec<Qwen4LayerWeights>, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let gdn_qk = config.linear_num_key_heads * config.linear_key_head_dim;
    let gdn_v = config.linear_num_value_heads * config.linear_value_head_dim;
    let gdn_qkv = 2 * gdn_qk + gdn_v;
    let ple_channels = config.ple_embed_dim * config.hc_count;
    let mut refs = Vec::with_capacity(config.num_hidden_layers);
    for (layer, kind) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        let attn_hyper = build_hyper_refs(
            &format!("{prefix}.attn_hyper_connection"),
            Some(layer),
            hc_wide,
            hc_rank,
        )?;
        let mlp_hyper = build_hyper_refs(
            &format!("{prefix}.mlp_hyper_connection"),
            Some(layer),
            hc_wide,
            hc_rank,
        )?;
        let attention = if kind == LayerType::FullAttention {
            Some(build_qsa_refs(
                &format!("{prefix}.self_attn"),
                Some(layer),
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
            )?)
        } else {
            None
        };
        let gdn = if kind == LayerType::LinearAttention {
            Some(build_gdn_refs(
                &format!("{prefix}.linear_attn"),
                Some(layer),
                hidden,
                gdn_qkv,
                gdn_v,
                config,
            )?)
        } else {
            None
        };
        let ple = if layer == 1 {
            Some(build_ple_refs(
                &format!("{prefix}.ple"),
                Some(layer),
                hidden,
                ple_channels,
                config,
            )?)
        } else {
            None
        };
        let moe = build_moe_refs(&format!("{prefix}.mlp"), Some(layer), hidden, config)?;
        refs.push(Qwen4LayerWeights {
            layer,
            kind,
            attn_hyper,
            mlp_hyper,
            attention,
            gdn,
            ple,
            moe,
        });
    }
    Ok(refs)
}

fn build_mtp_refs(config: &Qwen4Config) -> Result<Qwen4MtpWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    Ok(Qwen4MtpWeights {
        fc_embedding: TensorRef::new(
            "mtp.fc_embedding.weight",
            TensorRole::MtpEmbeddingProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        fc_hidden: TensorRef::new(
            "mtp.fc_hidden.weight",
            TensorRole::MtpHiddenProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        pre_fc_norm_embedding: TensorRef::new(
            "mtp.pre_fc_norm_embedding.weight",
            TensorRole::MtpNorm,
            None,
            vec![hidden],
            DType::BF16,
        )?,
        pre_fc_norm_hidden: TensorRef::new(
            "mtp.pre_fc_norm_hidden.weight",
            TensorRole::MtpNorm,
            None,
            vec![hc_wide],
            DType::BF16,
        )?,
        attn_hyper: build_hyper_refs("mtp.layers.0.attn_hyper_connection", None, hc_wide, hc_rank)?,
        mlp_hyper: build_hyper_refs("mtp.layers.0.mlp_hyper_connection", None, hc_wide, hc_rank)?,
        attention: build_qsa_refs(
            "mtp.layers.0.self_attn",
            None,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
        )?,
        moe: build_moe_refs("mtp.layers.0.mlp", None, hidden, config)?,
        final_hyper: build_hyper_read_refs("mtp.hyper_connection_mixer", None, hc_wide, hc_rank)?,
    })
}

fn build_gdn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    qkv: usize,
    value: usize,
    config: &Qwen4Config,
) -> Result<GdnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(GdnWeights {
        a_log: tr(
            "A_log",
            TensorRole::GdnALog,
            vec![config.linear_num_value_heads],
        )?,
        dt_bias: tr(
            "dt_bias",
            TensorRole::GdnDtBias,
            vec![config.linear_num_value_heads],
        )?,
        in_proj_a: tr(
            "in_proj_a.weight",
            TensorRole::GdnA,
            vec![config.linear_num_value_heads, hidden],
        )?,
        in_proj_b: tr(
            "in_proj_b.weight",
            TensorRole::GdnB,
            vec![config.linear_num_value_heads, hidden],
        )?,
        qkv: tr("in_proj_qkv.weight", TensorRole::GdnQkv, vec![qkv, hidden])?,
        z: tr("in_proj_z.weight", TensorRole::GdnZ, vec![value, hidden])?,
        norm: tr(
            "norm.weight",
            TensorRole::GdnNorm,
            vec![config.linear_value_head_dim],
        )?,
        conv: tr(
            "conv1d.weight",
            TensorRole::GdnConv,
            vec![qkv, 1, config.linear_conv_kernel_dim],
        )?,
        output: tr(
            "out_proj.weight",
            TensorRole::GdnOutput,
            vec![hidden, value],
        )?,
    })
}

fn build_qsa_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
) -> Result<QsaWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(QsaWeights {
        indexer_qk: tr(
            "indexer.index_qk_proj.weight",
            TensorRole::QsaIndexerQk,
            vec![indexer_qk_dim, hidden],
        )?,
        indexer_k_norm: tr(
            "indexer.k_layernorm.weight",
            TensorRole::QsaIndexerKNorm,
            vec![config.indexer_head_dim],
        )?,
        indexer_q_norm: tr(
            "indexer.q_layernorm.weight",
            TensorRole::QsaIndexerQNorm,
            vec![config.indexer_head_dim],
        )?,
        k_norm: tr("k_norm.weight", TensorRole::QsaKNorm, vec![config.head_dim])?,
        k: tr("k_proj.weight", TensorRole::QsaK, vec![kv_dim, hidden])?,
        output: tr("o_proj.weight", TensorRole::QsaOutput, vec![hidden, q_dim])?,
        q_norm: tr("q_norm.weight", TensorRole::QsaQNorm, vec![config.head_dim])?,
        q: tr("q_proj.weight", TensorRole::QsaQ, vec![2 * q_dim, hidden])?,
        v: tr("v_proj.weight", TensorRole::QsaV, vec![kv_dim, hidden])?,
    })
}

fn build_ple_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    channels: usize,
    config: &Qwen4Config,
) -> Result<PleWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(PleWeights {
        conv: tr(
            "conv1d.weight",
            TensorRole::PleConv,
            vec![channels, 1, config.ple_conv_kernel_size],
        )?,
        key: tr(
            "key_proj.weight",
            TensorRole::PleKey,
            vec![channels, hidden],
        )?,
        norm_conv: tr("norm_conv.weight", TensorRole::PleNorm, vec![channels])?,
        norm_key: tr("norm_key.weight", TensorRole::PleNorm, vec![channels])?,
        norm_query: tr("norm_query.weight", TensorRole::PleNorm, vec![channels])?,
        value: tr(
            "value_proj.weight",
            TensorRole::PleValue,
            vec![hidden, hidden],
        )?,
    })
}

fn build_moe_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    config: &Qwen4Config,
) -> Result<MoeWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>, dtype: DType| {
        TensorRef::new(format!("{prefix}.{suffix}"), role, layer, shape, dtype)
    };
    Ok(MoeWeights {
        experts_down: tr(
            "experts.down_proj",
            TensorRole::RoutedDown,
            vec![config.num_experts, hidden, config.moe_intermediate_size],
            ROUTED_DOWN_DTYPE,
        )?,
        experts_gate_up: tr(
            "experts.gate_up_proj",
            TensorRole::RoutedGateUp,
            vec![config.num_experts, 2 * config.moe_intermediate_size, hidden],
            ROUTED_GATE_UP_DTYPE,
        )?,
        expert_source_layout: ExpertSourceLayout::PackedFused {
            gate_up: format!("{prefix}.experts.gate_up_proj"),
            down: format!("{prefix}.experts.down_proj"),
            sidecars: Vec::new(),
        },
        gate: tr(
            "gate.weight",
            TensorRole::Router,
            vec![config.num_experts, hidden],
            DType::BF16,
        )?,
        shared_down: tr(
            "shared_expert.down_proj.weight",
            TensorRole::SharedExpertDown,
            vec![hidden, config.shared_expert_intermediate_size],
            DType::BF16,
        )?,
        shared_gate: tr(
            "shared_expert.gate_proj.weight",
            TensorRole::SharedExpertGate,
            vec![config.shared_expert_intermediate_size, hidden],
            DType::BF16,
        )?,
        shared_up: tr(
            "shared_expert.up_proj.weight",
            TensorRole::SharedExpertUp,
            vec![config.shared_expert_intermediate_size, hidden],
            DType::BF16,
        )?,
        shared_gate_scalar: tr(
            "shared_expert_gate.weight",
            TensorRole::SharedExpertGateScalar,
            vec![1, hidden],
            DType::BF16,
        )?,
    })
}

pub fn ple_valid_rows_for_shard(shard: usize) -> usize {
    crate::ple::PLE_VALID_ROWS
        .saturating_sub((shard as u64).saturating_mul(PLE_SHARD_ROWS as u64))
        .min(PLE_SHARD_ROWS as u64) as usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    Config(String),
    InvalidShape {
        name: String,
        shape: Vec<usize>,
    },
    ShapeOverflow(String),
    DescriptorMismatch(String),
    PlacementSetMismatch {
        expected: usize,
        actual: usize,
    },
    MissingPlacement {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingExternalDescriptor {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingResident {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    PleShardCount {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 manifest config: {message}"),
            Self::InvalidShape { name, shape } => write!(f, "{name}: invalid shape {shape:?}"),
            Self::ShapeOverflow(name) => write!(f, "{name}: shape overflow"),
            Self::DescriptorMismatch(name) => {
                write!(f, "{name}: external descriptor mismatch")
            }
            Self::PlacementSetMismatch { expected, actual } => {
                write!(f, "Qwen4 placement set expected {expected}, got {actual}")
            }
            Self::MissingPlacement {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident placement {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingExternalDescriptor {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing external descriptor {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingResident {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident tensor {name}[layer {layer:?}] on device {device}"
            ),
            Self::PleShardCount { expected, actual } => {
                write!(f, "Qwen4 PLE shard count expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ple_shards_exclude_padding() {
        let total: usize = (0..PLE_SHARD_COUNT).map(ple_valid_rows_for_shard).sum();
        assert_eq!(total, 320_001_446);
        assert_eq!(ple_valid_rows_for_shard(127), 2_499_922);
    }
    fn pinned_config() -> Qwen4Config {
        Qwen4Config {
            architecture_id: 16,
            model_type: "qwen4_exp".into(),
            text_model_type: "qwen4_exp_text".into(),
            dtype: SourceDType::BF16,
            hidden_size: 2560,
            vocab_size: 248320,
            num_hidden_layers: 48,
            max_position_embeddings: 262144,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000_000.0,
            attention_bias: false,
            layer_types: (0..48)
                .map(|layer| {
                    if layer % 4 == 3 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect(),
            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            recurrent_state_dtype: RecurrentStateDType::F32,
            indexer_n_heads: 4,
            indexer_kv_heads: 1,
            indexer_head_dim: 128,
            indexer_budget: 2048,
            indexer_compress_ratio: 4,
            num_experts: 512,
            num_experts_per_tok: 10,
            moe_intermediate_size: 640,
            shared_expert_intermediate_size: 640,
            norm_topk_prob: true,
            output_gate_type: "sigmoid".into(),
            hc_count: 4,
            hc_lowrank: 320,
            ple_layer_ids: vec![2],
            ple_conv_kernel_size: 4,
            ple_embed_dim: 2560,
            split_ngram_parts: 128,
            heads_per_ngram: 8,
            ngram_size: 3,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            eos_token_id: 248044,
            tie_word_embeddings: false,
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            mtp: Qwen4MtpConfig {
                num_hidden_layers: 1,
                layer_types: vec![LayerType::FullAttention],
                hybrid: true,
                rope_theta: 10_000_000.0,
            },
        }
    }

    #[test]
    fn manifest_matches_pinned_text_inventory_and_rejects_legacy_names() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        // The source has 1,658 records, of which 333 are vision records.  The
        // text manifest accounts for all 1,325 remaining records; its one MTP
        // embedding alias is excluded by source_tensor_count and its three
        // exact I64 PLE arrays are included through `metadata`.
        assert_eq!(manifest.source_tensor_count(), 1325);
        assert_eq!(manifest.source_tensor_count() + 333, 1658);
        let expect = |name: &str, layer: Option<usize>, shape: &[usize]| {
            let entry = manifest
                .entry(name, layer)
                .unwrap_or_else(|| panic!("missing exact inventory tensor {name} layer {layer:?}"));
            assert_eq!(entry.logical_shape, shape, "{name} shape");
        };
        expect(
            "model.language_model.layers.0.linear_attn.A_log",
            Some(0),
            &[48],
        );
        expect(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            Some(0),
            &[10240, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            Some(3),
            &[640, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Some(3),
            &[12288, 2560],
        );
        expect(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            Some(1),
            &[PLE_SHARD_ROWS, PLE_ROW_WIDTH],
        );
        expect("mtp.fc_hidden.weight", None, &[2560, 2560]);
        expect("mtp.layers.0.self_attn.q_proj.weight", None, &[12288, 2560]);
        expect(
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            None,
            &[10240],
        );
        assert!(manifest
            .entry("model.layers.0.linear_attn.in_proj_qkv.weight", Some(0))
            .is_none());
        assert!(manifest
            .entry(
                "model.language_model.layers.0.self_attn.qkv_proj.weight",
                Some(0)
            )
            .is_none());
        assert!(manifest
            .entry("model.language_model.layers.0.mlp.router.weight", Some(0))
            .is_none());
        assert!(manifest.metadata.iter().any(|tensor| {
            tensor.name == "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
                && tensor.source_dtype == "I64"
                && tensor.shape == [3]
        }));
        assert!(matches!(
            manifest
                .entry(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                    Some(1)
                )
                .expect("PLE shard"),
            WeightEntry {
                residency: WeightResidency::ExternalRows { .. },
                ..
            }
        ));
    }
    #[test]
    fn final_hyper_manifest_has_read_operands_only() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        assert!(
            manifest
                .entry(
                    "model.language_model.hyper_connection_mixer.block_inject_weight.weight",
                    None,
                )
                .is_none(),
            "final HC must not invent a write operand"
        );
        let root = build_root_refs(&config).expect("root references");
        assert_eq!(
            root.final_hyper.hc_norm.name,
            "model.language_model.hyper_connection_mixer.hc_norm.weight"
        );
        let mtp = build_mtp_refs(&config).expect("MTP references");
        assert_eq!(
            mtp.final_hyper.input_mix_up.name,
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight"
        );
    }

    #[test]
    fn qwen4_moe_keeps_fused_source_replicated_for_single_execution() {
        let config = pinned_config();
        let manifest = Qwen4Manifest::build(&config).expect("pinned config manifest");
        for name in [
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
        ] {
            let entry = manifest
                .entry(name, Some(0))
                .unwrap_or_else(|| panic!("missing Qwen4 expert source {name}"));
            assert_eq!(
                entry.policy,
                ShardPolicy::Replicate,
                "{name} must be replicated for the sealed Single route"
            );
        }

        let layers = build_layer_refs(&config).expect("Qwen4 layer references");
        match &layers[0].moe.expert_source_layout {
            ExpertSourceLayout::PackedFused {
                gate_up,
                down,
                sidecars,
            } => {
                assert_eq!(
                    gate_up,
                    "model.language_model.layers.0.mlp.experts.gate_up_proj"
                );
                assert_eq!(down, "model.language_model.layers.0.mlp.experts.down_proj");
                assert!(sidecars.is_empty());
            }
            other => panic!("pinned Qwen4 source inventory must stay PackedFused: {other:?}"),
        }
    }
    #[test]
    fn matrices_quantize_by_group_width_and_unrepresentable_k_stays_bf16() {
        let tensor = |name: &str, shape: &[usize]| {
            TensorRef::new(
                name,
                TensorRole::RoutedGateUp,
                Some(0),
                shape.to_vec(),
                DType::BF16,
            )
            .expect("tensor reference")
        };
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.gate_up_proj",
                &[2, 4, 256],
            )
            .dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor(
                "model.language_model.layers.0.mlp.experts.down_proj",
                &[2, 4, 128],
            )
            .dtype,
            ROUTED_DOWN_DTYPE,
            "a K the aligned group cannot carry falls to the row-local one"
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.gate_up_proj", &[2, 4, 256]).dtype,
            ROUTED_GATE_UP_DTYPE
        );
        assert_eq!(
            tensor("mtp.layers.0.mlp.experts.down_proj", &[2, 4, 128]).dtype,
            ROUTED_DOWN_DTYPE
        );

        // Rank-2 matrices the packed set admits all take the eight-bit class,
        // whatever their alignment above the 128-weight admission boundary:
        // the block format carries any K, so there is no fallback tier.
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        ] {
            for k in [256usize, 384] {
                assert_eq!(
                    tensor(name, &[2, k]).dtype,
                    DType::Q8_0,
                    "rank-2 matrix {name} K={k} must ship eight-bit"
                );
            }
        }

        // The token embedding and the language head take the same eight-bit
        // class tier, which every MoE recipe in this tree uses for them.
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::Q8_0,
                "{name} is an eight-bit class"
            );
        }

        // The thinnest-over-widest reduction keeps source bytes whatever its K:
        // the hyper-connection mixer is four rows over a
        // `hc_count * hidden` reduction.
        for name in [
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        ] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} keeps source bytes"
            );
        }

        // K = 320 is neither 256- nor 128-aligned (the hyper-connection up
        // projection), and K = 160 is the PLE row width; both stay source BF16.
        for (name, k) in [
            (
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                320usize,
            ),
            (
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                160,
            ),
        ] {
            assert_eq!(
                tensor(name, &[2, k]).dtype,
                DType::BF16,
                "{name} K={k} must stay source-exact BF16"
            );
        }

        // The MTP connector matrices still have a source-BF16 owner.
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_eq!(
                tensor(name, &[2, 256]).dtype,
                DType::BF16,
                "{name} stays source-exact BF16 until its owner reads quantized weights"
            );
        }
    }

    /// The trunk recipe is a decision, not a default: the wide attention/GDN
    /// projections ship eight-bit because the paired candidate tiers are worse
    /// here — four bits on a dense projection (which writes straight into the
    /// residual stream) produced a model that opened a reasoning block it could
    /// not close, and BF16 is the unpacked path this packing exists to replace.
    /// Either fallback changes the dtype assertion or the count below.  The
    /// eight-bit tier is what an unset `HIPFIRE_QWEN4_TRUNK_TIER` declares; the
    /// six-bit rung is the selector's, and is asserted below.
    #[test]
    fn mtp_tier_knob_reaches_the_five_attention_matrices() {
        let config = pinned_config();
        // Measured on the published rung: these five ship qt=3 (Q8F16) while the
        // trunk carries the mq6 rung, and the knob must move exactly these.
        let mtp_attention = [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        ];
        let trunk_attention = "model.language_model.layers.2.linear_attn.in_proj_qkv.weight";
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(false)));
        let shipped = Qwen4Manifest::build(&config).expect("default manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(Some(true)));
        let source = Qwen4Manifest::build(&config).expect("source-tier manifest");
        MTP_SOURCE_TIER_OVERRIDE.with(|value| value.set(None));
        for name in mtp_attention {
            let default_entry = shipped
                .entry(name, None)
                .expect("default MTP attention entry");
            let source_entry = source
                .entry(name, None)
                .expect("source-tier MTP attention entry");
            assert_ne!(default_entry.dtype, DType::BF16, "{name} ships quantized");
            assert_eq!(
                source_entry.dtype,
                DType::BF16,
                "{name} must follow HIPFIRE_QWEN4_MTP_TIER=source"
            );
        }
        // The rank-3 routed experts keep their packed tier: the expert writer
        // carries no BF16 arm, and the knob is scoped to rank-2 matrices.
        for name in [
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        ] {
            let default_entry = shipped.entry(name, None).expect("default expert entry");
            let source_entry = source.entry(name, None).expect("source-tier expert entry");
            assert_eq!(
                default_entry.dtype, source_entry.dtype,
                "{name} must not move with the matrix knob"
            );
        }
        let default_trunk = shipped
            .entry(trunk_attention, Some(2))
            .expect("trunk entry");
        let source_trunk = source.entry(trunk_attention, Some(2)).expect("trunk entry");
        assert_eq!(
            default_trunk.dtype, source_trunk.dtype,
            "trunk tier must not move"
        );
        let norm = source
            .entry("mtp.layers.0.self_attn.q_norm.weight", None)
            .expect("MTP norm entry");
        assert_eq!(
            norm.dtype,
            DType::BF16,
            "MTP norms are already source-exact"
        );
    }

    #[test]
    fn mtp_source_tier_knob_only_answers_to_its_own_value() {
        assert!(!qwen4_mtp_source_tier(None), "unset keeps the shipped tier");
        assert!(!qwen4_mtp_source_tier(Some("mq6")));
        assert!(qwen4_mtp_source_tier(Some("source")));
    }

    #[test]
    fn trunk_attention_projections_are_pinned_to_eight_bits() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        let mut packed = 0usize;
        for entry in &manifest.weights {
            let trunk_attention =
                entry.name.contains(".linear_attn.") || entry.name.contains(".self_attn.");
            if !trunk_attention || entry.logical_shape.len() != 2 {
                continue;
            }
            let k = *entry.logical_shape.last().expect("rank checked above");
            if k % 128 != 0 {
                continue;
            }
            assert_eq!(
                entry.dtype,
                DType::Q8_0,
                "{} is a trunk attention projection and must ship at eight bits",
                entry.name
            );
            packed += 1;
        }
        // 36 linear-attention layers x (qkv, z, in_proj_a, in_proj_b, out_proj)
        // + 12 full-attention layers x (q, k, v, o, indexer qk)
        // + the MTP layer's five full-attention projections.
        assert_eq!(packed, 245, "trunk packing count");
    }

    /// The trunk's declared tier follows `HIPFIRE_QWEN4_TRUNK_TIER`, and only
    /// the trunk's does.
    ///
    /// The default the line above pins is the eight-bit recipe; `mq6` is the
    /// only value that selects the six-bit rung, and it moves the two hundred
    /// and forty rank-2 trunk attention/GDN matrices — not the MTP namespace,
    /// which reaches the manifest through its own constructor and keeps its own
    /// targets.
    #[test]
    fn trunk_tier_selector_moves_the_declared_trunk_target() {
        assert_eq!(qwen4_trunk_tier(None), DType::Q8_0, "unset knob default");
        assert_eq!(qwen4_trunk_tier(Some("mq6")), DType::MQ6G256V2);
        assert_eq!(
            qwen4_trunk_tier(Some("mfp4e8")),
            DType::Q8_0,
            "a retired value must not resurrect a third tier"
        );

        let config = pinned_config();
        let six = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(Some("mq6")))
            .expect("six-bit trunk manifest");
        let eight = Qwen4Manifest::build_with_trunk_tier(&config, qwen4_trunk_tier(None))
            .expect("eight-bit trunk manifest");
        assert_eq!(
            six.weights.len(),
            eight.weights.len(),
            "tier, not the entry set"
        );

        let mut moved = 0usize;
        let mut held = 0usize;
        for (six_entry, eight_entry) in six.weights.iter().zip(&eight.weights) {
            assert_eq!(six_entry.name, eight_entry.name, "entries must not move");
            let trunk_matrix = qwen4_trunk_matrix(&six_entry.name, &six_entry.logical_shape)
                && six_entry.layer.is_some();
            // The layer index is what separates the trunk declaration from the
            // MTP namespace's: MTP names are layerless by construction.
            if trunk_matrix {
                assert_eq!(
                    six_entry.dtype,
                    DType::MQ6G256V2,
                    "{} trunk target under HIPFIRE_QWEN4_TRUNK_TIER=mq6",
                    six_entry.name
                );
                assert_eq!(
                    eight_entry.dtype,
                    DType::Q8_0,
                    "{} trunk target with the knob unset",
                    eight_entry.name
                );
                moved += 1;
            } else {
                assert_eq!(
                    six_entry.dtype, eight_entry.dtype,
                    "{} must not follow the trunk selector",
                    six_entry.name
                );
            }
            // Under either declaration the trunk keeps admitting both rungs,
            // which is what keeps a published artifact of the other tier
            // loadable by this build.
            if trunk_matrix {
                for tier in [DType::BF16, DType::Q8_0, DType::MQ6G256V2] {
                    assert!(
                        six_entry.dtype_constraint.accepts(tier),
                        "{} must keep admitting {tier:?}",
                        six_entry.name
                    );
                }
            }
        }
        assert_eq!(moved, 240, "trunk matrices the selector moves");

        // The MTP namespace's five attention projections are the same class
        // under a separate declaration: Q8F16 under the six-bit build too.
        for entry in &six.weights {
            if entry.name.starts_with("mtp.")
                && qwen4_trunk_matrix(&entry.name, &entry.logical_shape)
            {
                assert_eq!(entry.dtype, DType::Q8_0, "{} MTP tier", entry.name);
                held += 1;
            }
        }
        assert_eq!(held, 5, "MTP attention projections left alone");
    }

    #[test]
    fn manifest_admits_quantized_matrices_and_keeps_source_exact_payloads_bf16() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        for (name, layer, dtype) in [
            (
                "model.language_model.layers.3.mlp.experts.gate_up_proj",
                Some(3),
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "model.language_model.layers.3.mlp.experts.down_proj",
                Some(3),
                ROUTED_DOWN_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.gate_up_proj",
                None,
                ROUTED_GATE_UP_DTYPE,
            ),
            (
                "mtp.layers.0.mlp.experts.down_proj",
                None,
                ROUTED_DOWN_DTYPE,
            ),
        ] {
            let entry = manifest.entry(name, layer).expect("routed expert entry");
            assert_eq!(entry.dtype, dtype, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(dtype), "{name} source");
            assert!(
                !entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must not accept an aligned-K family its geometry is not"
            );
        }

        // Dense matrices are quantized targets that still accept BF16 source
        // bytes, which is what lets an older artifact load against a newer
        // manifest.  The trunk's wide attention/GDN projections are also the
        // class whose packed tier the artifact chooses: the six-bit rung
        // carries MQ6G256V2 (qt47) there and the eight-bit rung Q8F16 (qt3).
        // Both must satisfy the entry's source contract, or a published rung
        // of one tier cannot be loaded by the tree that produced it.
        let assert_quantized = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("quantized matrix entry");
            assert_eq!(entry.dtype, DType::Q8_0, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(DType::Q8_0));
            assert!(
                entry.dtype_constraint.accepts(DType::MQ6G256V2),
                "{name} must admit the six-bit trunk rung"
            );
        };
        for (name, layer) in [
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                Some(3),
            ),
            (
                "model.language_model.layers.2.linear_attn.in_proj_qkv.weight",
                Some(2),
            ),
            // The MTP layer's attention projections are the same class under
            // the model namespace, and reach the manifest through their own
            // constructor.
            ("mtp.layers.0.self_attn.q_proj.weight", None),
        ] {
            assert_quantized(name, layer);
        }

        let assert_bf16 = |name: &str, layer: Option<usize>| {
            let entry = manifest.entry(name, layer).expect("BF16 entry");
            assert_eq!(entry.dtype, DType::BF16, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G128V2));
        };
        for name in ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"] {
            assert_bf16(name, None);
        }
        // The eight-bit classes: BF16 in the checkpoint, Q8 in the artifact.
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            let entry = manifest.entry(name, None).expect("eight-bit class entry");
            assert_eq!(entry.dtype, DType::Q8_0, "{name} target dtype");
            assert!(entry.dtype_constraint.accepts(DType::BF16));
            assert!(entry.dtype_constraint.accepts(DType::Q8_0));
            assert!(!entry.dtype_constraint.accepts(DType::MQ4G256V2));
            // The head and the embedding are eight-bit by recipe; a six-bit
            // payload there is the same class of mistake as four bits.
            assert!(!entry.dtype_constraint.accepts(DType::MQ6G256V2));
        }
        // The shared expert, the router, and the PLE projections stay
        // source-exact: they ride the sealed-MoE route or select among 512
        // experts, and neither is a wide attention projection.
        assert_bf16("model.language_model.layers.3.mlp.gate.weight", Some(3));
        assert_bf16(
            "model.language_model.layers.3.mlp.shared_expert.up_proj.weight",
            Some(3),
        );
        let ple = manifest
            .weights
            .iter()
            .find(|entry| entry.name.contains(".ple.") && entry.name.contains("key"))
            .map(|entry| (entry.name.clone(), entry.layer))
            .expect("PLE key projection entry");
        assert_bf16(&ple.0, ple.1);
        // The hyper-connection mixer and block-inject matrices are the thin
        // side of the widest reduction in the network; they keep source bytes.
        assert_bf16(
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            None,
        );
        assert_bf16(
            "model.language_model.layers.3.attn_hyper_connection.block_inject_weight.weight",
            Some(3),
        );

        let ple = manifest
            .entry(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                Some(1),
            )
            .expect("PLE shard");
        assert_eq!(ple.dtype, DType::Q8_0);
        assert!(ple.dtype_constraint.accepts(DType::Q8_0));
        // A sealed artifact written before the Q8F16 tier carries BF16 rows and
        // must stay loadable: both tiers are admitted by the same declaration.
        assert!(ple.dtype_constraint.accepts(DType::BF16));
        assert!(!ple.dtype_constraint.accepts(DType::MQ4G128V2));
    }
}
