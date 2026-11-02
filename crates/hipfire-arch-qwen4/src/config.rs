// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Qwen4 (Qwen3.8-Flash-Next) configuration parsing and admission.
//!
//! The offline source/quantizer parser accepts two JSON envelopes: an HF
//! metadata envelope (`{"config": ...}`) and a raw Transformers `config.json`.
//! Runtime serving admission uses the strict nested conditional-generation
//! parser, which additionally requires the outer model type and canonical
//! architecture entry before allocation.
//!
//! Every parsed config carries a typed [`Qwen4Profile`]. The production
//! parsers only ever produce [`Qwen4Profile::FlashNext`], whose geometry is
//! pinned exactly. A developer-only synthetic profile is selected solely by an
//! explicit top-level `hipfire_profile` marker through the `from_developer_*`
//! parsers (tests and the `qwen4-dev-profiles` feature); it is never inferred
//! from geometry, and every production parser refuses a config that carries
//! the marker.

use crate::ple::{PleHashMetadata, PLE_VALID_ROWS};
use rdna_compute::{TrunkIu4Default, TRUNK_IU4_DEFAULT_KEY};
use serde_json::Value;
use std::fmt;

/// Primary HFQ architecture id reserved for Qwen4.
pub const ARCH_ID: u32 = 16;
/// Stable source model type used by the outer Qwen4 conditional config.
pub const MODEL_TYPE: &str = "qwen4_exp";
/// Stable source model type used by the text config.
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";
/// The only architecture entry admitted by this crate.
pub const ARCHITECTURE_NAME: &str = "Qwen4ExpForConditionalGeneration";
/// Stable profile id of the pinned Qwen3.8-Flash-Next checkpoint.
pub const FLASH_NEXT_PROFILE_ID: &str = "qwen3.8-flash-next";
/// Stable profile id of the AR-only Flash-Next profile ([`Qwen4Profile::FlashNextArOnly`]).
pub const FLASH_NEXT_AR_ONLY_PROFILE_ID: &str = "qwen3.8-flash-next-ar";
/// Stable profile id of the head-dropped packed Flash-Next profile
/// ([`Qwen4Profile::FlashNextHeadless`]).
pub const FLASH_NEXT_HEADLESS_PROFILE_ID: &str = "qwen3.8-flash-next-headless";
/// Top-level JSON key of the explicit developer-profile marker.
pub const PROFILE_KEY: &str = "hipfire_profile";
/// Spelling current Transformers normalizes `full_attention` to.  It is an
/// exact synonym of [`LayerType::FullAttention`] only when the QSA indexer
/// fields are present and valid.
pub const QSA_LAYER_TYPE_SPELLING: &str = "qwen_sparse_attention";

// Developer-profile bounds.  Synthetic geometry is arbitrary but checked: it
// must sit inside these bounds and every derived width must fit `i32`.
const MAX_HIDDEN: usize = 1 << 16;
const MAX_LAYERS: usize = 512;
const MAX_VOCAB: usize = 1 << 21;
const MAX_CONTEXT_POSITIONS: usize = 1 << 24;
const MAX_ATTENTION_HEADS: usize = 1024;
const MAX_HEAD_DIM: usize = 1024;
const MAX_GDN_HEADS: usize = 1024;
const MAX_GDN_HEAD_DIM: usize = 1024;
const MAX_CONV_KERNEL: usize = 16;
const MAX_INDEXER_HEADS: usize = 256;
const MAX_INDEXER_DIM: usize = 1024;
const MAX_INDEXER_BUDGET: usize = 1 << 20;
const MAX_INDEXER_RATIO: usize = 1024;
const MAX_EXPERTS: usize = 1 << 13;
const MAX_INTERMEDIATE: usize = 1 << 20;
const MAX_HC_COUNT: usize = 16;
const MAX_HC_LOWRANK: usize = 1 << 12;
const MAX_PLE_EMBED_DIM: usize = 1 << 20;
const MAX_PLE_HEADS_PER_NGRAM: usize = 64;
const MAX_PLE_SHARDS: usize = 4096;
const MAX_PLE_VOCAB_BASE: u64 = 1 << 40;
const MAX_PLE_PADDING: usize = 1 << 20;
const MAX_PROFILE_ID_LEN: usize = 64;
/// Largest derived linear extent (kernels index with `i32`).
const MAX_WIDTH: usize = i32::MAX as usize;
/// Largest total FFN weight element count a synthetic profile may imply.
const MAX_ELEMENTS: u64 = 1 << 40;

/// Decoder layer kind in the Qwen4 hybrid trunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Gated DeltaNet / linear-attention layer.
    LinearAttention,
    /// Qwen Sparse Attention layer (the full-attention branch).
    FullAttention,
}

impl LayerType {
    /// Parse the exact Transformers spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "linear_attention" => Some(Self::LinearAttention),
            "full_attention" => Some(Self::FullAttention),
            _ => None,
        }
    }

    /// Return the exact Transformers spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LinearAttention => "linear_attention",
            Self::FullAttention => "full_attention",
        }
    }
}

impl fmt::Display for LayerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dtype of the recurrent Gated DeltaNet state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecurrentStateDType {
    F32,
}

impl RecurrentStateDType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "float32" | "f32" => Some(Self::F32),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "float32",
        }
    }
}

/// Source/default weight dtype named by the Qwen4 checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceDType {
    BF16,
}

impl SourceDType {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "bfloat16" | "bf16" => Some(Self::BF16),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BF16 => "bfloat16",
        }
    }
}

/// Nested native MTP shape.  This is intentionally data-only: generation and
/// acceptance remain owned by the runtime engine.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4MtpConfig {
    /// Number of decoder layers in the MTP module.
    pub num_hidden_layers: usize,
    /// MTP's per-layer attention kinds.
    pub layer_types: Vec<LayerType>,
    /// Whether the MTP module uses its hybrid implementation path.
    pub hybrid: bool,
    /// MTP RoPE base, retained for the later typed executor.
    pub rope_theta: f64,
}

impl Qwen4MtpConfig {
    /// Validate the pinned Flash-Next native MTP contract: the shape contract
    /// of [`Self::validate_shape`] plus `rope_theta = 10,000,000`.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_shape()?;
        if !approximately_equal(self.rope_theta, 10_000_000.0) {
            return Err(format!(
                "qwen4: MTP rope_theta={} (expected 10000000)",
                self.rope_theta
            ));
        }
        Ok(())
    }

    /// Shape contract every profile shares: one hybrid full-attention layer
    /// and a finite positive RoPE base.
    pub fn validate_shape(&self) -> Result<(), String> {
        if self.num_hidden_layers != 1 {
            return Err(format!(
                "qwen4: MTP num_hidden_layers={} (expected 1)",
                self.num_hidden_layers
            ));
        }
        if self.layer_types != [LayerType::FullAttention] {
            return Err(format!(
                "qwen4: MTP layer_types={:?} (expected [full_attention])",
                self.layer_types
            ));
        }
        if !self.hybrid {
            return Err("qwen4: MTP hybrid must be true".into());
        }
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(format!(
                "qwen4: MTP rope_theta={} must be finite and positive",
                self.rope_theta
            ));
        }
        Ok(())
    }
}

/// Which equations a developer profile executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4EquationSource {
    /// The pinned upstream Transformers sparse-MoE decoder.
    PinnedUpstream,
    /// A dense FFN that is NOT in upstream: hipfire's explicit developer
    /// equation (`down(act(gate(x)) * up(x))`), never an upstream claim.
    NonUpstreamDense,
}

impl Qwen4EquationSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PinnedUpstream => "pinned-upstream",
            Self::NonUpstreamDense => "non-upstream-dense",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "pinned-upstream" => Some(Self::PinnedUpstream),
            "non-upstream-dense" => Some(Self::NonUpstreamDense),
            _ => None,
        }
    }
}

/// Identity of a developer-only synthetic profile (explicit marker only; it
/// is never inferred from geometry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4SyntheticProfile {
    pub id: String,
    pub equation_source: Qwen4EquationSource,
}

/// The shape profile a config was admitted under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen4Profile {
    /// The pinned Qwen3.8-Flash-Next checkpoint (exact production pins).
    FlashNext,
    /// The pinned Flash-Next geometry without the native MTP head: the
    /// artifact carries no `mtp.*` tensors (the stock-FP8 recipe ships the
    /// MTP experts as BF16 source and omits the head).  Never parsed from a
    /// config: an artifact's recipe selects it through
    /// [`Qwen4Config::into_ar_only`], so `config.mtp` is `None` and every MTP
    /// entry point sees a head-free profile instead of a config that claims a
    /// head the artifact does not carry.
    FlashNextArOnly,
    /// The pinned PACKED Flash-Next trunk loaded without its native MTP head
    /// (the TP trunk shard runs AR-only, `HIPFIRE_QWEN4_EP_TRUNK_SHARD`).  Unlike
    /// [`Self::FlashNextArOnly`] it is NOT the stock-FP8 recipe: the artifact is the
    /// shipped packed `mq4-xts` one (its `mtp.*` tensors stay in the file, unread), the
    /// recipe is [`crate::weights::Qwen4TierRecipe::FlashNextPacked`], and the
    /// stock-FP8 numerical pins (F32 GDN state, BF16 trunk) do not apply.  Never parsed
    /// from a config: artifact admission derives it through
    /// [`Qwen4Config::into_headless`] when
    /// [`crate::weights::trunk_shard::trunk_shard_drops_mtp_head`] holds.
    FlashNextHeadless,
    /// Developer-only synthetic geometry with pinned HC/GDN/QSA equations.
    Synthetic(Qwen4SyntheticProfile),
}

impl Qwen4Profile {
    pub fn id(&self) -> &str {
        match self {
            Self::FlashNext => FLASH_NEXT_PROFILE_ID,
            Self::FlashNextArOnly => FLASH_NEXT_AR_ONLY_PROFILE_ID,
            Self::FlashNextHeadless => FLASH_NEXT_HEADLESS_PROFILE_ID,
            Self::Synthetic(profile) => &profile.id,
        }
    }

    /// The pinned Flash-Next geometry, with or without the MTP head.
    pub const fn is_flash_next(&self) -> bool {
        matches!(
            self,
            Self::FlashNext | Self::FlashNextArOnly | Self::FlashNextHeadless
        )
    }

    /// The stock-FP8 Flash-Next profile ([`Self::FlashNextArOnly`]): head-free AND bound
    /// to the stock-FP8 recipe's numerical pins (F32 GDN state, BF16 trunk). The head-free
    /// predicate that does not imply the recipe is [`Self::is_headless`].
    pub const fn is_ar_only(&self) -> bool {
        matches!(self, Self::FlashNextArOnly)
    }

    /// Flash-Next geometry admitted without the native MTP head, under either recipe
    /// ([`Self::FlashNextArOnly`] stock FP8, [`Self::FlashNextHeadless`] packed).
    pub const fn is_headless(&self) -> bool {
        matches!(self, Self::FlashNextArOnly | Self::FlashNextHeadless)
    }

    /// The packed (non-stock) Flash-Next trunk, with or without its head: the profiles
    /// whose forward takes the Flash-Next packed-kernel routes.
    pub const fn is_packed_flash_next(&self) -> bool {
        matches!(self, Self::FlashNext | Self::FlashNextHeadless)
    }

    pub const fn is_synthetic(&self) -> bool {
        matches!(self, Self::Synthetic(_))
    }

    /// Synthetic profiles are developer-only; Flash-Next is production.
    pub const fn developer_only(&self) -> bool {
        self.is_synthetic()
    }

    pub const fn equation_source(&self) -> Qwen4EquationSource {
        match self {
            Self::FlashNext | Self::FlashNextArOnly | Self::FlashNextHeadless => {
                Qwen4EquationSource::PinnedUpstream
            }
            Self::Synthetic(profile) => profile.equation_source,
        }
    }
}

/// Feed-forward block of every decoder layer.  Dense is a genuine gated MLP,
/// never a one-expert MoE or `num_experts = 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Qwen4FfnConfig {
    Dense {
        intermediate_size: usize,
    },
    Moe {
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        shared_intermediate_size: Option<usize>,
        norm_topk_prob: bool,
    },
}

impl Qwen4FfnConfig {
    /// The pinned Flash-Next sparse MoE.
    pub const fn flash_next() -> Self {
        Self::Moe {
            num_experts: 512,
            top_k: 10,
            intermediate_size: 640,
            shared_intermediate_size: Some(640),
            norm_topk_prob: true,
        }
    }

    pub const fn is_moe(&self) -> bool {
        matches!(self, Self::Moe { .. })
    }

    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }

    /// Dense intermediate width, or the routed-expert intermediate width.
    pub const fn intermediate_size(&self) -> usize {
        match self {
            Self::Dense { intermediate_size }
            | Self::Moe {
                intermediate_size, ..
            } => *intermediate_size,
        }
    }

    /// Values mirrored by the retained scalar checkpoint fields.
    fn scalar_mirror(&self) -> (usize, usize, usize, usize, bool) {
        match self {
            Self::Dense { .. } => (0, 0, 0, 0, false),
            Self::Moe {
                num_experts,
                top_k,
                intermediate_size,
                shared_intermediate_size,
                norm_topk_prob,
            } => (
                *num_experts,
                *top_k,
                *intermediate_size,
                shared_intermediate_size.unwrap_or(0),
                *norm_topk_prob,
            ),
        }
    }
}

/// Optional per-layer n-gram embedding (PLE) geometry, with explicit row
/// geometry.  Nothing here is inherited from the Flash-Next constants unless
/// the profile is Flash-Next ([`Self::flash_next`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4PleConfig {
    /// Zero-based decoder layer carrying the PLE module.
    pub layer: usize,
    /// One-indexed source layer id (`ple_layer_ids[0]`).
    pub layer_id: usize,
    pub embed_dim: usize,
    pub conv_kernel_size: usize,
    pub heads_per_ngram: usize,
    /// Always 3: bigram and trigram heads and three stored multipliers.
    pub ngram_size: usize,
    pub ngram_vocab_size_base: u64,
    /// `make_ngram_vocab_size_divisible_by`: the physical row padding multiple.
    pub padding_multiple: usize,
    /// `2 * heads_per_ngram` stored head tables (bigram half, trigram half).
    pub head_count: usize,
    /// Stored hash multipliers (3).
    pub multiplier_count: usize,
    /// `embed_dim / (2 * heads_per_ngram)` values per row.
    pub row_width: usize,
    /// `split_ngram_parts`: external row shards.
    pub shard_count: usize,
    /// `padded_rows / shard_count` rows per shard.
    pub shard_rows: usize,
    /// Sum of the head tables' row counts.
    pub valid_rows: u64,
    /// `valid_rows` rounded up to `padding_multiple`.
    pub padded_rows: u64,
}

impl Qwen4PleConfig {
    /// Build and self-validate a PLE geometry.  Layer placement against the
    /// decoder is checked by [`Qwen4Config::validate`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_ids: &[usize],
        embed_dim: usize,
        conv_kernel_size: usize,
        split_ngram_parts: usize,
        heads_per_ngram: usize,
        ngram_size: usize,
        ngram_vocab_size_base: u64,
        make_ngram_vocab_size_divisible_by: usize,
        valid_rows: Option<u64>,
    ) -> Result<Self, String> {
        let layer_id = match layer_ids {
            [id] if *id >= 1 => *id,
            _ => {
                return Err(format!(
                    "qwen4: ple_layer_ids={layer_ids:?} (exactly one one-indexed layer id is supported)"
                ))
            }
        };
        expect(ngram_size, 3, "ngram_size")?;
        bounded(
            heads_per_ngram,
            1,
            MAX_PLE_HEADS_PER_NGRAM,
            "heads_per_ngram",
        )?;
        bounded(embed_dim, 1, MAX_PLE_EMBED_DIM, "ple_embed_dim")?;
        let head_count = 2 * heads_per_ngram;
        if embed_dim % head_count != 0 {
            return Err(format!(
                "qwen4: ple_embed_dim={embed_dim} is not divisible by 2*heads_per_ngram={head_count}"
            ));
        }
        let row_width = embed_dim / head_count;
        bounded(conv_kernel_size, 1, MAX_CONV_KERNEL, "ple_conv_kernel_size")?;
        bounded(split_ngram_parts, 1, MAX_PLE_SHARDS, "split_ngram_parts")?;
        if ngram_vocab_size_base == 0 || ngram_vocab_size_base > MAX_PLE_VOCAB_BASE {
            return Err(format!(
                "qwen4: ngram_vocab_size_base={ngram_vocab_size_base} is outside 1..={MAX_PLE_VOCAB_BASE}"
            ));
        }
        bounded(
            make_ngram_vocab_size_divisible_by,
            1,
            MAX_PLE_PADDING,
            "make_ngram_vocab_size_divisible_by",
        )?;
        // Upstream sizes the head tables as the `head_count` successive
        // primes at or above `ngram_vocab_size_base`; the row count is their
        // sum.  A declared count must agree with it.
        let head_sizes = ple_head_vocab_sizes(ngram_vocab_size_base, head_count)?;
        let derived_rows = head_sizes
            .iter()
            .try_fold(0u64, |sum, size| sum.checked_add(*size))
            .ok_or_else(|| "qwen4: PLE row count overflows".to_string())?;
        let valid_rows = match valid_rows {
            None => derived_rows,
            Some(declared) if declared == derived_rows => declared,
            Some(declared) => {
                return Err(format!(
                    "qwen4: PLE valid_rows={declared} disagrees with the {head_count} successive primes >= ngram_vocab_size_base={ngram_vocab_size_base} ({derived_rows} rows)"
                ))
            }
        };
        let multiple = make_ngram_vocab_size_divisible_by as u64;
        let padded_rows = (valid_rows
            .checked_add(multiple - 1)
            .ok_or_else(|| "qwen4: PLE padded rows overflow".to_string())?
            / multiple)
            .checked_mul(multiple)
            .ok_or_else(|| "qwen4: PLE padded rows overflow".to_string())?;
        let shards = split_ngram_parts as u64;
        if padded_rows % shards != 0 {
            return Err(format!(
                "qwen4: PLE padded_rows={padded_rows} is not divisible by split_ngram_parts={split_ngram_parts}"
            ));
        }
        let shard_rows_wide = padded_rows / shards;
        let shard_rows = usize::try_from(shard_rows_wide)
            .map_err(|_| "qwen4: PLE shard_rows does not fit usize".to_string())?;
        if valid_rows <= (shards - 1) * shard_rows_wide {
            return Err("qwen4: PLE valid_rows leave a trailing shard without valid rows".into());
        }
        let table_bytes = padded_rows
            .checked_mul(row_width as u64)
            .and_then(|elements| elements.checked_mul(2));
        if table_bytes.is_none() {
            return Err("qwen4: PLE table byte size overflows".into());
        }
        Ok(Self {
            layer: layer_id - 1,
            layer_id,
            embed_dim,
            conv_kernel_size,
            heads_per_ngram,
            ngram_size,
            ngram_vocab_size_base,
            padding_multiple: make_ngram_vocab_size_divisible_by,
            head_count,
            multiplier_count: 3,
            row_width,
            shard_count: split_ngram_parts,
            shard_rows,
            valid_rows,
            padded_rows,
        })
    }

    /// The pinned Flash-Next PLE geometry.
    pub fn flash_next() -> Self {
        Self::new(
            &[2],
            2560,
            4,
            128,
            8,
            3,
            20_000_000,
            128,
            Some(PLE_VALID_ROWS),
        )
        .expect("pinned Flash-Next PLE geometry must satisfy its invariants")
    }

    /// Dilation of the PLE short convolution: upstream spaces its taps one
    /// n-gram apart.
    pub const fn conv_dilation(&self) -> usize {
        self.ngram_size
    }

    /// Rows of PLE short-convolution history, `(kernel - 1) * dilation`.
    pub const fn conv_history_rows(&self) -> usize {
        (self.conv_kernel_size - 1) * self.ngram_size
    }

    /// Valid (non-padding) rows of external shard `shard`; 0 past the last.
    pub fn shard_valid_rows(&self, shard: usize) -> usize {
        if shard >= self.shard_count {
            return 0;
        }
        self.valid_rows
            .saturating_sub((shard as u64).saturating_mul(self.shard_rows as u64))
            .min(self.shard_rows as u64) as usize
    }

    /// Check stored hash metadata against this geometry.
    pub fn check_hash_metadata(&self, metadata: &PleHashMetadata) -> Result<(), String> {
        if metadata.head_count() != self.head_count {
            return Err(format!(
                "head count {} (expected {})",
                metadata.head_count(),
                self.head_count
            ));
        }
        if metadata.padding_multiple() != self.padding_multiple as u64 {
            return Err(format!(
                "padding multiple {} (expected {})",
                metadata.padding_multiple(),
                self.padding_multiple
            ));
        }
        if metadata.valid_rows() != self.valid_rows {
            return Err(format!(
                "valid rows {} (expected {})",
                metadata.valid_rows(),
                self.valid_rows
            ));
        }
        if metadata.padded_rows() != self.padded_rows {
            return Err(format!(
                "padded rows {} (expected {})",
                metadata.padded_rows(),
                self.padded_rows
            ));
        }
        let expected = ple_head_vocab_sizes(self.ngram_vocab_size_base, self.head_count)?;
        if metadata.head_vocab_sizes() != expected.as_slice() {
            let index = metadata
                .head_vocab_sizes()
                .iter()
                .zip(&expected)
                .position(|(got, wanted)| got != wanted)
                .unwrap_or(0);
            return Err(format!(
                "head vocabulary sizes are not the {} successive primes >= ngram_vocab_size_base={} (head {index}: {} vs {})",
                self.head_count,
                self.ngram_vocab_size_base,
                metadata.head_vocab_sizes()[index],
                expected[index]
            ));
        }
        Ok(())
    }
}

/// Which parser surface is reading the config.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(any(test, feature = "qwen4-dev-profiles")), allow(dead_code))]
enum ParseMode {
    /// Production: any developer marker is refused.
    Production,
    /// Developer: the explicit marker selects a synthetic profile.
    Developer,
}

/// A validated `hipfire_profile` marker.
struct ProfileMarker {
    profile: Qwen4SyntheticProfile,
    ffn: Qwen4FfnConfig,
    ple_valid_rows: Option<u64>,
}

/// Fully typed Qwen4 text configuration.
///
/// The scalar checkpoint fields (`num_experts`, `ple_embed_dim`, ...) are
/// retained source descriptors; `profile`, `ffn`, `ple` and `mtp` are
/// authoritative and `validate` enforces that the two agree.  `validate` is
/// strict by design: a parseable but different Qwen-family configuration must
/// not reach a Qwen4 allocator or executor.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4Config {
    /// Canonical architecture id for the future registry entry.
    pub architecture_id: u32,
    /// Outer source model type (`qwen4_exp`).
    pub model_type: String,
    /// Inner text model type (`qwen4_exp_text`).
    pub text_model_type: String,
    /// Shape profile: pinned Flash-Next or an explicit developer profile.
    pub profile: Qwen4Profile,
    /// Source dtype (`bfloat16` in the pinned checkpoint).
    pub dtype: SourceDType,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    /// Long-context RoPE scaling, resolved by the engine
    /// ([`hipfire_runtime::rope::rope_scaling`]); shared by the MTP head.
    pub rope_scaling: rdna_compute::rope::RopeScaling,
    pub attention_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub full_attention_interval: usize,

    // Gated DeltaNet dimensions.
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub recurrent_state_dtype: RecurrentStateDType,

    // QSA indexer dimensions.
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,

    /// Authoritative feed-forward shape.
    pub ffn: Qwen4FfnConfig,
    // Retained scalar mirrors of `ffn` (zero / false for a dense FFN).
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    /// Optional original-id routed-expert membership, validated against this geometry.
    pub expert_keep: Option<crate::ExpertKeep>,
    pub output_gate_type: String,

    // Hyper-connections.
    pub hc_count: usize,
    pub hc_lowrank: usize,

    /// Authoritative optional PLE geometry.
    pub ple: Option<Qwen4PleConfig>,
    // Retained scalar mirrors of `ple` (empty / zero when PLE is absent).
    pub ple_layer_ids: Vec<usize>,
    pub ple_conv_kernel_size: usize,
    pub ple_embed_dim: usize,
    pub split_ngram_parts: usize,
    pub heads_per_ngram: usize,
    pub ngram_size: usize,
    pub ngram_vocab_size_base: u64,
    pub make_ngram_vocab_size_divisible_by: usize,

    // Token/generation metadata used by PLE history and MTP.
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    // Retained scalar mirrors of `mtp` (0 when the head is absent).
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    /// Authoritative optional native MTP head.
    pub mtp: Option<Qwen4MtpConfig>,
    /// The dense IU4 trunk default the artifact's bake tooling embedded
    /// (top-level metadata key [`TRUNK_IU4_DEFAULT_KEY`]); `None` for every
    /// unmarked artifact. Only the loader's route resolution reads it (see
    /// [`rdna_compute::qwen4_trunk_iu4_route`]): the shape contract above does
    /// not depend on it.
    pub trunk_iu4_default: Option<TrunkIu4Default>,
    /// `Some((rank, parts))` on the RANK-LOCAL config of a TP trunk shard
    /// (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`, [`Self::trunk_shard_local`]): the GDN key/value heads,
    /// the q heads and the shared-expert intermediate are this rank's share; `None` on every
    /// parsed (global) config.
    pub trunk_shard: Option<(usize, usize)>,
}

impl Qwen4Config {
    /// Payload slots are compact; router and pointer-table ids stay original.
    pub(crate) fn kept_expert_ids(&self, layer: usize, mtp: bool) -> Option<&[u16]> {
        self.expert_keep.as_ref().and_then(|keep| {
            if mtp {
                keep.mtp_kept()
            } else {
                Some(keep.kept(layer))
            }
        })
    }
    /// Parse an offline source/quantizer JSON input.
    ///
    /// This deliberately accepts either the HF metadata envelope or a raw
    /// Transformers `config.json`; runtime serving admission must use the
    /// strict [`Self::from_metadata_json`] contract instead.  Production
    /// surface: a `hipfire_profile` developer marker is always refused.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("qwen4: configuration is not valid JSON: {e}"))?;
        Self::from_value(&value)
    }

    /// Parse the strict nested conditional-generation envelope used by a
    /// runtime serving artifact.
    pub fn from_metadata_json(json: &str) -> Result<Self, String> {
        Self::from_nested_metadata_json(json)
    }

    /// Parse only the nested conditional-generation envelope used by a
    /// Qwen4 serving artifact.
    ///
    /// `from_value` is the explicit source/quantizer parser and remains
    /// compatible with raw text `config.json` inputs. Runtime admission is
    /// stricter: a source must identify the outer conditional-generation
    /// model, its nested text model, and the one canonical architecture entry.
    /// This keeps an arbitrary text-shaped config from being classified as the
    /// reserved arch 16 route.
    pub fn from_nested_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("qwen4: configuration is not valid JSON: {e}"))?;
        Self::from_nested_value(&value)
    }

    /// Strict runtime classifier for the nested text+MTP Qwen4 envelope.
    /// Always refuses a `hipfire_profile` developer marker.
    pub fn from_nested_value(value: &Value) -> Result<Self, String> {
        check_nested_envelope(value)?;
        Self::parse_value(value, ParseMode::Production)
    }

    /// Parse the strict nested runtime envelope from HFQ metadata.
    pub fn from_nested_metadata_json(json: &str) -> Result<Self, String> {
        Self::from_nested_json(json)
    }

    /// Developer parser: the same strict nested envelope as the runtime
    /// classifier, but an explicit top-level `hipfire_profile` marker selects
    /// a synthetic developer profile.  An unmarked config is still validated
    /// as the exact Flash-Next profile; synthetic is never inferred from
    /// geometry.  Compiled only for tests and `qwen4-dev-profiles`.
    #[cfg(any(test, feature = "qwen4-dev-profiles"))]
    pub fn from_developer_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| format!("qwen4: configuration is not valid JSON: {e}"))?;
        Self::from_developer_value(&value)
    }

    /// [`Self::from_developer_json`] for HFQ metadata JSON.
    #[cfg(any(test, feature = "qwen4-dev-profiles"))]
    pub fn from_developer_metadata_json(json: &str) -> Result<Self, String> {
        Self::from_developer_json(json)
    }

    /// [`Self::from_developer_json`] for an already decoded value.
    #[cfg(any(test, feature = "qwen4-dev-profiles"))]
    pub fn from_developer_value(value: &Value) -> Result<Self, String> {
        check_nested_envelope(value)?;
        Self::parse_value(value, ParseMode::Developer)
    }

    /// Parse from an already decoded metadata/config value (production
    /// surface; refuses a developer marker).
    pub fn from_value(value: &Value) -> Result<Self, String> {
        Self::parse_value(value, ParseMode::Production)
    }

    fn parse_value(value: &Value, mode: ParseMode) -> Result<Self, String> {
        let config = match value.get("config") {
            Some(config) => config,
            None => value,
        };
        if !config.is_object() {
            return Err("qwen4: `config` must be a JSON object".into());
        }

        let text = match config.get("text_config") {
            Some(text) if text.is_object() => text,
            Some(_) => return Err("qwen4: `text_config` must be a JSON object".into()),
            None => config,
        };

        let outer_model_type = config.get("model_type").and_then(Value::as_str);
        if let Some(model_type) = outer_model_type {
            if model_type != MODEL_TYPE && model_type != TEXT_MODEL_TYPE {
                return Err(format!(
                    "qwen4: unsupported outer model_type={model_type:?}"
                ));
            }
            if config.get("text_config").is_some() && model_type != MODEL_TYPE {
                return Err(format!(
                    "qwen4: nested config requires outer model_type={MODEL_TYPE:?}, got {model_type:?}"
                ));
            }
        }
        let text_model_type = required_string(text, "model_type")?;
        if text_model_type != TEXT_MODEL_TYPE {
            return Err(format!(
                "qwen4: expected text_config.model_type={TEXT_MODEL_TYPE:?}, got {text_model_type:?}"
            ));
        }
        if let Some(architectures) = config.get("architectures") {
            let values = architectures
                .as_array()
                .ok_or_else(|| "qwen4: `architectures` must be an array".to_string())?;
            if !values
                .iter()
                .filter_map(Value::as_str)
                .any(|name| name == ARCHITECTURE_NAME)
            {
                return Err(format!(
                    "qwen4: architectures does not contain {ARCHITECTURE_NAME:?}"
                ));
            }
        }

        let marker = match find_profile_marker(value, config, text)? {
            None => None,
            Some(_) if mode == ParseMode::Production => {
                return Err(format!(
                    "qwen4: developer profile marker `{PROFILE_KEY}` is not admitted by the production parser"
                ))
            }
            Some(marker) => Some(parse_profile_marker(marker)?),
        };
        let synthetic = marker.is_some();

        // A raw text config has the inner model type at its root; retain the
        // canonical outer family id in the typed config.
        let model_type = MODEL_TYPE.to_string();
        let dtype = SourceDType::parse(
            &optional_string(text, "dtype").unwrap_or_else(|| "bfloat16".into()),
        )
        .ok_or_else(|| "qwen4: dtype must be bfloat16".to_string())?;
        let rope_parameters = text.get("rope_parameters");
        let rope_theta = rope_parameters
            .and_then(|v| v.get("rope_theta"))
            .and_then(Value::as_f64)
            .or_else(|| text.get("rope_theta").and_then(Value::as_f64));
        let partial_rotary_factor = text
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .or_else(|| {
                rope_parameters
                    .and_then(|v| v.get("partial_rotary_factor"))
                    .and_then(Value::as_f64)
            });
        let (rope_theta, partial_rotary_factor) = if synthetic {
            (
                rope_theta.ok_or_else(|| {
                    "qwen4: synthetic profile requires an explicit rope_theta".to_string()
                })?,
                partial_rotary_factor.ok_or_else(|| {
                    "qwen4: synthetic profile requires an explicit partial_rotary_factor"
                        .to_string()
                })?,
            )
        } else {
            (
                rope_theta.unwrap_or(10_000_000.0),
                partial_rotary_factor.unwrap_or(0.25),
            )
        };
        if synthetic && text.get("eos_token_id").map_or(true, Value::is_null) {
            return Err("qwen4: synthetic profile requires an explicit eos_token_id".into());
        }

        let qsa_proof = qsa_proof(text);
        let layer_types = parse_layers(
            required_value(text, "layer_types")?,
            "layer_types",
            &qsa_proof,
        )?;

        let (profile, ffn, ple, mtp) = match marker {
            None => {
                let ffn = Qwen4FfnConfig::Moe {
                    num_experts: required_usize(text, "num_experts")?,
                    top_k: required_usize(text, "num_experts_per_tok")?,
                    intermediate_size: required_usize(text, "moe_intermediate_size")?,
                    shared_intermediate_size: Some(required_usize(
                        text,
                        "shared_expert_intermediate_size",
                    )?),
                    norm_topk_prob: optional_bool(text, "norm_topk_prob").unwrap_or(true),
                };
                let ple = Some(
                    Qwen4PleConfig::new(
                        &parse_usize_array(
                            required_value(text, "ple_layer_ids")?,
                            "ple_layer_ids",
                        )?,
                        required_usize(text, "ple_embed_dim")?,
                        required_usize(text, "ple_conv_kernel_size")?,
                        required_usize(text, "split_ngram_parts")?,
                        required_usize(text, "heads_per_ngram")?,
                        required_usize(text, "ngram_size")?,
                        required_u64(text, "ngram_vocab_size_base")?,
                        required_usize(text, "make_ngram_vocab_size_divisible_by")?,
                        Some(PLE_VALID_ROWS),
                    )
                    .map_err(|error| format!("{error} [flash-next PLE geometry]"))?,
                );
                let mtp_value = required_value(text, "mtp")?;
                if !mtp_value.is_object() {
                    return Err("qwen4: `text_config.mtp` must be an object".into());
                }
                let mtp = Some(parse_mtp(mtp_value, &qsa_proof)?);
                required_usize(text, "mtp_num_hidden_layers")?;
                required_bool(text, "mtp_use_dedicated_embeddings")?;
                (Qwen4Profile::FlashNext, ffn, ple, mtp)
            }
            Some(marker) => {
                check_ffn_text_agreement(text, &marker.ffn)?;
                let ple = parse_synthetic_ple(text, marker.ple_valid_rows)?;
                let mtp = match text.get("mtp").filter(|value| !value.is_null()) {
                    None => {
                        if optional_usize(text, "mtp_num_hidden_layers")?.unwrap_or(0) != 0 {
                            return Err(
                                "qwen4: mtp_num_hidden_layers is set but text_config.mtp is absent"
                                    .into(),
                            );
                        }
                        if optional_bool(text, "mtp_use_dedicated_embeddings") == Some(true) {
                            return Err(
                                "qwen4: mtp_use_dedicated_embeddings is set but text_config.mtp is absent"
                                    .into(),
                            );
                        }
                        None
                    }
                    Some(mtp_value) => {
                        if !mtp_value.is_object() {
                            return Err("qwen4: `text_config.mtp` must be an object".into());
                        }
                        required_usize(text, "mtp_num_hidden_layers")?;
                        required_bool(text, "mtp_use_dedicated_embeddings")?;
                        Some(parse_mtp(mtp_value, &qsa_proof)?)
                    }
                };
                (
                    Qwen4Profile::Synthetic(marker.profile),
                    marker.ffn,
                    ple,
                    mtp,
                )
            }
        };

        let (num_experts, num_experts_per_tok, moe_intermediate_size, shared, norm_topk_prob) =
            ffn.scalar_mirror();
        let trunk_iu4_default = parse_trunk_iu4_default(value, &profile)?;
        let cfg = Self {
            architecture_id: ARCH_ID,
            model_type,
            text_model_type: text_model_type.to_string(),
            profile,
            dtype,
            hidden_size: required_usize(text, "hidden_size")?,
            vocab_size: required_usize(text, "vocab_size")?,
            num_hidden_layers: required_usize(text, "num_hidden_layers")?,
            max_position_embeddings: required_usize(text, "max_position_embeddings")?,
            num_attention_heads: required_usize(text, "num_attention_heads")?,
            num_key_value_heads: required_usize(text, "num_key_value_heads")?,
            head_dim: required_usize(text, "head_dim")?,
            partial_rotary_factor,
            rope_theta,
            rope_scaling: hipfire_runtime::rope::rope_scaling(text)
                .map_err(|error| format!("qwen4: {error}"))?,
            attention_bias: optional_bool(text, "attention_bias").unwrap_or(false),
            layer_types,
            full_attention_interval: required_usize(text, "full_attention_interval")?,
            linear_num_key_heads: required_usize(text, "linear_num_key_heads")?,
            linear_num_value_heads: required_usize(text, "linear_num_value_heads")?,
            linear_key_head_dim: required_usize(text, "linear_key_head_dim")?,
            linear_value_head_dim: required_usize(text, "linear_value_head_dim")?,
            linear_conv_kernel_dim: required_usize(text, "linear_conv_kernel_dim")?,
            recurrent_state_dtype: RecurrentStateDType::parse(&required_string(
                text,
                "mamba_ssm_dtype",
            )?)
            .ok_or_else(|| "qwen4: mamba_ssm_dtype must be float32".to_string())?,
            indexer_n_heads: required_usize(text, "indexer_n_heads")?,
            indexer_kv_heads: required_usize(text, "indexer_kv_heads")?,
            indexer_head_dim: required_usize(text, "indexer_head_dim")?,
            indexer_budget: required_usize(text, "indexer_budget")?,
            indexer_compress_ratio: required_usize(text, "indexer_compress_ratio")?,
            ffn,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size: shared,
            norm_topk_prob,
            expert_keep: crate::ExpertKeep::from_metadata(value)?,
            output_gate_type: optional_string(text, "output_gate_type")
                .unwrap_or_else(|| "sigmoid".into()),
            hc_count: required_usize(text, "hc_count")?,
            hc_lowrank: required_usize(text, "hc_lowrank")?,
            ple_layer_ids: ple.as_ref().map_or_else(Vec::new, |p| vec![p.layer_id]),
            ple_conv_kernel_size: ple.as_ref().map_or(0, |p| p.conv_kernel_size),
            ple_embed_dim: ple.as_ref().map_or(0, |p| p.embed_dim),
            split_ngram_parts: ple.as_ref().map_or(0, |p| p.shard_count),
            heads_per_ngram: ple.as_ref().map_or(0, |p| p.heads_per_ngram),
            ngram_size: ple.as_ref().map_or(0, |p| p.ngram_size),
            ngram_vocab_size_base: ple.as_ref().map_or(0, |p| p.ngram_vocab_size_base),
            make_ngram_vocab_size_divisible_by: ple.as_ref().map_or(0, |p| p.padding_multiple),
            ple,
            eos_token_id: parse_eos(text.get("eos_token_id"), 248_044)?,
            tie_word_embeddings: optional_bool(text, "tie_word_embeddings")
                .or_else(|| optional_bool(config, "tie_word_embeddings"))
                .unwrap_or(false),
            mtp_num_hidden_layers: optional_usize(text, "mtp_num_hidden_layers")?.unwrap_or(0),
            mtp_use_dedicated_embeddings: optional_bool(text, "mtp_use_dedicated_embeddings")
                .unwrap_or(false),
            mtp,
            trunk_iu4_default,
            trunk_shard: None,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate the profile: exact pinned Qwen3.8-Flash-Next geometry for
    /// [`Qwen4Profile::FlashNext`], bounded checked geometry with pinned
    /// HC/GDN/QSA equations for a synthetic developer profile; then the typed
    /// `ffn`/`ple`/`mtp` and the retained scalar mirrors must agree.
    pub fn validate(&self) -> Result<(), String> {
        if let Some((rank, parts)) = self.trunk_shard {
            // A rank-local shard config is the global config scaled by its own constructor:
            // validate the reconstructed global one against every pin, then require this
            // config to be exactly what the constructor makes of it.
            let global = self.trunk_shard_global()?;
            global.validate()?;
            if global.trunk_shard_local(rank, parts)? != *self {
                return Err(format!(
                    "qwen4: the rank-local trunk-shard config (rank {rank} of {parts}) is not the \
                     shard of its global config"
                ));
            }
            return self.validate_typed_agreement();
        }
        expect(self.architecture_id, ARCH_ID, "architecture_id")?;
        expect_str(&self.model_type, MODEL_TYPE, "model_type")?;
        expect_str(&self.text_model_type, TEXT_MODEL_TYPE, "text_model_type")?;
        expect_eq(self.dtype, SourceDType::BF16, "dtype")?;
        if let Some(keep) = &self.expert_keep {
            expect(keep.n_experts(), self.num_experts, "expert_keep.n_experts")?;
            expect(
                keep.n_layers(),
                self.num_hidden_layers,
                "expert_keep.layers",
            )?;
            // The masked router kernel is compiled for exactly 512 experts and top-10.
            if self.num_experts != 512 || self.num_experts_per_tok != 10 {
                return Err(format!(
                    "qwen4: expert_keep requires num_experts=512 and num_experts_per_tok=10 (got {} and {})",
                    self.num_experts, self.num_experts_per_tok
                ));
            }
            if !self.norm_topk_prob {
                return Err("qwen4: expert_keep requires norm_topk_prob=true".into());
            }
            for layer in 0..keep.n_layers() {
                if keep.kept(layer).len() < self.num_experts_per_tok {
                    return Err(format!(
                        "qwen4: expert_keep layer {layer} has fewer than top_k experts"
                    ));
                }
            }
            if keep
                .mtp_kept()
                .is_some_and(|row| row.len() < self.num_experts_per_tok)
            {
                return Err("qwen4: expert_keep MTP has fewer than top_k experts".into());
            }
        }
        match &self.profile {
            Qwen4Profile::FlashNext
            | Qwen4Profile::FlashNextArOnly
            | Qwen4Profile::FlashNextHeadless => self.validate_flash_next()?,
            Qwen4Profile::Synthetic(profile) => self.validate_synthetic(profile)?,
        }
        self.validate_typed_agreement()
    }

    /// Shared-expert intermediate rows of `rank` of `parts` in `SHARED_UNIT`-wide units
    /// (640 = 384 + 256 at two parts).
    fn trunk_shard_shared(total: usize, rank: usize, parts: usize) -> Result<usize, String> {
        const UNIT: usize = 128;
        if total % UNIT != 0 || parts == 0 || total / UNIT < parts {
            return Err(format!(
                "qwen4: shared intermediate {total} cannot shard over {parts} parts"
            ));
        }
        let units = total / UNIT;
        let (base, extra) = (units / parts, units % parts);
        Ok((base + usize::from(rank < extra)) * UNIT)
    }

    /// The rank-local config of `rank` of `parts` of a TP trunk shard: GDN key/value heads and
    /// q heads divided by `parts` (KV heads stay: the q heads of a rank are an equal share of
    /// every KV group), the shared-expert intermediate this rank's share. Defined only for a
    /// validated global head-free Flash-Next config ([`Qwen4Profile::is_headless`]; the
    /// native MTP head reads the global geometry and is not sharded yet). Whether a load
    /// drops the head for the shard is ONE decision,
    /// [`crate::weights::trunk_shard::trunk_shard_keeps_mtp`]; the slice that keeps the
    /// head replicated on the global config relaxes the head refusal below.
    pub fn trunk_shard_local(&self, rank: usize, parts: usize) -> Result<Self, String> {
        if self.trunk_shard.is_some() {
            return Err("qwen4: the config is already a trunk-shard config".into());
        }
        if !self.profile.is_headless() || self.mtp.is_some() {
            return Err(
                "qwen4: the trunk shard needs a head-free Flash-Next profile (no native MTP head; \
                 the artifact is admitted headless under HIPFIRE_QWEN4_EP_TRUNK_SHARD)"
                    .into(),
            );
        }
        if parts < 2 || rank >= parts {
            return Err(format!(
                "qwen4: trunk shard rank {rank} of {parts} parts is invalid"
            ));
        }
        let divides = |n: usize| n % parts == 0;
        if !divides(self.linear_num_key_heads)
            || !divides(self.linear_num_value_heads)
            || !divides(self.num_attention_heads)
            || (self.num_attention_heads / parts) % self.num_key_value_heads != 0
        {
            return Err(format!(
                "qwen4: heads {}/{}/{} (kv {}) do not shard over {parts} parts",
                self.linear_num_key_heads,
                self.linear_num_value_heads,
                self.num_attention_heads,
                self.num_key_value_heads
            ));
        }
        let shared = Self::trunk_shard_shared(self.shared_expert_intermediate_size, rank, parts)?;
        let mut local = self.clone();
        local.linear_num_key_heads /= parts;
        local.linear_num_value_heads /= parts;
        local.num_attention_heads /= parts;
        local.shared_expert_intermediate_size = shared;
        if let Qwen4FfnConfig::Moe {
            shared_intermediate_size,
            ..
        } = &mut local.ffn
        {
            *shared_intermediate_size = Some(shared);
        }
        local.trunk_shard = Some((rank, parts));
        Ok(local)
    }

    /// The global config a rank-local trunk-shard config was cut from (`self` when it is not
    /// one): heads scaled back, the pinned shared-expert intermediate restored.
    pub fn trunk_shard_global(&self) -> Result<Self, String> {
        let Some((rank, parts)) = self.trunk_shard else {
            return Ok(self.clone());
        };
        if parts < 2 || rank >= parts {
            return Err(format!(
                "qwen4: trunk shard rank {rank} of {parts} parts is invalid"
            ));
        }
        let mut global = self.clone();
        global.trunk_shard = None;
        global.linear_num_key_heads *= parts;
        global.linear_num_value_heads *= parts;
        global.num_attention_heads *= parts;
        global.ffn = Qwen4FfnConfig::flash_next();
        global.shared_expert_intermediate_size = 640;
        Ok(global)
    }

    /// The geometry ranks must agree on (`check_replica`): the global config.
    pub fn replica_geometry(&self) -> Self {
        self.trunk_shard_global().unwrap_or_else(|_| self.clone())
    }

    /// Why a peer loaded as rank `rank` of `ranks` is not the same model as `rank0` (the
    /// config of rank 0), or `None` when it is. Ranks agree on the GLOBAL geometry
    /// ([`Self::replica_geometry`]) although their rank-local configs differ; on top of
    /// that the shard must be on for every rank or for none, over `ranks` parts, each rank
    /// holding its own slice (another rank's local config would size state and scratch for
    /// another rank's weights).
    pub fn replica_mismatch(&self, rank0: &Self, rank: usize, ranks: usize) -> Option<String> {
        if self.replica_geometry() != rank0.replica_geometry() {
            return Some("the loaded peer's model configuration differs from rank 0".to_string());
        }
        let expected = rank0.trunk_shard.map(|(_, parts)| (rank, parts));
        if self.trunk_shard != expected
            || rank0.trunk_shard.is_some_and(|(_, parts)| parts != ranks)
        {
            return Some(format!(
                "the loaded peer's trunk shard is {:?}, expected {expected:?} (rank 0: {:?}, {ranks} ranks)",
                self.trunk_shard, rank0.trunk_shard
            ));
        }
        None
    }

    /// The AR-only form of a validated Flash-Next config: the
    /// [`Qwen4Profile::FlashNextArOnly`] profile with no MTP head (`mtp` and
    /// its scalar mirrors cleared).  Selected by an artifact recipe that omits
    /// the head (`Qwen4TierRecipe::StockFp8`), never by config contents, so a
    /// head-free artifact is admitted as head-free instead of under a config
    /// that names a head nothing loads.  Only Flash-Next converts.
    pub fn into_ar_only(self) -> Result<Self, String> {
        match self.profile {
            Qwen4Profile::FlashNext => {}
            Qwen4Profile::FlashNextArOnly => return Ok(self),
            Qwen4Profile::FlashNextHeadless => {
                return Err(format!(
                "qwen4: profile {} is the packed head-dropped form, not the stock-FP8 AR-only one",
                self.profile.id()
            ))
            }
            Qwen4Profile::Synthetic(_) => {
                return Err(format!(
                    "qwen4: profile {} has no AR-only form (only Flash-Next does)",
                    self.profile.id()
                ))
            }
        }
        self.drop_head(Qwen4Profile::FlashNextArOnly)
    }

    /// The head-dropped PACKED form of a validated Flash-Next config: the
    /// [`Qwen4Profile::FlashNextHeadless`] profile with no MTP head (`mtp` and its scalar
    /// mirrors cleared), geometry unchanged. Selected by the trunk shard's load
    /// ([`crate::weights::trunk_shard::trunk_shard_drops_mtp_head`]), never by config
    /// contents or the artifact recipe (the artifact is the shipped packed one). Unlike
    /// [`Self::into_ar_only`] it does not bind the stock-FP8 numerical pins.
    pub fn into_headless(self) -> Result<Self, String> {
        match self.profile {
            Qwen4Profile::FlashNext => {}
            Qwen4Profile::FlashNextHeadless => return Ok(self),
            Qwen4Profile::FlashNextArOnly => {
                return Err(format!(
                "qwen4: profile {} is the stock-FP8 AR-only form; it has no packed headless form",
                self.profile.id()
            ))
            }
            Qwen4Profile::Synthetic(_) => {
                return Err(format!(
                    "qwen4: profile {} has no headless form (only Flash-Next does)",
                    self.profile.id()
                ))
            }
        }
        self.drop_head(Qwen4Profile::FlashNextHeadless)
    }

    fn drop_head(mut self, profile: Qwen4Profile) -> Result<Self, String> {
        self.validate()?;
        self.profile = profile;
        self.mtp = None;
        self.mtp_num_hidden_layers = 0;
        self.mtp_use_dedicated_embeddings = false;
        self.validate()?;
        Ok(self)
    }

    /// Exact pinned Qwen3.8-Flash-Next geometry.
    fn validate_flash_next(&self) -> Result<(), String> {
        expect(self.hidden_size, 2560, "hidden_size")?;
        expect(self.vocab_size, 248_320, "vocab_size")?;
        expect(self.num_hidden_layers, 48, "num_hidden_layers")?;
        expect(
            self.max_position_embeddings,
            262_144,
            "max_position_embeddings",
        )?;
        expect(self.num_attention_heads, 24, "num_attention_heads")?;
        expect(self.num_key_value_heads, 2, "num_key_value_heads")?;
        expect(self.head_dim, 256, "head_dim")?;
        expect_float(self.partial_rotary_factor, 0.25, "partial_rotary_factor")?;
        expect_float(self.rope_theta, 10_000_000.0, "rope_theta")?;
        expect(self.attention_bias, false, "attention_bias")?;
        expect(self.full_attention_interval, 4, "full_attention_interval")?;
        expect(self.linear_num_key_heads, 16, "linear_num_key_heads")?;
        expect(self.linear_num_value_heads, 48, "linear_num_value_heads")?;
        expect(self.linear_key_head_dim, 128, "linear_key_head_dim")?;
        expect(self.linear_value_head_dim, 128, "linear_value_head_dim")?;
        expect(self.linear_conv_kernel_dim, 4, "linear_conv_kernel_dim")?;
        expect_eq(
            self.recurrent_state_dtype,
            RecurrentStateDType::F32,
            "mamba_ssm_dtype",
        )?;
        expect(self.indexer_n_heads, 4, "indexer_n_heads")?;
        expect(self.indexer_kv_heads, 1, "indexer_kv_heads")?;
        expect(self.indexer_head_dim, 128, "indexer_head_dim")?;
        expect(self.indexer_budget, 2048, "indexer_budget")?;
        expect(self.indexer_compress_ratio, 4, "indexer_compress_ratio")?;
        expect(self.num_experts, 512, "num_experts")?;
        expect(self.num_experts_per_tok, 10, "num_experts_per_tok")?;
        expect(self.moe_intermediate_size, 640, "moe_intermediate_size")?;
        expect(
            self.shared_expert_intermediate_size,
            640,
            "shared_expert_intermediate_size",
        )?;
        expect(self.norm_topk_prob, true, "norm_topk_prob")?;
        expect_str(&self.output_gate_type, "sigmoid", "output_gate_type")?;
        expect(self.hc_count, 4, "hc_count")?;
        expect(self.hc_lowrank, 320, "hc_lowrank")?;
        expect(
            self.ple_layer_ids.as_slice(),
            [2usize].as_slice(),
            "ple_layer_ids",
        )?;
        expect(self.ple_conv_kernel_size, 4, "ple_conv_kernel_size")?;
        expect(self.ple_embed_dim, 2560, "ple_embed_dim")?;
        expect(self.split_ngram_parts, 128, "split_ngram_parts")?;
        expect(self.heads_per_ngram, 8, "heads_per_ngram")?;
        expect(self.ngram_size, 3, "ngram_size")?;
        expect(
            self.ngram_vocab_size_base,
            20_000_000,
            "ngram_vocab_size_base",
        )?;
        expect(
            self.make_ngram_vocab_size_divisible_by,
            128,
            "make_ngram_vocab_size_divisible_by",
        )?;
        if self.eos_token_id as usize >= self.vocab_size {
            return Err(format!(
                "qwen4: eos_token_id={} is outside vocab_size={}",
                self.eos_token_id, self.vocab_size
            ));
        }
        expect(self.tie_word_embeddings, false, "tie_word_embeddings")?;
        // The AR-only profile carries no head, so its scalar mirror is 0
        // ("0 when the head is absent"); the checkpoint profile pins one layer.
        let mtp_layers = if self.profile.is_headless() { 0 } else { 1 };
        expect(
            self.mtp_num_hidden_layers,
            mtp_layers,
            "mtp_num_hidden_layers",
        )?;
        expect(
            self.mtp_use_dedicated_embeddings,
            false,
            "mtp_use_dedicated_embeddings",
        )?;
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "qwen4: layer_types length={} (expected {})",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        let expected_layers =
            scheduled_layers(self.num_hidden_layers, self.full_attention_interval);
        if self.layer_types != expected_layers {
            return Err(format!(
                "qwen4: layer_types do not follow 3 linear + 1 full order: {:?}",
                self.layer_types
            ));
        }
        if self.n_linear_layers() != 36 || self.n_full_layers() != 12 {
            return Err(format!(
                "qwen4: layer counts linear={} full={} (expected 36/12)",
                self.n_linear_layers(),
                self.n_full_layers()
            ));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
            || self.ple_embed_dim % (2 * self.heads_per_ngram) != 0
        {
            return Err("qwen4: head or PLE dimensions are not divisible".into());
        }
        if self.ffn != Qwen4FfnConfig::flash_next() {
            return Err(format!(
                "qwen4: typed ffn {:?} is not the pinned MoE",
                self.ffn
            ));
        }
        if self.ple.as_ref() != Some(&Qwen4PleConfig::flash_next()) {
            return Err("qwen4: typed ple is not the pinned Flash-Next PLE".into());
        }
        if self.profile.is_headless() {
            return match self.mtp {
                None => Ok(()),
                Some(_) => Err("qwen4: a head-free Flash-Next profile carries no MTP head".into()),
            };
        }
        self.mtp
            .as_ref()
            .ok_or_else(|| "qwen4: Flash-Next requires the native MTP head".to_string())?
            .validate()
    }

    /// Developer-only synthetic geometry: bounded, checked, with the pinned
    /// HC/GDN/QSA equations.  Unsupported semantics are rejected, not
    /// approximated.
    fn validate_synthetic(&self, profile: &Qwen4SyntheticProfile) -> Result<(), String> {
        validate_profile_id(&profile.id)?;
        match (&self.ffn, profile.equation_source) {
            (Qwen4FfnConfig::Moe { .. }, Qwen4EquationSource::PinnedUpstream)
            | (Qwen4FfnConfig::Dense { .. }, Qwen4EquationSource::NonUpstreamDense) => {}
            (ffn, source) => {
                return Err(format!(
                    "qwen4: ffn {ffn:?} cannot use equation_source={:?} (moe requires pinned-upstream, dense requires non-upstream-dense)",
                    source.as_str()
                ))
            }
        }
        bounded(self.hidden_size, 1, MAX_HIDDEN, "hidden_size")?;
        bounded(self.vocab_size, 1, MAX_VOCAB, "vocab_size")?;
        bounded(self.num_hidden_layers, 1, MAX_LAYERS, "num_hidden_layers")?;
        bounded(
            self.max_position_embeddings,
            1,
            MAX_CONTEXT_POSITIONS,
            "max_position_embeddings",
        )?;
        bounded(
            self.num_attention_heads,
            1,
            MAX_ATTENTION_HEADS,
            "num_attention_heads",
        )?;
        bounded(
            self.num_key_value_heads,
            1,
            self.num_attention_heads,
            "num_key_value_heads",
        )?;
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(format!(
                "qwen4: num_attention_heads={} is not divisible by num_key_value_heads={}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        bounded(self.head_dim, 1, MAX_HEAD_DIM, "head_dim")?;
        if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
            return Err(format!(
                "qwen4: rope_theta={} must be finite and positive",
                self.rope_theta
            ));
        }
        if !(self.partial_rotary_factor.is_finite()
            && self.partial_rotary_factor > 0.0
            && self.partial_rotary_factor <= 1.0)
        {
            return Err(format!(
                "qwen4: partial_rotary_factor={} must be in (0, 1]",
                self.partial_rotary_factor
            ));
        }
        let rotary = self.head_dim as f64 * self.partial_rotary_factor;
        let rotary_rounded = rotary.round();
        if (rotary - rotary_rounded).abs() > 1e-6
            || rotary_rounded < 2.0
            || (rotary_rounded as usize) % 2 != 0
        {
            return Err(format!(
                "qwen4: head_dim*partial_rotary_factor={rotary} must be an even integer >= 2"
            ));
        }
        expect(self.attention_bias, false, "attention_bias")?;

        // Gated DeltaNet.
        bounded(
            self.linear_num_key_heads,
            1,
            MAX_GDN_HEADS,
            "linear_num_key_heads",
        )?;
        bounded(
            self.linear_num_value_heads,
            1,
            MAX_GDN_HEADS,
            "linear_num_value_heads",
        )?;
        if self.linear_num_value_heads % self.linear_num_key_heads != 0 {
            return Err(format!(
                "qwen4: linear_num_value_heads={} is not divisible by linear_num_key_heads={}",
                self.linear_num_value_heads, self.linear_num_key_heads
            ));
        }
        bounded(
            self.linear_key_head_dim,
            1,
            MAX_GDN_HEAD_DIM,
            "linear_key_head_dim",
        )?;
        bounded(
            self.linear_value_head_dim,
            1,
            MAX_GDN_HEAD_DIM,
            "linear_value_head_dim",
        )?;
        bounded(
            self.linear_conv_kernel_dim,
            1,
            MAX_CONV_KERNEL,
            "linear_conv_kernel_dim",
        )?;
        expect_eq(
            self.recurrent_state_dtype,
            RecurrentStateDType::F32,
            "mamba_ssm_dtype",
        )?;
        expect_str(&self.output_gate_type, "sigmoid", "output_gate_type")?;

        // QSA indexer.
        bounded(
            self.indexer_n_heads,
            1,
            MAX_INDEXER_HEADS,
            "indexer_n_heads",
        )?;
        expect(self.indexer_kv_heads, 1, "indexer_kv_heads")?;
        bounded(
            self.indexer_head_dim,
            1,
            MAX_INDEXER_DIM,
            "indexer_head_dim",
        )?;
        bounded(self.indexer_budget, 1, MAX_INDEXER_BUDGET, "indexer_budget")?;
        bounded(
            self.indexer_compress_ratio,
            1,
            MAX_INDEXER_RATIO,
            "indexer_compress_ratio",
        )?;
        if self.indexer_budget % self.indexer_compress_ratio != 0 {
            return Err(format!(
                "qwen4: indexer_budget={} is not divisible by indexer_compress_ratio={}",
                self.indexer_budget, self.indexer_compress_ratio
            ));
        }
        if rotary_rounded as usize > self.indexer_head_dim {
            return Err(format!(
                "qwen4: rotary dimension {rotary_rounded} exceeds indexer_head_dim={}",
                self.indexer_head_dim
            ));
        }

        // FFN.
        match &self.ffn {
            Qwen4FfnConfig::Dense { intermediate_size } => {
                bounded(
                    *intermediate_size,
                    1,
                    MAX_INTERMEDIATE,
                    "dense intermediate_size",
                )?;
            }
            Qwen4FfnConfig::Moe {
                num_experts,
                top_k,
                intermediate_size,
                shared_intermediate_size,
                ..
            } => {
                bounded(*num_experts, 1, MAX_EXPERTS, "num_experts")?;
                bounded(*top_k, 1, *num_experts, "num_experts_per_tok")?;
                bounded(
                    *intermediate_size,
                    1,
                    MAX_INTERMEDIATE,
                    "moe_intermediate_size",
                )?;
                if let Some(shared) = shared_intermediate_size {
                    bounded(
                        *shared,
                        1,
                        MAX_INTERMEDIATE,
                        "shared_expert_intermediate_size",
                    )?;
                }
            }
        }

        // Hyper-connections.
        bounded(self.hc_count, 2, MAX_HC_COUNT, "hc_count")?;
        bounded(self.hc_lowrank, 1, MAX_HC_LOWRANK, "hc_lowrank")?;

        if self.eos_token_id as usize >= self.vocab_size {
            return Err(format!(
                "qwen4: eos_token_id={} is outside vocab_size={}",
                self.eos_token_id, self.vocab_size
            ));
        }
        expect(self.tie_word_embeddings, false, "tie_word_embeddings")?;
        expect(
            self.mtp_use_dedicated_embeddings,
            false,
            "mtp_use_dedicated_embeddings",
        )?;

        // Layer schedule: explicit layer_types must follow the interval.  A
        // trunk may be hybrid or linear-only (a prefix shorter than the
        // interval); a missing branch simply has no layers of that kind.
        bounded(
            self.full_attention_interval,
            1,
            MAX_LAYERS,
            "full_attention_interval",
        )?;
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "qwen4: layer_types length={} (expected {})",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        let expected_layers =
            scheduled_layers(self.num_hidden_layers, self.full_attention_interval);
        if self.layer_types != expected_layers {
            return Err(format!(
                "qwen4: layer_types do not follow full_attention_interval={}: {:?}",
                self.full_attention_interval, self.layer_types
            ));
        }

        // Optional PLE and MTP.
        if let Some(ple) = &self.ple {
            if ple.layer >= self.num_hidden_layers {
                return Err(format!(
                    "qwen4: ple_layer_ids={} is outside num_hidden_layers={}",
                    ple.layer_id, self.num_hidden_layers
                ));
            }
        }
        if let Some(mtp) = &self.mtp {
            mtp.validate_shape()?;
        }

        // Derived extents must fit the kernels' i32 indexing; the total FFN
        // weight count must stay under the element bound.
        let hidden = self.hidden_size;
        let wide = hidden.checked_mul(self.hc_count);
        let gdn_qk = self
            .linear_num_key_heads
            .checked_mul(self.linear_key_head_dim);
        let gdn_v = self
            .linear_num_value_heads
            .checked_mul(self.linear_value_head_dim);
        let widths = [
            ("hyper-connection width", wide),
            (
                "hyper-connection low-rank down extent",
                wide.and_then(|wide| wide.checked_mul(self.hc_lowrank)),
            ),
            (
                "attention q width",
                self.num_attention_heads
                    .checked_mul(self.head_dim)
                    .and_then(|width| width.checked_mul(2)),
            ),
            (
                "attention kv width",
                self.num_key_value_heads.checked_mul(self.head_dim),
            ),
            ("GDN q/k width", gdn_qk),
            ("GDN value width", gdn_v),
            (
                "GDN qkv width",
                gdn_qk
                    .and_then(|qk| qk.checked_mul(2))
                    .and_then(|qk| gdn_v.and_then(|v| qk.checked_add(v))),
            ),
            (
                "indexer width",
                self.indexer_n_heads.checked_mul(self.indexer_head_dim),
            ),
            (
                "PLE channel width",
                self.ple
                    .as_ref()
                    .map_or(Some(0), |_| self.hidden_size.checked_mul(self.hc_count)),
            ),
        ];
        for (name, width) in widths {
            match width {
                Some(width) if width <= MAX_WIDTH => {}
                _ => return Err(format!("qwen4: derived {name} overflows i32 indexing")),
            }
        }
        let ffn_elements: Option<u64> = match &self.ffn {
            Qwen4FfnConfig::Dense { intermediate_size } => {
                (hidden as u64).checked_mul(*intermediate_size as u64 * 3)
            }
            Qwen4FfnConfig::Moe {
                num_experts,
                intermediate_size,
                shared_intermediate_size,
                ..
            } => (hidden as u64)
                .checked_mul(3 * *intermediate_size as u64)
                .and_then(|per_expert| per_expert.checked_mul(*num_experts as u64))
                .and_then(|routed| {
                    (hidden as u64)
                        .checked_mul(3 * shared_intermediate_size.unwrap_or(0) as u64)
                        .and_then(|shared| routed.checked_add(shared))
                }),
        };
        let layer_elements =
            ffn_elements.and_then(|ffn| ffn.checked_mul(self.num_hidden_layers as u64));
        match layer_elements {
            Some(elements) if elements <= MAX_ELEMENTS => {}
            _ => {
                return Err("qwen4: FFN weight element count overflows the developer bound".into())
            }
        }
        Ok(())
    }

    /// The typed `ffn`/`ple`/`mtp` are authoritative; the retained scalar
    /// checkpoint fields must mirror them exactly.
    fn validate_typed_agreement(&self) -> Result<(), String> {
        let (experts, top_k, intermediate, shared, norm) = self.ffn.scalar_mirror();
        expect(self.num_experts, experts, "num_experts (typed ffn mirror)")?;
        expect(
            self.num_experts_per_tok,
            top_k,
            "num_experts_per_tok (typed ffn mirror)",
        )?;
        expect(
            self.moe_intermediate_size,
            intermediate,
            "moe_intermediate_size (typed ffn mirror)",
        )?;
        expect(
            self.shared_expert_intermediate_size,
            shared,
            "shared_expert_intermediate_size (typed ffn mirror)",
        )?;
        expect(
            self.norm_topk_prob,
            norm,
            "norm_topk_prob (typed ffn mirror)",
        )?;
        match &self.ple {
            Some(ple) => {
                if ple.layer + 1 != ple.layer_id {
                    return Err("qwen4: typed ple layer/layer_id disagree".into());
                }
                if ple.layer >= self.num_hidden_layers
                    || self.layer_types.get(ple.layer) != Some(&LayerType::LinearAttention)
                {
                    return Err(format!(
                        "qwen4: PLE layer {} must be an existing linear-attention layer",
                        ple.layer
                    ));
                }
                expect(
                    self.ple_layer_ids.as_slice(),
                    [ple.layer_id].as_slice(),
                    "ple_layer_ids (typed ple mirror)",
                )?;
                expect(
                    self.ple_conv_kernel_size,
                    ple.conv_kernel_size,
                    "ple_conv_kernel_size (typed ple mirror)",
                )?;
                expect(
                    self.ple_embed_dim,
                    ple.embed_dim,
                    "ple_embed_dim (typed ple mirror)",
                )?;
                expect(
                    self.split_ngram_parts,
                    ple.shard_count,
                    "split_ngram_parts (typed ple mirror)",
                )?;
                expect(
                    self.heads_per_ngram,
                    ple.heads_per_ngram,
                    "heads_per_ngram (typed ple mirror)",
                )?;
                expect(
                    self.ngram_size,
                    ple.ngram_size,
                    "ngram_size (typed ple mirror)",
                )?;
                expect(
                    self.ngram_vocab_size_base,
                    ple.ngram_vocab_size_base,
                    "ngram_vocab_size_base (typed ple mirror)",
                )?;
                expect(
                    self.make_ngram_vocab_size_divisible_by,
                    ple.padding_multiple,
                    "make_ngram_vocab_size_divisible_by (typed ple mirror)",
                )?;
            }
            None => {
                if !self.ple_layer_ids.is_empty()
                    || self.ple_conv_kernel_size != 0
                    || self.ple_embed_dim != 0
                    || self.split_ngram_parts != 0
                    || self.heads_per_ngram != 0
                    || self.ngram_size != 0
                    || self.ngram_vocab_size_base != 0
                    || self.make_ngram_vocab_size_divisible_by != 0
                {
                    return Err("qwen4: PLE scalar fields are set but typed ple is absent".into());
                }
            }
        }
        let mtp_layers = self.mtp.as_ref().map_or(0, |mtp| mtp.num_hidden_layers);
        expect(
            self.mtp_num_hidden_layers,
            mtp_layers,
            "mtp_num_hidden_layers (typed mtp mirror)",
        )
    }

    /// Number of Gated DeltaNet layers.
    pub fn n_linear_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::LinearAttention)
            .count()
    }

    /// Number of QSA/full-attention layers.
    pub fn n_full_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|kind| **kind == LayerType::FullAttention)
            .count()
    }

    /// RoPE width of the full-attention q/k rows: `head_dim *
    /// partial_rotary_factor` (64 for Flash-Next).  Validation guarantees an
    /// even integer >= 2.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor).round() as usize
    }

    /// The full-attention rotary embedding (the MTP head shares it).
    pub fn rope(&self) -> rdna_compute::rope::Rope {
        rdna_compute::rope::Rope {
            theta: self.rope_theta,
            scaling: self.rope_scaling,
        }
    }

    /// Positions the RoPE covers: `max_position_embeddings`, or the YaRN
    /// `factor x original` context.
    pub fn context_limit(&self) -> usize {
        self.rope_scaling
            .context_limit(self.max_position_embeddings)
    }

    /// Number of values in each sharded n-gram row (0 without PLE).
    pub fn ple_row_width(&self) -> usize {
        self.ple.as_ref().map_or(0, |ple| ple.row_width)
    }

    /// Dilation of the PLE short convolution: upstream spaces its taps one
    /// n-gram apart (0 without PLE).
    pub fn ple_conv_dilation(&self) -> usize {
        self.ple.as_ref().map_or(0, Qwen4PleConfig::conv_dilation)
    }

    /// Rows of PLE short-convolution history, `(kernel - 1) * dilation`
    /// (0 without PLE).
    pub fn ple_conv_history_rows(&self) -> usize {
        self.ple
            .as_ref()
            .map_or(0, Qwen4PleConfig::conv_history_rows)
    }

    /// The capacity selected by QSA: indexer budget plus the incomplete tail.
    pub fn qsa_selected_capacity(&self) -> usize {
        self.indexer_budget + self.indexer_compress_ratio.saturating_sub(1)
    }

    /// Bytes of one QSA layer's context-sized storage, including its
    /// optional derived F16 arena (four-token-rounded capacity) and pooled
    /// BF16 mirror (`ceil(max_seq / compress)` blocks of `index_dim` BF16
    /// channels). The MTP head passes its plain F32 format and never owns a
    /// derived arena.
    pub fn qsa_context_arena_bytes(
        &self,
        max_seq: usize,
        layout: impl Into<rdna_compute::tensor_ops::QsaStateLayout>,
    ) -> Option<usize> {
        let layout = layout.into();
        let format = layout.kv;
        let kv_row = format.kv_row_bytes(self.num_key_value_heads, self.head_dim);
        let raw = self.indexer_kv_heads.checked_mul(self.indexer_head_dim)?;
        let index_row = raw.checked_mul(format.index_dtype().size())?;
        let pooled_rows = max_seq.div_ceil(self.indexer_compress_ratio);
        max_seq
            .checked_mul(kv_row.checked_mul(2)?.checked_add(index_row)?)?
            .checked_add(pooled_rows.checked_mul(index_row)?)?
            .checked_add(pooled_rows.checked_mul(layout.pooled_bf16_bytes_per_block(raw))?)?
            .checked_add(if layout.f16_arena {
                max_seq
                    .div_ceil(4)
                    .checked_mul(4)?
                    .checked_mul(self.num_key_value_heads)?
                    .checked_mul(self.head_dim)?
                    .checked_mul(4)?
            } else {
                0
            })
    }
}

#[cfg(test)]
impl Qwen4Config {
    /// The complete HFQ metadata JSON of this config: the nested envelope
    /// accepted by [`Self::from_developer_metadata_json`] (plus a top-level
    /// `hipfire_profile` marker for a synthetic profile, none for
    /// Flash-Next) and, when PLE is present, a top-level version-1
    /// `qwen4_ple` object: exactly the canonical metadata for Flash-Next, a
    /// valid generalized one using the profile's prime-derived head tables.
    pub(crate) fn developer_metadata_json(&self) -> String {
        self.developer_value().to_string()
    }

    pub(crate) fn developer_value(&self) -> Value {
        use serde_json::json;
        let mut text = json!({
            "model_type": self.text_model_type,
            "dtype": self.dtype.as_str(),
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "max_position_embeddings": self.max_position_embeddings,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "partial_rotary_factor": self.partial_rotary_factor,
            "rope_parameters": {"rope_theta": self.rope_theta},
            "attention_bias": self.attention_bias,
            "layer_types": self.layer_types.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
            "full_attention_interval": self.full_attention_interval,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "mamba_ssm_dtype": self.recurrent_state_dtype.as_str(),
            "indexer_n_heads": self.indexer_n_heads,
            "indexer_kv_heads": self.indexer_kv_heads,
            "indexer_head_dim": self.indexer_head_dim,
            "indexer_budget": self.indexer_budget,
            "indexer_compress_ratio": self.indexer_compress_ratio,
            "output_gate_type": self.output_gate_type,
            "hc_count": self.hc_count,
            "hc_lowrank": self.hc_lowrank,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
        });
        let object = text.as_object_mut().expect("text config is an object");
        let marker_ffn = match &self.ffn {
            Qwen4FfnConfig::Moe {
                num_experts,
                top_k,
                intermediate_size,
                shared_intermediate_size,
                norm_topk_prob,
            } => {
                object.insert("num_experts".into(), json!(num_experts));
                object.insert("num_experts_per_tok".into(), json!(top_k));
                object.insert("moe_intermediate_size".into(), json!(intermediate_size));
                object.insert("norm_topk_prob".into(), json!(norm_topk_prob));
                let mut marker = json!({
                    "kind": "moe",
                    "num_experts": num_experts,
                    "top_k": top_k,
                    "intermediate_size": intermediate_size,
                    "norm_topk_prob": norm_topk_prob,
                });
                if let Some(shared) = shared_intermediate_size {
                    object.insert("shared_expert_intermediate_size".into(), json!(shared));
                    marker["shared_intermediate_size"] = json!(shared);
                }
                marker
            }
            Qwen4FfnConfig::Dense { intermediate_size } => {
                json!({"kind": "dense", "intermediate_size": intermediate_size})
            }
        };
        if let Some(ple) = &self.ple {
            object.insert("ple_layer_ids".into(), json!([ple.layer_id]));
            object.insert("ple_conv_kernel_size".into(), json!(ple.conv_kernel_size));
            object.insert("ple_embed_dim".into(), json!(ple.embed_dim));
            object.insert("split_ngram_parts".into(), json!(ple.shard_count));
            object.insert("heads_per_ngram".into(), json!(ple.heads_per_ngram));
            object.insert("ngram_size".into(), json!(ple.ngram_size));
            object.insert(
                "ngram_vocab_size_base".into(),
                json!(ple.ngram_vocab_size_base),
            );
            object.insert(
                "make_ngram_vocab_size_divisible_by".into(),
                json!(ple.padding_multiple),
            );
        }
        if let Some(mtp) = &self.mtp {
            object.insert(
                "mtp_num_hidden_layers".into(),
                json!(self.mtp_num_hidden_layers),
            );
            object.insert(
                "mtp_use_dedicated_embeddings".into(),
                json!(self.mtp_use_dedicated_embeddings),
            );
            object.insert(
                "mtp".into(),
                json!({
                    "hybrid": mtp.hybrid,
                    "layer_types": mtp.layer_types.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
                    "num_hidden_layers": mtp.num_hidden_layers,
                    "rope_theta": mtp.rope_theta,
                }),
            );
        }
        let mut envelope = json!({
            "architectures": [ARCHITECTURE_NAME],
            "model_type": self.model_type,
            "text_config": text,
        });
        if let Qwen4Profile::Synthetic(profile) = &self.profile {
            let mut marker = json!({
                "id": profile.id,
                "developer_only": true,
                "equation_source": profile.equation_source.as_str(),
                "ffn": marker_ffn,
            });
            if let Some(ple) = &self.ple {
                marker["ple"] = json!({"valid_rows": ple.valid_rows});
            }
            envelope[PROFILE_KEY] = marker;
        }
        if let Some(ple) = &self.ple {
            envelope["qwen4_ple"] = ple_metadata_object(ple);
        }
        envelope
    }
}

/// The `qwen4_ple` metadata object of a PLE geometry: the canonical
/// Flash-Next metadata for the pinned geometry, otherwise the upstream rule
/// (the `head_count` successive primes at or above `ngram_vocab_size_base`).
#[cfg(test)]
fn ple_metadata_object(ple: &Qwen4PleConfig) -> Value {
    use serde_json::json;
    if *ple == Qwen4PleConfig::flash_next() {
        let canonical = PleHashMetadata::qwen4();
        return json!({
            "version": 1,
            "multipliers": canonical.multipliers(),
            "head_vocab_sizes": canonical.head_vocab_sizes(),
            "head_offsets": canonical.head_offsets(),
            "padded_rows": canonical.padded_rows(),
        });
    }
    let sizes = ple_head_vocab_sizes(ple.ngram_vocab_size_base, ple.head_count)
        .expect("a validated PLE geometry has derivable head tables");
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut offset = 0u64;
    for size in &sizes {
        offsets.push(offset);
        offset += size;
    }
    json!({
        "version": 1,
        "multipliers": [3, 5, 7],
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "padded_rows": ple.padded_rows,
    })
}

/// Largest context any Qwen4 load admits (the `memory.max_seq` ceiling, the
/// Flash-Next native 262,144 under YaRN factor 4); a model admits only up to
/// its own [`Qwen4Config::context_limit`]. Past 15,360 pooled blocks (61,440
/// tokens) the QSA selector keeps its score rows in global memory instead of
/// LDS; whether a context fits is decided by the state allocation, which names
/// the MiB it needed when it does not.
pub const QWEN4_MAX_CONTEXT: usize = 1_048_576;

/// Context a Qwen4 load gets when `max_seq` is omitted.
pub const QWEN4_DEFAULT_CONTEXT: usize = 32768;

pub(crate) fn compact_test_config() -> Qwen4Config {
    let layers: Vec<_> = (0..48)
        .map(|idx| {
            if idx % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let value = serde_json::json!({
        "architectures": [ARCHITECTURE_NAME],
        "model_type": MODEL_TYPE,
        "text_config": {
            "model_type": TEXT_MODEL_TYPE,
            "dtype": "bfloat16",
            "hidden_size": 2560,
            "vocab_size": 248320,
            "num_hidden_layers": 48,
            "max_position_embeddings": 262144,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {"rope_theta": 10000000.0},
            "attention_bias": false,
            "layer_types": layers,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "norm_topk_prob": true,
            "hc_count": 4,
            "hc_lowrank": 320,
            "ple_layer_ids": [2],
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "split_ngram_parts": 128,
            "heads_per_ngram": 8,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "make_ngram_vocab_size_divisible_by": 128,
            "eos_token_id": 248044,
            "tie_word_embeddings": false,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "output_gate_type": "sigmoid",
            "mtp": {
                "hybrid": true,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "rope_theta": 10000000.0
            }
        }
    });
    Qwen4Config::from_value(&value).expect("compact Qwen4 GPU fixture config")
}

/// Raw nested developer JSON of a small synthetic profile that passes
/// `from_developer_value`: attention/GDN/QSA dimensions are fixed and tiny,
/// independent of `hidden_size`.  Layers alternate linear/full
/// (`full_attention_interval = 2`, so `layers >= 2`).  When `with_ple` the PLE
/// is layer id 1 (zero-based layer 0, a linear layer) with the
/// head-count-2/row-width-8/24-padded-row shape; when `with_mtp` a one-layer
/// full-attention hybrid head is present.
#[cfg(test)]
pub(crate) fn synthetic_fixture_value(
    ffn: &Qwen4FfnConfig,
    hidden_size: usize,
    num_hidden_layers: usize,
    with_ple: bool,
    with_mtp: bool,
) -> Value {
    use serde_json::json;
    let layer_types: Vec<&str> = scheduled_layers(num_hidden_layers, 2)
        .iter()
        .map(|kind| kind.as_str())
        .collect();
    let mut text = json!({
        "model_type": TEXT_MODEL_TYPE,
        "dtype": "bfloat16",
        "hidden_size": hidden_size,
        "vocab_size": 64,
        "num_hidden_layers": num_hidden_layers,
        "max_position_embeddings": 4096,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "partial_rotary_factor": 0.5,
        "rope_parameters": {"rope_theta": 10000.0},
        "attention_bias": false,
        "layer_types": layer_types,
        "full_attention_interval": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 16,
        "indexer_budget": 8,
        "indexer_compress_ratio": 4,
        "output_gate_type": "sigmoid",
        "hc_count": 2,
        "hc_lowrank": 4,
        "eos_token_id": 3,
        "tie_word_embeddings": false,
    });
    let object = text.as_object_mut().expect("text config is an object");
    let marker_ffn = match ffn {
        Qwen4FfnConfig::Moe {
            num_experts,
            top_k,
            intermediate_size,
            shared_intermediate_size,
            norm_topk_prob,
        } => {
            object.insert("num_experts".into(), json!(num_experts));
            object.insert("num_experts_per_tok".into(), json!(top_k));
            object.insert("moe_intermediate_size".into(), json!(intermediate_size));
            object.insert("norm_topk_prob".into(), json!(norm_topk_prob));
            let mut marker = json!({
                "kind": "moe",
                "num_experts": num_experts,
                "top_k": top_k,
                "intermediate_size": intermediate_size,
                "norm_topk_prob": norm_topk_prob,
            });
            if let Some(shared) = shared_intermediate_size {
                object.insert("shared_expert_intermediate_size".into(), json!(shared));
                marker["shared_intermediate_size"] = json!(shared);
            }
            marker
        }
        Qwen4FfnConfig::Dense { intermediate_size } => {
            json!({"kind": "dense", "intermediate_size": intermediate_size})
        }
    };
    let equation_source = if ffn.is_moe() {
        Qwen4EquationSource::PinnedUpstream
    } else {
        Qwen4EquationSource::NonUpstreamDense
    };
    let mut marker = json!({
        "id": "synthetic-test",
        "developer_only": true,
        "equation_source": equation_source.as_str(),
        "ffn": marker_ffn,
    });
    if with_ple {
        object.insert("ple_layer_ids".into(), json!([1]));
        object.insert("ple_conv_kernel_size".into(), json!(2));
        object.insert("ple_embed_dim".into(), json!(16));
        object.insert("split_ngram_parts".into(), json!(1));
        object.insert("heads_per_ngram".into(), json!(1));
        object.insert("ngram_size".into(), json!(3));
        object.insert("ngram_vocab_size_base".into(), json!(7));
        object.insert("make_ngram_vocab_size_divisible_by".into(), json!(8));
        marker["ple"] = json!({"valid_rows": 18});
    }
    if with_mtp {
        object.insert("mtp_num_hidden_layers".into(), json!(1));
        object.insert("mtp_use_dedicated_embeddings".into(), json!(false));
        object.insert(
            "mtp".into(),
            json!({
                "hybrid": true,
                "layer_types": ["full_attention"],
                "num_hidden_layers": 1,
                "rope_theta": 10000.0,
            }),
        );
    }
    json!({
        "architectures": [ARCHITECTURE_NAME],
        "model_type": MODEL_TYPE,
        "text_config": text,
        PROFILE_KEY: marker,
    })
}

/// A validated synthetic developer config (see [`synthetic_fixture_value`]);
/// `hidden_size`/`num_hidden_layers` are the only knobs that vary the trunk.
#[cfg(test)]
pub(crate) fn synthetic_test_config(
    ffn: Qwen4FfnConfig,
    hidden_size: usize,
    num_hidden_layers: usize,
    with_ple: bool,
    with_mtp: bool,
) -> Qwen4Config {
    let value = synthetic_fixture_value(&ffn, hidden_size, num_hidden_layers, with_ple, with_mtp);
    Qwen4Config::from_developer_value(&value).expect("synthetic Qwen4 developer fixture config")
}

/// Deterministic Miller-Rabin primality for every `u64`.
fn is_prime(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for prime in BASES {
        if n % prime == 0 {
            return n == prime;
        }
    }
    let mul_mod = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let (mut odd, mut twos) = (n - 1, 0u32);
    while odd % 2 == 0 {
        odd /= 2;
        twos += 1;
    }
    'witness: for base in BASES {
        let (mut result, mut power, mut exponent) = (1u64, base % n, odd);
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = mul_mod(result, power);
            }
            power = mul_mod(power, power);
            exponent >>= 1;
        }
        if result == 1 || result == n - 1 {
            continue;
        }
        for _ in 1..twos {
            result = mul_mod(result, result);
            if result == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// Row counts of the `head_count` PLE head tables: the successive primes at
/// or above `ngram_vocab_size_base` (Flash-Next: 20,000,003, 20,000,023, ...;
/// the tiny profile with base 7 and two heads: 7, 11).
pub fn ple_head_vocab_sizes(
    ngram_vocab_size_base: u64,
    head_count: usize,
) -> Result<Vec<u64>, String> {
    let mut sizes = Vec::with_capacity(head_count);
    let mut candidate = ngram_vocab_size_base.max(2);
    while sizes.len() < head_count {
        if is_prime(candidate) {
            sizes.push(candidate);
        }
        candidate = candidate
            .checked_add(1)
            .ok_or_else(|| "qwen4: PLE head vocabulary search overflows u64".to_string())?;
    }
    Ok(sizes)
}

fn approximately_equal(left: f64, right: f64) -> bool {
    let scale = right.abs().max(1.0);
    (left - right).abs() <= scale * 1e-9
}

fn bounded(value: usize, min: usize, max: usize, field: &str) -> Result<(), String> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={value} is outside {min}..={max}"))
    }
}

/// `layer % interval == interval - 1` is a full-attention layer.  `interval`
/// MUST be positive.
fn scheduled_layers(layers: usize, interval: usize) -> Vec<LayerType> {
    (0..layers)
        .map(|layer| {
            if layer % interval == interval - 1 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

/// Identity checks of the strict nested envelope.
fn check_nested_envelope(value: &Value) -> Result<(), String> {
    let config = value.get("config").unwrap_or(value);
    let object = config
        .as_object()
        .ok_or_else(|| "qwen4: nested `config` must be a JSON object".to_string())?;
    if object.get("model_type").and_then(Value::as_str) != Some(MODEL_TYPE) {
        return Err(format!(
            "qwen4: nested runtime admission requires outer model_type={MODEL_TYPE:?}"
        ));
    }
    let text = object
        .get("text_config")
        .ok_or_else(|| "qwen4: nested runtime admission requires `text_config`".to_string())?
        .as_object()
        .ok_or_else(|| "qwen4: `text_config` must be a JSON object".to_string())?;
    if text.get("model_type").and_then(Value::as_str) != Some(TEXT_MODEL_TYPE) {
        return Err(format!(
            "qwen4: nested runtime admission requires text_config.model_type={TEXT_MODEL_TYPE:?}"
        ));
    }
    let architectures = object
        .get("architectures")
        .ok_or_else(|| "qwen4: nested runtime admission requires `architectures`".to_string())?
        .as_array()
        .ok_or_else(|| "qwen4: `architectures` must be an array".to_string())?;
    if architectures.len() != 1
        || architectures.first().and_then(Value::as_str) != Some(ARCHITECTURE_NAME)
    {
        return Err(format!(
            "qwen4: nested runtime admission requires architectures=[{ARCHITECTURE_NAME:?}]"
        ));
    }
    Ok(())
}

/// Locate the explicit profile marker: at the root of the (unwrapped) config,
/// or at the root of the envelope that wraps it.  Declaring it in both places
/// with different content, or inside `text_config`, is refused.
fn find_profile_marker<'a>(
    value: &'a Value,
    config: &'a Value,
    text: &'a Value,
) -> Result<Option<&'a Value>, String> {
    if !std::ptr::eq(text, config) && text.get(PROFILE_KEY).is_some() {
        return Err(format!(
            "qwen4: `{PROFILE_KEY}` must be declared at the top level of the config, not inside text_config"
        ));
    }
    let inner = config.get(PROFILE_KEY);
    if std::ptr::eq(value, config) {
        return Ok(inner);
    }
    match (value.get(PROFILE_KEY), inner) {
        (Some(outer), Some(inner)) if outer != inner => Err(format!(
            "qwen4: conflicting `{PROFILE_KEY}` declarations in the envelope and the config"
        )),
        (Some(outer), _) => Ok(Some(outer)),
        (None, inner) => Ok(inner),
    }
}

fn validate_profile_id(id: &str) -> Result<(), String> {
    if id.is_empty()
        || id.len() > MAX_PROFILE_ID_LEN
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(format!(
            "qwen4: profile id {id:?} must be 1..={MAX_PROFILE_ID_LEN} characters of [A-Za-z0-9._-]"
        ));
    }
    if id.eq_ignore_ascii_case(FLASH_NEXT_PROFILE_ID)
        || id.eq_ignore_ascii_case(FLASH_NEXT_AR_ONLY_PROFILE_ID)
        || id.eq_ignore_ascii_case(FLASH_NEXT_HEADLESS_PROFILE_ID)
    {
        return Err(format!(
            "qwen4: profile id {id:?} is reserved for the pinned Flash-Next profiles"
        ));
    }
    Ok(())
}

fn reject_unknown_keys(
    object: &serde_json::Map<String, Value>,
    allowed: &[&str],
    context: &str,
) -> Result<(), String> {
    match object.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(format!("qwen4: {context} has unknown field `{key}`")),
        None => Ok(()),
    }
}

fn parse_profile_marker(marker: &Value) -> Result<ProfileMarker, String> {
    let object = marker
        .as_object()
        .ok_or_else(|| format!("qwen4: `{PROFILE_KEY}` must be a JSON object"))?;
    reject_unknown_keys(
        object,
        &["id", "developer_only", "equation_source", "ffn", "ple"],
        PROFILE_KEY,
    )?;
    let id = required_string(marker, "id")?;
    validate_profile_id(id)?;
    match object.get("developer_only") {
        Some(Value::Bool(true)) => {}
        _ => {
            return Err(format!(
                "qwen4: `{PROFILE_KEY}.developer_only` must be exactly true"
            ))
        }
    }
    let equation_source = required_string(marker, "equation_source")?;
    let equation_source = Qwen4EquationSource::parse(equation_source)
        .ok_or_else(|| format!("qwen4: unsupported equation_source={equation_source:?}"))?;
    let ffn = parse_marker_ffn(required_value(marker, "ffn")?)?;
    let ple_valid_rows = match object.get("ple") {
        None => None,
        Some(ple) => {
            let ple_object = ple
                .as_object()
                .ok_or_else(|| format!("qwen4: `{PROFILE_KEY}.ple` must be a JSON object"))?;
            reject_unknown_keys(ple_object, &["valid_rows"], "hipfire_profile.ple")?;
            Some(required_u64(ple, "valid_rows")?)
        }
    };
    Ok(ProfileMarker {
        profile: Qwen4SyntheticProfile {
            id: id.to_string(),
            equation_source,
        },
        ffn,
        ple_valid_rows,
    })
}

fn parse_marker_ffn(value: &Value) -> Result<Qwen4FfnConfig, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("qwen4: `{PROFILE_KEY}.ffn` must be a JSON object"))?;
    match required_string(value, "kind")? {
        "moe" => {
            reject_unknown_keys(
                object,
                &[
                    "kind",
                    "num_experts",
                    "top_k",
                    "intermediate_size",
                    "shared_intermediate_size",
                    "norm_topk_prob",
                ],
                "hipfire_profile.ffn",
            )?;
            let shared_intermediate_size = match object.get("shared_intermediate_size") {
                None | Some(Value::Null) => None,
                Some(_) => Some(required_usize(value, "shared_intermediate_size")?),
            };
            Ok(Qwen4FfnConfig::Moe {
                num_experts: required_usize(value, "num_experts")?,
                top_k: required_usize(value, "top_k")?,
                intermediate_size: required_usize(value, "intermediate_size")?,
                shared_intermediate_size,
                norm_topk_prob: required_bool(value, "norm_topk_prob")?,
            })
        }
        "dense" => {
            reject_unknown_keys(
                object,
                &["kind", "intermediate_size"],
                "hipfire_profile.ffn",
            )?;
            Ok(Qwen4FfnConfig::Dense {
                intermediate_size: required_usize(value, "intermediate_size")?,
            })
        }
        other => Err(format!("qwen4: unsupported ffn kind {other:?}")),
    }
}

/// The nested HF text config must say exactly what the explicit marker says.
fn check_ffn_text_agreement(text: &Value, ffn: &Qwen4FfnConfig) -> Result<(), String> {
    match ffn {
        Qwen4FfnConfig::Moe {
            num_experts,
            top_k,
            intermediate_size,
            shared_intermediate_size,
            norm_topk_prob,
        } => {
            expect(
                required_usize(text, "num_experts")?,
                *num_experts,
                "text_config.num_experts vs hipfire_profile.ffn.num_experts",
            )?;
            expect(
                required_usize(text, "num_experts_per_tok")?,
                *top_k,
                "text_config.num_experts_per_tok vs hipfire_profile.ffn.top_k",
            )?;
            expect(
                required_usize(text, "moe_intermediate_size")?,
                *intermediate_size,
                "text_config.moe_intermediate_size vs hipfire_profile.ffn.intermediate_size",
            )?;
            expect(
                optional_usize(text, "shared_expert_intermediate_size")?.unwrap_or(0),
                shared_intermediate_size.unwrap_or(0),
                "text_config.shared_expert_intermediate_size vs hipfire_profile.ffn.shared_intermediate_size",
            )?;
            expect(
                optional_bool(text, "norm_topk_prob").unwrap_or(true),
                *norm_topk_prob,
                "text_config.norm_topk_prob vs hipfire_profile.ffn.norm_topk_prob",
            )
        }
        Qwen4FfnConfig::Dense { intermediate_size } => {
            for key in [
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
            ] {
                if optional_usize(text, key)?.unwrap_or(0) != 0 {
                    return Err(format!(
                        "qwen4: text_config.{key} must be absent or 0 for a dense ffn profile"
                    ));
                }
            }
            if text
                .get("norm_topk_prob")
                .is_some_and(|value| !value.is_null())
            {
                return Err(
                    "qwen4: text_config.norm_topk_prob must be absent for a dense ffn profile"
                        .into(),
                );
            }
            if let Some(declared) = optional_usize(text, "intermediate_size")? {
                expect(
                    declared,
                    *intermediate_size,
                    "text_config.intermediate_size vs hipfire_profile.ffn.intermediate_size",
                )?;
            }
            Ok(())
        }
    }
}

/// Synthetic PLE is optional: absent `ple_layer_ids` (or an empty list) means
/// no PLE and none of the companion geometry may be present.  A present PLE
/// requires the complete explicit geometry plus `hipfire_profile.ple`.
fn parse_synthetic_ple(
    text: &Value,
    valid_rows: Option<u64>,
) -> Result<Option<Qwen4PleConfig>, String> {
    const COMPANION_KEYS: [&str; 7] = [
        "ple_conv_kernel_size",
        "ple_embed_dim",
        "split_ngram_parts",
        "heads_per_ngram",
        "ngram_size",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
    ];
    let ids = match text.get("ple_layer_ids") {
        None | Some(Value::Null) => Vec::new(),
        Some(value) => parse_usize_array(value, "ple_layer_ids")?,
    };
    if ids.is_empty() {
        if let Some(key) = COMPANION_KEYS
            .iter()
            .find(|key| text.get(**key).is_some_and(|value| !value.is_null()))
        {
            return Err(format!(
                "qwen4: `{key}` is set but ple_layer_ids is absent or empty (PLE needs its complete explicit geometry)"
            ));
        }
        if valid_rows.is_some() {
            return Err(format!(
                "qwen4: `{PROFILE_KEY}.ple` is set but the config has no PLE"
            ));
        }
        return Ok(None);
    }
    Qwen4PleConfig::new(
        &ids,
        required_usize(text, "ple_embed_dim")?,
        required_usize(text, "ple_conv_kernel_size")?,
        required_usize(text, "split_ngram_parts")?,
        required_usize(text, "heads_per_ngram")?,
        required_usize(text, "ngram_size")?,
        required_u64(text, "ngram_vocab_size_base")?,
        required_usize(text, "make_ngram_vocab_size_divisible_by")?,
        valid_rows,
    )
    .map(Some)
}

fn parse_mtp(mtp_value: &Value, qsa_proof: &Result<(), String>) -> Result<Qwen4MtpConfig, String> {
    Ok(Qwen4MtpConfig {
        num_hidden_layers: required_usize(mtp_value, "num_hidden_layers")?,
        layer_types: parse_layers(
            required_value(mtp_value, "layer_types")?,
            "mtp.layer_types",
            qsa_proof,
        )?,
        hybrid: required_bool(mtp_value, "hybrid")?,
        rope_theta: required_f64(mtp_value, "rope_theta")?,
    })
}

/// Proof that the QSA indexer fields exist and are valid, required before the
/// `qwen_sparse_attention` layer spelling is read as full attention.
fn qsa_proof(text: &Value) -> Result<(), String> {
    let heads = required_usize(text, "indexer_n_heads")?;
    let kv_heads = required_usize(text, "indexer_kv_heads")?;
    let head_dim = required_usize(text, "indexer_head_dim")?;
    let budget = required_usize(text, "indexer_budget")?;
    let ratio = required_usize(text, "indexer_compress_ratio")?;
    if heads == 0 || head_dim == 0 || budget == 0 || ratio == 0 {
        return Err("qwen4: QSA indexer dimensions must be positive".into());
    }
    if kv_heads != 1 {
        return Err(format!(
            "qwen4: indexer_kv_heads={kv_heads} (QSA requires 1)"
        ));
    }
    if budget % ratio != 0 {
        return Err(format!(
            "qwen4: indexer_budget={budget} is not divisible by indexer_compress_ratio={ratio}"
        ));
    }
    Ok(())
}

fn required_value<'a>(object: &'a Value, key: &str) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("qwen4: missing {key}"))
}

fn required_usize(object: &Value, key: &str) -> Result<usize, String> {
    let value = required_value(object, key)?;
    let value = value
        .as_u64()
        .ok_or_else(|| format!("qwen4: {key} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| format!("qwen4: {key} does not fit usize"))
}

/// `Ok(None)` for an absent or null key; an error for a non-integer value.
fn optional_usize(object: &Value, key: &str) -> Result<Option<usize>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => required_usize(object, key).map(Some),
    }
}

fn required_u64(object: &Value, key: &str) -> Result<u64, String> {
    required_value(object, key)?
        .as_u64()
        .ok_or_else(|| format!("qwen4: {key} must be a non-negative integer"))
}

fn required_f64(object: &Value, key: &str) -> Result<f64, String> {
    required_value(object, key)?
        .as_f64()
        .ok_or_else(|| format!("qwen4: {key} must be a number"))
}

fn required_bool(object: &Value, key: &str) -> Result<bool, String> {
    required_value(object, key)?
        .as_bool()
        .ok_or_else(|| format!("qwen4: {key} must be boolean"))
}

fn required_string<'a>(object: &'a Value, key: &str) -> Result<&'a str, String> {
    required_value(object, key)?
        .as_str()
        .ok_or_else(|| format!("qwen4: {key} must be a string"))
}

fn optional_bool(object: &Value, key: &str) -> Option<bool> {
    object.get(key).and_then(Value::as_bool)
}

fn optional_string(object: &Value, key: &str) -> Option<String> {
    object.get(key).and_then(Value::as_str).map(str::to_string)
}

/// The artifact's trunk default marker: absent is `None`; a present marker
/// must be a string this build knows and the pinned Flash-Next profile (the
/// only profile the trunk route runs) must carry it. Anything else is
/// refused rather than guessed.
fn parse_trunk_iu4_default(
    envelope: &Value,
    profile: &Qwen4Profile,
) -> Result<Option<TrunkIu4Default>, String> {
    let Some(marker) = envelope.get(TRUNK_IU4_DEFAULT_KEY) else {
        return Ok(None);
    };
    let marker = marker
        .as_str()
        .ok_or_else(|| format!("qwen4: `{TRUNK_IU4_DEFAULT_KEY}` must be a string"))?;
    let default = TrunkIu4Default::from_marker(marker).ok_or_else(|| {
        format!(
            "qwen4: `{TRUNK_IU4_DEFAULT_KEY}` {marker:?} is not a trunk default this build knows"
        )
    })?;
    if !matches!(profile, Qwen4Profile::FlashNext) {
        return Err(format!(
            "qwen4: `{TRUNK_IU4_DEFAULT_KEY}` is only meaningful on the Flash-Next profile"
        ));
    }
    Ok(Some(default))
}

fn parse_layers(
    value: &Value,
    key: &str,
    qsa_proof: &Result<(), String>,
) -> Result<Vec<LayerType>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("qwen4: {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let name = value
                .as_str()
                .ok_or_else(|| format!("qwen4: {key}[{index}] must be a string"))?;
            if name == QSA_LAYER_TYPE_SPELLING {
                qsa_proof.as_ref().map_err(|reason| {
                    format!(
                        "qwen4: {key}[{index}]={name:?} requires valid QSA indexer fields: {reason}"
                    )
                })?;
                return Ok(LayerType::FullAttention);
            }
            LayerType::parse(name)
                .ok_or_else(|| format!("qwen4: unsupported {key}[{index}]={name:?}"))
        })
        .collect()
}

fn parse_usize_array(value: &Value, key: &str) -> Result<Vec<usize>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("qwen4: {key} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let number = value
                .as_u64()
                .ok_or_else(|| format!("qwen4: {key}[{index}] must be a non-negative integer"))?;
            usize::try_from(number).map_err(|_| format!("qwen4: {key}[{index}] does not fit usize"))
        })
        .collect()
}

fn parse_eos(value: Option<&Value>, default: u32) -> Result<u32, String> {
    let value = match value {
        None | Some(Value::Null) => return Ok(default),
        Some(value) => value,
    };
    let number = match value {
        Value::Number(number) => number
            .as_u64()
            .ok_or_else(|| "qwen4: eos_token_id must be a non-negative integer".to_string())?,
        Value::Array(values) => values
            .first()
            .and_then(Value::as_u64)
            .ok_or_else(|| "qwen4: eos_token_id array must contain an integer".to_string())?,
        _ => return Err("qwen4: eos_token_id must be an integer or array".into()),
    };
    u32::try_from(number).map_err(|_| "qwen4: eos_token_id does not fit u32".into())
}

fn expect<T: PartialEq + fmt::Debug>(actual: T, wanted: T, field: &str) -> Result<(), String> {
    if actual == wanted {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual:?} (expected {wanted:?})"))
    }
}

fn expect_eq<T: PartialEq + fmt::Debug>(actual: T, wanted: T, field: &str) -> Result<(), String> {
    expect(actual, wanted, field)
}

fn expect_str(actual: &str, wanted: &str, field: &str) -> Result<(), String> {
    if actual == wanted {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual:?} (expected {wanted:?})"))
    }
}

fn expect_float(actual: f64, wanted: f64, field: &str) -> Result<(), String> {
    if approximately_equal(actual, wanted) {
        Ok(())
    } else {
        Err(format!("qwen4: {field}={actual} (expected {wanted})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn expert_keep_admission_ragged_and_refusals() {
        let mut metadata = fixture();
        let layers: Vec<Vec<u16>> = (0..48)
            .map(|l| (0..(16 + l * 496 / 47) as u16).collect())
            .collect();
        let keep = crate::ExpertKeep::new(512, layers, None, 10).unwrap();
        metadata[crate::EXPERT_KEEP_METADATA_KEY] = keep.to_metadata_value();
        let cfg = Qwen4Config::from_nested_value(&metadata).unwrap();
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.expert_keep, Some(keep));
        let manifest = crate::weights::Qwen4Manifest::build(&cfg).unwrap();
        assert!(manifest
            .clone()
            .with_expert_parallel(crate::weights::Qwen4EpPlacement { rank: 0, ranks: 2 })
            .unwrap_err()
            .to_string()
            .contains("expert_keep"));
        assert!(manifest
            .check_expert_tiers()
            .unwrap_err()
            .to_string()
            .contains("expert_keep"));
        let gate_up = manifest
            .weights
            .iter()
            .find(|entry| entry.name == "model.language_model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(gate_up.logical_shape[0], 16);
        let last = manifest
            .weights
            .iter()
            .find(|entry| entry.name == "model.language_model.layers.47.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(last.logical_shape[0], 512);
        let mut short = vec![(0..16).collect(); 48];
        short[4] = (0..9).collect();
        metadata[crate::EXPERT_KEEP_METADATA_KEY] = crate::ExpertKeep::new(512, short, None, 1)
            .unwrap()
            .to_metadata_value();
        assert!(Qwen4Config::from_nested_value(&metadata)
            .unwrap_err()
            .contains("top_k"));
        metadata[crate::EXPERT_KEEP_METADATA_KEY] =
            crate::ExpertKeep::new(512, vec![(0..16).collect(); 47], None, 10)
                .unwrap()
                .to_metadata_value();
        assert!(Qwen4Config::from_nested_value(&metadata)
            .unwrap_err()
            .contains("expert_keep.layers"));
    }

    #[test]
    fn expert_keep_requires_fixed_e512_top10_geometry() {
        let mut metadata = synthetic();
        let keep = crate::ExpertKeep::new(4, vec![vec![0, 1, 2]; 2], None, 2).unwrap();
        metadata[crate::EXPERT_KEEP_METADATA_KEY] = keep.to_metadata_value();
        let error = dev(&metadata).unwrap_err();
        assert!(error.contains("num_experts=512"), "{error}");
        // Without keep metadata the same synthetic geometry stays admitted.
        dev(&synthetic()).expect("dense synthetic is unaffected");
    }

    /// The rank-local config of a TP trunk shard: half the GDN/q heads, KV heads kept, the
    /// shared expert split 384 + 256, validated through its reconstructed global config,
    /// round-tripping, and refused on a profile that carries the MTP head.
    #[test]
    fn trunk_shard_local_config_is_the_shard_of_its_global_config() {
        let flash = compact_test_config();
        let ar = flash.clone().into_ar_only().unwrap();
        // MTP-bearing Flash-Next is refused: the head reads the global geometry.
        assert!(flash.trunk_shard_local(0, 2).is_err());
        let locals: Vec<Qwen4Config> = (0..2)
            .map(|r| ar.trunk_shard_local(r, 2).expect("local"))
            .collect();
        for (rank, local) in locals.iter().enumerate() {
            assert_eq!(local.trunk_shard, Some((rank, 2)));
            assert_eq!(local.linear_num_key_heads, ar.linear_num_key_heads / 2);
            assert_eq!(local.linear_num_value_heads, ar.linear_num_value_heads / 2);
            assert_eq!(local.num_attention_heads, ar.num_attention_heads / 2);
            assert_eq!(local.num_key_value_heads, ar.num_key_value_heads);
            assert_eq!(local.hidden_size, ar.hidden_size);
            assert_eq!(local.vocab_size, ar.vocab_size);
            local.validate().expect("the local config validates");
            assert_eq!(local.trunk_shard_global().unwrap(), ar);
            assert_eq!(local.replica_geometry(), ar);
        }
        assert_eq!(
            (
                locals[0].shared_expert_intermediate_size,
                locals[1].shared_expert_intermediate_size
            ),
            (384, 256)
        );
        // the typed ffn mirrors the scalar
        for local in &locals {
            assert_eq!(
                local.ffn.scalar_mirror().3,
                local.shared_expert_intermediate_size
            );
        }
        // ranks agree on the replica geometry although their configs differ
        assert_ne!(locals[0], locals[1]);
        assert_eq!(locals[0].replica_geometry(), locals[1].replica_geometry());
        // a tampered local config is not the shard of its global config
        let mut bad = locals[0].clone();
        bad.linear_num_value_heads += 1;
        assert!(bad.validate().is_err());
        let mut bad = locals[1].clone();
        bad.shared_expert_intermediate_size = 384;
        assert!(bad.validate().is_err());
        // sharding twice, a bad rank/part count and indivisible heads are refused
        assert!(locals[0].trunk_shard_local(0, 2).is_err());
        assert!(ar.trunk_shard_local(2, 2).is_err());
        assert!(ar.trunk_shard_local(0, 1).is_err());
        assert!(ar.trunk_shard_local(0, 5).is_err());
    }

    /// The AR-only profile is derived, never parsed: Flash-Next converts (the
    /// MTP head and its scalar mirrors cleared, geometry unchanged), it
    /// validates only head-free, its id is reserved, and a synthetic profile
    /// has no AR-only form.
    #[test]
    fn ar_only_profile_is_flash_next_without_mtp() {
        let flash = compact_test_config();
        let ar = flash
            .clone()
            .into_ar_only()
            .expect("Flash-Next has an AR-only form");
        assert_eq!(ar.profile, Qwen4Profile::FlashNextArOnly);
        assert_eq!(ar.profile.id(), FLASH_NEXT_AR_ONLY_PROFILE_ID);
        assert!(ar.profile.is_flash_next() && ar.profile.is_ar_only());
        assert!(!ar.profile.developer_only());
        assert!(ar.mtp.is_none());
        assert_eq!(ar.mtp_num_hidden_layers, 0);
        assert_eq!(
            (ar.hidden_size, ar.num_experts),
            (flash.hidden_size, flash.num_experts)
        );
        assert_eq!(ar.ple, flash.ple);
        ar.validate().expect("AR-only validates");
        assert_eq!(ar.clone().into_ar_only().unwrap(), ar);
        // A head bolted back on (or the mirror alone) is refused.
        let mut headed = ar.clone();
        headed.mtp = flash.mtp.clone();
        headed.mtp_num_hidden_layers = 1;
        assert!(headed.validate().is_err());
        let mut mirror = ar.clone();
        mirror.mtp_num_hidden_layers = 1;
        assert!(mirror.validate().is_err());
        // The checkpoint profile still requires its head.
        let mut headless = flash.clone();
        headless.mtp = None;
        headless.mtp_num_hidden_layers = 0;
        assert!(headless.validate().is_err());
        assert!(validate_profile_id(FLASH_NEXT_AR_ONLY_PROFILE_ID).is_err());
    }

    /// The head-dropped packed profile (trunk shard load) is derived like the stock AR-only
    /// one but is a DIFFERENT profile: it keeps the packed identity (not the stock-FP8
    /// numerical pins), is head-free everywhere, and neither profile converts into the other.
    #[test]
    fn headless_profile_is_packed_flash_next_without_mtp() {
        let flash = compact_test_config();
        let headless = flash.clone().into_headless().expect("headless form");
        assert_eq!(headless.profile, Qwen4Profile::FlashNextHeadless);
        assert_eq!(headless.profile.id(), FLASH_NEXT_HEADLESS_PROFILE_ID);
        let profile = &headless.profile;
        assert!(profile.is_flash_next() && profile.is_headless() && profile.is_packed_flash_next());
        assert!(!profile.is_ar_only(), "not the stock-FP8 profile");
        assert!(!profile.developer_only());
        assert!(headless.mtp.is_none());
        assert_eq!(headless.mtp_num_hidden_layers, 0);
        assert!(!headless.mtp_use_dedicated_embeddings);
        // everything but the profile and the head is the checkpoint geometry
        let mut same = headless.clone();
        same.profile = Qwen4Profile::FlashNext;
        same.mtp = flash.mtp.clone();
        same.mtp_num_hidden_layers = flash.mtp_num_hidden_layers;
        same.mtp_use_dedicated_embeddings = flash.mtp_use_dedicated_embeddings;
        assert_eq!(same, flash);
        headless.validate().expect("headless validates");
        assert_eq!(headless.clone().into_headless().unwrap(), headless);
        // the stock AR-only profile keeps its own, distinct, pins
        let stock = flash.clone().into_ar_only().unwrap();
        assert!(stock.profile.is_ar_only() && stock.profile.is_headless());
        assert!(!stock.profile.is_packed_flash_next());
        assert_ne!(stock.profile, headless.profile);
        assert!(stock.clone().into_headless().is_err());
        assert!(headless.clone().into_ar_only().is_err());
        // dropping the head must not move the numerical contract: the recurrent state keeps
        // the checkpoint's `auto` / explicit formats (the stock-FP8 profile pins F32)
        for request in ["", "auto", "q8", "q8_ef", "fp32"] {
            assert_eq!(
                crate::state::resolve_gdn_format(request, &headless),
                crate::state::resolve_gdn_format(request, &flash),
                "{request:?}"
            );
        }
        assert_ne!(
            crate::state::resolve_gdn_format("auto", &stock),
            crate::state::resolve_gdn_format("auto", &headless)
        );
        // a head bolted back on (or the mirror alone) is refused; the head-bearing profile
        // still requires its head
        let mut headed = headless.clone();
        headed.mtp = flash.mtp.clone();
        headed.mtp_num_hidden_layers = 1;
        assert!(headed.validate().is_err());
        let mut mirror = headless.clone();
        mirror.mtp_num_hidden_layers = 1;
        assert!(mirror.validate().is_err());
        assert!(validate_profile_id(FLASH_NEXT_HEADLESS_PROFILE_ID).is_err());
        // a synthetic profile has no headless form
        let mut synthetic = synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 64,
            },
            64,
            2,
            false,
            false,
        );
        assert!(synthetic.clone().into_headless().is_err());
        synthetic.profile = Qwen4Profile::FlashNextHeadless;
        assert!(synthetic.validate().is_err());
    }

    /// The load under the shard: the head is dropped (headless global config), then each
    /// rank's local config is cut from it. The two local configs differ, round trip through
    /// the global one, keep the head-free profile, and are mutually consistent replicas.
    #[test]
    fn headless_global_config_shards_into_replica_consistent_local_configs() {
        use crate::weights::trunk_shard::{drops_head_for, load_local_config_for};
        let flash = compact_test_config();
        let headless = flash.clone().into_headless().unwrap();
        // the packed head-bearing config is refused until the head policy flips
        assert!(flash.trunk_shard_local(0, 2).is_err());
        let placement = |rank| Some(crate::weights::Qwen4EpPlacement { rank, ranks: 2 });
        assert!(!crate::weights::trunk_shard::trunk_shard_keeps_mtp());
        assert!(drops_head_for(true) && !drops_head_for(false));
        let locals: Vec<Qwen4Config> = (0..2)
            .map(|r| {
                load_local_config_for(true, &headless, placement(r))
                    .expect("local config")
                    .expect("the switch is on")
            })
            .collect();
        for (rank, local) in locals.iter().enumerate() {
            assert_eq!(local.profile, Qwen4Profile::FlashNextHeadless);
            assert!(local.mtp.is_none() && local.mtp_num_hidden_layers == 0);
            assert_eq!(local.trunk_shard, Some((rank, 2)));
            local.validate().expect("the local config validates");
            assert_eq!(local.trunk_shard_global().unwrap(), headless);
            assert_eq!(local.replica_geometry(), headless);
        }
        assert_ne!(locals[0], locals[1]);
        // switch off: no local config; a head-bearing config or a non-EP2 load refuses
        assert_eq!(load_local_config_for(false, &flash, None).unwrap(), None);
        assert!(load_local_config_for(true, &flash, placement(0)).is_err());
        assert!(load_local_config_for(true, &headless, None).is_err());
        assert!(load_local_config_for(
            true,
            &headless,
            Some(crate::weights::Qwen4EpPlacement { rank: 0, ranks: 4 })
        )
        .is_err());
        // the census always cuts from the GLOBAL geometry, even when handed a local config
        let census =
            |config: &Qwen4Config| crate::weights::trunk_shard::census_trunk_for(true, config, 2);
        assert_eq!(census(&headless), census(&locals[0]));
        assert_eq!(census(&locals[0]), census(&locals[1]));
        assert!(crate::weights::trunk_shard::census_trunk_for(false, &headless, 2).is_none());
        assert!(crate::weights::trunk_shard::census_trunk_for(true, &headless, 1).is_none());

        // check_replica: rank 1's peer against rank 0's config
        assert_eq!(locals[1].replica_mismatch(&locals[0], 1, 2), None);
        assert_eq!(locals[0].replica_mismatch(&locals[1], 0, 2), None);
        // a peer holding the wrong rank's local config, or an unsharded one, or a different
        // model, is refused
        assert!(locals[0].replica_mismatch(&locals[0], 1, 2).is_some());
        assert!(headless.replica_mismatch(&locals[0], 1, 2).is_some());
        assert!(locals[1].replica_mismatch(&headless, 1, 2).is_some());
        assert!(locals[1].replica_mismatch(&locals[0], 1, 4).is_some());
        let mut other = locals[1].clone();
        other.rope_theta *= 2.0;
        assert!(other.replica_mismatch(&locals[0], 1, 2).is_some());
        // unsharded EP: the plain replica equality is unchanged
        assert_eq!(flash.replica_mismatch(&flash, 1, 2), None);
        assert!(headless.replica_mismatch(&flash, 1, 2).is_some());
    }
    fn fixture() -> Value {
        let layers: Vec<_> = (0..48)
            .map(|idx| {
                if idx % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        json!({
            "architectures": [ARCHITECTURE_NAME],
            "model_type": MODEL_TYPE,
            "text_config": {
                "model_type": TEXT_MODEL_TYPE,
                "dtype": "bfloat16",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "max_position_embeddings": 262144,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0},
                "attention_bias": false,
                "layer_types": layers,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "split_ngram_parts": 128,
                "heads_per_ngram": 8,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044,
                "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {
                    "hybrid": true,
                    "layer_types": ["full_attention"],
                    "num_hidden_layers": 1,
                    "rope_theta": 10000000.0
                }
            }
        })
    }

    #[test]
    fn parses_nested_metadata_and_exact_geometry() {
        let metadata = json!({"config": fixture()});
        let cfg = Qwen4Config::from_value(&metadata).expect("fixture should parse");
        assert_eq!(cfg.architecture_id, 16);
        assert_eq!(cfg.n_linear_layers(), 36);
        assert_eq!(cfg.n_full_layers(), 12);
        assert_eq!(cfg.layer_types[2], LayerType::LinearAttention);
        assert_eq!(cfg.layer_types[3], LayerType::FullAttention);
        assert_eq!(cfg.ple_row_width(), 160);
        assert_eq!(cfg.ple_conv_dilation(), 3);
        assert_eq!(cfg.ple_conv_history_rows(), 9);
        assert_eq!(cfg.qsa_selected_capacity(), 2051);
    }

    #[test]
    fn accepts_raw_text_config_without_metadata_wrapper() {
        let mut raw = fixture();
        let text = raw.as_object_mut().unwrap().remove("text_config").unwrap();
        let mut text = text;
        text.as_object_mut()
            .unwrap()
            .insert("model_type".into(), Value::String(TEXT_MODEL_TYPE.into()));
        let cfg = Qwen4Config::from_value(&text).expect("raw text config should parse");
        assert_eq!(cfg.text_model_type, TEXT_MODEL_TYPE);
    }
    #[test]
    fn strict_runtime_classifier_requires_nested_conditional_envelope() {
        let metadata = json!({"config": fixture()});
        let metadata_json = metadata.to_string();
        let cfg = Qwen4Config::from_metadata_json(&metadata_json).expect("nested config");
        assert_eq!(cfg.model_type, MODEL_TYPE);
        assert_eq!(cfg.text_model_type, TEXT_MODEL_TYPE);

        let mut raw = fixture();
        let text = raw.as_object_mut().unwrap().remove("text_config").unwrap();
        assert!(Qwen4Config::from_metadata_json(&text.to_string()).is_err());

        let mut missing_architecture = fixture();
        missing_architecture
            .as_object_mut()
            .unwrap()
            .remove("architectures");
        assert!(Qwen4Config::from_metadata_json(&missing_architecture.to_string()).is_err());
    }

    #[test]
    fn strict_runtime_classifier_accepts_conditional_config_with_vision_marker() {
        let mut metadata = fixture();
        metadata["vision_config"] = json!({"model_type": "vision"});
        let cfg = Qwen4Config::from_nested_value(&metadata).expect("text config");
        assert_eq!(cfg.architecture_id, ARCH_ID);
    }

    #[test]
    fn rejects_wrong_layer_order_before_allocation() {
        let mut value = fixture();
        value["text_config"]["layer_types"][0] = Value::String("full_attention".into());
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("layer_types"));
    }

    #[test]
    fn rejects_wrong_mtp_embedding_policy() {
        let mut value = fixture();
        value["text_config"]["mtp_use_dedicated_embeddings"] = Value::Bool(true);
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("mtp_use_dedicated_embeddings"));
    }

    #[test]
    fn rejects_wrong_recurrent_dtype() {
        let mut value = fixture();
        value["text_config"]["mamba_ssm_dtype"] = Value::String("bfloat16".into());
        let error = Qwen4Config::from_value(&value).unwrap_err();
        assert!(error.contains("mamba_ssm_dtype"));
    }

    // ---- Flash-Next typed contract and pins ------------------------------

    fn moe() -> Qwen4FfnConfig {
        Qwen4FfnConfig::Moe {
            num_experts: 4,
            top_k: 2,
            intermediate_size: 16,
            shared_intermediate_size: Some(16),
            norm_topk_prob: true,
        }
    }

    fn synthetic() -> Value {
        synthetic_fixture_value(&moe(), 32, 2, true, true)
    }

    fn dense_value() -> Value {
        synthetic_fixture_value(
            &Qwen4FfnConfig::Dense {
                intermediate_size: 48,
            },
            32,
            4,
            true,
            true,
        )
    }

    fn dev(value: &Value) -> Result<Qwen4Config, String> {
        Qwen4Config::from_developer_value(value)
    }

    #[test]
    fn flash_next_typed_contract_is_pinned() {
        let cfg = Qwen4Config::from_nested_value(&fixture()).unwrap();
        assert_eq!(cfg.profile, Qwen4Profile::FlashNext);
        assert_eq!(cfg.profile.id(), FLASH_NEXT_PROFILE_ID);
        assert!(cfg.profile.is_flash_next() && !cfg.profile.developer_only());
        assert_eq!(
            cfg.profile.equation_source(),
            Qwen4EquationSource::PinnedUpstream
        );
        assert_eq!(cfg.ffn, Qwen4FfnConfig::flash_next());
        let ple = cfg.ple.as_ref().expect("Flash-Next carries PLE");
        assert_eq!(*ple, Qwen4PleConfig::flash_next());
        assert_eq!(ple.layer, 1);
        assert_eq!(ple.layer_id, 2);
        assert_eq!(ple.head_count, 16);
        assert_eq!(ple.multiplier_count, 3);
        assert_eq!(ple.row_width, 160);
        assert_eq!(ple.shard_count, 128);
        assert_eq!(ple.shard_rows, 2_500_012);
        assert_eq!(ple.valid_rows, 320_001_446);
        assert_eq!(ple.padded_rows, 320_001_536);
        assert_eq!(ple.padding_multiple, 128);
        assert_eq!(ple.shard_valid_rows(0), 2_500_012);
        assert_eq!(ple.shard_valid_rows(127), 2_499_922);
        assert_eq!(ple.shard_valid_rows(128), 0);
        assert!(ple.check_hash_metadata(&PleHashMetadata::qwen4()).is_ok());
        let mtp = cfg.mtp.as_ref().expect("Flash-Next carries MTP");
        assert_eq!(mtp.num_hidden_layers, 1);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
        assert_eq!(cfg.num_experts, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 640);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn trunk_iu4_default_marker_is_read_from_the_envelope_root_only_when_known() {
        let key = TRUNK_IU4_DEFAULT_KEY;
        assert_eq!(key, "qwen4_trunk_iu4_default");
        // Unmarked artifacts (wrapped or raw) carry no default.
        assert_eq!(
            Qwen4Config::from_nested_value(&fixture())
                .unwrap()
                .trunk_iu4_default,
            None
        );
        let wrapped = json!({"config": fixture()});
        assert_eq!(
            Qwen4Config::from_nested_value(&wrapped)
                .unwrap()
                .trunk_iu4_default,
            None
        );

        // The marker rides at the root of the envelope, wrapped or raw.
        let mut marked_wrapped = json!({"config": fixture()});
        marked_wrapped[key] = json!("gfx1151-zab-v1");
        let mut marked_raw = fixture();
        marked_raw[key] = json!("gfx1151-zab-v1");
        for value in [&marked_wrapped, &marked_raw] {
            let cfg = Qwen4Config::from_nested_value(value).unwrap();
            assert_eq!(cfg.trunk_iu4_default, Some(TrunkIu4Default::Gfx1151ZabV1));
            assert_eq!(cfg.profile, Qwen4Profile::FlashNext);
        }

        // The marker is identity metadata, not shape: every other field of the
        // config is the unmarked one's.
        let mut stripped = Qwen4Config::from_nested_value(&marked_wrapped).unwrap();
        stripped.trunk_iu4_default = None;
        assert_eq!(stripped, Qwen4Config::from_nested_value(&wrapped).unwrap());

        // A marker inside the inner config is not the envelope marker.
        let mut inner = json!({"config": fixture()});
        inner["config"][key] = json!("gfx1151-zab-v1");
        assert_eq!(
            Qwen4Config::from_nested_value(&inner)
                .unwrap()
                .trunk_iu4_default,
            None
        );
    }

    #[test]
    fn trunk_iu4_default_marker_refuses_unknown_and_malformed_values() {
        let key = TRUNK_IU4_DEFAULT_KEY;
        for bad in [
            json!("gfx1151-zab-v2"),
            json!("GFX1151-ZAB-V1"),
            json!("gfx1201-zab-v1"),
            json!(""),
            json!(true),
            json!(1),
            json!(null),
            json!({"id": "gfx1151-zab-v1"}),
        ] {
            let mut value = json!({"config": fixture()});
            value[key] = bad.clone();
            let error = Qwen4Config::from_nested_value(&value).unwrap_err();
            assert!(error.contains(key), "{bad}: {error}");
        }
    }

    #[test]
    fn trunk_iu4_default_marker_is_refused_on_a_synthetic_profile() {
        let mut value = synthetic();
        value[TRUNK_IU4_DEFAULT_KEY] = json!("gfx1151-zab-v1");
        let error = dev(&value).unwrap_err();
        assert!(error.contains(TRUNK_IU4_DEFAULT_KEY), "{error}");
        assert!(error.contains("Flash-Next"), "{error}");
        // Unmarked synthetic profiles are unchanged.
        assert_eq!(dev(&synthetic()).unwrap().trunk_iu4_default, None);
    }

    #[test]
    fn trunk_iu4_route_follows_the_marker_arch_and_env_precedence() {
        use rdna_compute::{Qwen4TrunkMask, TrunkFamily, TrunkIu4MaskSource};
        let mut marked = json!({"config": fixture()});
        marked[TRUNK_IU4_DEFAULT_KEY] = json!("gfx1151-zab-v1");
        let marked = Qwen4Config::from_nested_value(&marked).unwrap();
        let stock = Qwen4Config::from_nested_value(&json!({"config": fixture()})).unwrap();
        let zab = TrunkFamily::GdnZ.bit() | TrunkFamily::GdnA.bit() | TrunkFamily::GdnB.bit();
        let resolve = |env: Option<&str>, arch: &str, cfg: &Qwen4Config| {
            Qwen4TrunkMask::resolve_iu4(env, arch, cfg.trunk_iu4_default).unwrap()
        };

        // Marked + exact gfx1151 + no env: the zab default, on every layer.
        let on = resolve(None, "gfx1151", &marked);
        assert!(matches!(on.source, TrunkIu4MaskSource::ArtifactDefault(_)));
        assert!((0..48).all(|layer| on.mask.bits(layer) == zab));
        // Marked on any other arch, and every unmarked artifact: unchanged (off).
        for arch in ["gfx1201", "gfx1100", "gfx1150", "gfx942"] {
            assert!(resolve(None, arch, &marked).mask.is_empty(), "{arch}");
        }
        for arch in ["gfx1151", "gfx1201", "gfx1100"] {
            let off = resolve(None, arch, &stock);
            assert_eq!(off.source, TrunkIu4MaskSource::Off, "{arch}");
            assert!(off.mask.is_empty(), "{arch}");
        }
        // Explicit env wins: `0` restores the stock route on the marked
        // artifact, any other value keeps the existing grammar.
        assert!(resolve(Some("0"), "gfx1151", &marked).mask.is_empty());
        assert!(resolve(Some(""), "gfx1151", &marked).mask.is_empty());
        assert_eq!(
            resolve(Some("gdn.qkv"), "gfx1151", &marked).mask.bits(5),
            TrunkFamily::GdnQkv.bit()
        );
        assert_eq!(
            resolve(Some("1"), "gfx1151", &stock).mask.bits(5),
            rdna_compute::TRUNK_ALL_BITS
        );
        // A marked mask names layers the 48-layer trunk has.
        assert!(on.mask.check_layers(marked.num_hidden_layers).is_ok());
    }

    #[test]
    fn flash_next_pins_are_not_relaxed_without_a_marker() {
        // Geometry that would be a legal synthetic profile is NOT inferred as
        // synthetic: without the explicit marker it fails the exact pins.
        let mut value = fixture();
        value["text_config"]["hidden_size"] = json!(5120);
        assert!(Qwen4Config::from_nested_value(&value)
            .unwrap_err()
            .contains("hidden_size"));
        assert!(dev(&value).unwrap_err().contains("hidden_size"));

        let mut value = synthetic();
        value.as_object_mut().unwrap().remove(PROFILE_KEY);
        assert!(dev(&value).is_err());
        assert!(Qwen4Config::from_nested_value(&value).is_err());
    }

    #[test]
    fn flash_next_pins_reject_each_pinned_field() {
        for (key, value) in [
            ("num_experts", json!(256)),
            ("num_experts_per_tok", json!(8)),
            ("moe_intermediate_size", json!(320)),
            ("shared_expert_intermediate_size", json!(320)),
            ("hc_count", json!(2)),
            ("ple_embed_dim", json!(1280)),
            ("split_ngram_parts", json!(64)),
            ("ple_layer_ids", json!([3])),
            ("indexer_budget", json!(1024)),
        ] {
            let mut fixture = fixture();
            fixture["text_config"][key] = value;
            assert!(
                Qwen4Config::from_nested_value(&fixture).is_err(),
                "{key} must stay pinned"
            );
        }
        let mut no_mtp = fixture();
        no_mtp["text_config"].as_object_mut().unwrap().remove("mtp");
        assert!(Qwen4Config::from_nested_value(&no_mtp).is_err());
        let mut no_ple = fixture();
        no_ple["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("ple_layer_ids");
        assert!(Qwen4Config::from_nested_value(&no_ple).is_err());
    }

    #[test]
    fn qwen_sparse_attention_spelling_is_full_attention_only_with_qsa_proof() {
        let mut value = fixture();
        for index in (3..48).step_by(4) {
            value["text_config"]["layer_types"][index] = json!(QSA_LAYER_TYPE_SPELLING);
        }
        value["text_config"]["mtp"]["layer_types"][0] = json!(QSA_LAYER_TYPE_SPELLING);
        let cfg = Qwen4Config::from_nested_value(&value).expect("QSA synonym with proof");
        let baseline = Qwen4Config::from_nested_value(&fixture()).unwrap();
        assert_eq!(cfg, baseline, "the synonym normalizes to FullAttention");

        // Missing / invalid proof fields refuse the spelling.
        let mut missing = value.clone();
        missing["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("indexer_budget");
        let error = Qwen4Config::from_nested_value(&missing).unwrap_err();
        assert!(error.contains("indexer_budget"), "{error}");

        let mut bad_kv = value.clone();
        bad_kv["text_config"]["indexer_kv_heads"] = json!(2);
        let error = Qwen4Config::from_nested_value(&bad_kv).unwrap_err();
        assert!(error.contains("QSA indexer"), "{error}");

        let mut bad_ratio = value.clone();
        bad_ratio["text_config"]["indexer_budget"] = json!(2050);
        assert!(Qwen4Config::from_nested_value(&bad_ratio).is_err());

        // A linear layer position still cannot be QSA by name.
        let mut wrong = value;
        wrong["text_config"]["layer_types"][0] = json!(QSA_LAYER_TYPE_SPELLING);
        assert!(Qwen4Config::from_nested_value(&wrong)
            .unwrap_err()
            .contains("layer_types"));
    }

    // ---- Production parsers refuse the developer marker -------------------

    #[test]
    fn production_parsers_reject_marker_regardless_of_content() {
        let synthetic = synthetic();
        let mut flash_shaped = fixture();
        flash_shaped[PROFILE_KEY] = synthetic[PROFILE_KEY].clone();
        let mut fake_flash_id = fixture();
        fake_flash_id[PROFILE_KEY] = json!({"id": FLASH_NEXT_PROFILE_ID, "developer_only": true});
        for value in [&synthetic, &flash_shaped, &fake_flash_id] {
            let text = value.to_string();
            assert!(Qwen4Config::from_value(value)
                .unwrap_err()
                .contains("production parser"));
            assert!(Qwen4Config::from_nested_value(value)
                .unwrap_err()
                .contains("production parser"));
            assert!(Qwen4Config::from_json(&text)
                .unwrap_err()
                .contains("production parser"));
            assert!(Qwen4Config::from_nested_json(&text)
                .unwrap_err()
                .contains("production parser"));
            assert!(Qwen4Config::from_metadata_json(&text)
                .unwrap_err()
                .contains("production parser"));
            assert!(Qwen4Config::from_nested_metadata_json(&text)
                .unwrap_err()
                .contains("production parser"));
            // Wrapped in the HFQ metadata envelope too.
            let wrapped = json!({"config": value});
            assert!(Qwen4Config::from_nested_value(&wrapped).is_err());
            assert!(Qwen4Config::from_metadata_json(&wrapped.to_string()).is_err());
        }
        // Even a marker with developer_only=false is refused by production.
        let mut disabled = synthetic.clone();
        disabled[PROFILE_KEY]["developer_only"] = json!(false);
        assert!(Qwen4Config::from_nested_value(&disabled)
            .unwrap_err()
            .contains("production parser"));
    }

    #[test]
    fn developer_parser_accepts_unmarked_flash_next_as_production_does() {
        let production = Qwen4Config::from_nested_value(&fixture()).unwrap();
        let developer = dev(&fixture()).unwrap();
        assert_eq!(production, developer);
        assert_eq!(
            Qwen4Config::from_developer_metadata_json(&fixture().to_string()).unwrap(),
            production
        );
        // The developer parser keeps the strict nested identity.
        let mut raw = fixture();
        let text = raw.as_object_mut().unwrap().remove("text_config").unwrap();
        assert!(Qwen4Config::from_developer_json(&text.to_string()).is_err());
        let mut no_architecture = fixture();
        no_architecture
            .as_object_mut()
            .unwrap()
            .remove("architectures");
        assert!(dev(&no_architecture).is_err());
        let mut wrong_model = fixture();
        wrong_model["model_type"] = json!("qwen4_exp_text");
        assert!(dev(&wrong_model).is_err());
        let mut wrong_arch = fixture();
        wrong_arch["architectures"] = json!(["Qwen4ExpForCausalLM"]);
        assert!(dev(&wrong_arch).is_err());
    }

    // ---- Synthetic developer profiles --------------------------------------

    #[test]
    fn synthetic_small_moe_with_ple_and_mtp() {
        let cfg = dev(&synthetic()).expect("synthetic MoE");
        assert_eq!(cfg.profile.id(), "synthetic-test");
        assert!(cfg.profile.is_synthetic() && cfg.profile.developer_only());
        assert!(!cfg.profile.is_flash_next());
        assert_eq!(
            cfg.profile.equation_source(),
            Qwen4EquationSource::PinnedUpstream
        );
        assert_eq!(cfg.ffn, moe());
        assert_eq!((cfg.num_experts, cfg.num_experts_per_tok), (4, 2));
        assert_eq!(cfg.shared_expert_intermediate_size, 16);
        assert_eq!((cfg.hidden_size, cfg.num_hidden_layers), (32, 2));
        assert_eq!((cfg.n_linear_layers(), cfg.n_full_layers()), (1, 1));
        let ple = cfg.ple.as_ref().unwrap();
        assert_eq!((ple.layer, ple.layer_id), (0, 1));
        assert_eq!((ple.head_count, ple.multiplier_count), (2, 3));
        assert_eq!(ple.row_width, 8);
        assert_eq!((ple.shard_count, ple.shard_rows), (1, 24));
        assert_eq!((ple.valid_rows, ple.padded_rows), (18, 24));
        assert_eq!(ple.padding_multiple, 8);
        assert_eq!(ple.shard_valid_rows(0), 18);
        assert_eq!(ple.shard_valid_rows(1), 0);
        assert_eq!(cfg.ple_row_width(), 8);
        assert_eq!(cfg.ple_conv_history_rows(), 3);
        assert_eq!(cfg.mtp.as_ref().unwrap().rope_theta, 10_000.0);
        assert_eq!(cfg.mtp_num_hidden_layers, 1);
    }

    #[test]
    fn synthetic_shared_expert_is_optional() {
        let ffn = Qwen4FfnConfig::Moe {
            num_experts: 3,
            top_k: 2,
            intermediate_size: 8,
            shared_intermediate_size: None,
            norm_topk_prob: false,
        };
        let value = synthetic_fixture_value(&ffn, 16, 2, false, false);
        let cfg = dev(&value).expect("MoE without shared expert");
        assert_eq!(cfg.ffn, ffn);
        assert_eq!(cfg.shared_expert_intermediate_size, 0);
        assert!(!cfg.norm_topk_prob);
    }

    #[test]
    fn synthetic_dense_has_zeroed_moe_mirrors_and_non_upstream_equations() {
        let cfg = dev(&dense_value()).expect("synthetic dense");
        assert_eq!(
            cfg.ffn,
            Qwen4FfnConfig::Dense {
                intermediate_size: 48
            }
        );
        assert!(cfg.ffn.is_dense() && !cfg.ffn.is_moe());
        assert_eq!(cfg.ffn.intermediate_size(), 48);
        assert_eq!(
            (
                cfg.num_experts,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
                cfg.shared_expert_intermediate_size
            ),
            (0, 0, 0, 0)
        );
        assert!(!cfg.norm_topk_prob);
        assert_eq!(
            cfg.profile.equation_source(),
            Qwen4EquationSource::NonUpstreamDense
        );
        assert_eq!(cfg.num_hidden_layers, 4);
    }

    #[test]
    fn dense_64_layer_5120_hidden_header_only() {
        let value = synthetic_fixture_value(
            &Qwen4FfnConfig::Dense {
                intermediate_size: 17_408,
            },
            5120,
            64,
            false,
            false,
        );
        let cfg = dev(&value).expect("dense 64/5120 header-only profile");
        assert_eq!((cfg.hidden_size, cfg.num_hidden_layers), (5120, 64));
        assert_eq!((cfg.n_linear_layers(), cfg.n_full_layers()), (32, 32));
        assert_eq!(cfg.hidden_size * cfg.hc_count, 10_240);
        assert!(cfg.ple.is_none() && cfg.mtp.is_none());
        // The same config is refused by every production parser.
        assert!(Qwen4Config::from_nested_value(&value).is_err());
        // A small MoE profile builds through the shared helper too.
        let small = synthetic_test_config(
            Qwen4FfnConfig::Moe {
                num_experts: 8,
                top_k: 2,
                intermediate_size: 64,
                shared_intermediate_size: Some(64),
                norm_topk_prob: true,
            },
            256,
            4,
            true,
            true,
        );
        assert_eq!((small.hidden_size, small.num_hidden_layers), (256, 4));
    }

    #[test]
    fn synthetic_without_ple_or_mtp_has_no_invented_state() {
        let value = synthetic_fixture_value(&moe(), 32, 2, false, false);
        let cfg = dev(&value).expect("no PLE, no MTP");
        assert!(cfg.ple.is_none() && cfg.mtp.is_none());
        assert!(cfg.ple_layer_ids.is_empty());
        assert_eq!(
            (
                cfg.ple_conv_kernel_size,
                cfg.ple_embed_dim,
                cfg.split_ngram_parts,
                cfg.heads_per_ngram,
                cfg.ngram_size,
                cfg.ngram_vocab_size_base,
                cfg.make_ngram_vocab_size_divisible_by
            ),
            (0, 0, 0, 0, 0, 0, 0)
        );
        assert_eq!(cfg.ple_row_width(), 0);
        assert_eq!(cfg.ple_conv_dilation(), 0);
        assert_eq!(cfg.ple_conv_history_rows(), 0);
        assert_eq!(cfg.mtp_num_hidden_layers, 0);
        assert!(!cfg.mtp_use_dedicated_embeddings);
        // PLE and MTP are independently optional.
        let only_ple = dev(&synthetic_fixture_value(&moe(), 32, 2, true, false)).unwrap();
        assert!(only_ple.ple.is_some() && only_ple.mtp.is_none());
        let only_mtp = dev(&synthetic_fixture_value(&moe(), 32, 2, false, true)).unwrap();
        assert!(only_mtp.ple.is_none() && only_mtp.mtp.is_some());
    }

    #[test]
    fn partial_or_unanchored_optional_blocks_are_rejected() {
        // PLE geometry without ple_layer_ids.
        let mut value = synthetic_fixture_value(&moe(), 32, 2, false, false);
        value["text_config"]["ple_embed_dim"] = json!(16);
        assert!(dev(&value).unwrap_err().contains("ple_embed_dim"));
        // The row count is derived from the vocabulary base; the marker's
        // `ple.valid_rows` is only an optional cross-check.
        let mut value = synthetic();
        value[PROFILE_KEY].as_object_mut().unwrap().remove("ple");
        assert_eq!(dev(&value).unwrap().ple.unwrap().valid_rows, 18);
        // Marker PLE rows without a PLE.
        let mut value = synthetic_fixture_value(&moe(), 32, 2, false, false);
        value[PROFILE_KEY]["ple"] = json!({"valid_rows": 18});
        assert!(dev(&value).unwrap_err().contains("no PLE"));
        // Missing individual PLE field.
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("ple_embed_dim");
        assert!(dev(&value).unwrap_err().contains("ple_embed_dim"));
        // MTP counts without a head.
        let mut value = synthetic_fixture_value(&moe(), 32, 2, false, false);
        value["text_config"]["mtp_num_hidden_layers"] = json!(1);
        assert!(dev(&value).unwrap_err().contains("mtp"));
        let mut value = synthetic_fixture_value(&moe(), 32, 2, false, false);
        value["text_config"]["mtp_use_dedicated_embeddings"] = json!(true);
        assert!(dev(&value).unwrap_err().contains("mtp"));
        // A head without its counts.
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("mtp_num_hidden_layers");
        assert!(dev(&value).unwrap_err().contains("mtp_num_hidden_layers"));
        // Dedicated embeddings are unsupported.
        let mut value = synthetic();
        value["text_config"]["mtp_use_dedicated_embeddings"] = json!(true);
        assert!(dev(&value)
            .unwrap_err()
            .contains("mtp_use_dedicated_embeddings"));
    }

    #[test]
    fn marker_identity_is_explicit_and_strict() {
        let cases: Vec<(&str, Box<dyn Fn(&mut Value)>)> = vec![
            (
                "developer_only missing",
                Box::new(|v: &mut Value| {
                    v[PROFILE_KEY]
                        .as_object_mut()
                        .unwrap()
                        .remove("developer_only");
                }),
            ),
            (
                "developer_only false",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["developer_only"] = json!(false)),
            ),
            (
                "developer_only string",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["developer_only"] = json!("true")),
            ),
            (
                "bad equation source",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["equation_source"] = json!("custom")),
            ),
            (
                "bad id characters",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["id"] = json!("has space")),
            ),
            (
                "empty id",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["id"] = json!("")),
            ),
            (
                "overlong id",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["id"] = json!("a".repeat(65))),
            ),
            (
                "reserved flash-next id",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["id"] = json!("Qwen3.8-Flash-Next")),
            ),
            (
                "unknown marker field",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["extra"] = json!(1)),
            ),
            (
                "unknown ffn kind",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["ffn"]["kind"] = json!("hybrid")),
            ),
            (
                "unknown ffn field",
                Box::new(|v: &mut Value| v[PROFILE_KEY]["ffn"]["extra"] = json!(1)),
            ),
            (
                "non-object marker",
                Box::new(|v: &mut Value| v[PROFILE_KEY] = json!("synthetic")),
            ),
            (
                "moe with non-upstream equations",
                Box::new(|v: &mut Value| {
                    v[PROFILE_KEY]["equation_source"] = json!("non-upstream-dense")
                }),
            ),
            (
                "marker inside text_config",
                Box::new(|v: &mut Value| {
                    let marker = v[PROFILE_KEY].clone();
                    v.as_object_mut().unwrap().remove(PROFILE_KEY);
                    v["text_config"][PROFILE_KEY] = marker;
                }),
            ),
        ];
        for (name, mutate) in cases {
            let mut value = synthetic();
            mutate(&mut value);
            assert!(dev(&value).is_err(), "{name} must be refused");
        }
        let mut dense = dense_value();
        dense[PROFILE_KEY]["equation_source"] = json!("pinned-upstream");
        assert!(dev(&dense).unwrap_err().contains("equation_source"));
    }

    #[test]
    fn marker_location_accepts_envelope_root_and_rejects_conflicts() {
        // Marker at the root of the raw config (S0 layout) is the base case.
        let raw = synthetic();
        let base = dev(&raw).unwrap();
        // Envelope-root marker with an unmarked wrapped config.
        let mut inner = raw.clone();
        let marker = inner.as_object_mut().unwrap().remove(PROFILE_KEY).unwrap();
        let wrapped = json!({"config": inner, PROFILE_KEY: marker.clone()});
        assert_eq!(dev(&wrapped).unwrap(), base);
        // Same marker declared in both places is allowed...
        let both = json!({"config": raw.clone(), PROFILE_KEY: marker});
        assert_eq!(dev(&both).unwrap(), base);
        // ...a different marker in the other place is a conflict.
        let mut other = raw.clone();
        other[PROFILE_KEY]["id"] = json!("other-profile");
        let conflicting = json!({"config": other, PROFILE_KEY: raw[PROFILE_KEY].clone()});
        assert!(dev(&conflicting).unwrap_err().contains("conflicting"));
        // The wrapped (HFQ metadata) raw config parses the same.
        assert_eq!(dev(&json!({"config": raw})).unwrap(), base);
    }

    #[test]
    fn marker_and_text_config_must_agree() {
        let mut value = synthetic();
        value["text_config"]["num_experts"] = json!(5);
        assert!(dev(&value).unwrap_err().contains("num_experts"));
        let mut value = synthetic();
        value["text_config"]["num_experts_per_tok"] = json!(3);
        assert!(dev(&value).unwrap_err().contains("num_experts_per_tok"));
        let mut value = synthetic();
        value["text_config"]["moe_intermediate_size"] = json!(8);
        assert!(dev(&value).unwrap_err().contains("moe_intermediate_size"));
        let mut value = synthetic();
        value["text_config"]["shared_expert_intermediate_size"] = json!(8);
        assert!(dev(&value)
            .unwrap_err()
            .contains("shared_expert_intermediate_size"));
        let mut value = synthetic();
        value["text_config"]["norm_topk_prob"] = json!(false);
        assert!(dev(&value).unwrap_err().contains("norm_topk_prob"));
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_experts");
        assert!(dev(&value).unwrap_err().contains("num_experts"));
        // A dense marker may not carry MoE text fields.
        let mut value = dense_value();
        value["text_config"]["num_experts"] = json!(4);
        assert!(dev(&value).unwrap_err().contains("num_experts"));
        let mut value = dense_value();
        value["text_config"]["intermediate_size"] = json!(47);
        assert!(dev(&value).unwrap_err().contains("intermediate_size"));
        let mut value = dense_value();
        value["text_config"]["intermediate_size"] = json!(48);
        assert!(dev(&value).is_ok());
    }

    #[test]
    fn synthetic_requires_explicit_rope_and_eos() {
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("rope_parameters");
        assert!(dev(&value).unwrap_err().contains("rope_theta"));
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("partial_rotary_factor");
        assert!(dev(&value).unwrap_err().contains("partial_rotary_factor"));
        let mut value = synthetic();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("eos_token_id");
        assert!(dev(&value).unwrap_err().contains("eos_token_id"));
    }

    #[test]
    fn synthetic_rejects_bad_dimensions_and_unsupported_semantics() {
        let cases: Vec<(&str, &str, Value)> = vec![
            ("hidden_size", "hidden_size", json!(0)),
            ("hidden_size", "hidden_size", json!(1usize << 17)),
            ("vocab_size", "vocab_size", json!(0)),
            ("num_hidden_layers", "layer_types", json!(3)),
            ("num_attention_heads", "num_key_value_heads", json!(1)),
            ("num_key_value_heads", "num_key_value_heads", json!(3)),
            ("head_dim", "head_dim", json!(0)),
            ("partial_rotary_factor", "partial_rotary_factor", json!(1.5)),
            ("partial_rotary_factor", "rotary", json!(0.4)),
            ("partial_rotary_factor", "partial_rotary_factor", json!(0.0)),
            ("attention_bias", "attention_bias", json!(true)),
            ("linear_num_key_heads", "linear_num_key_heads", json!(3)),
            ("linear_key_head_dim", "linear_key_head_dim", json!(0)),
            ("linear_conv_kernel_dim", "linear_conv_kernel_dim", json!(0)),
            ("mamba_ssm_dtype", "mamba_ssm_dtype", json!("bfloat16")),
            ("output_gate_type", "output_gate_type", json!("softmax")),
            ("indexer_kv_heads", "indexer_kv_heads", json!(2)),
            ("indexer_budget", "indexer_budget", json!(10)),
            ("indexer_head_dim", "indexer_head_dim", json!(4)),
            ("indexer_compress_ratio", "indexer_compress_ratio", json!(0)),
            ("hc_count", "hc_count", json!(1)),
            ("hc_count", "hc_count", json!(17)),
            ("hc_lowrank", "hc_lowrank", json!(0)),
            ("eos_token_id", "eos_token_id", json!(64)),
            ("tie_word_embeddings", "tie_word_embeddings", json!(true)),
            (
                "full_attention_interval",
                "full_attention_interval",
                json!(3),
            ),
            ("dtype", "dtype", json!("float16")),
            (
                "max_position_embeddings",
                "max_position_embeddings",
                json!(0),
            ),
        ];
        for (key, needle, replacement) in cases {
            let mut value = synthetic();
            value["text_config"][key] = replacement.clone();
            let error = dev(&value).expect_err(&format!("{key}={replacement} must fail"));
            assert!(error.contains(needle), "{key}={replacement}: {error}");
        }
        // Unsupported layer spelling is refused.
        let mut value = synthetic();
        value["text_config"]["layer_types"][1] = json!("sliding_attention");
        assert!(dev(&value).unwrap_err().contains("layer_types"));
        // A linear-only prefix (interval longer than the trunk) is a valid
        // developer profile with no QSA layers...
        let mut value = synthetic_fixture_value(&moe(), 32, 1, true, false);
        let linear_only = dev(&value).expect("linear-only developer profile");
        assert_eq!(
            (linear_only.n_linear_layers(), linear_only.n_full_layers()),
            (1, 0)
        );
        // ...but a schedule that contradicts the interval is not.
        value["text_config"]["full_attention_interval"] = json!(1);
        assert!(dev(&value).unwrap_err().contains("full_attention_interval"));
        // Wrong MTP shape.
        let mut value = synthetic();
        value["text_config"]["mtp"]["layer_types"] = json!(["linear_attention"]);
        assert!(dev(&value).unwrap_err().contains("MTP"));
        let mut value = synthetic();
        value["text_config"]["mtp"]["hybrid"] = json!(false);
        assert!(dev(&value).unwrap_err().contains("hybrid"));
        let mut value = synthetic();
        value["text_config"]["mtp"]["rope_theta"] = json!(0.0);
        assert!(dev(&value).unwrap_err().contains("rope_theta"));
        let mut value = synthetic();
        value["text_config"]["mtp"]["num_hidden_layers"] = json!(2);
        assert!(dev(&value).is_err());
    }

    #[test]
    fn synthetic_rejects_bad_ffn_geometry() {
        let set = |value: &mut Value, experts: u64, top_k: u64, inter: u64, shared: Option<u64>| {
            value["text_config"]["num_experts"] = json!(experts);
            value["text_config"]["num_experts_per_tok"] = json!(top_k);
            value["text_config"]["moe_intermediate_size"] = json!(inter);
            value[PROFILE_KEY]["ffn"]["num_experts"] = json!(experts);
            value[PROFILE_KEY]["ffn"]["top_k"] = json!(top_k);
            value[PROFILE_KEY]["ffn"]["intermediate_size"] = json!(inter);
            match shared {
                Some(shared) => {
                    value["text_config"]["shared_expert_intermediate_size"] = json!(shared);
                    value[PROFILE_KEY]["ffn"]["shared_intermediate_size"] = json!(shared);
                }
                None => {
                    value["text_config"]
                        .as_object_mut()
                        .unwrap()
                        .remove("shared_expert_intermediate_size");
                    value[PROFILE_KEY]["ffn"]
                        .as_object_mut()
                        .unwrap()
                        .remove("shared_intermediate_size");
                }
            }
        };
        let mut value = synthetic();
        set(&mut value, 4, 5, 16, Some(16));
        assert!(dev(&value).unwrap_err().contains("num_experts_per_tok"));
        let mut value = synthetic();
        set(&mut value, 4, 0, 16, Some(16));
        assert!(dev(&value).unwrap_err().contains("num_experts_per_tok"));
        let mut value = synthetic();
        set(&mut value, 0, 1, 16, Some(16));
        assert!(dev(&value).unwrap_err().contains("num_experts"));
        let mut value = synthetic();
        set(&mut value, 4, 2, 0, Some(16));
        assert!(dev(&value).unwrap_err().contains("moe_intermediate_size"));
        let mut value = synthetic();
        set(&mut value, 4, 2, 16, Some(0));
        assert!(dev(&value).is_err());
        // Total FFN weight overflow of the developer bound: 8192 experts of
        // 65536 x 3 x 2^20 elements.
        let mut value = synthetic_fixture_value(&moe(), 1 << 16, 2, false, false);
        set(&mut value, 8192, 2, 1 << 20, None);
        let error = dev(&value).unwrap_err();
        assert!(error.contains("FFN weight element count"), "{error}");
        // Dense out of bounds.
        let mut value = dense_value();
        value[PROFILE_KEY]["ffn"]["intermediate_size"] = json!(0);
        assert!(dev(&value).unwrap_err().contains("intermediate_size"));
        let mut value = dense_value();
        value[PROFILE_KEY]["ffn"]["intermediate_size"] = json!((1usize << 20) + 1);
        assert!(dev(&value).unwrap_err().contains("intermediate_size"));
    }

    #[test]
    fn synthetic_rejects_derived_width_overflow() {
        // hidden 65536 * hc 16 * lowrank 4096 exceeds i32 indexing.
        let mut value = synthetic_fixture_value(&moe(), 1 << 16, 2, false, false);
        value["text_config"]["hc_count"] = json!(16);
        value["text_config"]["hc_lowrank"] = json!(4096);
        let error = dev(&value).unwrap_err();
        assert!(error.contains("low-rank down extent"), "{error}");
    }

    #[test]
    fn synthetic_ple_geometry_is_explicit_and_checked() {
        let mutate = |key: &str, replacement: Value| {
            let mut value = synthetic();
            value["text_config"][key] = replacement;
            dev(&value)
        };
        assert!(mutate("ngram_size", json!(4))
            .unwrap_err()
            .contains("ngram_size"));
        assert!(mutate("ple_embed_dim", json!(15))
            .unwrap_err()
            .contains("ple_embed_dim"));
        assert!(mutate("heads_per_ngram", json!(0))
            .unwrap_err()
            .contains("heads_per_ngram"));
        assert!(mutate("heads_per_ngram", json!(65))
            .unwrap_err()
            .contains("heads_per_ngram"));
        assert!(mutate("ple_conv_kernel_size", json!(0))
            .unwrap_err()
            .contains("ple_conv_kernel_size"));
        assert!(mutate("split_ngram_parts", json!(5))
            .unwrap_err()
            .contains("split_ngram_parts"));
        assert!(mutate("split_ngram_parts", json!(4))
            .unwrap_err()
            .contains("trailing shard"));
        assert!(mutate("ngram_vocab_size_base", json!(0))
            .unwrap_err()
            .contains("ngram_vocab_size_base"));
        assert!(mutate("ngram_vocab_size_base", json!(10))
            .unwrap_err()
            .contains("disagrees"));
        assert!(mutate("make_ngram_vocab_size_divisible_by", json!(0))
            .unwrap_err()
            .contains("make_ngram_vocab_size_divisible_by"));
        assert!(mutate("ple_layer_ids", json!([1, 2]))
            .unwrap_err()
            .contains("ple_layer_ids"));
        assert!(mutate("ple_layer_ids", json!([0]))
            .unwrap_err()
            .contains("ple_layer_ids"));
        assert!(mutate("ple_layer_ids", json!([3]))
            .unwrap_err()
            .contains("ple_layer_ids"));
        // Layer 2 (index 1) is the full-attention layer: PLE is linear-only.
        assert!(mutate("ple_layer_ids", json!([2]))
            .unwrap_err()
            .contains("linear-attention"));
        // The marker's valid_rows only cross-checks the derived count.
        for rows in [json!(u64::MAX), json!(13), json!(0), json!(24)] {
            let mut value = synthetic();
            value[PROFILE_KEY]["ple"]["valid_rows"] = rows.clone();
            assert!(
                dev(&value).unwrap_err().contains("valid_rows"),
                "valid_rows={rows} must fail"
            );
        }
        let mut value = synthetic();
        value[PROFILE_KEY]["ple"]["extra"] = json!(1);
        assert!(dev(&value).is_err());
        // Multi-shard geometry derives rows per shard from padding.
        let mut value = synthetic();
        value["text_config"]["split_ngram_parts"] = json!(3);
        let cfg = dev(&value).unwrap();
        let ple = cfg.ple.unwrap();
        assert_eq!(
            (ple.shard_count, ple.shard_rows, ple.padded_rows),
            (3, 8, 24)
        );
        assert_eq!(
            (0..3).map(|i| ple.shard_valid_rows(i)).collect::<Vec<_>>(),
            vec![8, 8, 2]
        );
    }

    #[test]
    fn typed_config_and_scalar_mirrors_must_agree() {
        let mut cfg = dev(&synthetic()).unwrap();
        assert!(cfg.validate().is_ok());
        cfg.num_experts += 1;
        assert!(cfg.validate().unwrap_err().contains("typed ffn mirror"));
        let mut cfg = dev(&synthetic()).unwrap();
        cfg.ple_embed_dim += 16;
        assert!(cfg.validate().unwrap_err().contains("typed ple mirror"));
        let mut cfg = dev(&synthetic()).unwrap();
        cfg.mtp_num_hidden_layers = 0;
        assert!(cfg.validate().unwrap_err().contains("typed mtp mirror"));
        let mut cfg = dev(&synthetic_fixture_value(&moe(), 32, 2, false, false)).unwrap();
        cfg.ple_layer_ids = vec![1];
        assert!(cfg.validate().unwrap_err().contains("typed ple is absent"));
        let mut cfg = dev(&dense_value()).unwrap();
        cfg.num_experts = 4;
        assert!(cfg.validate().unwrap_err().contains("typed ffn mirror"));
        // A typed ffn / mtp that no longer matches the pinned Flash-Next is
        // refused.
        let mut flash = Qwen4Config::from_nested_value(&fixture()).unwrap();
        flash.ffn = Qwen4FfnConfig::Dense {
            intermediate_size: 640,
        };
        assert!(flash.validate().is_err());
        let mut flash = Qwen4Config::from_nested_value(&fixture()).unwrap();
        flash.mtp = None;
        assert!(flash.validate().is_err());
        // A synthetic config cannot be relabeled Flash-Next.
        let mut relabeled = dev(&synthetic()).unwrap();
        relabeled.profile = Qwen4Profile::FlashNext;
        assert!(relabeled.validate().is_err());
    }

    #[test]
    fn developer_metadata_json_round_trips_with_ple_metadata() {
        for cfg in [
            Qwen4Config::from_nested_value(&fixture()).unwrap(),
            dev(&synthetic()).unwrap(),
            dev(&dense_value()).unwrap(),
            synthetic_test_config(moe(), 16, 2, false, false),
            synthetic_test_config(
                Qwen4FfnConfig::Dense {
                    intermediate_size: 24,
                },
                5120,
                64,
                true,
                true,
            ),
        ] {
            let json = cfg.developer_metadata_json();
            let parsed = Qwen4Config::from_developer_metadata_json(&json).unwrap();
            assert_eq!(parsed, cfg);
            if cfg.profile.is_flash_next() {
                assert!(!json.contains(PROFILE_KEY));
                assert_eq!(Qwen4Config::from_metadata_json(&json).unwrap(), cfg);
            } else {
                assert!(Qwen4Config::from_metadata_json(&json).is_err());
            }
            match &cfg.ple {
                Some(ple) => {
                    let metadata = PleHashMetadata::from_json_for(&json, ple)
                        .expect("generated qwen4_ple metadata matches the geometry");
                    ple.check_hash_metadata(&metadata).unwrap();
                    if cfg.profile.is_flash_next() {
                        assert_eq!(metadata, PleHashMetadata::qwen4());
                        assert_eq!(PleHashMetadata::from_json(&json).unwrap(), metadata);
                    }
                }
                None => assert!(!json.contains("qwen4_ple")),
            }
        }
    }

    #[test]
    fn synthetic_ple_matches_hash_metadata_geometry() {
        let cfg = dev(&synthetic()).unwrap();
        let ple = cfg.ple.as_ref().unwrap();
        let metadata = PleHashMetadata::from_parts([3, 5, 7], &[7, 11], &[0, 7], 24, 8).unwrap();
        assert!(ple.check_hash_metadata(&metadata).is_ok());
        let wrong_heads =
            PleHashMetadata::from_parts([3, 5, 7], &[7, 7, 4, 7], &[0, 7, 14, 18], 32, 8).unwrap();
        assert!(ple.check_hash_metadata(&wrong_heads).is_err());
        let wrong_padding =
            PleHashMetadata::from_parts([3, 5, 7], &[7, 11], &[0, 7], 128, 128).unwrap();
        assert!(ple.check_hash_metadata(&wrong_padding).is_err());
        let small_vocab = PleHashMetadata::from_parts([3, 5, 7], &[6, 12], &[0, 6], 24, 8).unwrap();
        assert!(ple
            .check_hash_metadata(&small_vocab)
            .unwrap_err()
            .contains("successive primes"));
    }

    #[test]
    fn ple_row_counts_are_successive_primes() {
        use crate::ple::PLE_HEAD_VOCAB_SIZES;
        assert_eq!(ple_head_vocab_sizes(7, 2).unwrap(), vec![7, 11]);
        assert_eq!(
            ple_head_vocab_sizes(20_000_000, 16).unwrap(),
            PLE_HEAD_VOCAB_SIZES.to_vec()
        );
        assert_eq!(Qwen4PleConfig::flash_next().valid_rows, PLE_VALID_ROWS);
        assert!(!is_prime(0) && !is_prime(1) && is_prime(2) && is_prime(3));
        assert!(!is_prime(561) && !is_prime(25) && is_prime(18_446_744_073_709_551_557));
        assert!(ple_head_vocab_sizes(u64::MAX - 10, 3)
            .unwrap_err()
            .contains("overflows"));
        // Explicit row counts must match the derived primes.
        assert!(Qwen4PleConfig::new(&[1], 16, 2, 1, 1, 3, 7, 8, Some(18)).is_ok());
        assert!(Qwen4PleConfig::new(&[1], 16, 2, 1, 1, 3, 7, 8, Some(17))
            .unwrap_err()
            .contains("disagrees"));
        assert_eq!(
            Qwen4PleConfig::new(&[1], 16, 2, 1, 1, 3, 7, 8, None)
                .unwrap()
                .valid_rows,
            18
        );
    }
}
