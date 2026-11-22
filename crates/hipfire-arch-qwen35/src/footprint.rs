// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Device footprint of what a Qwen3.5 load owns besides the KV rows, as pure
//! functions of the model's config and file headers.
//!
//! The capacity projection (`hipfire_loader::admission::qwen35_card_projection`,
//! `hipfire fit`) and the carrier's post-weight context cap both charge these
//! terms, so the numbers `hipfire fit` prints are the numbers the load budgets.
//! Every term is the sum of what the matching allocator asks ROCr for, each
//! allocation charged at what ROCr occupies ([`device_alloc_charge`]); the
//! oracle is `vram-ledger/LEDGER.md` (default `qwen3.8:27b` on a 7900 XTX).
//!
//! Each function names the allocator it mirrors. They are read-only mirrors,
//! not shared code: when an allocator's layout changes, its mirror changes
//! with it, and the tests below pin the 27B figures.

use crate::dflash_spec::{dflash_window_override, DEFAULT_DFLASH_CTX_CAP};
use crate::qwen35::forward::qwen35_flash_partials_len;
use crate::qwen35::weights::{delta_net_state_ef_enabled, DeltaNetCarry, DeltaNetStatePlan};
use crate::qwen35::{LayerType, PrefillBatchScratch, Qwen35Config, StateQuant, PREFILL_MAX_BATCH};
use crate::speculative::DeltaNetSnapshot;
use hip_bridge::{device_alloc_charge, SlabMode, SLAB_ALIGN};
use hipfire_runtime::dflash::{mq_x_rot_elems, DflashConfig};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::gdn_stream::{stream_workspace_plane_bytes, GdnScanAlgorithm};

/// Sum of [`device_alloc_charge`] over one allocation list.
fn charged(sizes: impl IntoIterator<Item = usize>) -> u64 {
    sizes
        .into_iter()
        .map(|size| device_alloc_charge(size) as u64)
        .sum()
}

// ── Weights ──────────────────────────────────────────────────────────────

/// Device bytes of one weight tensor of `bytes` payload bytes. A tensor the
/// slab mode packs ([`SlabMode::max_tensor_bytes`], 2 MiB by default) is a
/// view into a shared slab at its exact size, 4 KiB aligned; anything larger
/// is its own `hipMalloc`. Slab tail waste (at most one partly filled 32 MiB
/// slab per mode) is not charged.
fn weight_tensor_charge(bytes: usize, slab: SlabMode) -> u64 {
    if bytes <= slab.max_tensor_bytes() {
        bytes.div_ceil(SLAB_ALIGN).saturating_mul(SLAB_ALIGN) as u64
    } else {
        device_alloc_charge(bytes) as u64
    }
}

/// Device bytes the trunk's weights occupy once loaded: every tensor of the
/// file at its payload size, ROCr-rounded the way the load allocates it.
/// Device weights are exactly the file (14,337 MiB for the 14.99 GB
/// `qwen3.8:27b` file, `vram-ledger/LEDGER.md` §1), so this replaces the old
/// `file + file / 8 + 512 MiB` estimate, which over-counted by 2.25 GiB.
///
/// Under the qwen35 default slab mode (`64m`, [`crate::carrier::weight_slab_default`])
/// the dense loader reserves one slab per layer for the layer's tensors above
/// 1 MiB (`load.rs` `reserve_layer_slab`), so ROCr's 2 MiB rounding is paid
/// once per layer; every other packed tensor costs its 4 KiB-aligned size and
/// a tensor above the mode's limit is its own `hipMalloc`. Before the packing
/// the 27B paid 283 MiB of per-tensor rounding (ledger §1).
///
/// It charges the steady state after the load: the Z|beta|alpha fold copy
/// stands in for Z (no unfused Z is uploaded). The token embedding is charged
/// only when it stays on the card: on a discrete GPU (`unified_memory` false)
/// with `memory.embed_host` on, the load pins it in host RAM
/// ([`crate::qwen35::load::host_placed_token_embd`], 1,288 MiB on the 27B) and
/// [`token_embd_host_bytes`] reports it instead.
pub fn weights_device_bytes(hfq: &HfqFile, unified_memory: bool) -> u64 {
    let host_embed = crate::qwen35::load::host_placed_token_embd(hfq, unified_memory)
        .map(|info| info.name.as_str());
    let tensors = hfq
        .tensor_infos()
        .iter()
        .filter(|tensor| Some(tensor.name.as_str()) != host_embed);
    let slab = SlabMode::from_env_explicit()
        .or_else(|| crate::carrier::weight_slab_default(""))
        .unwrap_or(hip_bridge::DEFAULT_SLAB_MODE);
    if slab != SlabMode::Up64M {
        return tensors
            .map(|tensor| weight_tensor_charge(tensor.data_size, slab))
            .sum();
    }
    let mut layer_slabs: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut total = 0u64;
    for tensor in tensors {
        let size = tensor.data_size;
        let layer = tensor
            .name
            .strip_prefix("layers.")
            .or_else(|| tensor.name.split_once(".layers.").map(|(_, rest)| rest))
            .and_then(|rest| rest.split_once('.'))
            .map(|(index, _)| index);
        match layer {
            Some(index) if size > (1 << 20) && size <= slab.max_tensor_bytes() => {
                *layer_slabs.entry(index).or_default() += size.next_multiple_of(SLAB_ALIGN);
            }
            _ => total += weight_tensor_charge(size, slab),
        }
    }
    total + charged(layer_slabs.into_values())
}

/// Pinned host bytes of the token embedding a load of `hfq` keeps off the card
/// ([`weights_device_bytes`] leaves them out): the table's payload, 0 when it
/// stays in VRAM.
pub fn token_embd_host_bytes(hfq: &HfqFile, unified_memory: bool) -> u64 {
    crate::qwen35::load::host_placed_token_embd(hfq, unified_memory)
        .map_or(0, |info| info.data_size as u64)
}

/// A DFlash2 draft's selector codebooks stay in host memory
/// (`DflashWeights::predecessor_codebook`/`successor_codebook`: "never VRAM"),
/// so their file bytes are not device bytes.
fn is_host_side_selector_codebook(name: &str) -> bool {
    name.contains("predecessor") || name.contains("successor")
}

/// [`weights_device_bytes`] for a DFlash draft file: its device-resident
/// tensors (everything but the host-side selector codebooks).
pub fn dflash_weights_device_bytes(hfq: &HfqFile) -> u64 {
    let slab = SlabMode::from_env();
    hfq.tensor_infos()
        .iter()
        .filter(|tensor| !is_host_side_selector_codebook(&tensor.name))
        .map(|tensor| weight_tensor_charge(tensor.data_size, slab))
        .sum()
}

/// Whether a draft's matrix weights use an MQ (FWHT-rotated) format, which
/// makes the draft scratch own the rotation buffers
/// (`DflashWeights::has_mq`). Read off the draft's `fc.weight` quant type: the
/// MQ set of `dflash_lm_head_refusal` (MQ4/6/5/3 G256, MQ2..6 G256V2).
pub fn dflash_draft_has_mq(hfq: &HfqFile) -> bool {
    const MQ_QUANT_TYPES: [u8; 9] = [13, 15, 17, 31, 44, 47, 48, 49, 50];
    hfq.tensor_infos()
        .iter()
        .find(|tensor| tensor.name == "fc.weight" || tensor.name.ends_with(".fc.weight"))
        .is_some_and(|tensor| MQ_QUANT_TYPES.contains(&tensor.quant_type))
}

// ── GDN recurrent state, stream carry and checkpoint snapshots ───────────

/// Microtile of the persistent GDN stream carry the default dense route
/// provisions (`Scan64`): `provision_gdn_stream` sizes the carry from the
/// resolved plans' largest `micro_rows`. A route that resolves `Sequential`
/// (1 row) holds less, so this is the safe side.
const CARRY_MICRO_ROWS: usize = GdnScanAlgorithm::Scan64.micro_rows();

/// One entry per `hipMalloc` of a batch-1 `DeltaNetState` of `quant`, with a
/// stream carry of `carry_micro_rows` when `Some`: S, scales, EF residual,
/// conv state, then the four carry planes of every linear-attention layer
/// (`DeltaNetState::new_batched_with_carry`).
fn dn_state_allocation_sizes(
    config: &Qwen35Config,
    quant: StateQuant,
    carry_micro_rows: Option<usize>,
) -> Vec<usize> {
    let plan = DeltaNetStatePlan::new(config, quant, 1, delta_net_state_ef_enabled(quant));
    let s_dim = config.linear_key_head_dim;
    let n_heads = config.linear_num_value_heads;
    let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim;
    let conv_bytes = conv_channels * config.conv_kernel_dim.saturating_sub(1) * 4;
    let (s_bytes, scale_bytes) = match quant {
        StateQuant::FP32 => (plan.s_per_layer * 4, n_heads * 4),
        StateQuant::Q8 | StateQuant::Q4 => (plan.s_per_layer, n_heads * s_dim * 4),
    };
    let carry = carry_micro_rows.and_then(|rows| DeltaNetCarry::geometry_for(config, rows));
    let mut sizes = Vec::with_capacity(plan.n_layers * 8);
    for _ in 0..plan.n_layers {
        sizes.push(s_bytes);
        sizes.push(scale_bytes);
        if let Some(ef) = plan.ef_per_layer {
            sizes.push(ef * 2);
        }
        sizes.push(conv_bytes);
        if let Some(g) = &carry {
            sizes.extend([
                g.state_bytes(),
                g.conv_bytes(),
                g.raw_qkv_bytes(),
                g.raw_ab_bytes(),
            ]);
        }
    }
    sizes
}

/// Device bytes of one checkpoint snapshot of the default (Q8 + error
/// feedback, canonical stream carry) DeltaNet state: the 27B's 404,226,048 raw
/// bytes occupy 404,750,336 as one packed allocation (they occupied 530 MiB as
/// separate allocations, `vram-ledger/LEDGER.md` §2).
/// [`DeltaNetSnapshot::device_charge_of`] is the one place that knows how a
/// snapshot sits in device memory.
pub fn dn_snapshot_device_charge(config: &Qwen35Config) -> u64 {
    DeltaNetSnapshot::device_charge_of(dn_state_allocation_sizes(
        config,
        StateQuant::Q8,
        Some(CARRY_MICRO_ROWS),
    ))
}

/// Bytes of the checkpoint rings one load can populate at once, from the
/// charge of one snapshot ([`dn_snapshot_device_charge`] before the load,
/// `DeltaNetSnapshot::device_charge_for` after it).
///
/// Each snapshot mirrors the DeltaNet state; only the first of a ring is
/// charged. `take_dn_checkpoint` adds further ones (up to the cap,
/// `HIPFIRE_CACHE_CKPT_MAX`) only from memory the admitted context does not
/// need (`Gpu::kv_fit_floor`). Two rings can be live together: the AR prefill
/// ring (`LoadedModel.prefill_checkpoints`, cap `ckpt_max()`) and, when a
/// DFlash or native-MTP speculator is attached, that speculator's own ring
/// (cap = the same `HIPFIRE_CACHE_CKPT_MAX` knob). `ckpt_max()`,
/// `mtp_cache_policy().checkpoint_cap` and the DFlash builder all read that
/// one knob with default 8 and floor 1, so a single `cap` serves every ring.
pub fn checkpoint_ring_bytes(snapshot_charge: usize, speculator_ring: bool) -> usize {
    snapshot_charge.saturating_mul(1 + usize::from(speculator_ring))
}

/// Device bytes of the resident DeltaNet state of the default load: the
/// recurrent state and canonical stream carry (the same tensors a snapshot
/// mirrors) plus the stream binding's edge workspace
/// (`provision_gdn_stream`'s `gdn_stream_workspace_alloc`, one Bf16-output
/// workspace). 393 MiB for the 27B (ledger: 391 MiB raw "GDN state + carry",
/// plus 144 MiB of rounding before the state was packed into one allocation).
pub fn dn_state_device_charge(config: &Qwen35Config) -> u64 {
    let linear_layers = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    if linear_layers == 0 {
        return 0;
    }
    let workspace = stream_workspace_plane_bytes(
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        2,
    );
    dn_snapshot_device_charge(config) + charged(workspace)
}

// ── Flash-attention partials ─────────────────────────────────────────────

/// Bytes of the shared flash-attention partials buffer per token of KV
/// capacity: `Qwen35Scratch::flash_partials` holds
/// `batch x n_heads x ceil(S / 128) x (2 + head_dim)` F32 for a capacity of
/// `S` tokens (`batch` 16 unless `HIPFIRE_FLASH_PARTIALS_BATCH` says
/// otherwise), i.e. 3,096 B/token on the 27B (24 heads, head_dim 256).
/// Sized from the load's provisional capacity, not the admitted one
/// (`vram-ledger/LEDGER.md` §1: 455 MiB at 153,871 tokens). Exact gfx1100
/// below 8K sizes the buffer by a finer tile; the legacy tile is the rate
/// that matters at the contexts where this term is large.
pub fn flash_partials_bytes_per_token(config: &Qwen35Config) -> usize {
    const TILE: usize = 128;
    let batch = hipfire_runtime::config::get()
        .flash_partials_batch
        .filter(|&n| (1..=PREFILL_MAX_BATCH).contains(&n));
    let one_tile =
        4 * qwen35_flash_partials_len("", config.n_heads, config.head_dim, TILE, TILE, TILE, batch);
    one_tile.div_ceil(TILE)
}

// ── DFlash side ──────────────────────────────────────────────────────────

/// What the paired DFlash draft and its owners hold on the card, itemised.
/// 27B + the published mq4 draft at the default block 16 (W = 2048):
/// weights 911 MiB, scratch 491 MiB, ring 230 MiB, verify 30 MiB, rollback
/// snapshot 386 MiB, GDN tape 34 MiB: 2.0 GiB (the ledger measures the same
/// owners at 2.0 GiB before ROCr rounding).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DflashSide {
    /// The draft's device-resident tensors ([`dflash_weights_device_bytes`]).
    pub draft_weights: u64,
    /// `DflashScratch`: per-layer draft K/V over the window, the `fc`/FWHT
    /// rotation buffers, `target_hidden` and its projection, concat K/V.
    pub draft_scratch: u64,
    /// `HiddenStateRingBuffer`: one `window x hidden` F32 ring plus one
    /// staging buffer per extracted target layer.
    pub hidden_ring: u64,
    /// `VerifyScratch`: the verify block's hidden, logits, rotated hidden,
    /// argmax and persistent prefill-batch scratch.
    pub verify_scratch: u64,
    /// `DeltaNetSnapshot` of the target's state, the verify rollback copy.
    pub target_snapshot: u64,
    /// `GdnTape`: the verify forward's per-layer qkv/alpha/beta capture and
    /// replay scratch.
    pub gdn_tape: u64,
}

impl DflashSide {
    pub fn total(&self) -> u64 {
        [
            self.draft_weights,
            self.draft_scratch,
            self.hidden_ring,
            self.verify_scratch,
            self.target_snapshot,
            self.gdn_tape,
        ]
        .iter()
        .fold(0u64, |sum, bytes| sum.saturating_add(*bytes))
    }
}

/// The draft-side facts [`dflash_side`] sizes from.
pub struct DraftFootprintInput<'a> {
    pub config: &'a DflashConfig,
    /// Device bytes of the draft's weights ([`dflash_weights_device_bytes`]).
    pub weights_device_bytes: u64,
    /// The draft carries MQ weights, so its scratch owns the FWHT rotation
    /// buffers ([`dflash_draft_has_mq`]).
    pub has_mq: bool,
    /// `speculation.dflash_block`, resolved per serving configuration
    /// (`DflashBlockPolicy::resolve`); the draft's own width applies when it is
    /// not length-generic ([`DflashConfig::runtime_block_size`]).
    pub block_request: usize,
    /// The context the load sizes, for a draft with a full-attention layer
    /// (its long-reach ring spans the whole context) or without a window
    /// (capped at [`DEFAULT_DFLASH_CTX_CAP`]). An all-sliding DFlash2 draft,
    /// every published one, does not depend on it.
    pub context_rows: usize,
}

/// How the draft scratch holds context (`DraftCtxMode`).
enum DraftContext {
    /// One contiguous `rows`-row context.
    Legacy { rows: usize },
    /// Sliding `w`-row rings; a draft with a full-attention last layer also
    /// holds one `w_full`-row long-reach layer.
    Windowed { w: usize, long_reach: Option<usize> },
}

/// One entry per `alloc_tensor` of a `DflashScratch` of `block` rows over
/// `context` (`DflashScratch::new_with_mq_inner`, `new_windowed`), F32 unless
/// the sizing is the F16 FWHT twin.
fn draft_scratch_allocations(
    cfg: &DflashConfig,
    block: usize,
    context: &DraftContext,
    with_mq: bool,
) -> Vec<usize> {
    let (b, h, inter) = (block, cfg.hidden, cfg.intermediate);
    let (qd, kvd, fw) = (cfg.q_dim(), cfg.kv_dim(), cfg.feature_width());
    let (rows, long_reach) = match *context {
        DraftContext::Legacy { rows } => (rows, None),
        DraftContext::Windowed { w, long_reach } => (w, long_reach),
    };
    let tot = rows + b;
    let mut sizes: Vec<usize> = Vec::with_capacity(32 + 2 * cfg.n_layers);
    if with_mq {
        let widest = mq_x_rot_elems(inter, qd, fw, None);
        sizes.extend([widest * 4, widest * 2]);
    }
    if let (Some(kernel), Some(group)) = (cfg.conv_kernel_size, cfg.conv_group_size) {
        sizes.extend([b * h * 4, b * 2 * kernel * (h / group) * 4]);
    }
    if let (Some(rank), Some(top_k)) = (cfg.selector_rank, cfg.selector_top_k) {
        sizes.extend([b * rank * 4, b * top_k * 4, b * top_k * 4]);
    }
    // The long-reach layer replaces the last layer's `rows`-row pair.
    let ring_layers = cfg.n_layers - usize::from(long_reach.is_some() && cfg.n_layers > 0);
    for _ in 0..ring_layers {
        sizes.extend([rows * kvd * 4, rows * kvd * 4]);
    }
    sizes.extend([
        b * h * 4,     // x
        b * h * 4,     // x_norm
        b * qd * 4,    // q
        b * kvd * 4,   // k_noise
        b * kvd * 4,   // v_noise
        b * inter * 4, // gate
        b * inter * 4, // up
        b * inter * 4, // gate_up
        b * qd * 4,    // attn_out
        b * h * 4,     // residual
        rows * fw * 4, // target_hidden
        rows * h * 4,  // target_hidden_proj
        tot * kvd * 4, // k_cat
        tot * kvd * 4, // v_cat
        b * 4,         // positions_q
        tot * 4,       // positions_k
        b * 4,         // noise_tokens
    ]);
    if let Some(w_full) = long_reach {
        let w_full = w_full.max(rows);
        let full_tot = w_full + b;
        sizes.extend([
            w_full * kvd * 4,   // k_full_cached
            w_full * kvd * 4,   // v_full_cached
            full_tot * kvd * 4, // k_cat_full
            full_tot * kvd * 4, // v_cat_full
            full_tot * 4,       // positions_k replacement
        ]);
        // The base `positions_k` ([rows + b]) is freed once the full-width one
        // exists.
        if let Some(at) = sizes.iter().rposition(|&size| size == tot * 4) {
            sizes.remove(at);
        }
    }
    sizes
}

/// What `load_dflash_state` + `alloc_dflash_owners` (singleton) leave on the
/// card for `draft` paired to `target`. Mirrors the allocators the ledger
/// itemises: draft weights, `DflashScratch` (`hipfire-runtime/src/dflash.rs`),
/// `HiddenStateRingBuffer` and `VerifyScratch` (`speculative.rs`),
/// `DeltaNetSnapshot` and `GdnTape`.
pub fn dflash_side(target: &Qwen35Config, draft: &DraftFootprintInput<'_>) -> DflashSide {
    let cfg = draft.config;
    let block = cfg.runtime_block_size(draft.block_request);
    let max_n = block + 1;
    // The window is the draft-declared one unless `HIPFIRE_DFLASH_WINDOW`
    // overrides it; without one the contiguous context is capped.
    let window = dflash_window_override()
        .ok()
        .flatten()
        .or(cfg.declared_window);
    let (context, ring_rows) = match window {
        Some(w) => (
            DraftContext::Windowed {
                w,
                long_reach: (!cfg.all_layers_sliding).then_some(draft.context_rows),
            },
            w,
        ),
        None => {
            let rows = draft.context_rows.min(DEFAULT_DFLASH_CTX_CAP);
            (DraftContext::Legacy { rows }, rows)
        }
    };
    let draft_scratch = charged(draft_scratch_allocations(
        cfg,
        block,
        &context,
        draft.has_mq,
    ));

    // HiddenStateRingBuffer: per extracted layer a `ring_rows` ring and a
    // staging buffer wide enough for a verify block or a prefill chunk.
    let staging_rows = max_n.max(PREFILL_MAX_BATCH);
    let hidden_ring = charged(
        (0..cfg.target_layer_ids.len())
            .flat_map(|_| [ring_rows * target.dim * 4, staging_rows * target.dim * 4]),
    );

    // VerifyScratch::with_prefill: final hidden, logits, FWHT-rotated hidden,
    // argmax, and the persistent prefill-batch scratch (charged at its raw
    // projected bytes: the projection does not list its ~25 allocations).
    let hidden_k = target.dim.next_power_of_two();
    let verify_scratch = charged([
        max_n * target.dim * 4,
        max_n * target.vocab_size * 4,
        max_n * hidden_k * 4,
        max_n * 4,
    ]) + PrefillBatchScratch::projected_allocation_bytes(target, max_n)
        .unwrap_or(0);

    // GdnTape::new_capture_only: per linear layer qkv, alpha, beta, then the
    // six replay scratches. The multi-layer replay tables it arms are a few
    // KiB.
    let k_dim = target.linear_num_key_heads * target.linear_key_head_dim;
    let v_dim = target.linear_num_value_heads * target.linear_value_head_dim;
    let qkv_dim = 2 * k_dim + v_dim;
    let heads = target.linear_num_value_heads;
    let linear_layers = target
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let gdn_tape = charged(
        (0..linear_layers)
            .flat_map(|_| [max_n * qkv_dim * 4, max_n * heads * 4, max_n * heads * 4])
            .chain([
                max_n * k_dim * 4,
                max_n * k_dim * 4,
                max_n * v_dim * 4,
                max_n * v_dim * 4,
                max_n * v_dim * 4,
                max_n * v_dim * 4,
            ]),
    );

    DflashSide {
        draft_weights: draft.weights_device_bytes,
        draft_scratch,
        hidden_ring,
        verify_scratch,
        target_snapshot: dn_snapshot_device_charge(target),
        gdn_tape,
    }
}

/// [`dflash_side`] for the DFlash draft file at `path`; reads only its header.
/// `Err` names the file when it is not an openable DFlash HFQ.
pub fn dflash_side_for_file(
    target: &Qwen35Config,
    path: &std::path::Path,
    block_request: usize,
    context_rows: usize,
) -> Result<DflashSide, String> {
    let hfq = HfqFile::open(path).map_err(|e| format!("DFlash draft {}: {e}", path.display()))?;
    let config = DflashConfig::from_hfq(&hfq).ok_or_else(|| {
        format!(
            "DFlash draft {}: failed to parse DflashConfig from HFQ metadata",
            path.display()
        )
    })?;
    Ok(dflash_side(
        target,
        &DraftFootprintInput {
            config: &config,
            weights_device_bytes: dflash_weights_device_bytes(&hfq),
            has_mq: dflash_draft_has_mq(&hfq),
            block_request,
            context_rows,
        },
    ))
}

/// Device footprint of an attached native-MTP head, as the load charges it
/// (S0 census, 2026-11-22: the loader's lazy charge, the head upload and the
/// ring snapshot; every term from the same functions the load uses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpSide {
    /// The head file's tensors at device size, each ROCr-rounded.
    pub head_weights: u64,
    /// The 65,536-row draft lm_head sliced from the trunk's on exact gfx1100
    /// (`draft_vocab`); 0 elsewhere or when the head ships its own.
    pub draft_vocab: u64,
    /// The speculator's checkpoint ring: one more DeltaNet snapshot beside the
    /// AR prefill ring's.
    pub ring_snapshot: u64,
    /// First request's `MtpSpecState` (rollback snapshot, verify scratch, GDN
    /// tape, head scratch and Q8 head KV sized for `context_rows`) plus the
    /// prompt-fill transient: [`crate::mtp_spec::mtp_request_bytes_for`].
    pub state: u64,
}

impl MtpSide {
    /// Everything the head adds on top of the no-head load.
    pub fn total(&self) -> u64 {
        self.head_weights + self.draft_vocab + self.ring_snapshot + self.state
    }
}

/// [`MtpSide`] for the head at `head` (`base_offset` into a bundled trailer, 0
/// for a sidecar) beside the trunk `target`, for context `context_rows` (the
/// provisional window the load sizes the head KV and flash partials at),
/// chain depth `requested_k` on device `arch`. `draft_vocab_bytes` is the
/// device size of the sliced draft lm_head the load attaches on this arch
/// (the caller knows the trunk's lm_head layout), 0 when none.
pub fn mtp_side_for_file(
    target: &Qwen35Config,
    head: &std::path::Path,
    base_offset: u64,
    requested_k: usize,
    context_rows: usize,
    arch: &str,
    draft_vocab_bytes: u64,
) -> Result<MtpSide, String> {
    let hfq = HfqFile::open_at_offset(head, base_offset)
        .map_err(|e| format!("MTP head {}: {e}", head.display()))?;
    let head_weights = charged(hfq.tensor_infos().iter().map(|t| t.data_size));
    let (head_config, compressed_vocab) =
        crate::mtp_head::read_mtp_head_sizing(head, base_offset, context_rows)?;
    let snapshot = usize::try_from(dn_snapshot_device_charge(target))
        .map_err(|_| "DeltaNet snapshot sizing overflow".to_string())?;
    // The batched prompt fill is the larger layout and the one every measured
    // head took; the loader charges it too when the head's dtypes are unknown.
    let state = crate::mtp_spec::mtp_request_bytes_for(
        arch,
        crate::qwen35::prefill_max_batch_planned(arch),
        target,
        snapshot,
        &head_config,
        compressed_vocab,
        true,
        requested_k,
    )? as u64;
    Ok(MtpSide {
        head_weights,
        draft_vocab: if compressed_vocab.is_none() {
            draft_vocab_bytes
        } else {
            0
        },
        ring_snapshot: snapshot as u64,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    /// The 27B's text config as the published `qwen3.8-27b.mq4-xts` header
    /// writes it.
    fn qwen38_27b() -> Qwen35Config {
        let layers: Vec<&str> = (0..64)
            .map(|i| {
                if i % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let metadata = serde_json::json!({"config": {"text_config": {
            "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 24,
            "num_key_value_heads": 4, "head_dim": 256, "intermediate_size": 17408,
            "vocab_size": 248320, "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_value_head_dim": 128, "layer_types": layers,
            "max_position_embeddings": 262144,
        }}});
        crate::qwen35::config_from_metadata_json(&metadata.to_string()).unwrap()
    }

    /// The published mq4 DFlash2 draft's header (`qwen38-27b-dflash-mq4.hfq`).
    fn qwen38_27b_draft() -> DflashConfig {
        let metadata = serde_json::json!({
            "config": {"use_sliding_window": true, "sliding_window": 2048,
                "layer_types": vec!["sliding_attention"; 5]},
            "dflash": {"block_size": 8, "mask_token_id": 248070,
                "target_layer_ids": [5, 19, 33, 47, 61], "num_target_layers": 64,
                "num_hidden_layers": 5, "hidden_size": 5120, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
                "vocab_size": 248320,
                "dflash_config": {"conv_group_size": 16, "conv_kernel_size": 2,
                    "selector_rank": 256, "selector_top_k": 16}}});
        DflashConfig::from_metadata(&metadata).unwrap()
    }

    #[test]
    fn snapshot_of_the_27b_occupies_its_packed_size_not_530_mib() {
        let config = qwen38_27b();
        // Raw bytes: the DFlash snapshots in the daemon's `max_seq fit` line
        // (checkpoints 808,452,096 = 2 x 404,226,048).
        let raw: usize = dn_state_allocation_sizes(&config, StateQuant::Q8, Some(CARRY_MICRO_ROWS))
            .iter()
            .sum();
        assert_eq!(raw, 404_226_048);
        // Occupancy: one packed allocation, rounded once to 2 MiB. Separate
        // allocations paid 144 MiB of ROCr rounding on the 1.5 MiB EF
        // residuals and the 3 MiB / 2.5 MiB carry planes (ledger §2: "530 MiB").
        let charge = dn_snapshot_device_charge(&config);
        assert_eq!(charge, 404_750_336);
        assert!(charge - (raw as u64) < 2 * MIB);
    }

    #[test]
    fn resident_dn_state_adds_the_stream_workspace() {
        // Ledger "GDN recurrent state + canonical stream carry": 391 MiB raw
        // (386 + 5 of workspace), now one packed allocation plus the workspace.
        assert_eq!(dn_state_device_charge(&qwen38_27b()), 411_582_464);
    }

    #[test]
    fn flash_partials_are_3096_bytes_per_token_on_the_27b() {
        let rate = flash_partials_bytes_per_token(&qwen38_27b());
        assert_eq!(rate, 3_096);
        // Ledger: 455 MiB at the provisional 153,871 tokens.
        assert_eq!((rate as u64 * 153_871) / MIB, 454);
    }

    #[test]
    fn dflash_side_of_the_27b_is_about_two_gib() {
        let target = qwen38_27b();
        let draft = qwen38_27b_draft();
        assert_eq!(draft.declared_window, Some(2048));
        assert!(draft.all_layers_sliding);
        let side = dflash_side(
            &target,
            &DraftFootprintInput {
                config: &draft,
                // Ledger: the draft's 911 MiB of device tensors (the 1.21 GB
                // file less its 242 MiB of host-side selector codebooks).
                weights_device_bytes: 911 * MIB,
                has_mq: true,
                block_request: 16,
                context_rows: 262_144,
            },
        );
        // Ring: 5 layers x (2048 x 5120 x 4 B ring + 256 x 5120 x 4 B staging
        // rounded to 6 MiB) = 230 MiB (ledger 225 raw).
        assert_eq!(side.hidden_ring, 5 * (40 * MIB + 6 * MIB));
        // Rollback snapshot = a checkpoint snapshot.
        assert_eq!(side.target_snapshot, 404_750_336);
        // Tape: ledger 34 MiB.
        assert_eq!(side.gdn_tape / MIB, 34);
        // Scratch: ledger 475 MiB measured as one site.
        assert!(
            (470 * MIB..=500 * MIB).contains(&side.draft_scratch),
            "{}",
            side.draft_scratch / MIB
        );
        assert!(
            (25 * MIB..=50 * MIB).contains(&side.verify_scratch),
            "{}",
            side.verify_scratch / MIB
        );
        // Whole side: ~2.0 GiB measured (2.2 GiB with ROCr rounding), against
        // the 1.77 GiB the old `file + file/8 + 512 MiB` reserve held.
        let total = side.total();
        assert!(
            (2_000 * MIB..=2_300 * MIB).contains(&total),
            "{} MiB",
            total / MIB
        );
    }

    #[test]
    fn an_all_sliding_draft_does_not_grow_with_the_context() {
        let target = qwen38_27b();
        let draft = qwen38_27b_draft();
        let side = |context_rows| {
            dflash_side(
                &target,
                &DraftFootprintInput {
                    config: &draft,
                    weights_device_bytes: 0,
                    has_mq: true,
                    block_request: 16,
                    context_rows,
                },
            )
            .total()
        };
        assert_eq!(side(4_096), side(262_144));
    }

    #[test]
    fn a_draft_with_a_full_attention_layer_holds_a_long_reach_ring() {
        let target = qwen38_27b();
        let mut draft = qwen38_27b_draft();
        draft.all_layers_sliding = false;
        let side = |context_rows| {
            dflash_side(
                &target,
                &DraftFootprintInput {
                    config: &draft,
                    weights_device_bytes: 0,
                    has_mq: false,
                    block_request: 16,
                    context_rows,
                },
            )
            .draft_scratch
        };
        // K, V and the two concat buffers span the context: 4 x kv_dim x 4 B
        // per row, 16 KiB per token here (kv_dim 1024).
        let growth = side(65_536) - side(32_768);
        let per_token = 4 * 1024 * 4;
        assert!(
            growth.abs_diff(32_768 * per_token) <= 8 * MIB,
            "{} MiB",
            growth / MIB
        );
    }

    #[test]
    fn selector_codebooks_are_host_side() {
        assert!(is_host_side_selector_codebook(
            "candidate_selector.predecessor_codebook"
        ));
        assert!(is_host_side_selector_codebook(
            "candidate_selector.successor_codebook"
        ));
        assert!(!is_host_side_selector_codebook(
            "candidate_selector.hidden_projection.weight"
        ));
        assert!(!is_host_side_selector_codebook("fc.weight"));
    }

    #[test]
    fn weight_tensors_round_to_the_rocr_granule_unless_slab_packed() {
        // Default slab: tensors up to 2 MiB are exact (4 KiB aligned) slab
        // views; bigger ones take whole 2 MiB granules.
        let slab = SlabMode::Up2M;
        assert_eq!(weight_tensor_charge(100, slab), 4096);
        assert_eq!(weight_tensor_charge(2 << 20, slab), 2 << 20);
        assert_eq!(weight_tensor_charge((2 << 20) + 1, slab), 4 << 20);
        // No slab: a tensor between 1 and 2 MiB takes a whole 2 MiB granule.
        assert_eq!(weight_tensor_charge((1 << 20) + 1, SlabMode::Off), 2 << 20);
        // The 27B's 47,349,760 B MLP projection (45.2 MiB) occupies 46 MiB.
        assert_eq!(weight_tensor_charge(47_349_760, slab), 46 << 20);
    }

    #[test]
    fn a_draft_file_is_sized_from_its_header_with_codebooks_left_on_the_host() {
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};
        let tensor = |name: &str, quant_type: u8, bytes: usize| HfqMemTensor {
            name: name.to_owned(),
            quant_type,
            shape: vec![4, 4],
            group_size: 0,
            data: vec![0u8; bytes],
        };
        let metadata = serde_json::json!({
            "architecture": "dflash",
            "config": {"use_sliding_window": true, "sliding_window": 2048,
                "layer_types": vec!["sliding_attention"; 5]},
            "dflash": {"block_size": 8, "mask_token_id": 248070,
                "target_layer_ids": [5, 19, 33, 47, 61], "num_target_layers": 64,
                "num_hidden_layers": 5, "hidden_size": 5120, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 17408,
                "vocab_size": 248320,
                "dflash_config": {"conv_group_size": 16, "conv_kernel_size": 2,
                    "selector_rank": 256, "selector_top_k": 16}}});
        let path = std::env::temp_dir().join(format!(
            "hipfire-footprint-draft-{}.hfq",
            std::process::id()
        ));
        write_hfqm_package_mem(
            &path,
            5,
            &metadata.to_string(),
            &[
                // An MQ4V2 matrix of 3 MiB: one 4 MiB allocation.
                tensor("fc.weight", 44, 3 << 20),
                tensor("candidate_selector.predecessor_codebook", 1, 3 << 20),
                tensor("candidate_selector.successor_codebook", 1, 3 << 20),
            ],
        )
        .unwrap();
        let target = qwen38_27b();
        let side = dflash_side_for_file(&target, &path, 16, 8192).unwrap();
        let _ = std::fs::remove_file(&path);
        // The two host-side codebooks do not count; the matrix is rounded.
        assert_eq!(side.draft_weights, 4 * MIB);
        // The same draft built by hand: MQ weights, so the FWHT buffers exist.
        let by_hand = dflash_side(
            &target,
            &DraftFootprintInput {
                config: &qwen38_27b_draft(),
                weights_device_bytes: 4 * MIB,
                has_mq: true,
                block_request: 16,
                context_rows: 8192,
            },
        );
        assert_eq!(side, by_hand);
        // A file that is not a DFlash HFQ names itself in the error.
        let missing = dflash_side_for_file(&target, std::path::Path::new("/no/such.hfq"), 16, 0);
        assert!(missing.unwrap_err().contains("/no/such.hfq"));
    }
}
