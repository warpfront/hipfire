// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed-expert residency for cards whose VRAM cannot hold every expert.
//!
//! Flash-Next's routed experts are ~65 GB, the rest of the model ~5 GB. On a
//! discrete card the experts of trunk layers at or past the VRAM layer count
//! ([`EXPERT_VRAM_LAYERS_ENV`]) and the MTP layer's are fulfilled into pinned,
//! device-mapped host RAM instead of VRAM. The sealed MoE kernels already read
//! every expert through per-layer pointer tables, so a host-mapped layer is
//! read over PCIe (zero-copy) with no kernel or dispatch change: a decode
//! token reads only its routed experts' bytes from host RAM.

use crate::gpu_forward::Qwen4ForwardScratchMode;
use crate::weights::Qwen4BandScope;
use crate::Qwen4Config;
use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};
use rdna_compute::uma_memory::{
    amdgpu_gtt_used_bytes, read_sysfs_u64, TTM_PAGES_LIMIT, TTM_PAGE_BYTES,
};

/// `N` keeps the routed experts of trunk layers `0..N` in VRAM; `auto` picks
/// the largest `N` that fits the card's free VRAM. Unset keeps every expert
/// resident where they fit ([`resolve_expert_vram_layers`]) and is `auto`
/// otherwise. Set, it also keeps host memory out of reclaim from before the
/// HIP runtime loads (hip-bridge owns the name), as a process about to load
/// a Qwen4 model on a discrete GPU does unset
/// ([`keeps_host_memory_out_of_reclaim`]).
pub const EXPERT_VRAM_LAYERS_ENV: &str = hip_bridge::QWEN4_EXPERT_VRAM_LAYERS_ENV;

/// VRAM left free by `auto` beyond the resident non-expert weights: forward
/// scratch, KV and state for [`AUTO_VRAM_RESERVE_MAX_SEQ`] tokens plus
/// headroom, without native MTP. Measured on a gfx1201 R9700 at N=16 with a
/// [`AUTO_VRAM_RESERVE_CHUNK`]-row prefill chunk: every context-sized buffer
/// is allocated at load, and a 32,700-token prefill plus decode at that
/// context peaked at 6,199 MiB beyond the non-expert and expert weights,
/// 457 MiB under this reserve. [`auto_vram_reserve`] adds what a longer
/// context, a different chunk and native MTP take; a shorter context keeps it.
pub const AUTO_VRAM_RESERVE_BYTES: u64 = 6656 << 20;

/// Device memory the HIP runtime holds on an exact gfx1201 beyond every
/// allocation the load and the forward make: code objects, HSA queues and
/// their context-save areas, kernel scratch and the ROCr pools. Measured on an
/// R9700 as the device-wide sysfs `mem_info_vram_used` minus every live
/// allocation `rocprofv3 --memory-allocation-trace` reports, beyond the
/// 57 MiB idle baseline:
/// - pruned W3-slab pack fully resident, compact scratch, 1024-row chunk:
///   929 MiB after load, 1,330 MiB at the peak of a 32K-token request
///   (receipt `slab-scratch-notes/rcpt/h1-proof-run1/p1024`);
/// - shipped `mq3-xts` with 30 routed layers in pinned host RAM, full scratch,
///   1024-row chunk: 974 MiB after load, 1,373 MiB at the peak (receipt
///   `slab-scratch-notes/rcpt/h2a-run2/c1024`). The same load at 4096 and
///   8192 rows (no trace; the traced allocations rebased by the census
///   difference) peaks at about 1,380 and 1,383 MiB, at 256 rows at about
///   956 MiB.
///
/// The growth after load appears only once a request runs at least 512 rows.
/// The constant is the largest peak rounded up to 1,408 MiB.
/// The historical floor's stale F32 context used to cover it; with the floor
/// retired on this route the census carries it as [`ReserveCensus::runtime`].
pub const QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES: u64 = 1408 << 20;

/// The measured HIP runtime overhead the census charges: only where it was
/// measured, an exact gfx1201 discrete (not unified-memory) single-device
/// load. `None` elsewhere, where the historical floor still applies.
pub fn runtime_overhead_bytes(arch: &str, unified_memory: bool, ranks: usize) -> Option<u64> {
    (arch == "gfx1201" && !unified_memory && ranks == 1)
        .then_some(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES)
}

/// Context [`AUTO_VRAM_RESERVE_BYTES`] was measured at (`hipfire run`'s
/// legacy-KV `max_seq`).
pub const AUTO_VRAM_RESERVE_MAX_SEQ: usize = 32768;

/// Prefill chunk [`AUTO_VRAM_RESERVE_BYTES`] was measured at; its forward
/// scratch then carried speculative logits and argmax for every chunk row.
pub const AUTO_VRAM_RESERVE_CHUNK: usize = 1536;

/// Host RAM that must remain available after the pinned experts are placed.
pub const HOST_RAM_HEADROOM_BYTES: u64 = 4 << 30;

/// Slack `Gpu::upload_raw_host_mapped` adds to every host-mapped tensor.
pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertVramLayers {
    Layers(usize),
    Auto,
}

/// Parse [`EXPERT_VRAM_LAYERS_ENV`]. `Ok(None)` = unset
/// ([`resolve_expert_vram_layers`] picks the placement).
pub fn expert_vram_layers_from_env() -> Result<Option<ExpertVramLayers>, String> {
    match hipfire_config::developer_var(EXPERT_VRAM_LAYERS_ENV) {
        Ok(value) => parse_expert_vram_layers(&value).map(Some),
        Err(_) => Ok(None),
    }
}

/// The placement for a discrete-GPU load: an explicit `policy` always wins.
/// Unset (`None`) keeps every routed expert resident (`Ok(None)`) when
/// `fits_resident` reports that the whole expert footprint, the non-expert
/// weights and the `auto` reserve fit in VRAM, and is `auto` otherwise.
/// `fits_resident` runs only when unset.
pub fn resolve_expert_vram_layers(
    policy: Option<ExpertVramLayers>,
    fits_resident: impl FnOnce() -> Result<bool, String>,
) -> Result<Option<ExpertVramLayers>, String> {
    match policy {
        Some(policy) => Ok(Some(policy)),
        None => Ok((!fits_resident()?).then_some(ExpertVramLayers::Auto)),
    }
}

/// A profile without routed experts (a dense FFN) has nothing to place in
/// host RAM: its gate/up/down projections are ordinary resident weights, so
/// the placement is always fully resident (`resident_split` counts them as
/// non-expert bytes, `routed_expert_bytes` is 0).  An explicit
/// [`EXPERT_VRAM_LAYERS_ENV`] request (a layer count or `auto`) is refused
/// instead of being honored by moving dense weights as fake experts; unset
/// is fine.
pub fn refuse_host_expert_request(
    policy: Option<ExpertVramLayers>,
    profile: &str,
) -> Result<(), String> {
    let Some(policy) = policy else {
        return Ok(());
    };
    let value = match policy {
        ExpertVramLayers::Layers(layers) => layers.to_string(),
        ExpertVramLayers::Auto => "auto".to_string(),
    };
    Err(format!(
        "{EXPERT_VRAM_LAYERS_ENV}={value} requests host-mapped routed experts, but profile \
         {profile} has a dense FFN with no routed experts (its gate/up/down are resident \
         weights); unset {EXPERT_VRAM_LAYERS_ENV}"
    ))
}

/// Whether every routed expert (`expert_bytes`), the non-expert weights and
/// `reserve` ([`auto_vram_reserve`]) fit in `free_vram`.
pub fn fits_fully_resident(
    free_vram: u64,
    non_expert_bytes: u64,
    expert_bytes: u64,
    reserve: u64,
) -> bool {
    non_expert_bytes
        .checked_add(expert_bytes)
        .and_then(|bytes| bytes.checked_add(reserve))
        .is_some_and(|need| need <= free_vram)
}

/// Whether a process about to load the model with HFQ `arch_id` on the GPUs
/// `device_archs` (every card it may use) keeps host memory out of reclaim
/// before the HIP runtime loads (`hip_bridge::keep_host_memory_out_of_reclaim`),
/// as an explicit [`EXPERT_VRAM_LAYERS_ENV`] does: a Qwen4 model on discrete
/// GPUs. It rests on those static facts, not on the placement the load
/// resolves: unset, that depends on the free VRAM and the reserve at load
/// time, after the runtime has read the switches, and the upload of tens of
/// GB out of the mapped file needs them under host-memory pressure whatever
/// the placement. Unified memory never host-maps experts, and an
/// unrecognized arch keeps ROCm's defaults.
pub fn keeps_host_memory_out_of_reclaim(arch_id: u32, device_archs: &[&str]) -> bool {
    arch_id == crate::ARCH_ID
        && !device_archs.is_empty()
        && device_archs
            .iter()
            .all(|arch| hipfire_config::is_discrete_memory_arch(arch))
}

/// Bytes of every routed expert payload (trunk and MTP layers).
pub fn routed_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    routed_expert_bytes_ranks(weights, bytes_of, &EpExpertSpans::single())
}

/// [`routed_expert_bytes`] for one rank of an expert-parallel load with the
/// sealed `spans`: the largest rank's share of every routed expert stack,
/// which is what every rank budgets so a layer is resident on all ranks or
/// none.
pub fn routed_expert_bytes_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    spans: &EpExpertSpans,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| {
            is_routed_expert(&entry.name)
                && !entry.residency.is_external()
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .try_fold(0u64, |total, entry| {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            Ok(total + rank_share(entry, bytes, spans.largest_rank(), spans)?)
        })
}

/// [`routed_expert_bytes`] for one pipeline-parallel band: every expert of
/// the owned trunk layers (all 512, never a modulo-rank share) plus the MTP
/// head's experts only on the output band, from the exact per-entry payload
/// sizes. `Single` equals [`routed_expert_bytes`].
pub fn routed_expert_bytes_scope(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
) -> Result<u64, String> {
    let bytes = band_weight_census(weights, &bytes_of, scope)?;
    bytes
        .trunk
        .checked_add(bytes.mtp)
        .ok_or_else(|| "band routed-expert byte census overflows".to_string())
}

fn parse_expert_vram_layers(value: &str) -> Result<ExpertVramLayers, String> {
    let value = value.trim();
    if value == "auto" {
        return Ok(ExpertVramLayers::Auto);
    }
    value
        .parse::<usize>()
        .map(ExpertVramLayers::Layers)
        .map_err(|_| {
            format!("{EXPERT_VRAM_LAYERS_ENV}={value:?} is neither `auto` nor a layer count")
        })
}

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj")
        || name.ends_with(".mlp.experts.down_proj")
        || name.ends_with(crate::expert_arena::ARENA_SUFFIX)
}

/// Resident bytes outside the routed experts, and the routed-expert bytes of
/// one trunk layer (layer 0), from each entry's payload size. Tied aliases
/// have no payload of their own and external rows stay in the file.
pub fn resident_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    resident_split_ranks(weights, bytes_of, &EpExpertSpans::single())
}

/// [`resident_split`] for one rank of an expert-parallel load with the sealed
/// `spans`: the non-expert bytes are replicated on every rank, the layer-0
/// routed-expert bytes are the largest rank's share.
pub fn resident_split_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    spans: &EpExpertSpans,
) -> Result<(u64, u64), String> {
    let largest = spans.largest_rank();
    let mut non_expert = 0u64;
    let mut layer_experts = 0u64;
    // Slab arenas differ in size from layer to layer (the hot set is not
    // uniform), so the per-layer charge is the largest arena, never layer 0.
    let mut layer_arena_max = 0u64;
    for entry in weights {
        if entry.residency.is_external() || matches!(entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        if !is_routed_expert(&entry.name) {
            non_expert += bytes;
        } else if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
            layer_arena_max = layer_arena_max.max(rank_share(entry, bytes, largest, spans)?);
        } else if !entry.name.starts_with("mtp.") && entry.layer == Some(0) {
            layer_experts += rank_share(entry, bytes, largest, spans)?;
        }
    }
    Ok((non_expert, layer_experts + layer_arena_max))
}

/// Gate/up and down payload bytes of the largest trunk layer of a packed
/// (one tensor per projection per layer) manifest, separately: what the S1
/// expert page layout splits over the layer's experts
/// ([`crate::slowlink_admission::PanelPageLayout::new`]). The MTP layer is not
/// a trunk layer and is excluded, like [`resident_split`]. A manifest with no
/// packed projection pair is an error: S1 never sizes an arena layout.
pub fn routed_layer_payload_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    let mut per_layer: std::collections::BTreeMap<usize, (u64, u64)> =
        std::collections::BTreeMap::new();
    for entry in weights {
        if entry.residency.is_external()
            || matches!(entry.policy, ShardPolicy::Tied { .. })
            || entry.name.starts_with("mtp.")
        {
            continue;
        }
        let gate_up = entry.name.ends_with(".mlp.experts.gate_up_proj");
        let down = entry.name.ends_with(".mlp.experts.down_proj");
        let Some(layer) = entry.layer.filter(|_| gate_up || down) else {
            continue;
        };
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        let slot = per_layer.entry(layer).or_default();
        let target = if gate_up { &mut slot.0 } else { &mut slot.1 };
        *target = target
            .checked_add(bytes)
            .ok_or_else(|| "routed layer payload byte census overflows".to_string())?;
    }
    let gate_up = per_layer.values().map(|bytes| bytes.0).max().unwrap_or(0);
    let down = per_layer.values().map(|bytes| bytes.1).max().unwrap_or(0);
    if gate_up == 0 || down == 0 {
        return Err("manifest has no packed routed gate/up and down expert payload".to_string());
    }
    Ok((gate_up, down))
}

/// Stored dtype of the language head, which the native MTP draft head ranks
/// with.
pub fn language_head_dtype(weights: &[WeightEntry]) -> Option<rdna_compute::DType> {
    weights
        .iter()
        .find(|entry| entry.name == crate::weights::QWEN4_LM_HEAD)
        .map(|entry| entry.dtype)
}

/// VRAM `auto` leaves free beyond the non-expert weights for a load whose
/// trunk QSA context storage is `context` with a `chunk_rows` prefill chunk:
/// [`AUTO_VRAM_RESERVE_BYTES`], the trunk QSA arenas' committed bytes in
/// `qsa_format` past those of [`AUTO_VRAM_RESERVE_MAX_SEQ`] legacy tokens
/// (the measured layout), the chunk-sized forward resources' change from the
/// measured [`AUTO_VRAM_RESERVE_CHUNK`] layout, `mtp_bytes` when a native MTP
/// speculator attaches (`mtp_spec::native_mtp_device_bytes`), and
/// `gather_bytes` for the gathered QSA prefill attention's workspace
/// (`rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes`). This is
/// the historical measured formula: the floor of [`AutoReserve`], which
/// takes the larger of it and the bottom-up [`ReserveCensus`]. Only
/// committed bytes are charged: VMM context arenas commit the first chunk's
/// pages (inside the measured layout), their virtual extent nothing, so the
/// admitted context does not size a VMM reserve; legacy arenas commit all
/// of `max_seq`. Callers on a discrete card pass legacy storage whatever the
/// backend ([`crate::Qwen4ContextCommit::for_expert_reserve`]).
///
/// `mode` is the scratch plan the forward will actually allocate
/// ([`crate::gpu_forward::qwen4_forward_scratch_mode`]); only the current
/// forward is charged in it. The measured baseline is a historical
/// subtraction of the original layout and stays `Full` with its all-row
/// speculative logits, so a compact plan's saving reaches the placement
/// instead of being rebased away.
pub fn auto_vram_reserve(
    config: &Qwen4Config,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    qsa_layout: impl Into<rdna_compute::tensor_ops::QsaStateLayout>,
    mtp_bytes: Option<u64>,
    gather_bytes: Option<u64>,
) -> Result<u64, String> {
    auto_vram_reserve_with_forward(
        config,
        context,
        chunk_rows,
        mode,
        qsa_layout,
        mtp_bytes,
        gather_bytes,
        None,
    )
}

/// [`auto_vram_reserve`] with the forward term replaced by `forward` when
/// given: a decode-share load charges only its dedicated forward
/// ([`crate::gpu_forward::qwen4_decode_share_dedicated_bytes`]); `None` is
/// the ordinary `qwen4_forward_device_bytes(config, chunk_rows, mode)`.
#[allow(clippy::too_many_arguments)]
pub fn auto_vram_reserve_with_forward(
    config: &Qwen4Config,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    qsa_layout: impl Into<rdna_compute::tensor_ops::QsaStateLayout>,
    mtp_bytes: Option<u64>,
    gather_bytes: Option<u64>,
    forward: Option<u64>,
) -> Result<u64, String> {
    let qsa_layout = qsa_layout.into();
    let qsa_format = qsa_layout.kv;
    let overflow = |seq| format!("QSA context state for max_seq {seq} overflows");
    let trunk = |bytes: Option<usize>| {
        bytes
            .and_then(|bytes| bytes.checked_mul(config.n_full_layers()))
            .and_then(|bytes| u64::try_from(bytes).ok())
    };
    let committed = trunk(context.committed_layer_bytes(config, qsa_format))
        .ok_or_else(|| overflow(context.max_seq))?;
    let measured_context =
        trunk(config.qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, qsa_format))
            .ok_or_else(|| overflow(AUTO_VRAM_RESERVE_MAX_SEQ))?;
    // The derived arenas' (BF-1 F16 K/V, pooled BF16 mirror) extra storage
    // must fit through the admitted context, including on UMA VMM loads that
    // otherwise charge only the first mapped chunk. Do not subtract it with
    // the historical (arena-off) measured layout.
    let arena = if qsa_layout.f16_arena || qsa_layout.pooled_bf16 {
        let capacity = crate::Qwen4ContextCommit {
            tokens: context.max_seq,
            ..*context
        };
        let plain = capacity
            .committed_layer_bytes(config, qsa_format)
            .ok_or_else(|| overflow(context.max_seq))?;
        let with = capacity
            .committed_layer_bytes(config, qsa_layout)
            .ok_or_else(|| overflow(context.max_seq))?;
        trunk(Some(
            with.checked_sub(plain)
                .ok_or_else(|| overflow(context.max_seq))?,
        ))
        .ok_or_else(|| overflow(context.max_seq))?
    } else {
        0
    };
    let context = committed
        .saturating_sub(measured_context)
        .checked_add(arena)
        .ok_or_else(|| overflow(context.max_seq))?;
    let overflow = || format!("forward resources for a {chunk_rows}-row chunk overflow");
    let measured = crate::gpu_forward::Qwen4GpuForwardScratch::device_bytes(
        config,
        AUTO_VRAM_RESERVE_CHUNK,
        Qwen4ForwardScratchMode::Full,
    )
    .ok()
    .and_then(|scratch| {
        let spec = AUTO_VRAM_RESERVE_CHUNK.checked_mul(config.vocab_size.checked_mul(4)? + 4)?;
        scratch.checked_add(spec as u64)
    })
    .ok_or_else(overflow)?;
    let forward = match forward {
        Some(bytes) => bytes,
        None => crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows, mode)
            .ok_or_else(overflow)?,
    };
    AUTO_VRAM_RESERVE_BYTES
        .checked_add(context)
        .and_then(|bytes| bytes.checked_add(forward))
        .and_then(|bytes| bytes.checked_sub(measured))
        .and_then(|bytes| bytes.checked_add(mtp_bytes.unwrap_or(0)))
        .and_then(|bytes| bytes.checked_add(gather_bytes.unwrap_or(0)))
        .ok_or_else(|| "auto expert VRAM reserve overflows".to_string())
}

/// Number of itemised terms of the kernels' pool workspace
/// (`rdna_compute::scratch::qwen4_prefill_workspace_terms`).
pub const WORKSPACE_TERM_COUNT: usize = 11;

/// A bottom-up census of what one load's first full-context request takes
/// beyond the resident weights, term by term: the state's QSA context at the
/// admitted `max_seq` as the actual backend maps it, the rest of the request
/// state, the chunk-sized forward resources, the lazy F16 weight shadows, the
/// kernels' pool workspace (itemised by the engine), the QSA split
/// selector's scratch, native MTP and the QSA gather workspace. `auto`
/// keeps the larger of this and the historical measured formula
/// ([`auto_vram_reserve`]) free ([`AutoReserve`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReserveCensus {
    pub qsa_context: u64,
    pub state: u64,
    pub forward: u64,
    pub f16_shadows: u64,
    /// The shared-expert INT8 weight shadows ([`shared_i8_shadow_bytes`]).
    pub shared_i8_shadows: u64,
    pub workspace: [(&'static str, u64); WORKSPACE_TERM_COUNT],
    pub qsa_selector: u64,
    pub mtp: u64,
    pub gather: u64,
    /// The HIP runtime's own device memory ([`runtime_overhead_bytes`]).
    pub runtime: u64,
}

impl ReserveCensus {
    /// Every term, named, in log order.
    pub fn terms(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        [
            ("QSA context", self.qsa_context),
            ("state", self.state),
            ("forward", self.forward),
            ("F16 weight shadows", self.f16_shadows),
            ("INT8 shared-expert shadows", self.shared_i8_shadows),
            ("QSA selector", self.qsa_selector),
            ("native MTP", self.mtp),
            ("QSA gather", self.gather),
            ("HIP runtime", self.runtime),
        ]
        .into_iter()
        .chain(self.workspace.iter().copied())
    }

    /// Sum of every term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("reserve census overflows at '{name}'"))
        })
    }

    pub fn describe(&self) -> String {
        self.describe_with(None)
    }

    /// [`Self::describe`] with `forward_note` appended to the `forward` term
    /// (the decode-share dedicated charge).
    pub fn describe_with(&self, forward_note: Option<&str>) -> String {
        let mut text = String::new();
        for (name, bytes) in self.terms() {
            if !text.is_empty() {
                text.push_str(", ");
            }
            text.push_str(&format!("{name} {} MiB", bytes >> 20));
            if let (Some(note), "forward") = (forward_note, name) {
                text.push_str(&format!(" ({note})"));
            }
        }
        text
    }
}

/// The state-derived census terms ([`ReserveCensus::qsa_context`],
/// [`ReserveCensus::state`], [`ReserveCensus::forward`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCensus {
    pub qsa_context: u64,
    pub state: u64,
    pub forward: u64,
}

/// The census terms the state and forward determine, for a load whose
/// context storage is `context`: pass the ACTUAL backend with the whole
/// admitted context mapped (`Qwen4ContextCommit::new(backend, max_seq,
/// max_seq, granularity)`), never the legacy-forced expert-reserve layout.
///
/// - `qsa_context`: the trunk QSA context arenas as `context` commits them
///   (VMM: each arena's mapped pages rounded to the driver granularity).
/// - `state`: the rest of `Qwen4State::device_bytes` (GDN recurrent and
///   conv with the snapshot arena's mirror, the QSA fixed arenas, PLE conv,
///   hyper feedback), the QSA context counted once in `qsa_context`. For a
///   `Q8Ef` GDN format the state also carries the one live F16 residual
///   plane per GDN layer (`Qwen4State::device_bytes` adds it once, 54 MiB over
///   36 layers: not per ring row, snapshot mirror or session); this census
///   adds nothing on top, so the plane is never charged twice.
/// - `forward`: [`crate::gpu_forward::qwen4_forward_device_bytes`] at
///   `chunk_rows` in `mode`.
pub fn state_census(
    config: &Qwen4Config,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Result<StateCensus, String> {
    state_census_with_forward(config, format, context, chunk_rows, mode, None)
}

/// [`state_census`] with the forward term replaced by `forward` when given
/// (the decode-share dedicated charge).
pub fn state_census_with_forward(
    config: &Qwen4Config,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    forward: Option<u64>,
) -> Result<StateCensus, String> {
    let overflow = |what: &str| format!("reserve census {what} overflows");
    let qsa_context = context
        .committed_layer_bytes(config, format.layout())
        .and_then(|bytes| bytes.checked_mul(config.n_full_layers()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("QSA context"))?;
    let state = crate::state::Qwen4State::device_bytes(config, format, context)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("state"))?;
    let state = state
        .checked_sub(qsa_context)
        .ok_or_else(|| overflow("state without its QSA context"))?;
    let forward = match forward {
        Some(bytes) => bytes,
        None => crate::gpu_forward::qwen4_forward_device_bytes(config, chunk_rows, mode)
            .ok_or_else(|| overflow("forward resources"))?,
    };
    Ok(StateCensus {
        qsa_context,
        state,
        forward,
    })
}

/// The forward scratch one pipeline-parallel rank allocates, split by who
/// needs it, so every duplicate across ranks is explicit.
///
/// - `stage_shared`: fields every stage's trunk loop needs (hyper-connection
///   streams and mixers, rotation, projections, attention mixer output, every
///   MoE buffer, the BF16 scratch). Each stage carries its own copy: a declared
///   per-stage duplicate, never shared across devices.
/// - `band_local`: GDN fields only where the band has a GDN layer, QSA fields
///   only where it has a QSA layer.
/// - `input_only`: token ids, embedding rows and the PLE fields, input band
///   only.
/// - `output_only`: the language-model logits and the speculative
///   logit/argmax rows, output band only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardCensus {
    pub stage_shared: u64,
    pub band_local: u64,
    pub input_only: u64,
    pub output_only: u64,
}

impl ForwardCensus {
    /// All four terms, checked.
    pub fn total(&self) -> Result<u64, String> {
        [
            self.stage_shared,
            self.band_local,
            self.input_only,
            self.output_only,
        ]
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(|| "forward census overflows".to_string())
    }
}

/// [`crate::gpu_forward::qwen4_forward_device_bytes`] for one pipeline-parallel
/// band: the same plan (`Qwen4GpuForwardScratch::plan`, owning slots only,
/// borrowed views uncharged, plus the speculative logits/argmax), but a field
/// is charged only on a rank that runs its operation. `Single` equals
/// `qwen4_forward_device_bytes` exactly (pinned by a test over both modes).
///
/// Band-aware sizing: the projection scratch spans the QSA q/gate width only
/// with a QSA layer and the GDN qkv width only with a GDN layer; the BF16
/// scratch spans the GDN value width only with a GDN layer. In the compact
/// plan the GDN recurrent output is a view of the QSA q/gate owner, so a
/// compact band with GDN but no QSA layer is refused.
pub fn forward_census_scope(
    config: &Qwen4Config,
    rows: usize,
    mode: Qwen4ForwardScratchMode,
    scope: &Qwen4BandScope,
) -> Result<ForwardCensus, String> {
    use crate::config::LayerType;
    use rdna_compute::DType;
    let overflow = || "forward census size overflows".to_string();
    if rows == 0 {
        return Err("forward census rows is zero".to_string());
    }
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let dims = crate::gpu_forward::program_dims(config);
    let layout = crate::program::Qwen4ScratchLayout::for_rows(dims, rows)
        .map_err(|error| error.to_string())?;
    let has = |kind: LayerType| {
        scope
            .band
            .clone()
            .any(|layer| config.layer_types.get(layer) == Some(&kind))
    };
    let (has_gdn, has_qsa) = (
        has(LayerType::LinearAttention),
        has(LayerType::FullAttention),
    );
    let compact = mode == Qwen4ForwardScratchMode::FnSymGfx1201;
    let hidden = layout.hidden;
    let wide = layout.wide;
    let q_width = layout.q_width;
    let qsa_qgate = 2 * q_width;
    let gdn_value = layout.gdn_value;
    let max_projection = wide
        .max(hidden)
        .max(if has_qsa { qsa_qgate } else { 0 })
        .max(if has_gdn { layout.gdn_qkv } else { 0 });
    let max_rotation = wide.max(hidden).max(config.ffn.intermediate_size());
    let ple_embed_dim = config
        .ple
        .as_ref()
        .map_or(0, |ple| ple.head_count * ple.row_width);
    let ple_channels = wide;
    let ffn_batch_rows = layout.routed_rows.max(layout.dense_rows);
    let i32_bytes = std::mem::size_of::<i32>();
    let rows_of = |width: usize| mul(rows, width);
    let route_i32_bytes = mul(layout.slots, i32_bytes)?;
    let grouped_bound = layout.grouped_rows;
    let routed_width = mul(config.num_experts_per_tok, config.moe_intermediate_size)?;
    let top_k = config.num_experts_per_tok;
    let moe = config.moe_intermediate_size;

    let mut arena = 0usize;
    // MoE buffers whose compact size differs from the full plan.
    let (mut gate_batch, mut up_batch, mut rot_batch) =
        (ffn_batch_rows, ffn_batch_rows, rows_of(routed_width)?);
    let (mut down_expanded, mut y_gate_up, mut y_down) = (
        rows_of(mul(top_k, hidden)?)?,
        mul(grouped_bound, 2 * moe)?,
        mul(grouped_bound, hidden)?,
    );
    if compact {
        if ple_channels != wide {
            return Err(
                "compact scratch requires PLE channels equal to the hyper-connection \
                 width"
                    .to_string(),
            );
        }
        let f32_bytes = std::mem::size_of::<f32>();
        let bf16_bytes = DType::BF16.size();
        let fallback_rows = rows.min(crate::gpu_forward::QWEN4_COMPACT_FALLBACK_ROWS);
        let fallback_grouped =
            crate::program::grouped_m_total_bound(mul(fallback_rows, top_k)?, config.num_experts)
                .map_err(|error| error.to_string())?;
        let facade = |bytes: usize| bytes.div_ceil(f32_bytes);
        let fallback_batch = mul(fallback_rows, routed_width)?;
        (gate_batch, up_batch, rot_batch) = (fallback_batch, fallback_batch, fallback_batch);
        let hidden_f32_bytes = mul(hidden, f32_bytes)?;
        down_expanded = facade(
            mul(mul(fallback_rows, top_k)?, hidden_f32_bytes)?
                .max(mul(rows, hidden_f32_bytes)?)
                .max(mul(mul(rows, top_k)?, 8)?),
        );
        y_gate_up = facade(
            mul(fallback_grouped, mul(2 * moe, f32_bytes)?)?
                .max(mul(grouped_bound, mul(moe, bf16_bytes)?)?),
        );
        y_down = facade(
            mul(fallback_grouped, hidden_f32_bytes)?
                .max(mul(grouped_bound, mul(hidden, bf16_bytes)?)?),
        );
        // The arena is `Qwen4GpuForwardScratch::plan`'s own sizing, so a
        // geometry the plan refuses is refused here too, whatever the band.
        arena =
            crate::gpu_forward::Qwen4GpuForwardScratch::compact_arena_elements(config, rows, scope)
                .map_err(|error| error.to_string())?;
    }

    // Bytes of one owning slot (every slot holds at least one element).
    let slot = |elements: usize, dtype: DType| -> Result<u64, String> {
        u64::try_from(
            elements
                .max(1)
                .checked_mul(dtype.size())
                .ok_or_else(overflow)?,
        )
        .map_err(|_| overflow())
    };
    let sum = |slots: &[Result<u64, String>]| -> Result<u64, String> {
        slots.iter().try_fold(0u64, |total, bytes| {
            total.checked_add(bytes.clone()?).ok_or_else(overflow)
        })
    };
    // Compact plan: the step-local fields are uncharged views of the arena.
    let owned = |view: bool, bytes: Result<u64, String>| -> Result<u64, String> {
        if compact && view {
            Ok(0)
        } else {
            bytes
        }
    };
    let f32 = DType::F32;
    // The MoE buffers: views of the arena in the compact plan, every slot at
    // least one element otherwise.
    let moe_slot = |elements: usize| -> Result<u64, String> {
        if compact {
            Ok(0)
        } else {
            slot(elements, f32)
        }
    };
    let stage_shared = sum(&[
        slot(rows_of(wide)?, f32),                           // Streams
        owned(true, slot(rows_of(wide)?, f32)),              // HcNormalized (view)
        owned(true, slot(rows_of(config.hc_lowrank)?, f32)), // HcLow (view)
        owned(true, slot(rows_of(wide)?, f32)),              // HcUp (view)
        slot(rows_of(hidden)?, f32),                         // HcMixed
        slot(rows_of(config.hc_count)?, f32),                // HcGates
        slot(rows_of(max_rotation)?, f32),                   // Rotation
        // Projection: the compact phase arena owner.
        if compact {
            slot(arena, f32)
        } else {
            slot(rows_of(max_projection)?, f32)
        },
        owned(true, slot(rows_of(max_projection)?, f32)), // Projection2 (view)
        // GdnBf16: also the QSA and hyper-connection BF16 scratch.
        slot(
            if has_gdn { gdn_value } else { 0 }
                .max(hidden)
                .max(config.hc_lowrank),
            DType::BF16,
        ),
        slot(rows_of(hidden)?, f32), // AttentionOutput
        slot(rows_of(dims_router_columns(config))?, f32), // RouterLogits
        owned(true, slot(rows_of(hidden)?, f32)), // MoeXRot (view)
        owned(true, slot(rows_of(2 * moe)?, f32)), // MoeGateUp (view)
        owned(
            true,
            slot(rows_of(config.shared_expert_intermediate_size)?, f32),
        ), // MoeGate (view)
        owned(
            true,
            slot(rows_of(config.shared_expert_intermediate_size)?, f32),
        ), // MoeUp (view)
        owned(
            true,
            slot(rows_of(config.shared_expert_intermediate_size)?, f32),
        ), // MoeHidden (view)
        slot(rows_of(hidden)?, f32), // FfnOutput
        owned(true, slot(rows_of(hidden)?, f32)), // MoeSharedOutput (view)
        moe_slot(gate_batch),
        moe_slot(up_batch),
        moe_slot(rot_batch),
        slot(route_i32_bytes, DType::Raw), // MoeTopkIndices
        slot(rows_of(top_k)?, f32),        // MoeTopkWeights
        moe_slot(down_expanded),
        owned(
            true,
            slot(rows_of(config.shared_expert_intermediate_size)?, f32),
        ), // MoeScalar (view)
        slot(mul(config.num_experts, i32_bytes)?, DType::Raw), // MoeExpertTokenCounts
        slot(mul(config.num_experts + 1, i32_bytes)?, DType::Raw), // MoeExpertOffsets
        slot(mul(grouped_bound, i32_bytes)?, DType::Raw),      // MoeSortedSlotIndex
        slot(mul(grouped_bound / 16, i32_bytes)?, DType::Raw), // MoeExpertTileIds
        slot(route_i32_bytes, DType::Raw),                     // MoeInversePerm
        moe_slot(y_gate_up),
        moe_slot(y_down),
    ])?;
    let gdn_local = if has_gdn {
        let value_rows = rows_of(gdn_value)?;
        sum(&[
            owned(true, slot(rows_of(config.linear_num_value_heads)?, f32)), // GdnA (view)
            owned(true, slot(rows_of(config.linear_num_value_heads)?, f32)), // GdnB (view)
            owned(true, slot(rows_of(config.linear_num_value_heads)?, f32)), // GdnGate (view)
            owned(true, slot(rows_of(config.linear_num_value_heads)?, f32)), // GdnBeta (view)
            owned(true, slot(value_rows, f32)), // GdnRecurrentOutput (view)
            owned(true, slot(value_rows, f32)), // GdnZ (view)
            owned(true, slot(value_rows, f32)), // GdnOutput (view)
        ])?
    } else {
        0
    };
    let qsa_local = if has_qsa {
        let kv_width = mul(config.num_key_value_heads, config.head_dim)?;
        let selected = mul(mul(rows, config.qsa_selected_capacity())?, i32_bytes)?;
        sum(&[
            owned(true, slot(rows_of(layout.index_width)?, f32)), // QsaIndex (view)
            owned(true, slot(rows_of(qsa_qgate)?, f32)),          // QsaQgate (view)
            owned(true, slot(rows_of(kv_width)?, f32)),           // QsaK (view)
            owned(true, slot(rows_of(kv_width)?, f32)),           // QsaV (view)
            owned(true, slot(rows_of(q_width)?, f32)),            // QsaOutput (view)
            owned(true, slot(selected, DType::Raw)),              // QsaSelected (view)
        ])?
    } else {
        0
    };
    let band_local = gdn_local.checked_add(qsa_local).ok_or_else(overflow)?;
    let input_only = if scope.input {
        sum(&[
            slot(rows_of(i32_bytes)?, DType::Raw),          // TokenIds
            owned(true, slot(rows_of(hidden)?, f32)),       // EmbeddingRot (view)
            owned(true, slot(rows_of(hidden)?, f32)),       // Embeddings (view)
            slot(rows_of(ple_embed_dim)?, DType::BF16),     // PleStaged
            slot(rows_of(ple_embed_dim.max(hidden))?, f32), // PleRows
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleKey (view)
            owned(true, slot(rows_of(hidden)?, f32)),       // PleValue (view)
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleQuery (view)
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleGated (view)
            // PleNormed (dedicated): the rows a verify rollback reads after
            // the forward in the compact plan, every row otherwise.
            slot(
                if compact {
                    mul(
                        rows.min(crate::gpu_forward::QWEN4_PLE_NORMED_KEEP_ROWS),
                        ple_channels,
                    )?
                } else {
                    rows_of(ple_channels)?
                },
                f32,
            ),
            // PleNormedWide: an uncharged arena view in the compact plan,
            // absent from `Full`.
            Ok(0),
            owned(true, slot(rows_of(ple_channels)?, f32)), // PleOutput (view)
        ])?
    } else {
        0
    };
    let output_only = if scope.output {
        let logit_row_bytes = config
            .vocab_size
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or_else(overflow)?;
        let spec = crate::gpu_forward::qwen4_spec_logit_rows(rows)
            .checked_mul(logit_row_bytes)
            .ok_or_else(overflow)?;
        slot(config.vocab_size, f32)?
            .checked_add(u64::try_from(spec).map_err(|_| overflow())?)
            .ok_or_else(overflow)?
    } else {
        0
    };
    Ok(ForwardCensus {
        stage_shared,
        band_local,
        input_only,
        output_only,
    })
}

/// `router_columns` of the config's program dims (one placeholder column for
/// a dense FFN).
fn dims_router_columns(config: &Qwen4Config) -> usize {
    crate::gpu_forward::router_columns(crate::gpu_forward::program_dims(config))
}

/// [`state_census`] for one pipeline-parallel band: the GDN and QSA state of
/// the band's own layers, the PLE conv history and `hyper_feedback` only on
/// the input band (`Qwen4State::device_bytes_scoped`), the QSA context of
/// the band's own QSA layers and the band's own forward scratch
/// ([`forward_census_scope`]). `Single` equals [`state_census`].
pub fn state_census_scope(
    config: &Qwen4Config,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    scope: &Qwen4BandScope,
) -> Result<StateCensus, String> {
    scope.validate(config)?;
    let overflow = |what: &str| format!("reserve census {what} overflows");
    let qsa_layers = scope
        .band
        .clone()
        .filter(|&layer| {
            config.layer_types.get(layer) == Some(&crate::config::LayerType::FullAttention)
        })
        .count();
    let qsa_context = context
        .committed_layer_bytes(config, format.layout())
        .and_then(|bytes| bytes.checked_mul(qsa_layers))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("QSA context"))?;
    let state = crate::state::Qwen4State::device_bytes_scoped(config, format, context, scope)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("state"))?;
    let state = state
        .checked_sub(qsa_context)
        .ok_or_else(|| overflow("state without its QSA context"))?;
    let forward = forward_census_scope(config, chunk_rows, mode, scope)?.total()?;
    Ok(StateCensus {
        qsa_context,
        state,
        forward,
    })
}

/// Refuse a pipeline band combined with expert parallelism: PP never reuses
/// the modulo-rank expert split, so any `ranks > 1` with a band that does not
/// cover the whole model is an error. `Single` with any rank count passes.
pub fn refuse_band_with_expert_parallel(
    config: &Qwen4Config,
    scope: &Qwen4BandScope,
    ranks: usize,
) -> Result<(), String> {
    let whole = scope.input && scope.output && scope.band == (0..config.num_hidden_layers);
    if ranks > 1 && !whole {
        return Err(format!(
            "pipeline-parallel band {}..{} cannot combine with {ranks}-way expert parallelism",
            scope.band.start, scope.band.end
        ));
    }
    Ok(())
}

/// One pipeline-parallel rank's byte census (Single: the whole model). The
/// weight terms come from the manifest entries the band owns and the same
/// payload sizes as [`resident_split`]; EP is not involved, no routed expert
/// is split by rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BandCensus {
    pub band: std::ops::Range<usize>,
    pub input: bool,
    pub output: bool,
    /// MoE layers owned: the band's trunk layers plus the MTP layer on output.
    pub moe_layers: usize,
    /// Owned resident non-routed payloads (tied aliases and external rows have
    /// none).
    pub non_expert_bytes: u64,
    /// Routed experts of the owned trunk layers, every expert, exact codec bytes.
    pub routed_trunk_bytes: u64,
    /// Routed experts of the MTP head's layer (output band only).
    pub routed_mtp_bytes: u64,
    /// Routed bytes of the largest single owned trunk layer.
    pub max_layer_routed_bytes: u64,
    /// The separately owned materialization of a tied source the source's
    /// owner also holds (the MTP embedding of `embed_tokens` on an output
    /// band that is not the input band). Zero for `Single`.
    pub endpoint_duplicate_bytes: u64,
    pub f16_shadow_bytes: u64,
    /// `state.forward` is `forward.total()`.
    pub state: StateCensus,
    /// The band's forward scratch by role: the per-stage shared copy is an
    /// explicit duplicate across ranks, the rest is owned once.
    pub forward: ForwardCensus,
    /// Native MTP reserves of this band (all zero when the load is AR): the
    /// output band's head and request buffers, every band's row capture.
    pub mtp: NativeMtpCensus,
}

impl BandCensus {
    /// Trunk plus MTP routed expert bytes, checked.
    pub fn routed_bytes(&self) -> Result<u64, String> {
        self.routed_trunk_bytes
            .checked_add(self.routed_mtp_bytes)
            .ok_or_else(|| "band routed-expert byte census overflows".to_string())
    }

    /// Every resident weight byte of the band: non-expert, routed and the
    /// endpoint duplicate, checked. Shadows and state are separate terms.
    pub fn weight_bytes(&self) -> Result<u64, String> {
        self.routed_bytes()?
            .checked_add(self.non_expert_bytes)
            .and_then(|bytes| bytes.checked_add(self.endpoint_duplicate_bytes))
            .ok_or_else(|| "band weight byte census overflows".to_string())
    }
}

/// Byte terms of [`band_weight_census`].
struct BandWeightBytes {
    non_expert: u64,
    trunk: u64,
    mtp: u64,
    max_layer: u64,
    endpoint_duplicate: u64,
}

/// Owned routed-expert bytes of trunk layers and of the MTP head (exact sums
/// over the entries' payloads), the owned non-expert bytes, the largest owned
/// trunk layer's routed bytes and the tied endpoint duplicates, for `scope`.
/// Sums are checked; `Single` equals [`routed_expert_bytes`] and
/// [`resident_split`]'s non-expert bytes.
///
/// `weights` may be the full manifest or one already filtered to the band
/// (`Qwen4Manifest::with_band_scope`), which drops a tied source another band
/// owns. Then the source's payload is requested through `bytes_of` from a
/// clone of the tied entry renamed to the source with a replicated policy
/// (same logical shape and dtype): no source is uploaded, the bytes are the
/// output band's own materialization, counted as an endpoint duplicate and
/// never as an alias.
///
/// A non-`Single` scope refuses expert-sharded entries before any `bytes_of`
/// call: pipeline bands never reuse the expert-parallel split.
fn band_weight_census(
    weights: &[WeightEntry],
    bytes_of: &impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
) -> Result<BandWeightBytes, String> {
    if !(scope.input && scope.output)
        && weights.iter().any(|entry| {
            matches!(
                entry.policy,
                ShardPolicy::ExpertSharded { .. } | ShardPolicy::ExpertTensorSharded { .. }
            )
        })
    {
        return Err(
            "pipeline-parallel bands cannot be combined with expert parallelism".to_string(),
        );
    }
    let overflow = |what: &str| format!("band census {what} overflows");
    let payload = |entry: &WeightEntry| {
        bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))
    };
    let (mut non_expert, mut trunk, mut mtp, mut duplicate) = (0u64, 0u64, 0u64, 0u64);
    let mut per_layer = std::collections::BTreeMap::<usize, u64>::new();
    for entry in weights {
        if entry.residency.is_external() {
            continue;
        }
        if let ShardPolicy::Tied { source } = &entry.policy {
            if scope.owns_entry(entry) {
                let origin = weights.iter().find(|candidate| &candidate.name == source);
                let materialized = match origin {
                    Some(origin) if scope.owns_entry(origin) => None,
                    Some(origin) => Some(payload(origin)?),
                    None => {
                        let mut request = entry.clone();
                        request.name = source.clone();
                        request.policy = ShardPolicy::Replicate;
                        Some(payload(&request)?)
                    }
                };
                if let Some(bytes) = materialized {
                    duplicate = duplicate
                        .checked_add(bytes)
                        .ok_or_else(|| overflow("endpoint duplicates"))?;
                }
            }
            continue;
        }
        if !scope.owns_entry(entry) {
            continue;
        }
        let bytes = payload(entry)?;
        if !is_routed_expert(&entry.name) {
            non_expert = non_expert
                .checked_add(bytes)
                .ok_or_else(|| overflow("non-expert bytes"))?;
        } else if entry.name.starts_with("mtp.") {
            mtp = mtp
                .checked_add(bytes)
                .ok_or_else(|| overflow("MTP routed bytes"))?;
        } else {
            let layer = entry
                .layer
                .ok_or_else(|| format!("routed expert '{}' has no trunk layer", entry.name))?;
            trunk = trunk
                .checked_add(bytes)
                .ok_or_else(|| overflow("trunk routed bytes"))?;
            let slot = per_layer.entry(layer).or_default();
            *slot = slot
                .checked_add(bytes)
                .ok_or_else(|| overflow("layer routed bytes"))?;
        }
    }
    let max_layer = per_layer.values().copied().max().unwrap_or(0);
    Ok(BandWeightBytes {
        non_expert,
        trunk,
        mtp,
        max_layer,
        endpoint_duplicate: duplicate,
    })
}

/// The band-aware byte census of one pipeline-parallel rank, from the
/// manifest's own entries and the actual state backend: routed bytes (owned
/// trunk layers; MTP only on output), non-expert bytes, F16 shadows,
/// [`moe_layer_count_scope`] and [`state_census_scope`]. Expert-parallel
/// counts are untouched ([`routed_expert_bytes_ranks`] and friends); a band
/// is validated first and no routed stack is split by rank.
#[allow(clippy::too_many_arguments)]
pub fn band_census(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
) -> Result<BandCensus, String> {
    band_census_native_mtp(
        config, weights, bytes_of, scope, format, context, chunk_rows, mode, None,
    )
}

/// [`band_census`] for a load whose native MTP is `mtp` (`None` = AR, every
/// MTP reserve zero; the output band still counts the head's weights, which
/// the band loads whether or not the head attaches). The weight terms are the
/// same entries either way; `mtp` adds [`BandCensus::mtp`] by
/// [`native_mtp_census_scope`].
#[allow(clippy::too_many_arguments)]
pub fn band_census_native_mtp(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    scope: &Qwen4BandScope,
    format: crate::Qwen4StateFormat,
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mode: Qwen4ForwardScratchMode,
    mtp: Option<&NativeMtpCensusInputs>,
) -> Result<BandCensus, String> {
    scope.validate(config)?;
    let weights_census = band_weight_census(weights, &bytes_of, scope)?;
    let mtp = match mtp {
        Some(inputs) => {
            native_mtp_census_scope(config, weights, context, chunk_rows, inputs, scope)?
        }
        None => NativeMtpCensus::default(),
    };
    Ok(BandCensus {
        band: scope.band.clone(),
        input: scope.input,
        output: scope.output,
        moe_layers: moe_layer_count_scope(config, scope),
        non_expert_bytes: weights_census.non_expert,
        routed_trunk_bytes: weights_census.trunk,
        routed_mtp_bytes: weights_census.mtp,
        max_layer_routed_bytes: weights_census.max_layer,
        endpoint_duplicate_bytes: weights_census.endpoint_duplicate,
        f16_shadow_bytes: f16_shadow_bytes_scope(weights, config.num_hidden_layers, scope)?,
        state: state_census_scope(config, format, context, chunk_rows, mode, scope)?,
        forward: forward_census_scope(config, chunk_rows, mode, scope)?,
        mtp,
    })
}

/// The native MTP request of a load, as the census sizes it: the resolved
/// values the Single load passes to `mtp_spec::native_mtp_device_bytes`
/// (the head dtype comes from the manifest's language head).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMtpCensusInputs {
    /// Draft window K the request sizes (`native_mtp_window_rows` rows). The
    /// Single load passes `native_mtp_proposal_drafts` (n-gram takeover
    /// width); a PP load has no n-gram pool or copy lane, so passes its K.
    pub max_k: usize,
    /// The GDN storage format and ring rows of the verify-row capture where
    /// the route captures rows (`mtp_spec::native_mtp_capture`), else `None`.
    /// A `Q8Ef` format charges its plain Q8 ring (`row_capture_bytes*` size by
    /// `storage()`): the live residual plane is charged once by the state
    /// census, never per ring row, so the reserve adds no residual bytes.
    pub row_capture: Option<(crate::GdnStateFormat, usize)>,
    /// Whether the request allocates the batched prompt-fill scratch
    /// (`mtp_spec::native_mtp_fill_scratch`).
    pub batched_fill: bool,
}

/// One band's native MTP reserve terms. [`Self::reserve`] of the `Single`
/// scope is exactly `mtp_spec::native_mtp_device_bytes`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeMtpCensus {
    /// What the attached head keeps (`Qwen4MtpGpu`): output band only.
    pub head_resident: u64,
    /// The draft head's build scratch, released at attach: output band only.
    pub head_build_scratch: u64,
    /// What the first speculative request keeps beyond the capture: the
    /// verify (wide-capture) hidden rows, the pending and row hidden carries
    /// and the prompt-fill scratch (`native_mtp_request_bytes` without row
    /// capture): output band only.
    pub request: u64,
    /// This band's own GDN verify-row capture (ring, inputs, output block and
    /// pointer table for its GDN layers): every band, never the whole model's
    /// charged once per rank.
    pub row_capture: u64,
}

impl NativeMtpCensus {
    /// Bytes the band reserves for native MTP: the resident head plus the
    /// larger of its build scratch (released at attach) and what the first
    /// request allocates after it; a band without the head reserves its row
    /// capture alone. Checked.
    pub fn reserve(&self) -> Result<u64, String> {
        let overflow = || "native MTP band reserve overflows".to_string();
        let request = self
            .request
            .checked_add(self.row_capture)
            .ok_or_else(overflow)?;
        self.head_resident
            .checked_add(self.head_build_scratch.max(request))
            .ok_or_else(overflow)
    }
}

/// The native MTP reserve of one pipeline band (`Single` = the whole model).
///
/// Every band charges the row capture of its OWN GDN layers
/// (`Qwen4State::row_capture_bytes_scoped`, `native_mtp_window_rows` rows): each
/// band arms capture before its target snapshot. The output band, which owns
/// the head and the request buffers, additionally charges
/// `Qwen4MtpGpu::device_bytes` and `native_mtp_request_bytes` WITHOUT capture
/// (the capture is band-scoped here, not the whole model's), so the `Single`
/// scope equals `native_mtp_device_bytes` and no rank pays the full-model
/// capture. `context` is the census context, as for the head's QSA arenas.
pub fn native_mtp_census_scope(
    config: &Qwen4Config,
    weights: &[WeightEntry],
    context: &crate::Qwen4ContextCommit,
    chunk_rows: usize,
    mtp: &NativeMtpCensusInputs,
    scope: &Qwen4BandScope,
) -> Result<NativeMtpCensus, String> {
    scope.validate(config)?;
    if config.mtp.is_none() {
        return Err(format!(
            "native MTP census for profile {}, which has no MTP head",
            config.profile.id()
        ));
    }
    let overflow = |what: &str| format!("native MTP census {what} overflows");
    let gdn_layers = scope
        .band
        .clone()
        .filter(|&layer| {
            config.layer_types.get(layer) == Some(&crate::config::LayerType::LinearAttention)
        })
        .count();
    let row_capture = match mtp.row_capture {
        Some((gdn, ring_rows)) => crate::state::Qwen4State::row_capture_bytes_scoped(
            config,
            gdn,
            ring_rows.max(crate::mtp_spec::native_mtp_window_rows(mtp.max_k)),
            gdn_layers,
        )
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| overflow("row capture"))?,
        None => 0,
    };
    if !scope.output {
        return Ok(NativeMtpCensus {
            row_capture,
            ..NativeMtpCensus::default()
        });
    }
    let head_dtype = language_head_dtype(weights)
        .ok_or_else(|| "native MTP census: manifest has no language head".to_string())?;
    let (resident, scratch) =
        crate::mtp_gpu::Qwen4MtpGpu::device_bytes(config, context, head_dtype, false)
            .ok_or_else(|| overflow("head"))?;
    let request = crate::mtp_spec::native_mtp_request_bytes(
        config,
        context.max_seq,
        chunk_rows,
        mtp.max_k,
        None,
        mtp.batched_fill,
    )
    .ok_or_else(|| overflow("request"))?;
    Ok(NativeMtpCensus {
        head_resident: u64::try_from(resident).map_err(|_| overflow("head"))?,
        head_build_scratch: u64::try_from(scratch).map_err(|_| overflow("head scratch"))?,
        request,
        row_capture,
    })
}

/// Name suffixes, below a trunk layer's `model.language_model.layers.{i}.`
/// prefix, of the BF16 matrices the long-prefill F16 WMMA routes read through
/// a model-lifetime F16 shadow (`Gpu::ensure_bf16_f16_shadow`, keyed by the
/// weight pointer): hyper-connection down mixes, the router, the shared
/// expert's gate/up/down and the PLE key/value projections. The direct BF16
/// readers (`input_mix_weight_up`, `shared_expert_gate`) and every other
/// tensor are not shadowed.
const F16_SHADOW_LAYER_SUFFIXES: [&str; 8] = [
    "attn_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp.gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "ple.key_proj.weight",
    "ple.value_proj.weight",
];

/// The final hyper-connection's down mix, shadowed like the per-layer ones.
const F16_SHADOW_ROOT: &str =
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";

/// Device bytes of the lazy F16 weight shadows the long-prefill routes can
/// create for the trunk: for every resident, non-tied BF16 rank-2 entry of
/// [`F16_SHADOW_ROOT`] or a [`F16_SHADOW_LAYER_SUFFIXES`] tensor of a trunk
/// layer `0..num_layers`, the whole F16 matrix (`2 * rows * cols`, the full
/// payload). A conservative cold bound: the shadows exist only when the F16
/// WMMA arms run (shadow-capable device, `route_rows >= 512`, K a multiple
/// of 64), but they are charged whenever the manifest owns the weights,
/// never from the current warm cache. Callers gate it with
/// [`f16_shadow_charge`]. Entries are the manifest's own, so a profile or dtype
/// without them (an MQ-quantized trunk) charges nothing.
pub fn f16_shadow_bytes(weights: &[WeightEntry], num_layers: usize) -> Result<u64, String> {
    f16_shadow_bytes_owned(weights, num_layers, |_| true, true)
}

/// [`f16_shadow_bytes`] for one pipeline-parallel band: only the shadows of
/// the band's own trunk layers (the PLE projections ride with layer 1 on the
/// input band), and the final hyper-connection's down mix only on the output
/// band. `Single` equals [`f16_shadow_bytes`]. The MTP head has no shadows.
pub fn f16_shadow_bytes_scope(
    weights: &[WeightEntry],
    num_layers: usize,
    scope: &Qwen4BandScope,
) -> Result<u64, String> {
    f16_shadow_bytes_owned(
        weights,
        num_layers,
        |layer| scope.owns_layer(layer),
        scope.output,
    )
}

/// Largest `route_rows` ([`rdna_compute::Gpu::route_rows`]) any prefill
/// forward of at most `chunk_rows` rows reaches on `arch`: the chunk itself,
/// or the canonical prefill rows when the forward runs under the canonical
/// routing scope, decided by the SAME pure helper the forward reads
/// ([`crate::gpu_forward::qwen4_route_scope_flags`]). The F16 WMMA route and
/// its lazy shadows are gated on THIS count, so the census must read it, not
/// the raw chunk.
pub fn f16_shadow_route_rows(arch: &str, chunk_rows: usize) -> usize {
    use rdna_compute::prefill_invariant as invariant;
    f16_shadow_route_rows_with(
        arch,
        chunk_rows,
        invariant::qwen4_append_wmma_enabled(),
        invariant::qwen4_append_wmma_min_rows(),
        invariant::prefill_pm_invariant_enabled(),
    )
}

/// [`f16_shadow_route_rows`] over explicit knob values (the process snapshot
/// of `HIPFIRE_QWEN4_APPEND_WMMA`, its `_MIN_ROWS` and
/// `HIPFIRE_PREFILL_PM_INVARIANT`), so the CPU proof does not depend on the
/// environment.
pub fn f16_shadow_route_rows_with(
    arch: &str,
    chunk_rows: usize,
    append_enabled: bool,
    append_min_rows: usize,
    pm_invariant_enabled: bool,
) -> usize {
    // The batch-invariant flag does not depend on the row count, and the append
    // route is open for rows in `append_min_rows..512`: some prefill forward of
    // at most `chunk_rows` rows runs scoped iff the largest sub-512-row size does.
    let probe_rows = chunk_rows.min(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS - 1);
    let (pm_invariant, append) = crate::gpu_forward::qwen4_route_scope_flags(
        true,
        arch,
        probe_rows,
        append_enabled,
        append_min_rows,
        pm_invariant_enabled,
    );
    if pm_invariant || append {
        chunk_rows.max(rdna_compute::prefill_invariant::QWEN4_PREFILL_CANON_ROWS)
    } else {
        chunk_rows
    }
}

/// The F16 weight shadow bytes a load of `weights` charges: the cold bound
/// [`f16_shadow_bytes`] when the device is shadow-capable
/// (`rdna_compute::gemm::qwen4_f16_shadow_capable`, shared with the runtime
/// predicate) and `route_rows` reaches the route's threshold
/// ([`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`]); 0 otherwise.
pub fn f16_shadow_charge(
    capable: bool,
    route_rows: usize,
    weights: &[WeightEntry],
    num_layers: usize,
) -> Result<u64, String> {
    if capable && route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
        f16_shadow_bytes(weights, num_layers)
    } else {
        Ok(0)
    }
}

/// One trunk layer whose shared-expert triple the INT8 route shadows: the
/// route's shape and the byte sizes of its two planes
/// ([`rdna_compute::shared_i8::shared_i8_shadow_plane_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedI8ShadowLayer {
    pub layer: usize,
    pub intermediate: usize,
    pub hidden: usize,
    pub down_rows: usize,
    /// Bytes of the interleaved gate/up plane.
    pub gate_up_bytes: u64,
    /// Bytes of the down plane.
    pub down_bytes: u64,
}

/// The layers [`shared_i8_shadow_bytes`] charges, in layer order: every trunk
/// layer `0..num_layers` whose BF16, rank-2, resident, non-tied
/// `mlp.shared_expert.{gate,up,down}_proj.weight` triple has the route's shape
/// (`gate`/`up` `[I, K]`, `down` `[M, I]`).
pub fn shared_i8_shadow_layers(
    weights: &[WeightEntry],
    num_layers: usize,
) -> Result<Vec<SharedI8ShadowLayer>, String> {
    use rdna_compute::DType;
    let find = |layer: usize, proj: &str| {
        let name = format!("model.language_model.layers.{layer}.mlp.shared_expert.{proj}.weight");
        weights.iter().find(|e| {
            e.name == name
                && e.dtype == DType::BF16
                && e.logical_shape.len() == 2
                && !e.residency.is_external()
                && !matches!(e.policy, ShardPolicy::Tied { .. })
        })
    };
    let mut layers = Vec::new();
    for layer in 0..num_layers {
        let (Some(gate), Some(up), Some(down)) = (
            find(layer, "gate_proj"),
            find(layer, "up_proj"),
            find(layer, "down_proj"),
        ) else {
            continue;
        };
        let (i, k) = (gate.logical_shape[0], gate.logical_shape[1]);
        if up.logical_shape != gate.logical_shape || down.logical_shape[1] != i {
            continue;
        }
        let Some((gate_up_bytes, down_bytes)) =
            rdna_compute::shared_i8::shared_i8_shadow_plane_bytes(i, k, down.logical_shape[0])
        else {
            continue;
        };
        layers.push(SharedI8ShadowLayer {
            layer,
            intermediate: i,
            hidden: k,
            down_rows: down.logical_shape[0],
            gate_up_bytes,
            down_bytes,
        });
    }
    Ok(layers)
}

/// Device bytes of the shared-expert INT8 shadows (`rdna_compute::shared_i8`) the long-prefill
/// route builds lazily: for every trunk layer `0..num_layers` whose BF16, rank-2, resident,
/// non-tied `mlp.shared_expert.{gate,up,down}_proj.weight` triple has the route's shape
/// (`gate`/`up` `[I, K]`, `down` `[M, I]`), the planes of
/// [`rdna_compute::shared_i8::shared_i8_shadow_layer_bytes`]. A conservative cold bound, like
/// [`f16_shadow_bytes`]. Callers gate it with [`shared_i8_charge`].
pub fn shared_i8_shadow_bytes(weights: &[WeightEntry], num_layers: usize) -> Result<u64, String> {
    let mut total = 0u64;
    for layer in shared_i8_shadow_layers(weights, num_layers)? {
        total = total
            .checked_add(layer.gate_up_bytes)
            .and_then(|bytes| bytes.checked_add(layer.down_bytes))
            .ok_or_else(|| {
                format!(
                    "shared-expert INT8 shadow bytes of layer {} overflow",
                    layer.layer
                )
            })?;
    }
    Ok(total)
}

/// The INT8 shadow bytes a load of `weights` charges: [`shared_i8_shadow_bytes`] when the
/// route can run (`enabled`: [`rdna_compute::Gpu::qwen4_shared_i8_enabled`], i.e. exact gfx1201
/// and `HIPFIRE_QWEN4_SHARED_INT8` not `0`) and `route_rows` reaches the route's threshold
/// ([`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`]); 0 otherwise.
pub fn shared_i8_charge(
    enabled: bool,
    route_rows: usize,
    weights: &[WeightEntry],
    num_layers: usize,
) -> Result<u64, String> {
    if enabled && route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
        shared_i8_shadow_bytes(weights, num_layers)
    } else {
        Ok(0)
    }
}

/// Whether `entry` is a lazily F16-shadowed BF16 owner of the trunk
/// ([`f16_shadow_bytes`]'s one predicate, shared with the physical-granule
/// census so the two can never name different owners).
pub(crate) fn is_f16_shadow_owner(
    entry: &WeightEntry,
    num_layers: usize,
    owns_layer: &impl Fn(usize) -> bool,
    owns_root: bool,
) -> bool {
    use rdna_compute::DType;
    const LAYER_PREFIX: &str = "model.language_model.layers.";
    let shadowed = |name: &str| {
        if name == F16_SHADOW_ROOT {
            return owns_root;
        }
        name.strip_prefix(LAYER_PREFIX)
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(layer, suffix)| {
                layer
                    .parse::<usize>()
                    .is_ok_and(|layer| layer < num_layers && owns_layer(layer))
                    && F16_SHADOW_LAYER_SUFFIXES.contains(&suffix)
            })
    };
    entry.dtype == DType::BF16
        && entry.logical_shape.len() == 2
        && !entry.residency.is_external()
        && !matches!(entry.policy, ShardPolicy::Tied { .. })
        && shadowed(&entry.name)
}

/// The F16 shadow allocation of one owner: `2 * rows * cols` bytes.
pub(crate) fn f16_shadow_entry_bytes(entry: &WeightEntry) -> Result<u64, String> {
    entry.logical_shape[0]
        .checked_mul(entry.logical_shape[1])
        .and_then(|elements| elements.checked_mul(2))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| format!("F16 shadow bytes of '{}' overflow", entry.name))
}

fn f16_shadow_bytes_owned(
    weights: &[WeightEntry],
    num_layers: usize,
    owns_layer: impl Fn(usize) -> bool,
    owns_root: bool,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| is_f16_shadow_owner(entry, num_layers, &owns_layer, owns_root))
        .try_fold(0u64, |total, entry| {
            f16_shadow_entry_bytes(entry)?
                .checked_add(total)
                .ok_or_else(|| format!("F16 shadow bytes of '{}' overflow", entry.name))
        })
}

/// Physical VRAM allocation granule model. The granule is NEVER a constant
/// here: a load takes it from the device's own query
/// (`Gpu::vmm_recommended_granularity`, `hipMemGetAllocationGranularity` with
/// the recommended flag for a pinned device allocation) and `0` models no
/// rounding. This is a MEASURED BACKING MODEL, not a proven API contract: that
/// query is documented for VMM physical mappings, and it is applied to
/// ordinary `hipMalloc` allocations because the j1786 Hold 1a run's external
/// 20 ms sampler saw 227 of 267 device-memory steps of at least 2 MiB land on
/// an exact multiple of 2 MiB (20 more on a multiple plus one 4 KiB page),
/// e.g. the 496,640,000 B language head became 497,025,024 B (237 granules),
/// a 445,644,800 B resident down arena 446,693,376 B (213) and the session
/// cache's 34,130,944 B and 34,183,168 B snapshots each 35,651,584 B (17).
/// The query's value on that device is not in the run's artifacts; its
/// equality with 2 MiB is an inference the placement log line prints for
/// confirmation. Allocations below one granule were not observed to round
/// (tiny allocations sub-allocate) and are charged at their payload. The
/// census otherwise sums payload bytes; [`granule_slack`] is the per-owner
/// difference, kept apart from the opaque runtime budget it must not double
/// count.
///
/// Bytes one device allocation of `bytes` occupies beyond its payload at
/// `granule`: the distance to the next granule boundary for an allocation of
/// at least one granule, else 0. `granule == 0` models no rounding.
pub fn granule_slack(bytes: u64, granule: u64) -> u64 {
    if granule == 0 || bytes < granule {
        0
    } else {
        (granule - bytes % granule) % granule
    }
}

/// [`granule_slack`] summed over one owner's allocations, checked.
pub fn granule_slack_total(
    sizes: impl IntoIterator<Item = u64>,
    granule: u64,
) -> Result<u64, String> {
    sizes.into_iter().try_fold(0u64, |total, bytes| {
        total
            .checked_add(granule_slack(bytes, granule))
            .ok_or_else(|| "device allocation granule slack overflows".to_string())
    })
}

/// The granule rounding of the incumbent base reserve, owner by owner. Every
/// term is the sum, over that owner's real device allocations, of
/// [`granule_slack`]; none is a measured residual. Not enumerated (and so
/// left in the runtime budget): kernel module code and HIP runtime state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GranuleSlack {
    /// Every resident non-routed-expert weight tensor, one allocation each.
    pub weights: u64,
    /// The lazy F16 weight shadows ([`f16_shadow_bytes`]), one allocation per
    /// owner; 0 when the reserve does not charge shadows.
    pub f16_shadows: u64,
    /// The forward scratch's owning slots and speculative buffers.
    pub forward: u64,
    /// The request state's tensors and snapshot-arena mirror (VMM context
    /// arenas are listed at their mapped, already granule-rounded size).
    pub state: u64,
    /// The kernels' pool workspace terms, one allocation each.
    pub workspace: u64,
    /// The native MTP head and its first-request allocations (0 without a
    /// kept head).
    pub native_mtp: u64,
    /// The QSA gathered-prefill workspace (one legacy allocation; a VMM
    /// workspace maps granule-rounded pages, so nothing is added) and the QSA
    /// split selector's one scratch allocation.
    pub qsa_workspace: u64,
}

impl GranuleSlack {
    /// Every term, named, in log order.
    pub fn terms(&self) -> [(&'static str, u64); 7] {
        [
            ("weight tensors", self.weights),
            ("F16 weight shadows", self.f16_shadows),
            ("forward scratch", self.forward),
            ("request state", self.state),
            ("kernel workspace", self.workspace),
            ("native MTP", self.native_mtp),
            ("QSA workspace", self.qsa_workspace),
        ]
    }

    /// Sum of every term, checked.
    pub fn total(&self) -> Result<u64, String> {
        self.terms().iter().try_fold(0u64, |total, (name, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| format!("granule slack overflows at '{name}'"))
        })
    }
}

/// Granule slack of the resident non-routed-expert weights of `weights`: the
/// same entries [`resident_split`] counts as non-expert, one device
/// allocation per tensor.
pub fn weight_granule_slack(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    granule: u64,
) -> Result<u64, String> {
    let mut sizes = Vec::new();
    for entry in weights {
        if entry.residency.is_external()
            || matches!(entry.policy, ShardPolicy::Tied { .. })
            || is_routed_expert(&entry.name)
        {
            continue;
        }
        sizes.push(bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?);
    }
    granule_slack_total(sizes, granule)
}

/// Granule slack of the lazy F16 weight shadows ([`f16_shadow_bytes`]'s
/// owners), one allocation per shadow.
pub fn f16_shadow_granule_slack(
    weights: &[WeightEntry],
    num_layers: usize,
    granule: u64,
) -> Result<u64, String> {
    let mut sizes = Vec::new();
    for entry in weights {
        if is_f16_shadow_owner(entry, num_layers, &|_| true, true) {
            sizes.push(f16_shadow_entry_bytes(entry)?);
        }
    }
    granule_slack_total(sizes, granule)
}

/// Byte alignment of every member of a weight arena root: the existing
/// device allocation alignment ([`crate::slowlink_admission::S1_ALIGN`]).
pub const WEIGHT_ARENA_ALIGNMENT: usize = 256;

/// Smallest device allocation the planner will request for a root (a
/// root cap and every root request are at least this).
pub const WEIGHT_ARENA_MIN_ROOT_BYTES: usize = 256;

fn arena_bytes(name: &str, bytes: u64) -> Result<usize, String> {
    usize::try_from(bytes)
        .map_err(|_| format!("arena member '{name}' of {bytes} B overflows usize"))
}

/// The members of the weight arena family: exactly the entries
/// [`resident_split`] counts as non-expert (resident, not an external row
/// table, not a tied alias, not a routed expert), each at its own
/// `bytes_of` payload. CPU-only metadata records are not manifest entries
/// and never appear.
pub fn weight_arena_members(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<Vec<(String, usize)>, String> {
    let mut members = Vec::new();
    for entry in weights {
        if entry.residency.is_external()
            || matches!(entry.policy, ShardPolicy::Tied { .. })
            || is_routed_expert(&entry.name)
        {
            continue;
        }
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        members.push((entry.name.clone(), arena_bytes(&entry.name, bytes)?));
    }
    Ok(members)
}

/// The members of the F16 shadow arena family: exactly
/// [`f16_shadow_bytes`]'s owners (the one shared predicate), each at its
/// `2 * rows * cols` shadow allocation.
pub fn f16_shadow_arena_members(
    weights: &[WeightEntry],
    num_layers: usize,
) -> Result<Vec<(String, usize)>, String> {
    let mut members = Vec::new();
    for entry in weights {
        if is_f16_shadow_owner(entry, num_layers, &|_| true, true) {
            members.push((
                entry.name.clone(),
                arena_bytes(&entry.name, f16_shadow_entry_bytes(entry)?)?,
            ));
        }
    }
    Ok(members)
}

/// The two-family root plan of one load's weights, built once and shared by
/// the census, the allocator and the F16 shadow cache. Each root is bounded
/// by the largest existing eligible allocation of either family (never below
/// [`WEIGHT_ARENA_MIN_ROOT_BYTES`]): a root never asks the device for more
/// contiguous memory than the incumbent's biggest single tensor did.
/// `shadow_capable` is the device's own F16 shadow predicate
/// (`Gpu::qwen4_f16_shadow_capable`); without it no shadow ever exists and the
/// shadow family is empty. `granule_bytes` is the device's allocation granule
/// from its single query.
pub fn model_weight_arena_plan(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    num_layers: usize,
    shadow_capable: bool,
    granule_bytes: usize,
) -> Result<std::sync::Arc<rdna_compute::byte_arena::ModelWeightArenaPlan>, String> {
    use rdna_compute::byte_arena::{ModelWeightArenaPlan, NamedByteArenaPlan};
    if granule_bytes == 0 {
        return Err("weight arena plan: the device allocation granule is zero".to_string());
    }
    let weight_members = weight_arena_members(weights, bytes_of)?;
    let shadow_members = if shadow_capable {
        f16_shadow_arena_members(weights, num_layers)?
    } else {
        Vec::new()
    };
    let max_root_bytes = weight_members
        .iter()
        .chain(&shadow_members)
        .map(|member| member.1)
        .max()
        .unwrap_or(0)
        .max(WEIGHT_ARENA_MIN_ROOT_BYTES);
    let plan = ModelWeightArenaPlan {
        weights: NamedByteArenaPlan::new(weight_members, WEIGHT_ARENA_ALIGNMENT, max_root_bytes)
            .map_err(|error| format!("weight arena plan: weights: {error}"))?,
        f16_shadows: NamedByteArenaPlan::new(
            shadow_members,
            WEIGHT_ARENA_ALIGNMENT,
            max_root_bytes,
        )
        .map_err(|error| format!("weight arena plan: F16 shadows: {error}"))?,
        granule_bytes,
    };
    plan.validate()
        .map_err(|error| format!("weight arena plan: {error}"))?;
    Ok(std::sync::Arc::new(plan))
}

/// Granule slack of the provided arena plan, `(weights, F16 shadows)`: each
/// family's root BACKING (every root request rounded up to the granule once)
/// minus the LOGICAL payload the census charges ([`resident_split`]'s
/// non-expert bytes and [`f16_shadow_bytes`]). The plan is only read, never
/// rebuilt; it must describe this manifest (its logical payloads equal the
/// census payloads) and the same `granule`. The shadow term is 0 unless
/// `charge_shadows` (the reserve charges them).
pub fn weight_arena_granule_slack(
    arenas: &rdna_compute::byte_arena::ModelWeightArenaPlan,
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    num_layers: usize,
    charge_shadows: bool,
    granule: u64,
) -> Result<(u64, u64), String> {
    if u64::try_from(arenas.granule_bytes).ok() != Some(granule) {
        return Err(format!(
            "weight arena plan granule {} B differs from the census granule {granule} B",
            arenas.granule_bytes
        ));
    }
    let granule_bytes = arenas.granule_bytes;
    let family = |plan: &rdna_compute::byte_arena::NamedByteArenaPlan,
                  payload: u64,
                  what: &str|
     -> Result<u64, String> {
        let logical = plan.logical_bytes()?;
        if logical != payload {
            return Err(format!(
                "weight arena plan {what} hold {logical} B but the manifest census charges {payload} B"
            ));
        }
        plan.backing_bytes(granule_bytes)?
            .checked_sub(logical)
            .ok_or_else(|| format!("weight arena plan {what} backing is below its payload"))
    };
    let weight_payload = resident_split(weights, bytes_of)?.0;
    let weight_slack = family(&arenas.weights, weight_payload, "weights")?;
    let shadow_slack = if charge_shadows {
        family(
            &arenas.f16_shadows,
            f16_shadow_bytes(weights, num_layers)?,
            "F16 shadows",
        )?
    } else {
        0
    };
    Ok((weight_slack, shadow_slack))
}

/// What [`base_granule_slack`] reads: the same load facts
/// [`placement_inputs`] read, plus the reserve it returned.
pub struct GranuleSlackArgs<'a> {
    pub gpu: &'a rdna_compute::Gpu,
    pub config: &'a Qwen4Config,
    pub weights: &'a [WeightEntry],
    pub bytes_of: &'a dyn Fn(&WeightEntry) -> Option<u64>,
    pub backend: crate::Qwen4KvBackend,
    pub max_seq: usize,
    pub vmm_granularity: usize,
    /// The native MTP head stays attached (as for [`placement_inputs`]).
    pub mtp_kept: bool,
    pub max_k: usize,
    pub state_format: crate::Qwen4StateFormat,
    pub scratch_mode: Qwen4ForwardScratchMode,
    /// The chunk the reserve was sized from.
    pub chunk_rows: usize,
    /// The reserve [`placement_inputs`] returned for these arguments: it
    /// says whether shadows are charged and carries the workspace, gather and
    /// selector terms.
    pub reserve: AutoReserve,
    pub granule: u64,
    /// The provided two-family root plan. `Some`: weights and F16 shadows
    /// are charged their root backing minus the logical payload, never
    /// rebuilt here ([`weight_arena_granule_slack`]). `None`: the individual
    /// allocation counterfactual ([`weight_granule_slack`],
    /// [`f16_shadow_granule_slack`]).
    pub weight_arenas: Option<&'a rdna_compute::byte_arena::ModelWeightArenaPlan>,
}

/// The granule slack of one load's base reserve ([`GranuleSlack`]), from the
/// owners' own allocation lists: the manifest's tensors and shadow owners,
/// `gpu_forward::qwen4_forward_allocation_sizes`,
/// `state::Qwen4State::device_allocation_sizes` (the state and QSA
/// context at the census context [`placement_inputs`] charges) and the
/// workspace terms of `reserve`.
pub fn base_granule_slack(args: &GranuleSlackArgs<'_>) -> Result<GranuleSlack, String> {
    let GranuleSlackArgs {
        gpu,
        config,
        weights,
        bytes_of,
        backend,
        max_seq,
        vmm_granularity,
        mtp_kept,
        max_k,
        state_format,
        scratch_mode,
        chunk_rows,
        reserve,
        granule,
        weight_arenas,
    } = args;
    let census_context =
        crate::Qwen4ContextCommit::new(*backend, *max_seq, *max_seq, *vmm_granularity);
    let forward =
        crate::gpu_forward::qwen4_forward_allocation_sizes(config, *chunk_rows, *scratch_mode)
            .ok_or_else(|| "forward scratch allocation sizes overflow".to_string())?;
    let state =
        crate::state::Qwen4State::device_allocation_sizes(config, *state_format, &census_context)
            .ok_or_else(|| "request state allocation sizes overflow".to_string())?;
    // The native MTP allocations of the SAME call `placement_inputs` charges
    // (`mtp_for(&census_context)`): zero without a kept head. The head's
    // build scratch (released at attach) and the first request's allocations
    // are alternatives of the logical peak; their BACKING is judged per phase
    // ([`native_mtp_granule_slack`]), never by which has the larger logical sum.
    let native_mtp = if *mtp_kept {
        let head = language_head_dtype(weights).ok_or("manifest has no language head")?;
        let drafts = crate::mtp_spec::native_mtp_proposal_drafts(*max_k, &gpu.arch);
        let row_capture =
            crate::mtp_spec::native_mtp_capture(gpu, config, state_format.gdn, drafts);
        let (resident, build_scratch) =
            crate::mtp_gpu::Qwen4MtpGpu::allocation_sizes(config, &census_context, head, false)
                .ok_or_else(|| "native MTP head allocation sizes overflow".to_string())?;
        let request = crate::mtp_spec::native_mtp_request_allocation_sizes(
            config,
            census_context.max_seq,
            *chunk_rows,
            drafts,
            row_capture,
            crate::mtp_spec::native_mtp_fill_scratch(gpu),
        )
        .ok_or_else(|| "native MTP request allocation sizes overflow".to_string())?;
        native_mtp_granule_slack(&resident, &build_scratch, &request, *granule)?
    } else {
        0
    };
    // The gathered-prefill workspace is one allocation on the legacy backend;
    // a VMM workspace maps granule-rounded pages (no second rounding). The
    // split selector reserves one scratch allocation (`qsa_select_scores`).
    let qsa_workspace = granule_slack_total(
        [
            if matches!(backend, crate::Qwen4KvBackend::Legacy) {
                reserve.census.gather
            } else {
                0
            },
            reserve.census.qsa_selector,
        ],
        *granule,
    )?;
    let charge_shadows = reserve.census.f16_shadows > 0;
    let (weights_slack, f16_shadows_slack) = match weight_arenas {
        Some(arenas) => weight_arena_granule_slack(
            arenas,
            weights,
            bytes_of,
            config.num_hidden_layers,
            charge_shadows,
            *granule,
        )?,
        None => (
            weight_granule_slack(weights, bytes_of, *granule)?,
            if charge_shadows {
                f16_shadow_granule_slack(weights, config.num_hidden_layers, *granule)?
            } else {
                0
            },
        ),
    };
    Ok(GranuleSlack {
        weights: weights_slack,
        f16_shadows: f16_shadows_slack,
        forward: granule_slack_total(forward, *granule)?,
        state: granule_slack_total(state, *granule)?,
        workspace: granule_slack_total(
            reserve.census.workspace.iter().map(|term| term.1),
            *granule,
        )?,
        native_mtp,
        qsa_workspace,
    })
}

/// Granule slack of the native MTP charge over the LOGICAL peak
/// `resident + max(build_scratch, request)` that `native_mtp_device_bytes`
/// (and so the census) holds. The two phases are alternatives, so the backing
/// peak is `resident backing + max(build backing, request backing)`; the
/// phase with the larger logical sum need not have the larger backing (a
/// smaller phase whose allocations round up more can win). Returned: that
/// backing peak minus the logical peak (never negative: a phase's backing is
/// at least its payload), checked.
pub fn native_mtp_granule_slack(
    resident: &[u64],
    build_scratch: &[u64],
    request: &[u64],
    granule: u64,
) -> Result<u64, String> {
    let overflow = || "native MTP granule slack overflows".to_string();
    let sum = |sizes: &[u64]| {
        sizes
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
            .ok_or_else(overflow)
    };
    let backing = |sizes: &[u64]| {
        sum(sizes)?
            .checked_add(granule_slack_total(sizes.iter().copied(), granule)?)
            .ok_or_else(overflow)
    };
    let logical_peak = sum(resident)?
        .checked_add(sum(build_scratch)?.max(sum(request)?))
        .ok_or_else(overflow)?;
    let backing_peak = backing(resident)?
        .checked_add(backing(build_scratch)?.max(backing(request)?))
        .ok_or_else(overflow)?;
    backing_peak.checked_sub(logical_peak).ok_or_else(overflow)
}

/// The reserve `auto` keeps free: the larger of the historical measured
/// formula (`historical`, [`auto_vram_reserve`], the floor, no longer
/// evidence of what a long context takes) and the bottom-up
/// [`ReserveCensus`]; where the census carries the measured HIP runtime
/// overhead ([`runtime_overhead_bytes`]) the floor is retired
/// (`historical` is `None`) and the census alone is the reserve. The
/// whole-layer stage buffers and safety margin ([`PlacementCharges`]) are
/// added on top of [`Self::bytes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoReserve {
    pub historical: Option<u64>,
    pub census: ReserveCensus,
    pub census_total: u64,
    /// `Some(S)` when a decode-share load charged only its dedicated forward:
    /// `S` bytes of prefill scratch are borrowed from the donor tail instead
    /// of reserved ([`DecodeShareForward`]).
    pub decode_share_borrowed: Option<u64>,
    /// Bytes of F16 + INT8 weight shadows that live in the donor tail instead
    /// of resident VRAM (their census terms are not charged); 0 when the
    /// shadows stay resident ([`DecodeShareForward::shadows_borrowed`]).
    pub decode_share_shadows: u64,
}

impl AutoReserve {
    pub fn new(historical: Option<u64>, census: ReserveCensus) -> Result<Self, String> {
        let census_total = census.total()?;
        Ok(Self {
            historical,
            census,
            census_total,
            decode_share_borrowed: None,
            decode_share_shadows: 0,
        })
    }

    /// Mark the forward term as the decode-share dedicated charge, with
    /// `borrowed` bytes of prefill scratch lent by the donor tail.
    pub fn with_decode_share(mut self, borrowed: u64) -> Self {
        self.decode_share_borrowed = Some(borrowed);
        self
    }

    /// Record `bytes` of F16 + INT8 weight shadows borrowed inside the donor
    /// tail (the census `f16_shadows` / `shared_i8_shadows` terms are 0).
    pub fn with_decode_share_shadows(mut self, bytes: u64) -> Self {
        self.decode_share_shadows = bytes;
        self
    }

    /// The census with its `forward` term annotated when decode-share charged
    /// the dedicated forward.
    fn census_text(&self) -> String {
        match self.decode_share_borrowed {
            Some(borrowed) => {
                let shadows = if self.decode_share_shadows == 0 {
                    String::new()
                } else {
                    format!(" + {} MiB weight shadows", self.decode_share_shadows >> 20)
                };
                self.census.describe_with(Some(&format!(
                    "decode-share dedicated; {} MiB prefill scratch{shadows} borrowed from the donor tail",
                    borrowed >> 20
                )))
            }
            None => self.census.describe(),
        }
    }

    /// The bytes `auto` reserves: `max(historical, census)`, or the census
    /// alone once the floor is retired.
    pub fn bytes(&self) -> u64 {
        match self.historical {
            Some(historical) => historical.max(self.census_total),
            None => self.census_total,
        }
    }

    /// The base reserve a whole-layer placement is sized from. Every load that
    /// is not an admitted S1 load (`slowlink == false`: the incumbent, fast
    /// links, unified memory, every opt-out) reads exactly [`Self::bytes`],
    /// byte for byte. An admitted S1 load reads the bottom-up census alone
    /// ([`Self::census_total`]): the historical floor is a stale measured
    /// formula (5,517,207,808 B at the 1,536-row tile against a 2,237,111,724 B
    /// census) that would put even the preferred ten resident layers out of reach, and the S1
    /// profile adds its own explicit terms and the safety margin on top
    /// ([`PlacementCharges`]); the Hold 1 gate verifies the measured peak
    /// against census + safety + allowance.
    pub fn placement_bytes(&self, slowlink: bool) -> u64 {
        if slowlink {
            self.census_total
        } else {
            self.bytes()
        }
    }

    /// Which formula set [`Self::bytes`].
    pub fn winner(&self) -> &'static str {
        match self.historical {
            None => "census (historical floor retired)",
            Some(historical) if self.census_total > historical => "census",
            Some(_) => "historical floor",
        }
    }

    /// The reserve and its parts for the placement log, every census term.
    pub fn describe(&self) -> String {
        match self.historical {
            Some(historical) => format!(
                "{} MiB = max({} MiB historical floor, {} MiB census [{}]): {} wins",
                self.bytes() >> 20,
                historical >> 20,
                self.census_total >> 20,
                self.census_text(),
                self.winner()
            ),
            None => format!(
                "{} MiB = census [{}] (historical floor retired: the census carries the \
                 measured HIP runtime overhead)",
                self.bytes() >> 20,
                self.census_text()
            ),
        }
    }
}

/// The byte terms of a decode-share load against the baseline forward charge. Pure arithmetic;
/// not yet wired into [`auto_vram_reserve`], [`state_census`] or the carriers.
///
/// Both of those take [`crate::gpu_forward::qwen4_forward_device_bytes`] as their forward term,
/// so substituting `dedicated_bytes` for it lowers [`AutoReserve::bytes`] by exactly
/// [`Self::reserve_delta`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeShareCharge {
    /// `S`: the donor span the borrowed prefill bank occupies
    /// ([`crate::gpu_forward::qwen4_decode_share_borrow_bytes`]).
    pub borrow_bytes: u64,
    /// `D_bytes`: the dedicated forward the load charges instead of the baseline
    /// ([`crate::gpu_forward::qwen4_decode_share_dedicated_bytes`]).
    pub dedicated_bytes: u64,
    /// `F`: the baseline forward charge
    /// ([`crate::gpu_forward::qwen4_forward_device_bytes`]).
    pub baseline_forward: u64,
    /// `O`: the donor guard plus the second device pointer tables.
    pub overhead: u64,
}

impl DecodeShareCharge {
    /// `F - D_bytes`: how far the reserve falls (zero if the dedicated bank is not smaller).
    pub fn reserve_delta(&self) -> u64 {
        self.baseline_forward.saturating_sub(self.dedicated_bytes)
    }

    /// `F - D_bytes - O`: the net bytes the expert budget gains (zero if the overhead eats it).
    pub fn expert_headroom(&self) -> u64 {
        self.reserve_delta().saturating_sub(self.overhead)
    }
}

/// The largest trunk-layer count whose routed experts fit in `free_vram`
/// after the non-expert weights and `reserve` ([`auto_vram_reserve`]). The S1
/// admission's [`crate::slowlink_admission::PlacementBudget::largest_fitting_layers`]
/// calls it (once the base budget fits) so there is one largest-fit
/// arithmetic; it has no layer floor, and zero is a valid answer.
pub fn auto_vram_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    reserve: u64,
) -> usize {
    let budget = free_vram
        .saturating_sub(non_expert_bytes)
        .saturating_sub(reserve);
    match budget.checked_div(layer_expert_bytes) {
        Some(layers) => (layers as usize).min(num_layers),
        None => num_layers,
    }
}

/// The trunk-layer count a whole-layer `auto` placement keeps resident:
/// [`auto_vram_layers`] against `base_reserve` ([`auto_vram_reserve`]) plus
/// every [`PlacementCharges`] term. Refuses, before anything is allocated,
/// when even zero resident layers cannot hold the non-expert weights and the
/// charged reserve (the load would fail at its first long request instead).
pub fn auto_placement_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    base_reserve: u64,
    charges: &PlacementCharges,
) -> Result<usize, String> {
    const MIB: u64 = 1 << 20;
    let reserve = charges.reserve(base_reserve)?;
    if non_expert_bytes
        .checked_add(reserve)
        .is_none_or(|need| free_vram < need)
    {
        return Err(format!(
            "{} MiB of free VRAM cannot hold the {} MiB of non-expert weights plus the {} MiB reserve ({} MiB base + {}) even with every routed expert in host RAM; free VRAM on this GPU",
            free_vram / MIB,
            non_expert_bytes / MIB,
            reserve / MIB,
            base_reserve / MIB,
            charges.describe()
        ));
    }
    Ok(auto_vram_layers(
        free_vram,
        non_expert_bytes,
        layer_expert_bytes,
        num_layers,
        reserve,
    ))
}

/// Whether every routed expert fits resident beside the non-expert weights and
/// the reserve with the placement charges of a fully resident load
/// ([`PlacementCharges::new`] with no dedicated stage): the decision the
/// unset-`HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` load makes before it falls back to
/// `auto` ([`resolve_expert_vram_layers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentFit {
    pub charges: PlacementCharges,
    /// `base_reserve` plus [`Self::charges`].
    pub reserve: u64,
    pub fits: bool,
}

/// [`ResidentFit`] from the measured free VRAM, the non-expert and total
/// routed-expert bytes, one layer's expert bytes and the base reserve
/// ([`AutoReserve::bytes`]). The one place the fully-resident decision is
/// computed, for the product loader and [`incumbent_placement_layers`].
pub fn resident_fit(
    free_vram: u64,
    non_expert_bytes: u64,
    routed_expert_bytes: u64,
    layer_expert_bytes: u64,
    base_reserve: u64,
) -> Result<ResidentFit, String> {
    let charges = PlacementCharges::new(layer_expert_bytes, false)?;
    let reserve = charges.reserve(base_reserve)?;
    let fits = fits_fully_resident(free_vram, non_expert_bytes, routed_expert_bytes, reserve);
    Ok(ResidentFit {
        charges,
        reserve,
        fits,
    })
}

/// The placement the product loader resolves for an UNSET
/// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` on a single discrete device, from its
/// measured inputs: `None` when everything fits resident, else
/// `Some(layers)` kept on the device by the `auto` placement with the
/// dedicated-stage charge (`dedicated_stage`, [`dedicated_slab_stage`]). Pure:
/// the arithmetic every single-device load, lab example and gate shares.
pub fn incumbent_placement_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    routed_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
    base_reserve: u64,
    dedicated_stage: bool,
) -> Result<Option<usize>, String> {
    let fit = resident_fit(
        free_vram,
        non_expert_bytes,
        routed_expert_bytes,
        layer_expert_bytes,
        base_reserve,
    )?;
    if fit.fits {
        return Ok(None);
    }
    let charges = PlacementCharges::new(layer_expert_bytes, dedicated_stage)?;
    auto_placement_layers(
        free_vram,
        non_expert_bytes,
        layer_expert_bytes,
        num_layers,
        base_reserve,
        &charges,
    )
    .map(Some)
}

/// Device bytes of an all-row F32 logits buffer for `rows` rows of `vocab`
/// logits: what a lab scorer that keeps every row's logits (`qwen4_kld`'s
/// cold-prefill route) allocates beside the load, and the product never does
/// (it keeps the last row and the speculative verify rows only, already in the
/// forward term). Pass it as `lab_device_bytes` to
/// [`incumbent_auto_resident_layers`].
pub fn all_row_logits_bytes(rows: usize, vocab: usize) -> Result<u64, String> {
    rows.checked_mul(vocab)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| format!("all-row logits bytes for {rows} rows x {vocab} overflow"))
}

/// `base_reserve` ([`AutoReserve::bytes`]) plus `lab_device_bytes` of device
/// memory the caller allocates after the placement: the one place the lab term
/// joins the reserve [`incumbent_auto_resident_layers`] sizes against.
pub fn reserve_with_lab_bytes(base_reserve: u64, lab_device_bytes: u64) -> Result<u64, String> {
    base_reserve
        .checked_add(lab_device_bytes)
        .ok_or_else(|| "auto expert VRAM reserve with lab device bytes overflows".to_string())
}

/// [`incumbent_placement_layers`] for a loaded single-device, no-MTP artifact
/// on `gpu`: sizes the inputs exactly as the product loader does
/// ([`placement_inputs`] at the requested prefill chunk, free VRAM from the
/// device, one rank, no native MTP head) and returns the layers the product
/// keeps resident (`None`: fully resident, also on unified memory). The shared
/// lab loader's answer when no explicit layer count is set.
///
/// `lab_device_bytes` is device memory the caller allocates after the
/// placement that the product never holds ([`all_row_logits_bytes`]); it is
/// added to the base reserve, so the placement leaves it free. `0` is the
/// product's own placement byte for byte.
#[allow(clippy::too_many_arguments)]
pub fn incumbent_auto_resident_layers(
    gpu: &rdna_compute::Gpu,
    config: &Qwen4Config,
    weights: &[WeightEntry],
    bytes_of: &dyn Fn(&WeightEntry) -> Option<u64>,
    backend: crate::Qwen4KvBackend,
    max_seq: usize,
    vmm_granularity: usize,
    state_format: crate::Qwen4StateFormat,
    scratch_mode: Qwen4ForwardScratchMode,
    dedicated_stage: bool,
    lab_device_bytes: u64,
) -> Result<Option<usize>, String> {
    if gpu.is_uma() {
        return Ok(None);
    }
    let routed_experts = !config.ffn.is_dense();
    let inputs = placement_inputs(&PlacementInputsArgs {
        gpu,
        config,
        weights,
        bytes_of,
        backend,
        use_ranges: false,
        max_seq,
        vmm_granularity,
        mtp_kept: false,
        max_k: 0,
        routed_experts,
        state_format,
        scratch_mode,
        spans: &EpExpertSpans::single(),
        chunk_rows: crate::gpu_forward::qwen4_prefill_chunk_requested(gpu, max_seq, scratch_mode),
    })?;
    let (free, _) = gpu
        .device_mem_info()
        .map_err(|error| format!("qwen4: device memory query: {error}"))?;
    let routed =
        routed_expert_bytes(weights, bytes_of).map_err(|error| format!("qwen4: {error}"))?;
    incumbent_placement_layers(
        free as u64,
        inputs.non_expert,
        routed,
        inputs.layer_experts,
        config.num_hidden_layers,
        reserve_with_lab_bytes(inputs.reserve.bytes(), lab_device_bytes)
            .map_err(|error| format!("qwen4: {error}"))?,
        dedicated_stage,
    )
    .map_err(|error| format!("qwen4: {error}"))
}

/// Trunk layers (ascending, the MTP head excluded) with at least one
/// host-mapped routed-expert tensor in `weights`: the placement a load actually
/// resolved, read back from the placed manifest rather than from whatever
/// policy produced it.
pub fn host_placed_trunk_layers(weights: &[WeightEntry]) -> Vec<usize> {
    weights
        .iter()
        .filter(|entry| {
            entry.residency == WeightResidency::HostMapped
                && is_routed_expert(&entry.name)
                && !entry.name.starts_with("mtp.")
        })
        .filter_map(|entry| entry.layer)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// sha256 (lowercase hex) of a resolved placement: the trunk layer count and
/// the host-placed layers ([`host_placed_trunk_layers`]). Equal digests mean
/// equal placement; every lab loader and receipt uses this one definition.
pub fn placement_digest(trunk_layers: usize, host_layers: &[usize]) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(b"qwen4-untiered-placement-v1");
    digest.update((trunk_layers as u64).to_le_bytes());
    digest.update((host_layers.len() as u64).to_le_bytes());
    for &layer in host_layers {
        digest.update((layer as u64).to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// Device bytes an expert-tier lease's frozen tables take beyond the payload:
/// the gate/up and down pointer tables (8 bytes per expert each) and the
/// 1-byte-per-expert codec table, over every trunk expert
/// (`expert_tier_owners::materialize`).
pub const EXPERT_TIER_TABLE_BYTES: u64 = crate::expert_slabs::TRUNK_LAYERS as u64
    * crate::expert_slabs::EXPERTS_PER_LAYER as u64
    * (2 * 8 + 1);

/// Device bytes kept free beyond every other budget term when sizing the
/// `HIPFIRE_QWEN4_EXPERT_HOTSET` payload (driver/allocator headroom).
pub const EXPERT_HOTSET_VRAM_MARGIN_BYTES: u64 = 512 << 20;

/// Device bytes the `HIPFIRE_QWEN4_EXPERT_HOTSET` payload may take:
/// `free_vram` less the non-expert weights, `reserve` ([`auto_vram_reserve`],
/// the caller's own value, not recomputed here), the lease's frozen tables
/// ([`EXPERT_TIER_TABLE_BYTES`]) and [`EXPERT_HOTSET_VRAM_MARGIN_BYTES`].
/// Every subtraction is checked; exactly zero left is `Ok(0)`, any shortfall
/// is `Err`.
pub fn expert_hotset_budget(
    free_vram: u64,
    non_expert_bytes: u64,
    reserve: u64,
) -> Result<u64, String> {
    free_vram
        .checked_sub(non_expert_bytes)
        .and_then(|bytes| bytes.checked_sub(reserve))
        .and_then(|bytes| bytes.checked_sub(EXPERT_TIER_TABLE_BYTES))
        .and_then(|bytes| bytes.checked_sub(EXPERT_HOTSET_VRAM_MARGIN_BYTES))
        .ok_or_else(|| {
            format!(
                "expert hotset has no device budget: free VRAM {free_vram} bytes is below the \
                 non-expert weights {non_expert_bytes} + reserve {reserve} + expert tier tables \
                 {EXPERT_TIER_TABLE_BYTES} + safety margin {EXPERT_HOTSET_VRAM_MARGIN_BYTES}"
            )
        })
}

/// Whether a load on `gpu` may build the long-prefill expert stage (the same
/// switch `Qwen4ExpertStage::new` reads). Admission and the constructor both
/// call this with the same `gpu`, so they share one input:
/// `HIPFIRE_QWEN4_EXPERT_STAGE` (`1`/`0`) when set, else
/// [`crate::host_link::expert_stage_default`] on the device's cached measured
/// host link. An explicit value or gfx1201 never triggers the H2D probe.
///
/// This probes (gfx1100 default), so it must never size an expert placement:
/// placement charges use [`expert_stage_placement_enabled`] and the stage
/// this switch alone turns on borrows donor layers (no dedicated VRAM).
pub fn expert_stage_enabled(gpu: &rdna_compute::Gpu) -> bool {
    expert_stage_switch(
        &gpu.arch,
        hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV)
            .ok()
            .as_deref(),
        || crate::host_link::host_link(gpu),
    )
}

/// [`expert_stage_enabled`] with the env value and the (lazily measured)
/// link supplied; the strict `1`/`0` spelling of `developer_bool`.
pub(crate) fn expert_stage_switch(
    arch: &str,
    env: Option<&str>,
    link: impl FnOnce() -> crate::host_link::HostLink,
) -> bool {
    match env {
        Some("1") => true,
        Some("0") => false,
        _ => crate::host_link::expert_stage_default(arch, link),
    }
}

/// The stage switch every placement charge reads: the explicit
/// `HIPFIRE_QWEN4_EXPERT_STAGE` value, else gfx1201 only. It is
/// [`expert_stage_enabled`] with the link taken as unmeasured, so it never
/// reads the host link (the gfx1100 narrow-link chunk is the only way the
/// link reaches the placement). A dedicated stage (slab or
/// tier layout) is therefore charged only where it can actually allocate
/// (`Qwen4ExpertStage::new` allocates those layouts only under this switch);
/// the borrowed stage, packed or slab, is charged nothing: it lends resident
/// donor layers, so a stage the measured link alone enables never takes
/// uncharged VRAM nor a resident layer.
pub fn expert_stage_placement_enabled(arch: &str) -> bool {
    expert_stage_placement_switch(
        arch,
        hipfire_config::developer_var(crate::gpu_forward::QWEN4_EXPERT_STAGE_ENV)
            .ok()
            .as_deref(),
    )
}

/// [`expert_stage_placement_enabled`] with the env value supplied.
fn expert_stage_placement_switch(arch: &str, env: Option<&str>) -> bool {
    expert_stage_switch(arch, env, || crate::host_link::HostLink::Unmeasured)
}

/// Device bytes the tier long-prefill staging allocates for `plan`: one VRAM
/// stage buffer per layer parity, each the largest host-tier payload of a
/// layer of that parity (projections 256-byte aligned) plus the zeroed tail
/// pad, as `gpu_forward::slab_stage` sizes them. An upper bound of what it
/// allocates, so the hot set budget can hold it. This is the Resident
/// placement; a decode-share plan stages from its BorrowPrefill placement
/// ([`expert_tier_stage_bytes_prefill`]).
pub fn expert_tier_stage_bytes(plan: &crate::expert_tier_plan::ExpertTierPlan) -> u64 {
    expert_tier_stage_bytes_for_entries(plan.entries())
}

/// [`expert_tier_stage_bytes`] over an explicit entry list in model id order
/// (`layer * 512 + expert`): the plan's own entries for the Resident
/// placement, `DecodeShareTail::borrow_entries` for BorrowPrefill (donors
/// count as host experts of their layer).
pub fn expert_tier_stage_bytes_for_entries(
    entries: &[crate::expert_tier_plan::ExpertTierEntry],
) -> u64 {
    use crate::expert_slabs::{EXPERTS_PER_LAYER, TRUNK_LAYERS};
    use crate::expert_tier_plan::Placement;
    let align = |len: u64| len.div_ceil(256) * 256;
    let mut largest = [0u64; 2];
    for (layer, experts) in entries
        .chunks(EXPERTS_PER_LAYER as usize)
        .take(TRUNK_LAYERS as usize)
        .enumerate()
    {
        let host: u64 = experts
            .iter()
            .filter(|entry| entry.tier == Placement::HostMapped)
            .map(|entry| align(entry.gate_up_len) + align(entry.down_len))
            .sum();
        largest[layer % 2] = largest[layer % 2].max(host);
    }
    largest
        .iter()
        .map(|bytes| bytes + crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64)
        .sum()
}

/// Stage bytes of the placement a long prefill reads: the BorrowPrefill
/// entries when the plan carries a decode-share donor tail, else the Resident
/// entries (byte-identical to [`expert_tier_stage_bytes`] for a non-sharing
/// plan).
pub fn expert_tier_stage_bytes_prefill(plan: &crate::expert_tier_plan::ExpertTierPlan) -> u64 {
    match plan.decode_share_tail() {
        Some(tail) => expert_tier_stage_bytes_for_entries(&tail.borrow_entries),
        None => expert_tier_stage_bytes(plan),
    }
}

/// Whether a whole-layer (slab arena) load allocates the two dedicated
/// long-prefill stage buffers at load: the probe-free placement switch is on
/// ([`expert_stage_placement_enabled`]; a stage the measured link alone
/// enables borrows resident slab layers instead and is charged nothing), the
/// artifact is a slab-arena recipe (the legacy aggregate layout borrows
/// resident donor layers and allocates nothing; a tier lease charges its own
/// stage in the hot set fixed point) and the load is not expert-parallel
/// (`Qwen4ExpertStage::new_slab` has no EP route). Not conditioned on the
/// prefill length: the buffers exist from load, even if every prompt stays
/// under `HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS`.
pub fn dedicated_slab_stage(stage_enabled: bool, slab_recipe: bool, expert_parallel: bool) -> bool {
    stage_enabled && slab_recipe && !expert_parallel
}

/// What a whole-layer placement (`auto`, or the unset fallback that decides
/// whether everything is resident) keeps free beyond [`AutoReserve::bytes`]
/// (the larger of the historical formula and the full-context census, which
/// already hold the chunk-sized forward resources and the pool workspace),
/// itemised for the placement log:
///
/// - `margin`: [`EXPERT_HOTSET_VRAM_MARGIN_BYTES`], the driver/allocator
///   safety margin the hot set budget keeps.
/// - `stage`: the two dedicated long-prefill stage buffers of a slab load,
///   `2 * (largest layer payload + ARENA_TAIL_PAD_BYTES)`
///   ([`dedicated_slab_stage`]); 0 when none are allocated.
/// - `slowlink`: the S1 slow-link prefill profile's complete census
///   ([`crate::slowlink_admission::SlowlinkCensus::total`]: tile arena, expert
///   page store, panel scratch, boundary state); 0 on every load that does
///   not select S1, which keeps that load's reserve and log byte for byte.
///
/// The hot set path does not use this: it subtracts its own margin and
/// tier-plan stage ([`expert_hotset_budget`], [`expert_tier_stage_bytes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementCharges {
    pub margin: u64,
    pub stage: u64,
    pub slowlink: u64,
}

impl PlacementCharges {
    /// `layer_expert_bytes` is the largest layer payload (`resident_split`);
    /// `dedicated_stage` is [`dedicated_slab_stage`].
    pub fn new(layer_expert_bytes: u64, dedicated_stage: bool) -> Result<Self, String> {
        let stage = if dedicated_stage {
            layer_expert_bytes
                .checked_add(crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64)
                .and_then(|buffer| buffer.checked_mul(2))
                .ok_or("long-prefill stage buffer bytes overflow")?
        } else {
            0
        };
        Ok(Self {
            margin: EXPERT_HOTSET_VRAM_MARGIN_BYTES,
            stage,
            slowlink: 0,
        })
    }

    /// These charges plus the S1 profile's census `bytes`.
    pub fn with_slowlink(self, bytes: u64) -> Self {
        Self {
            slowlink: bytes,
            ..self
        }
    }

    /// `base` ([`auto_vram_reserve`]) plus every charge.
    pub fn reserve(&self, base: u64) -> Result<u64, String> {
        base.checked_add(self.margin)
            .and_then(|bytes| bytes.checked_add(self.stage))
            .and_then(|bytes| bytes.checked_add(self.slowlink))
            .ok_or_else(|| "auto expert VRAM reserve with placement charges overflows".to_string())
    }

    /// The charges for the placement log, in MiB.
    pub fn describe(&self) -> String {
        const MIB: u64 = 1 << 20;
        let base = format!(
            "{} MiB safety margin + {} MiB long-prefill stage buffers",
            self.margin / MIB,
            self.stage / MIB
        );
        if self.slowlink == 0 {
            base
        } else {
            format!(
                "{base} + {} MiB slow-link prefill (S1) arena, expert pages and boundary state",
                self.slowlink / MIB
            )
        }
    }
}
/// Actual D1 CPU-owned expert storage (admission side).
///
/// Storage only, NOT a qualification or a default: the legacy default route still
/// admits at most [`crate::program::QWEN4_CPU_SPLICE_MAX_ROWS`] rows, and a 5..64-row
/// window additionally needs the explicit lab-wide scope owned by the routing slice.
/// Never a metadata bump of the qualified width: this section only reserves and reports
/// the real bytes/slots the [`hipfire_dispatch::cpu_exec`] backend consumes.
///
/// Two thread-local pools back every D1 splice on the calling thread
/// ([`hipfire_dispatch::cpu_exec`]: the `CPU_TRANSFER` pinned stage and the
/// `MOE_SCRATCH` host-slot scratch, both grown on demand by the first window that
/// needs them). Admission prewarms both for the actually requested window
/// ([`cpu_d1_required_storage_rows`]) so no first wide window grows them on the hot
/// path; the routing slice reads the binding capacity through
/// [`hipfire_dispatch::families::moe::cpu_d1_prewarmed_scratch_rows`] (the live
/// thread-local minimum of host-scratch rows and stage rows — never a constant).
///
/// Checked formulas (exact [`hipfire_dispatch::cpu_exec::moe_cpu_experts`] extents;
/// `None` on zero geometry or overflow → the caller refuses before any allocation):
/// - [`cpu_d1_host_slots`]: `rows * top_k` computed slots (64 x 10 = 640).
/// - [`cpu_d1_stage_values`]: `2 * (rows * dim + rows * top_k)` f32 stage values,
///   exactly the `(n_x + slots) * 8` stage bytes / 4 the splice stages the route,
///   weights, activation and residual through.
/// - [`cpu_d1_stage_capacity_rows`]: the inverse over live stage bytes.
/// Whether the CPU expert splice may arm for one layer. Every field is an
/// input; [`Self::armed`] is the single decision.
///
/// Computed host slots of a `rows`-row D1 window at `top_k` experts per row.
/// `None` for zero rows or on overflow: the caller refuses before any allocation.
pub fn cpu_d1_host_slots(rows: usize, top_k: usize) -> Option<usize> {
    if rows == 0 || top_k == 0 {
        return None;
    }
    rows.checked_mul(top_k)
}

/// f32 stage values the D1 splice's pinned transfer stage holds for a `rows`-row
/// window: `2 * (rows * dim + rows * top_k)`, exactly `moe_cpu_experts`'
/// `(n_x + slots) * 8` stage bytes / 4 (route ids, route weights, rotated
/// activation download plus the residual upload through the same stage).
/// `None` for zero geometry or on overflow: the caller refuses before any allocation.
pub fn cpu_d1_stage_values(dim: usize, top_k: usize, rows: usize) -> Option<usize> {
    if dim == 0 || top_k == 0 || rows == 0 {
        return None;
    }
    let route = rows.checked_mul(dim)?;
    let slots = rows.checked_mul(top_k)?;
    route.checked_add(slots)?.checked_mul(2)
}

/// Logical rows a `stage_bytes`-byte pinned stage serves at `(dim, top_k)`:
/// `bytes / (8 * (dim + top_k))`, the inverse of [`cpu_d1_stage_values`].
/// `0` for zero geometry, on overflow, or when the stage cannot hold one row.
pub fn cpu_d1_stage_capacity_rows(stage_bytes: usize, dim: usize, top_k: usize) -> usize {
    let per_row_values = match dim.checked_add(top_k) {
        Some(v) if v > 0 => v,
        _ => return 0,
    };
    let per_row_bytes = match per_row_values.checked_mul(8) {
        Some(v) if v > 0 => v,
        _ => return 0,
    };
    stage_bytes / per_row_bytes
}

/// Actually requested D1 storage rows (logical window rows) for this load, or `0`
/// for no prewarm. Prewarm is NOT unconditional: legacy default loads (no lab-wide
/// override, small K) keep on-demand growth and byte-identical placement; only an
/// explicit lab-wide scope on a CPU-owned single-device Flash-Next load reserves up
/// to the physical 64. Both lab arms with an admitted 64-row window therefore
/// prewarm the same real 64.
///
/// - `lab_override_present`: the canonical
///   [`rdna_compute::prefill_invariant::qwen4_wide_verify_override_present`] signal
///   (runtime override — including the `Some(8)` baseline — or the parsed developer
///   var, on an exact oracle arch). Presence only, never a capacity value.
/// - `cpu_owned_single_fn`: single-device Flash-Next scope with a CPU owner (D1 splice
///   or D2 split armed) and no EP/PP/lane route (those never take the CPU path).
/// - `requested_window_rows`: the admitted native/copy/verify need in rows — the
///   planned native-MTP capture rows the loader already derived (drafts + 1 saturating
///   at the physical bound, at least the row-exact floor). `0` (AR load, unadmitted
///   capture) requests nothing.
///
/// Returns `0` unless the scope holds and more than the legacy
/// [`crate::program::QWEN4_CPU_SPLICE_MAX_ROWS`] rows are actually requested;
/// else `requested` saturating at
/// [`hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS`].
pub fn cpu_d1_required_storage_rows(
    lab_override_present: bool,
    cpu_owned_single_fn: bool,
    requested_window_rows: usize,
) -> usize {
    if !lab_override_present || !cpu_owned_single_fn {
        return 0;
    }
    if requested_window_rows <= crate::program::QWEN4_CPU_SPLICE_MAX_ROWS {
        return 0;
    }
    requested_window_rows.min(hipfire_dispatch::cpu_split::CPU_SPLIT_STORAGE_MAX_ROWS)
}

/// The long-prefill stage a `whole-first` load runs with
/// (`HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY`), the existing G2 stage unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeFirstStage {
    /// No stage: switch off, no host layer, or fewer than two resident layers
    /// to donate.
    None,
    /// Legacy aggregate (packed) layout: the stage borrows the last two
    /// resident whole layers as its DMA buffers. Nothing is allocated; the
    /// restore copies are pinned host RAM.
    Borrowed {
        donors: [usize; 2],
        /// Resident device bytes the two donors lend (no incremental VRAM).
        capacity: u64,
        /// Pinned host bytes the donors are restored from.
        restore_copies: u64,
    },
    /// Slab layout: two dedicated VRAM stage buffers, charged in
    /// [`PlacementCharges::stage`].
    Dedicated { buffer: u64, allocated: u64 },
}

impl WholeFirstStage {
    /// Device bytes the stage allocates beyond the resident layers.
    pub fn allocated_vram(&self) -> u64 {
        match self {
            Self::Dedicated { allocated, .. } => *allocated,
            Self::None | Self::Borrowed { .. } => 0,
        }
    }

    /// Resident device bytes the stage borrows from donor layers.
    pub fn borrowed_vram(&self) -> u64 {
        match self {
            Self::Borrowed { capacity, .. } => *capacity,
            Self::None | Self::Dedicated { .. } => 0,
        }
    }
}

/// The whole-layer placement `HIPFIRE_QWEN4_EXPERT_HOTSET_POLICY=whole-first`
/// keeps: exactly the `hotset=0` whole-layer plan (no tier plan, owners or
/// sidecar), so the decode route is the whole-layer route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WholeFirstPlan {
    /// Whole resident layers: trunk layers `0..whole_layers`.
    pub whole_layers: usize,
    /// MoE layers of the model (trunk plus MTP).
    pub moe_layers: usize,
    /// Hot experts kept off a host layer: always 0 (no tier lease).
    pub hot_experts: usize,
    /// Largest layer payload the plan was budgeted with.
    pub layer_expert_bytes: u64,
    pub stage: WholeFirstStage,
}

impl WholeFirstPlan {
    /// The stage for `whole_layers` resident layers. `charges` are the
    /// loader's [`PlacementCharges`] for the load (a slab stage is charged in
    /// `charges.stage`); `borrowed_donors` is whether the stage is switched
    /// on and borrows donor layers (legacy aggregate recipe, no expert
    /// parallelism).
    pub fn stage_for(
        whole_layers: usize,
        moe_layers: usize,
        layer_expert_bytes: u64,
        charges: &PlacementCharges,
        borrowed_donors: bool,
    ) -> WholeFirstStage {
        if whole_layers >= moe_layers {
            WholeFirstStage::None
        } else if charges.stage > 0 {
            WholeFirstStage::Dedicated {
                buffer: charges.stage / 2,
                allocated: charges.stage,
            }
        } else if borrowed_donors && whole_layers >= 2 {
            let pair = layer_expert_bytes.saturating_mul(2);
            WholeFirstStage::Borrowed {
                donors: [whole_layers - 2, whole_layers - 1],
                capacity: pair,
                restore_copies: pair,
            }
        } else {
            WholeFirstStage::None
        }
    }

    /// The `auto` placement under `whole-first`: [`auto_placement_layers`]
    /// with the same inputs the `hotset=0` path passes (so never fewer whole
    /// layers), the stage that load runs with, and a check that the
    /// resident layers, the charged reserve (dedicated stage included) and
    /// the non-expert weights fit `free_vram`.
    #[allow(clippy::too_many_arguments)]
    pub fn auto(
        free_vram: u64,
        non_expert_bytes: u64,
        layer_expert_bytes: u64,
        num_layers: usize,
        moe_layers: usize,
        base_reserve: u64,
        charges: &PlacementCharges,
        borrowed_donors: bool,
    ) -> Result<Self, String> {
        let whole_layers = auto_placement_layers(
            free_vram,
            non_expert_bytes,
            layer_expert_bytes,
            num_layers,
            base_reserve,
            charges,
        )?;
        charges
            .reserve(base_reserve)
            .ok()
            .and_then(|reserve| reserve.checked_add(non_expert_bytes))
            .and_then(|bytes| {
                layer_expert_bytes
                    .checked_mul(whole_layers as u64)
                    .and_then(|layers| bytes.checked_add(layers))
            })
            .filter(|need| *need <= free_vram)
            .ok_or_else(|| {
                format!(
                    "whole-first plan of {whole_layers} layers does not fit {} MiB free VRAM",
                    free_vram >> 20
                )
            })?;
        Ok(Self {
            whole_layers,
            moe_layers,
            hot_experts: 0,
            layer_expert_bytes,
            stage: Self::stage_for(
                whole_layers,
                moe_layers,
                layer_expert_bytes,
                charges,
                borrowed_donors,
            ),
        })
    }

    /// The fully resident load (every MoE layer whole, no host layer, no
    /// stage).
    pub fn fully_resident(moe_layers: usize, layer_expert_bytes: u64) -> Self {
        Self {
            whole_layers: moe_layers,
            moe_layers,
            hot_experts: 0,
            layer_expert_bytes,
            stage: WholeFirstStage::None,
        }
    }

    /// The one log line (no newline, no prefix) of an opt-in load.
    /// `resident_bytes` is the measured routed-expert bytes left
    /// device-resident ([`resident_routed_expert_bytes`]). The layer range
    /// is half open, as the placement log prints it.
    pub fn describe(&self, resident_bytes: u64) -> String {
        const MIB: u64 = 1 << 20;
        let stage = match self.stage {
            WholeFirstStage::None => "stage allocated 0 MiB, borrowed 0 MiB".to_string(),
            WholeFirstStage::Borrowed {
                donors,
                capacity,
                restore_copies,
            } => format!(
                "stage allocated 0 MiB, borrowed {} MiB from resident layers {} and {} ({} MiB pinned host restore copies)",
                capacity / MIB,
                donors[0],
                donors[1],
                restore_copies / MIB
            ),
            WholeFirstStage::Dedicated { buffer, allocated } => format!(
                "stage allocated {} MiB (2 dedicated buffers of {} MiB, charged to the placement), borrowed 0 MiB",
                allocated / MIB,
                buffer / MIB
            ),
        };
        format!(
            "policy=whole-first, whole layers 0..{} ({} of {} MoE layers), hot experts {}, expert resident {} MiB, {stage}; hotset=0 whole-layer placement, sidecar not consulted",
            self.whole_layers,
            self.whole_layers,
            self.moe_layers,
            self.hot_experts,
            resident_bytes / MIB
        )
    }
}

/// Routed-expert bytes of the entries left [`WeightResidency::Resident`]
/// (after [`place_routed_experts`], what stays on the device).
pub fn resident_routed_expert_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    weights
        .iter()
        .filter(|entry| {
            entry.residency == WeightResidency::Resident
                && is_routed_expert(&entry.name)
                && !matches!(entry.policy, ShardPolicy::Tied { .. })
        })
        .try_fold(0u64, |total, entry| {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total
                .checked_add(bytes)
                .ok_or_else(|| "resident routed-expert byte census overflows".to_string())
        })
}

/// Mark the routed-expert entries of trunk layers `>= vram_layers`, and of
/// the MTP layer, [`WeightResidency::HostMapped`]. Returns the number of
/// entries moved to host RAM.
pub fn place_routed_experts(weights: &mut [WeightEntry], vram_layers: usize) -> usize {
    let mut moved = 0;
    for entry in weights.iter_mut() {
        if entry.residency != WeightResidency::Resident || !is_routed_expert(&entry.name) {
            continue;
        }
        let trunk_resident =
            !entry.name.starts_with("mtp.") && entry.layer.is_some_and(|layer| layer < vram_layers);
        if !trunk_resident {
            entry.residency = WeightResidency::HostMapped;
            moved += 1;
        }
    }
    moved
}

/// [`place_routed_experts`] for an arbitrary resident set: the routed-expert
/// entries of trunk layers listed in `resident` stay as they are and every
/// other [`WeightResidency::Resident`] routed entry, the MTP layer's always
/// included, becomes [`WeightResidency::HostMapped`]. Same skips as the
/// count-based form (non-routed or already-placed entries are untouched, an
/// entry with no layer is host), and `resident == [0, .., n)` is identical to
/// `place_routed_experts(weights, n)`. Returns the number of entries moved.
pub fn place_routed_experts_set(weights: &mut [WeightEntry], resident: &[usize]) -> usize {
    let mut moved = 0;
    for entry in weights.iter_mut() {
        if entry.residency != WeightResidency::Resident || !is_routed_expert(&entry.name) {
            continue;
        }
        let trunk_resident = !entry.name.starts_with("mtp.")
            && entry.layer.is_some_and(|layer| resident.contains(&layer));
        if !trunk_resident {
            entry.residency = WeightResidency::HostMapped;
            moved += 1;
        }
    }
    moved
}

/// Pinned host bytes the [`WeightResidency::HostMapped`] entries will take.
pub fn host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total += bytes + HOST_MAPPED_PAD_BYTES;
        }
    }
    Ok(total)
}

/// Pinned host bytes the manifest will hold after
/// `place_routed_experts(weights, vram_layers)` and `with_expert_tiers`,
/// computed without mutating or cloning `weights` so every host RAM/GTT check
/// can run before the manifest changes. `with_expert_tiers` drops every trunk
/// routed-expert tensor (non-`mtp.` with a layer), whatever its placement, so
/// those are never counted. Counted, each with [`HOST_MAPPED_PAD_BYTES`]: every
/// other entry already [`WeightResidency::HostMapped`], and every other
/// [`WeightResidency::Resident`] routed entry [`place_routed_experts`] would
/// move to host (the MTP layer's experts). Sums are checked.
pub fn hotset_host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    vram_layers: usize,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        let routed = is_routed_expert(&entry.name);
        let mtp = entry.name.starts_with("mtp.");
        if routed && !mtp && entry.layer.is_some() {
            continue;
        }
        let host = match entry.residency {
            WeightResidency::HostMapped => true,
            WeightResidency::Resident => {
                routed && !(entry.layer.is_some_and(|layer| layer < vram_layers) && !mtp)
            }
            _ => false,
        };
        if !host {
            continue;
        }
        let bytes =
            bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        total = bytes
            .checked_add(HOST_MAPPED_PAD_BYTES)
            .and_then(|charge| total.checked_add(charge))
            .ok_or_else(|| "hotset host-mapped byte total overflows".to_string())?;
    }
    Ok(total)
}

/// Pinned host bytes every rank of an expert-parallel load with the sealed
/// `spans` takes for its [`WeightResidency::HostMapped`] entries, indexed by
/// rank (`bytes_of` is the full source payload; a routed expert stack's share
/// on rank `r` is its per-expert bytes times the experts the seal gives `r`).
/// Every rank pins its own copy in the same host RAM, so [`check_host_ram`]
/// and [`check_gtt_cap`] take the sum of the ranks that have not pinned yet.
/// [`EpExpertSpans::single`] is a single entry equal to [`host_mapped_bytes`].
pub fn host_mapped_bytes_ranks(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
    spans: &EpExpertSpans,
) -> Result<Vec<u64>, String> {
    let mut per_rank = vec![0u64; spans.ranks()];
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            for (rank, charge) in per_rank.iter_mut().enumerate() {
                *charge += rank_share(entry, bytes, rank, spans)? + HOST_MAPPED_PAD_BYTES;
            }
        }
    }
    Ok(per_rank)
}

/// Two donor mirrors per rank share one host RAM/GTT budget: the sum over
/// ranks of each rank's `rank_layer_bytes` (one donor layer's gate/up plus
/// down share), twice, plus the padding of its four pinned allocations.
pub(crate) fn expert_stage_mirror_bytes(rank_layer_bytes: &[usize]) -> Result<u64, String> {
    if rank_layer_bytes.is_empty() {
        return Err("expert stage mirror host charge needs at least one rank".to_string());
    }
    rank_layer_bytes.iter().try_fold(0u64, |total, &bytes| {
        (bytes as u64)
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(4 * HOST_MAPPED_PAD_BYTES))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| "expert stage mirror host charge overflows".to_string())
    })
}

/// Pinned host RAM/GTT the borrowed slab stage's two donor restore copies take
/// (`gpu_forward::slab_stage`): the overwritten arena prefix of each parity
/// (`window`: the largest cold arena plus tail pad, or the whole donor
/// allocation when the donor is a little smaller than that,
/// `slab_stage::plan_slab_donors`) in one pinned allocation each, including
/// that allocation's padding. Taken at load and held on top of the placement's
/// pinned experts.
pub(crate) fn slab_stage_mirror_bytes(window: [usize; 2]) -> Result<u64, String> {
    (window[0] as u64)
        .checked_add(window[1] as u64)
        .and_then(|bytes| bytes.checked_add(2 * HOST_MAPPED_PAD_BYTES))
        .ok_or_else(|| "slab stage mirror host charge overflows".to_string())
}

/// Routed experts each rank of an expert-parallel load holds, read from the
/// load's seal (`Qwen4EpSeal::expert_spans`); [`Self::single`] for one rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpExpertSpans {
    experts: usize,
    local: Vec<usize>,
}

impl EpExpertSpans {
    /// A single-rank load: every routed stack is held whole.
    pub fn single() -> Self {
        Self {
            experts: 0,
            local: vec![0],
        }
    }

    /// `local[r]` of `experts` routed experts on rank `r`: every rank holds
    /// at least one and together they hold each expert once.
    pub fn new(experts: usize, local: Vec<usize>) -> Result<Self, String> {
        if local.len() < 2 || local.contains(&0) || local.iter().sum::<usize>() != experts {
            return Err(format!(
                "expert-parallel spans {local:?} do not split {experts} experts over two or more ranks"
            ));
        }
        Ok(Self { experts, local })
    }

    pub fn ranks(&self) -> usize {
        self.local.len()
    }

    /// Experts rank `rank` holds (`None` for a single-rank load or a rank
    /// outside the spans).
    pub fn local(&self, rank: usize) -> Option<usize> {
        self.local.get(rank).copied().filter(|_| self.ranks() > 1)
    }

    /// The rank holding the most experts (the lowest such rank).
    pub fn largest_rank(&self) -> usize {
        let max = self.local.iter().copied().max().unwrap_or(0);
        self.local.iter().position(|&n| n == max).unwrap_or(0)
    }
}

/// Rank `rank`'s share of an entry's payload: a routed expert stack (logical
/// shape `[n_experts, ..]`) holds its per-expert bytes times the experts the
/// seal gives the rank, everything else is replicated. An expert-slab arena
/// (logical shape `[bytes]`, no expert axis) keeps its even byte split; EP
/// refuses slab manifests today (`Qwen4Manifest::with_expert_parallel`).
fn rank_share(
    entry: &WeightEntry,
    bytes: u64,
    rank: usize,
    spans: &EpExpertSpans,
) -> Result<u64, String> {
    let ranks = spans.ranks();
    if ranks == 1 || !is_routed_expert(&entry.name) {
        return Ok(bytes);
    }
    if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
        let ranks = ranks as u64;
        if bytes % ranks != 0 {
            return Err(format!(
                "'{}' has {bytes} bytes, which do not split across {ranks} ranks",
                entry.name
            ));
        }
        return Ok(bytes / ranks);
    }
    let local = spans
        .local(rank)
        .ok_or_else(|| format!("expert-parallel rank {rank} is outside {ranks} ranks"))?;
    let experts = entry.logical_shape.first().copied().unwrap_or(0);
    if experts != spans.experts {
        return Err(format!(
            "'{}' has {experts} routed experts, the seal splits {}",
            entry.name, spans.experts
        ));
    }
    if bytes % experts as u64 != 0 {
        return Err(format!(
            "'{}' has {bytes} bytes, which are not a multiple of its {experts} experts",
            entry.name
        ));
    }
    Ok(bytes / experts as u64 * local as u64)
}

/// MoE layers of the model: the trunk's plus the MTP head's.
pub fn moe_layer_count(config: &Qwen4Config) -> usize {
    config.num_hidden_layers + usize::from(config.mtp.is_some())
}

/// MoE layers one pipeline-parallel band owns: its trunk layers, plus the MTP
/// head's layer only on the output band. `Single` equals [`moe_layer_count`].
/// Expert-parallel counts never use this: they stay [`moe_layer_count`].
pub fn moe_layer_count_scope(config: &Qwen4Config, scope: &Qwen4BandScope) -> usize {
    scope.band.len() + usize::from(scope.output && config.mtp.is_some())
}

/// How many of the model's MoE layers (trunk first, the MTP layer last) one
/// rank with `free_vram` keeps resident: all of them when the trunk, every
/// layer's experts and the reserve fit, otherwise the trunk layers `auto`
/// fits (the MTP layer is then always host-mapped, as in
/// [`place_routed_experts`]). `layer_expert_bytes_per_rank` is one MoE
/// layer's routed experts on the largest rank (`max_local_count` experts), so
/// a layer counted resident fits every rank of an uneven split.
pub fn ep_rank_layers(
    config: &Qwen4Config,
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
) -> usize {
    let moe_layers = moe_layer_count(config);
    let all_experts = layer_expert_bytes_per_rank.saturating_mul(moe_layers as u64);
    if fits_fully_resident(free_vram, non_expert_bytes, all_experts, reserve_bytes) {
        return moe_layers;
    }
    auto_vram_layers(
        free_vram,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        config.num_hidden_layers,
        reserve_bytes,
    )
}

/// The expert VRAM split every rank of an expert-parallel load applies: the
/// minimum over ranks of [`ep_rank_layers`], so each layer is resident on all
/// ranks or on none (the full prefill table relies on equal residency per
/// layer across ranks). A count of [`moe_layer_count`] means fully resident.
/// `free_per_rank` is every rank's free VRAM before the load; the result is
/// the `expert_vram_layers` override passed to every rank's load.
pub fn ep_vram_layers(
    config: &Qwen4Config,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
) -> usize {
    free_per_rank
        .iter()
        .map(|&free| {
            ep_rank_layers(
                config,
                free,
                non_expert_bytes,
                layer_expert_bytes_per_rank,
                reserve_bytes,
            )
        })
        .min()
        .unwrap_or(0)
}

/// The expert-parallel split with each rank's AR lane store reserved
/// ([`ep_vram_layers_with_lanes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpLaneReservedSplit {
    /// Device bytes of one rank's lane store held back from the experts.
    pub lane_bytes: u64,
    /// The split without the reservation ([`ep_vram_layers`]).
    pub baseline_layers: usize,
    /// The split with `lane_bytes` added to every rank's reserve; never above
    /// `baseline_layers`.
    pub reserved_layers: usize,
}

/// [`ep_vram_layers`] with `lane_bytes` (one rank's lane store,
/// [`crate::ep::ep_lane_store_device_bytes`]) added to the `reserve_bytes`
/// every rank keeps free, so the stage after the load finds the store's bytes
/// still free on every rank. `lane_bytes == 0` returns the baseline twice.
/// `Err` (the caller keeps the baseline split and says why) when the sum
/// overflows, or when some rank cannot hold the non-expert weights plus the
/// reserve plus the lane store even with every routed expert in host RAM.
pub fn ep_vram_layers_with_lanes(
    config: &Qwen4Config,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    lane_bytes: u64,
) -> Result<EpLaneReservedSplit, String> {
    const MIB: u64 = 1 << 20;
    let baseline_layers = ep_vram_layers(
        config,
        free_per_rank,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserve_bytes,
    );
    let reserved = reserve_bytes
        .checked_add(lane_bytes)
        .ok_or_else(|| "the EP lane reservation overflows the expert reserve".to_string())?;
    let floor = non_expert_bytes.checked_add(reserved).ok_or_else(|| {
        "the EP lane reservation overflows the non-expert weights plus reserve".to_string()
    })?;
    if let Some((rank, &free)) = free_per_rank
        .iter()
        .enumerate()
        .find(|(_, free)| **free < floor)
    {
        return Err(format!(
            "rank {rank}: {} MiB free VRAM cannot hold the {} MiB of non-expert weights plus the {} MiB reserve and the {} MiB lane store even with every routed expert in host RAM",
            free / MIB,
            non_expert_bytes / MIB,
            reserve_bytes / MIB,
            lane_bytes / MIB
        ));
    }
    let reserved_layers = ep_vram_layers(
        config,
        free_per_rank,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserved,
    );
    Ok(EpLaneReservedSplit {
        lane_bytes,
        baseline_layers,
        reserved_layers,
    })
}

/// An AR-lane reservation request of an expert-parallel load, already
/// resolved to one rank's lane-store bytes. `bytes` is `Err` when the bytes
/// cannot be planned (overflow, a backend whose lane staging refuses): the
/// plan then keeps the baseline split and reports the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpLaneRequest {
    pub lanes: usize,
    /// Staged row budget (`ep_lane_row_budget`).
    pub row_budget: usize,
    pub bytes: Result<u64, String>,
}

/// A lane reservation the plan applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpLaneReservation {
    pub lanes: usize,
    pub row_budget: usize,
    /// Device bytes of one rank's lane store held back from the experts.
    pub lane_bytes: u64,
    /// Resident expert layers without the reservation.
    pub baseline_layers: usize,
}

/// The expert VRAM split an EP load applies, and the lane reservation (if
/// any) behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpVramPlan {
    /// Resident MoE layers every rank applies.
    pub layers: usize,
    /// The reservation `layers` already accounts for (`None`: none was
    /// requested, or it was impossible and `unreserved` says why).
    pub reservation: Option<EpLaneReservation>,
    /// Why a requested reservation was not applied; the baseline split is
    /// kept and the later lane staging keeps its own refusal.
    pub unreserved: Option<String>,
}

/// [`ep_vram_layers`] and, when `lane` asks for it, the same split with the
/// lane store reserved on every rank ([`ep_vram_layers_with_lanes`]). Without
/// a request the plan is exactly the baseline split.
pub fn ep_vram_plan(
    config: &Qwen4Config,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    lane: Option<EpLaneRequest>,
) -> EpVramPlan {
    let baseline = || {
        ep_vram_layers(
            config,
            free_per_rank,
            non_expert_bytes,
            layer_expert_bytes_per_rank,
            reserve_bytes,
        )
    };
    let Some(lane) = lane else {
        return EpVramPlan {
            layers: baseline(),
            reservation: None,
            unreserved: None,
        };
    };
    let planned = lane.bytes.and_then(|bytes| {
        ep_vram_layers_with_lanes(
            config,
            free_per_rank,
            non_expert_bytes,
            layer_expert_bytes_per_rank,
            reserve_bytes,
            bytes,
        )
    });
    match planned {
        Ok(split) => EpVramPlan {
            layers: split.reserved_layers,
            reservation: Some(EpLaneReservation {
                lanes: lane.lanes,
                row_budget: lane.row_budget,
                lane_bytes: split.lane_bytes,
                baseline_layers: split.baseline_layers,
            }),
            unreserved: None,
        },
        Err(reason) => EpVramPlan {
            layers: baseline(),
            reservation: None,
            unreserved: Some(reason),
        },
    }
}

/// The one load log line of an EP lane reservation: the trade
/// (`lanes`, `max_seq`, bytes per rank, resident expert layers before and
/// after), or why the reservation was not applied. `None` when no
/// reservation was requested.
pub fn ep_lane_reservation_line(max_seq: usize, plan: &EpVramPlan) -> Option<String> {
    if let Some(reservation) = &plan.reservation {
        return Some(format!(
            "[daemon] qwen4 EP lane reservation: lanes={} max_seq={max_seq} -> {:.3} GiB/rank; resident expert layers {} -> {}",
            reservation.lanes,
            reservation.lane_bytes as f64 / GIB,
            reservation.baseline_layers,
            plan.layers
        ));
    }
    plan.unreserved.as_ref().map(|reason| {
        format!(
            "[daemon] qwen4 EP lane reservation skipped: {reason}; keeping the baseline plan (resident expert layers {})",
            plan.layers
        )
    })
}

/// One rank's line of the expert-parallel fit table: free VRAM, the
/// replicated trunk, the rank's routed-expert bytes (`MoE layers` of
/// `layer_expert_bytes_per_rank` each), the split applied (`applied_layers`
/// of the MoE layers resident, the rest pinned host RAM; the MTP layer is the
/// last), what this rank alone could keep, the reserve and the predicted
/// headroom. `layer_expert_bytes_per_rank` is the largest rank's layer bytes
/// (`max_local_count` experts), as every rank budgets it.
pub fn ep_fit_rank_line(
    config: &Qwen4Config,
    rank: usize,
    ranks: usize,
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    applied_layers: usize,
) -> String {
    let moe_layers = moe_layer_count(config);
    let applied = applied_layers.min(moe_layers);
    let own = ep_rank_layers(
        config,
        free_vram,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserve_bytes,
    );
    let resident_bytes = layer_expert_bytes_per_rank.saturating_mul(applied as u64);
    let headroom = i128::from(free_vram)
        - i128::from(non_expert_bytes)
        - i128::from(resident_bytes)
        - i128::from(reserve_bytes);
    let gib = |bytes: u64| bytes as f64 / GIB;
    format!(
        "qwen4 EP rank {rank}/{ranks}: free {:.2} GiB | trunk {:.2} GiB | experts {:.2} GiB/rank \
         ({moe_layers} MoE layers x {:.1} MiB) | resident expert layers {applied} of {moe_layers} \
         ({} host-mapped; this rank alone fits {own}) | reserve {:.2} GiB | headroom {:+.2} GiB",
        gib(free_vram),
        gib(non_expert_bytes),
        gib(layer_expert_bytes_per_rank.saturating_mul(moe_layers as u64)),
        layer_expert_bytes_per_rank as f64 / (1u64 << 20) as f64,
        moe_layers - applied,
        gib(reserve_bytes),
        headroom as f64 / GIB,
    )
}

/// The expert-parallel fit table for `ranks` cards (`free_per_rank[r]` is rank
/// `r`'s free VRAM): a header naming the context and chunk the reserve was
/// sized for and the split every rank applies ([`ep_vram_layers`], the minimum
/// over ranks), then one [`ep_fit_rank_line`] per rank at that split.
/// `non_expert_bytes` is the replicated trunk, `layer_expert_bytes_per_rank`
/// one MoE layer's routed experts on the largest rank (`resident_split_ranks`),
/// `reserve_bytes` ([`auto_vram_reserve`]) what `auto` keeps free.
pub fn ep_fit_report(
    config: &Qwen4Config,
    ranks: usize,
    free_per_rank: &[u64],
    non_expert_bytes: u64,
    layer_expert_bytes_per_rank: u64,
    reserve_bytes: u64,
    max_seq: usize,
    chunk_rows: usize,
) -> String {
    let split = ep_vram_layers(
        config,
        free_per_rank,
        non_expert_bytes,
        layer_expert_bytes_per_rank,
        reserve_bytes,
    );
    let mut report = format!(
        "qwen4 EP fit: {ranks} ranks, max_seq {max_seq}, prefill chunk {chunk_rows} rows, \
         VRAM split {split} of {} MoE layers on every rank (minimum over ranks)",
        moe_layer_count(config)
    );
    for (rank, free) in free_per_rank.iter().enumerate() {
        report.push('\n');
        report.push_str(&ep_fit_rank_line(
            config,
            rank,
            ranks,
            *free,
            non_expert_bytes,
            layer_expert_bytes_per_rank,
            reserve_bytes,
            split,
        ));
    }
    report
}

/// Refuse before any allocation when the pinned experts would not leave
/// [`HOST_RAM_HEADROOM_BYTES`] of the host RAM they can take: `mem_available`
/// (`MemAvailable - CMA`; CMA pageblocks serve only movable allocations)
/// plus `ttm_pool_bytes`, the estimate of freed GTT pages parked in TTM's
/// page pool ([`ttm_pool_estimate`]). Pinned pages cannot be reclaimed, so
/// over-committing them starves the rest of the host.
pub fn check_host_ram(
    host_bytes: u64,
    mem_available: Option<u64>,
    ttm_pool_bytes: u64,
) -> Result<(), String> {
    if host_bytes == 0 {
        return Ok(());
    }
    let Some(available) = mem_available else {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM, but MemAvailable is unreadable",
            host_bytes as f64 / GIB
        ));
    };
    let needed = host_bytes + HOST_RAM_HEADROOM_BYTES;
    if available.saturating_add(ttm_pool_bytes) < needed {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM plus {:.0} GiB headroom, but \
             MemAvailable - CMA is {:.1} GiB (plus {:.1} GiB estimated in TTM's page pool); free host \
             memory or keep more expert layers in VRAM ({EXPERT_VRAM_LAYERS_ENV})",
            host_bytes as f64 / GIB,
            HOST_RAM_HEADROOM_BYTES as f64 / GIB,
            available as f64 / GIB,
            ttm_pool_bytes as f64 / GIB
        ));
    }
    Ok(())
}

fn host_mapped_is_gtt() -> bool {
    std::env::var("HSA_USERPTR_FOR_PAGED_MEM").is_ok_and(|value| value.trim() == "0")
}

/// Host RAM held in TTM's page pool that the host-mapped experts can take
/// ([`rdna_compute::uma_memory::ttm_pool_bytes`]). amdgpu parks the
/// write-combined and uncached pages of freed GTT buffers there, up to
/// `page_pool_size` (half of RAM by default), so a Flash-Next process that
/// exits leaves its host-mapped experts' pages in it. A new GTT allocation
/// takes pages from the pool first, and TTM's shrinker frees it under memory
/// pressure (`echo 2 | sudo tee /proc/sys/vm/drop_caches` empties it at
/// once). 0 unless host-mapped memory is GTT-backed
/// (`HSA_USERPTR_FOR_PAGED_MEM=0`), or when sysfs or `/proc/meminfo` is
/// unreadable. Loads that hold a [`rdna_compute::Gpu`] use
/// [`ttm_pool_estimate_on`], which also counts the pool on a GTT-backed APU.
pub fn ttm_pool_estimate() -> u64 {
    if !host_mapped_is_gtt() {
        return 0;
    }
    rdna_compute::uma_memory::ttm_pool_bytes()
}

/// [`ttm_pool_estimate`] for a load on `gpu`. A GTT-backed unified-memory APU
/// (Strix Halo: a 1 GiB carve-out, the rest of the device pool is GTT) takes
/// every allocation, weights and scratch included, from GTT, so a previous
/// process's freed pages parked in TTM's page pool are available to it whether
/// or not `HSA_USERPTR_FOR_PAGED_MEM=0` is set (hip-bridge leaves it unset on
/// an APU, which ignores it). Without this count a reload after any exit
/// reads the ~80 GiB the last process parked as `MemAvailable` it does not
/// have.
pub fn ttm_pool_estimate_on(gpu: &rdna_compute::Gpu) -> u64 {
    if gpu.is_gtt_backed_apu() {
        return rdna_compute::uma_memory::ttm_pool_bytes();
    }
    ttm_pool_estimate()
}

/// GTT room on the host: TTM's cap and what amdgpu devices already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GttBudget {
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

/// The GTT budget when host-mapped memory is GTT-backed, i.e. when
/// `HSA_USERPTR_FOR_PAGED_MEM` is `0` (hip-bridge's default). It is `None`
/// under userptr, or when sysfs does not expose TTM's limit. `used_bytes`
/// sums `mem_info_gtt_used` over every amdgpu device.
pub fn gtt_budget() -> Option<GttBudget> {
    if !host_mapped_is_gtt() {
        return None;
    }
    let limit_bytes = read_sysfs_u64(TTM_PAGES_LIMIT.as_ref())?.checked_mul(TTM_PAGE_BYTES)?;
    Some(GttBudget {
        limit_bytes,
        used_bytes: amdgpu_gtt_used_bytes()?,
    })
}

/// Refuse before any allocation when GTT-backed host-mapped experts would not
/// fit under TTM's `pages_limit` beside what amdgpu devices already hold.
/// Past it `hipHostMalloc` fails, after the load has uploaded the VRAM
/// weights. `None` (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget {
        limit_bytes,
        used_bytes,
    }) = budget
    else {
        return Ok(());
    };
    let free = limit_bytes.saturating_sub(used_bytes);
    if host_bytes > free {
        return Err(format!(
            "routed experts need {:.1} GiB of GTT-backed host RAM, but TTM's GTT cap \
             ({TTM_PAGES_LIMIT} = {} pages, {:.1} GiB) has {:.1} GiB left beside the {:.1} GiB \
             amdgpu devices already hold; keep more expert layers in VRAM \
             ({EXPERT_VRAM_LAYERS_ENV}), raise ttm.pages_limit, or set \
             HSA_USERPTR_FOR_PAGED_MEM=1 for pageable userptr host memory, which host-memory \
             pressure can stall (ROCm/rocm-systems#12528)",
            host_bytes as f64 / GIB,
            limit_bytes / TTM_PAGE_BYTES,
            limit_bytes as f64 / GIB,
            free as f64 / GIB,
            used_bytes as f64 / GIB
        ));
    }
    Ok(())
}

/// Everything [`placement_inputs`] reads: the load's manifest and payload
/// sizes, the state and scratch plan it admitted, and the context geometry.
pub struct PlacementInputsArgs<'a> {
    pub gpu: &'a rdna_compute::Gpu,
    pub config: &'a Qwen4Config,
    pub weights: &'a [WeightEntry],
    pub bytes_of: &'a dyn Fn(&WeightEntry) -> Option<u64>,
    pub backend: crate::Qwen4KvBackend,
    /// Unified memory (the context is charged as committed, not legacy).
    pub use_ranges: bool,
    pub max_seq: usize,
    pub vmm_granularity: usize,
    /// The native MTP head stays attached after the placement
    /// (`native_mtp` and the host-mapped MTP policy kept it).
    pub mtp_kept: bool,
    pub max_k: usize,
    /// The profile has routed experts (not a dense FFN).
    pub routed_experts: bool,
    pub state_format: crate::Qwen4StateFormat,
    pub scratch_mode: Qwen4ForwardScratchMode,
    /// Every rank's routed-expert span, from the load's seal.
    pub spans: &'a EpExpertSpans,
    /// The chunk the reserve is sized from; never probes the host link here.
    pub chunk_rows: usize,
}

/// What the expert placement is sized from, for one rank of a `ranks`-way load.
#[derive(Clone, Copy, Debug)]
pub struct PlacementInputs {
    /// Replicated non-expert bytes.
    pub non_expert: u64,
    /// One MoE layer's routed-expert bytes on this rank (the largest layer).
    pub layer_experts: u64,
    pub reserve: AutoReserve,
    pub mtp_bytes: Option<u64>,
    pub gather_bytes: Option<u64>,
}

/// The `auto` reserve ([`AutoReserve`]: the larger of the historical measured
/// formula and a bottom-up census of the first full-context request,
/// itemised; the census alone where it carries the measured HIP runtime
/// overhead, [`runtime_overhead_bytes`]) with its native MTP and QSA gather
/// parts. The one place this is computed: the product loader, the
/// expert-parallel split and every example/gate that asks the S1 admission for
/// a profile read it, so there is one convention.
pub fn placement_inputs(args: &PlacementInputsArgs<'_>) -> Result<PlacementInputs, String> {
    placement_inputs_with_forward(args, None)
}

/// The census `(f16_shadows, shared_i8_shadows)` terms of a load: the charged
/// bytes, or `(0, 0)` when `shadows_borrowed > 0` (the shadows live in the
/// donor tail and are never resident: [`DecodeShareForward::shadows_borrowed`]).
pub fn census_shadow_terms(f16: u64, shared_i8: u64, shadows_borrowed: u64) -> (u64, u64) {
    if shadows_borrowed > 0 {
        (0, 0)
    } else {
        (f16, shared_i8)
    }
}

/// The forward charge of an admitted decode-share `on` load: the dedicated
/// bytes (resident bank, persistent token ids, speculative buffers) replace
/// `qwen4_forward_device_bytes` in BOTH the census and the historical floor,
/// and the prefill scratch spans `borrowed` bytes of the donor tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeShareForward {
    /// [`crate::gpu_forward::qwen4_decode_share_dedicated_bytes`].
    pub dedicated: u64,
    /// [`crate::gpu_forward::qwen4_decode_share_borrow_bytes`].
    pub borrowed: u64,
    /// Bytes of the weight-shadow region ([`crate::decode_share_shadows`])
    /// that live at the start of the donor tail instead of resident VRAM: the
    /// census `f16_shadows` and `shared_i8_shadows` terms are not charged. 0
    /// keeps the shadows resident (and charged) as without decode-share.
    pub shadows_borrowed: u64,
}

/// [`placement_inputs`] charging `share` as the forward term (`None` is
/// byte-identical to [`placement_inputs`]).
pub fn placement_inputs_with_forward(
    args: &PlacementInputsArgs<'_>,
    share: Option<DecodeShareForward>,
) -> Result<PlacementInputs, String> {
    let PlacementInputsArgs {
        gpu,
        config,
        weights,
        bytes_of,
        backend,
        use_ranges,
        max_seq,
        vmm_granularity,
        mtp_kept,
        max_k,
        routed_experts,
        state_format,
        scratch_mode,
        spans,
        chunk_rows,
    } = *args;
    let qsa_format = state_format.qsa;
    let (non_expert, layer_experts) = resident_split_ranks(weights, bytes_of, spans)
        .map_err(|error| format!("qwen4: {error}"))?;
    // `chunk_rows` is the caller's: it never probes the host link here.
    let context = crate::Qwen4ContextCommit::for_expert_reserve(
        backend,
        use_ranges,
        max_seq,
        chunk_rows,
        vmm_granularity,
    );
    // The context the census charges: the ACTUAL backend with the whole
    // admitted context mapped (a VMM arena maps ahead to `max_seq`, rounded
    // to the driver granularity), not the legacy-forced layout the historical
    // formula charges on a discrete card.
    let census_context = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, vmm_granularity);
    let mtp_for = |context: &crate::Qwen4ContextCommit| -> Result<Option<u64>, String> {
        if !mtp_kept {
            return Ok(None);
        }
        let head = language_head_dtype(weights).ok_or("qwen4: manifest has no language head")?;
        let drafts = crate::mtp_spec::native_mtp_proposal_drafts(max_k, &gpu.arch);
        let row_capture =
            crate::mtp_spec::native_mtp_capture(gpu, config, state_format.gdn, drafts);
        Ok(Some(
            crate::mtp_spec::native_mtp_device_bytes(
                config,
                context,
                chunk_rows,
                drafts,
                head,
                row_capture,
                crate::mtp_spec::native_mtp_fill_scratch(gpu),
            )
            .ok_or("qwen4: native MTP device bytes overflow")?,
        ))
    };
    // The gathered QSA prefill attention's workspace, committed as the
    // context is (none when the route is off: the reserve is then unchanged).
    // Bytes already committed by an earlier load in this process are out of
    // `free` and only growth past them is charged.
    let gather_for = |context: &crate::Qwen4ContextCommit| -> Result<Option<u64>, String> {
        rdna_compute::tensor_ops::qsa_gather_workspace_needed(
            gpu,
            qsa_format,
            state_format.qsa_f16_arena,
        )
        .then(|| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                context.tokens,
            )
            .map(|bytes| (bytes as u64).saturating_sub(gpu.qsa_gather_scratch_bytes() as u64))
            .ok_or("qwen4: QSA gather scratch size overflows".to_string())
        })
        .transpose()
    };
    let mtp_bytes = mtp_for(&context)?;
    let gather_bytes = gather_for(&context)?;
    // The measured HIP runtime overhead (exact gfx1201, discrete,
    // single-device): the census then carries it and the historical floor,
    // whose stale F32 context used to cover it, is retired.
    let runtime = runtime_overhead_bytes(&gpu.arch, use_ranges, spans.ranks());
    let historical = if runtime.is_some() {
        None
    } else {
        Some(
            auto_vram_reserve_with_forward(
                config,
                &context,
                chunk_rows,
                scratch_mode,
                state_format.layout(),
                mtp_bytes,
                gather_bytes,
                share.map(|share| share.dedicated),
            )
            .map_err(|error| format!("qwen4: {error}"))?,
        )
    };
    // The bottom-up census at the full admitted context.
    let state = state_census_with_forward(
        config,
        state_format,
        &census_context,
        chunk_rows,
        scratch_mode,
        share.map(|share| share.dedicated),
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    // Lazy BF16 -> F16 weight shadows of the owners the long-prefill F16 WMMA
    // arms read: the cold bound from the manifest, never the warm cache. The
    // capability is the runtime's own predicate (`Gpu::qwen4_f16_shadow_capable`,
    // the half of `qwen4_f16_wmma_applies` that does not depend on a call) and
    // the rows are the largest `route_rows` a prefill forward of this chunk can
    // reach ([`f16_shadow_route_rows`]), so census and runtime cannot disagree.
    let f16_shadows = f16_shadow_charge(
        gpu.qwen4_f16_shadow_capable(),
        f16_shadow_route_rows(&gpu.arch, chunk_rows),
        weights,
        config.num_hidden_layers,
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    // The shared-expert INT8 route's pool-allocated shadows (`Gpu::qwen4_shared_i8_shadow`,
    // built on the first prefill the route takes): the runtime's own enable predicate and the
    // same route-rows threshold as the F16 route they replace.
    let shared_i8_shadows = shared_i8_charge(
        gpu.qwen4_shared_i8_enabled(),
        f16_shadow_route_rows(&gpu.arch, chunk_rows),
        weights,
        config.num_hidden_layers,
    )
    .map_err(|error| format!("qwen4: {error}"))?;
    // A decode-share load that keeps the shadows in the donor tail
    // ([`DecodeShareForward::shadows_borrowed`]) never holds them resident:
    // they are rebuilt into the donor tail per borrow epoch, so neither term
    // is charged (the hot budget grows by that much).
    let shadows_borrowed = share.map_or(0, |share| share.shadows_borrowed);
    let (f16_shadows, shared_i8_shadows) =
        census_shadow_terms(f16_shadows, shared_i8_shadows, shadows_borrowed);
    // The kernels' pool workspaces at the largest requested chunk, exactly
    // (every short-tail remainder included). The dense GDN scan's scratch is
    // charged wherever `fn_gdn_dense::applies` can hold at load: exact gfx1201,
    // Q8 or Q8Ef GDN state (`qwen4_dense_gdn_admits`) and the pinned
    // 16-key/48-value-head, 128-dim, 10240-wide geometry (its opt-out flag
    // only lowers the charge, so it is ignored). The census charges A for both
    // formats and the thread-local EF scratch only for plain Q8; a Q8Ef live
    // residual is charged once by the state owner, not here.
    let dense_gdn = rdna_compute::scratch::qwen4_dense_gdn_admits(&gpu.arch, state_format.gdn)
        && config.linear_num_key_heads == 16
        && config.linear_num_value_heads == 48
        && config.linear_key_head_dim == 128
        && config.linear_value_head_dim == 128
        && 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
            == 10240;
    // The routed-expert I4 MoE terms exist only for a profile whose routed
    // experts run the grouped I4 route; a dense or synthetic profile
    // (no experts, or dimensions that route cannot take) charges none of
    // them and is never refused for them.
    let routed_i4 = routed_experts
        && config.num_experts > 0
        && config.num_experts_per_tok > 0
        && config.moe_intermediate_size > 0
        && config.hidden_size % 256 == 0
        && config.moe_intermediate_size % 128 == 0;
    let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
        hidden_size: config.hidden_size,
        moe_intermediate_size: config.moe_intermediate_size,
        num_experts: config.num_experts,
        top_k: config.num_experts_per_tok,
        hc_count: config.hc_count,
        hc_lowrank: config.hc_lowrank,
        gdn_format: dense_gdn.then_some(state_format.gdn),
        routed_i4,
        include_short_tails: true,
    };
    // Overflow (or a dimension an owning allocator rejects) refuses the load
    // with a clear error; there is no historical-only fallback.
    let workspace = rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk_rows, &dims)
        .ok_or("qwen4: prefill workspace bytes cannot be computed (arithmetic overflow, or a hidden/MoE dimension the kernel allocators reject)")?;
    let census = ReserveCensus {
        qsa_context: state.qsa_context,
        state: state.state,
        forward: state.forward,
        f16_shadows,
        shared_i8_shadows,
        workspace,
        // The split QSA selector's score/filter scratch (0 unless its route
        // can run).
        qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_scratch_bytes(
            gpu,
            state_format.layout().index_dtype(),
        ) as u64,
        mtp: mtp_for(&census_context)?.unwrap_or(0),
        gather: gather_for(&census_context)?.unwrap_or(0),
        runtime: runtime.unwrap_or(0),
    };
    let reserve =
        AutoReserve::new(historical, census).map_err(|error| format!("qwen4: {error}"))?;
    let reserve = match share {
        Some(share) => reserve
            .with_decode_share(share.borrowed)
            .with_decode_share_shadows(share.shadows_borrowed),
        None => reserve,
    };
    Ok(PlacementInputs {
        non_expert,
        layer_experts,
        reserve,
        mtp_bytes,
        gather_bytes,
    })
}

/// The real artifact's rank-2 BF16 tensors (name, rows, cols) from its
/// model-index metadata, no payload: the fixture every shadow-census test
/// reads. 547 tensors, 291 of them F16-shadow owners.
#[cfg(test)]
pub(crate) fn j1778_bf16_manifest() -> Vec<WeightEntry> {
    include_str!("../tests/fixtures/j1778_bf16_owners.tsv")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split('\t');
            let name = fields.next().expect("tensor name");
            let rows: usize = fields.next().expect("rows").parse().expect("rows");
            let cols: usize = fields.next().expect("cols").parse().expect("cols");
            WeightEntry::model(
                name,
                vec![rows, cols],
                rdna_compute::DType::BF16,
                hipfire_runtime::weight_manifest::ShardPolicy::Replicate,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_forward::Qwen4ForwardScratchMode::{FnSymGfx1201, Full};
    use crate::weights::Qwen4EpPlacement;
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use rdna_compute::tensor_ops::QsaKvFormat::{self, F32};
    use rdna_compute::uma_memory::ttm_pool_estimate_from;
    use rdna_compute::DType;

    fn entry(name: &str, layer: usize) -> WeightEntry {
        let mut entry = WeightEntry::model(
            name,
            vec![4, 4, 256],
            DType::MQ4G256V2,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry
    }

    fn arena_entry(layer: usize, bytes: usize) -> WeightEntry {
        let mut entry = WeightEntry::model(
            format!("model.language_model.layers.{layer}.mlp.experts.slab_arena"),
            vec![bytes],
            DType::Raw,
            ShardPolicy::Replicate,
        );
        entry.layer = Some(layer);
        entry
    }

    #[test]
    fn slab_arenas_are_routed_experts_and_the_layer_charge_is_the_largest_arena() {
        // Non-uniform arenas (the hot set is not spread evenly): layer 0 is the
        // smallest, layer 2 the largest. `element_bytes` doubles the shape product.
        let mut weights = vec![
            arena_entry(0, 100),
            arena_entry(1, 300),
            arena_entry(2, 700),
            arena_entry(3, 200),
            entry("model.language_model.layers.0.self_attn.q_proj.weight", 0),
        ];
        assert!(weights[..4]
            .iter()
            .all(|entry| is_routed_expert(&entry.name)));
        let (non_expert, layer_experts) = resident_split(&weights, |entry| {
            Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
        })
        .expect("split");
        assert_eq!(
            layer_experts,
            700 * 2,
            "the charge is the max arena, never layer 0's"
        );
        assert_eq!(non_expert, 4 * 4 * 256 * 2);
        // Admission totals every arena's physical bytes.
        assert_eq!(
            routed_expert_bytes(&weights, |entry| Some(
                entry.logical_shape.iter().product::<usize>() as u64 * 2
            )),
            Ok((100 + 300 + 700 + 200) * 2)
        );
        // Per-rank shares of a max arena split the same way.
        let (_, per_rank) = resident_split_ranks(
            &weights,
            |entry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2),
            &stride_spans(4, 2),
        )
        .expect("split");
        assert_eq!(per_rank, 700);
        // Layers past the budget host-map their arenas.
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        assert_eq!(weights[0].residency, WeightResidency::Resident);
        assert_eq!(weights[1].residency, WeightResidency::Resident);
        assert_eq!(weights[2].residency, WeightResidency::HostMapped);
        assert_eq!(weights[3].residency, WeightResidency::HostMapped);
    }

    #[test]
    fn layers_past_the_budget_and_mtp_go_to_host() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry(
                "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
                1,
            ),
            entry("model.language_model.layers.2.mlp.experts.gate_up_proj", 2),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
        ];
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        let host: Vec<_> = weights
            .iter()
            .filter(|entry| entry.residency == WeightResidency::HostMapped)
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(
            host,
            [
                "model.language_model.layers.2.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj"
            ]
        );
    }

    #[test]
    fn place_routed_experts_set_keeps_exactly_the_listed_trunk_layers() {
        let manifest = || {
            let mut weights = vec![
                entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
                entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
                entry(
                    "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
                    1,
                ),
                entry("model.language_model.layers.2.mlp.experts.gate_up_proj", 2),
                entry("model.language_model.layers.3.mlp.experts.down_proj", 3),
                // The MTP layer is host whatever its layer index says.
                entry("mtp.layers.0.mlp.experts.down_proj", 0),
                entry("mtp.layers.2.mlp.experts.down_proj", 2),
                entry("model.language_model.layers.2.mlp.experts.up_proj", 2),
            ];
            // Already host-mapped entries are skipped, not counted again.
            weights[7].residency = WeightResidency::HostMapped;
            weights
        };
        let mut weights = manifest();
        assert_eq!(place_routed_experts_set(&mut weights, &[3, 0]), 4);
        let residency: Vec<_> = weights.iter().map(|entry| entry.residency).collect();
        assert_eq!(
            residency,
            [
                WeightResidency::Resident,
                WeightResidency::HostMapped,
                WeightResidency::Resident,
                WeightResidency::HostMapped,
                WeightResidency::Resident,
                WeightResidency::HostMapped,
                WeightResidency::HostMapped,
                WeightResidency::HostMapped,
            ]
        );
        // A leading set is exactly the count-based placement.
        let mut by_count = manifest();
        let mut by_set = manifest();
        assert_eq!(
            place_routed_experts_set(&mut by_set, &[0, 1, 2]),
            place_routed_experts(&mut by_count, 3)
        );
        assert_eq!(
            by_set
                .iter()
                .map(|entry| entry.residency)
                .collect::<Vec<_>>(),
            by_count
                .iter()
                .map(|entry| entry.residency)
                .collect::<Vec<_>>()
        );
        // The empty set host-maps every routed expert.
        let mut none = manifest();
        assert_eq!(place_routed_experts_set(&mut none, &[]), 6);
        assert_eq!(none[2].residency, WeightResidency::Resident);
    }

    #[test]
    fn hotset_host_mapped_bytes_matches_place_then_drop_trunk_routed() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.2.mlp.experts.down_proj", 2),
            entry(
                "model.language_model.layers.1.mlp.shared_expert.down_proj.weight",
                1,
            ),
            entry("mtp.layers.0.mlp.experts.gate_up_proj", 0),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
            entry("model.language_model.embed_tokens.weight", 0),
        ];
        weights[5].layer = None;
        weights[5].residency = WeightResidency::HostMapped;
        let bytes_of = |entry: &WeightEntry| Some(entry.name.len() as u64 * 1000);
        let before: Vec<_> = weights.iter().map(|entry| entry.residency).collect();
        for vram_layers in [0usize, 1, 3] {
            let got = hotset_host_mapped_bytes(&weights, bytes_of, vram_layers).unwrap();
            let mut placed = weights.clone();
            place_routed_experts(&mut placed, vram_layers);
            placed.retain(|entry| {
                !(is_routed_expert(&entry.name)
                    && !entry.name.starts_with("mtp.")
                    && entry.layer.is_some())
            });
            assert_eq!(
                got,
                host_mapped_bytes(&placed, bytes_of).unwrap(),
                "vram_layers={vram_layers}"
            );
        }
        // MTP experts and the preexisting host entry only; the input never mutates.
        let expected =
            (weights[3].name.len() + weights[4].name.len() + weights[5].name.len()) as u64 * 1000
                + 3 * HOST_MAPPED_PAD_BYTES;
        assert_eq!(
            hotset_host_mapped_bytes(&weights, bytes_of, 3),
            Ok(expected)
        );
        assert_eq!(
            before,
            weights
                .iter()
                .map(|entry| entry.residency)
                .collect::<Vec<_>>()
        );
        assert!(hotset_host_mapped_bytes(&weights, |_| None, 3).is_err());
        assert!(hotset_host_mapped_bytes(&weights, |_| Some(u64::MAX), 3).is_err());
    }

    #[test]
    fn expert_hotset_budget_subtracts_weights_reserve_tables_and_margin_checked() {
        assert_eq!(EXPERT_TIER_TABLE_BYTES, 2 * 24576 * 8 + 24576);
        assert_eq!(EXPERT_HOTSET_VRAM_MARGIN_BYTES, 512 << 20);
        let (free, non_expert, reserve) = (30u64 << 30, 5_364_000_000u64, 6_656u64 << 20);
        assert_eq!(
            expert_hotset_budget(free, non_expert, reserve).unwrap(),
            free - non_expert - reserve - EXPERT_TIER_TABLE_BYTES - EXPERT_HOTSET_VRAM_MARGIN_BYTES
        );
        // Exactly zero left is a (useless but valid) budget; one byte short is not.
        let exact =
            non_expert + reserve + EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES;
        assert_eq!(expert_hotset_budget(exact, non_expert, reserve).unwrap(), 0);
        assert_eq!(
            expert_hotset_budget(exact + 1, non_expert, reserve).unwrap(),
            1
        );
        assert!(expert_hotset_budget(exact - 1, non_expert, reserve).is_err());
        assert!(expert_hotset_budget(0, 1, 0).is_err());
        assert!(expert_hotset_budget(u64::MAX, u64::MAX, u64::MAX).is_err());
    }

    #[test]
    fn expert_hotset_budget_margin_alone_can_exhaust_the_budget() {
        // Everything but the margin fits; the margin shortfall must not underflow.
        let free = EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES - 1;
        let err = expert_hotset_budget(free, 0, 0).unwrap_err();
        assert!(err.contains("safety margin"), "{err}");
        // Tables fit but not the margin, with weights and reserve consuming the rest.
        assert!(expert_hotset_budget(1 << 20, 0, 0).is_err());
        assert_eq!(
            expert_hotset_budget(
                EXPERT_TIER_TABLE_BYTES + EXPERT_HOTSET_VRAM_MARGIN_BYTES,
                0,
                0
            )
            .unwrap(),
            0
        );
        // Full-range inputs do not overflow the chain.
        assert_eq!(
            expert_hotset_budget(u64::MAX, 0, 0).unwrap(),
            u64::MAX - EXPERT_TIER_TABLE_BYTES - EXPERT_HOTSET_VRAM_MARGIN_BYTES
        );
    }

    #[test]
    fn f16_arena_reserve_covers_full_capacity_even_with_vmm() {
        let config = crate::config::compact_test_config();
        let layout = rdna_compute::tensor_ops::QsaStateLayout {
            f16_arena: true,
            ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with =
                auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
            let extra = if backend == crate::Qwen4KvBackend::Vmm {
                2 * (262148usize * 1024).div_ceil(2 << 20) * (2 << 20) * config.n_full_layers()
            } else {
                262148 * layout.f16_arena_bytes_per_token(2, 256) * config.n_full_layers()
            };
            assert_eq!(with - plain, extra as u64);
        }
    }

    #[test]
    fn pooled_bf16_reserve_covers_full_capacity_even_with_vmm() {
        use hipfire_runtime::kv_backend::{
            KvChunkPlan, DEFAULT_KV_CHUNK_TOKENS, DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
        };
        let config = crate::config::compact_test_config();
        let layout = rdna_compute::tensor_ops::QsaStateLayout {
            pooled_bf16: true,
            ..rdna_compute::tensor_ops::QsaStateLayout::F32
        };
        let blocks = 262145usize.div_ceil(4);
        let block_bytes = layout.pooled_bf16_bytes_per_block(128);
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let context = crate::Qwen4ContextCommit::new(backend, 262145, 8192, 2 << 20);
            let plain = auto_vram_reserve(&config, &context, 8192, Full, F32, None, None).unwrap();
            let with =
                auto_vram_reserve(&config, &context, 8192, Full, layout, None, None).unwrap();
            let per_layer = if backend == crate::Qwen4KvBackend::Vmm {
                KvChunkPlan::new(
                    block_bytes,
                    blocks,
                    DEFAULT_KV_CHUNK_TOKENS,
                    2 << 20,
                    DEFAULT_VMM_PHYSICAL_CHUNK_BYTES,
                )
                .and_then(|plan| plan.mapped_bytes_for_tokens(blocks))
                .unwrap()
            } else {
                blocks * block_bytes
            };
            assert_eq!(with - plain, (per_layer * config.n_full_layers()) as u64);
        }
    }

    #[test]
    fn auto_fits_the_measured_gfx1201_card() {
        // R9700 before load: 32548 MiB free; Flash-Next non-expert weights
        // 5.364 GB; one trunk layer's routed experts 1.3369 GB.
        let free = 32548u64 << 20;
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let reserve =
            |chunk| auto_vram_reserve(&config, &measured, chunk, Full, F32, None, None).unwrap();
        // Verify-sized spec logits free 1472 logit rows of the measured
        // 1536-row layout: one layer past the measured N = 16 fits.
        let measured_chunk = reserve(AUTO_VRAM_RESERVE_CHUNK);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, measured_chunk),
            17
        );
        // gfx1201's 4096-row chunk spends that and one layer more on
        // forward scratch.
        let reserve = reserve(crate::gpu_forward::qwen4_prefill_chunk_default(
            "gfx1201",
            Full,
            || crate::host_link::HostLink::Unmeasured,
        ));
        assert!(reserve > measured_chunk);
        assert_eq!(
            auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48, reserve),
            15
        );
        // A card that holds everything keeps every layer resident.
        assert_eq!(
            auto_vram_layers(u64::MAX / 2, 5_364_000_000, 1_336_900_000, 48, reserve),
            48
        );
        // No room past the reserve places every expert in host RAM.
        assert_eq!(
            auto_vram_layers(8 << 30, 5_364_000_000, 1_336_900_000, 48, reserve),
            0
        );
    }

    /// The card-E record (daemon p6, 763 tokens): 32550 MiB free, 5115 MiB
    /// non-expert, 1060 MiB per slab layer, 7927 MiB historical reserve.
    const EVIDENCE: (u64, u64, u64, u64) = (32550 << 20, 5115 << 20, 1060 << 20, 7927 << 20);

    /// A census of the five terms given, a workspace of 153,256,072 B (the
    /// 4096-row target table, in its first term) and zero MTP and gather.
    fn census_with(
        qsa_context: u64,
        state: u64,
        forward: u64,
        f16_shadows: u64,
        qsa_selector: u64,
    ) -> ReserveCensus {
        let mut workspace = [("workspace", 0u64); WORKSPACE_TERM_COUNT];
        workspace[0].1 = 153_256_072;
        ReserveCensus {
            qsa_context,
            state,
            forward,
            f16_shadows,
            shared_i8_shadows: 0,
            workspace,
            qsa_selector,
            mtp: 0,
            gather: 0,
            runtime: 0,
        }
    }

    /// The target profile's actual state formats: fp8 QSA K/V, Q8 GDN.
    fn target_state_format() -> crate::Qwen4StateFormat {
        crate::Qwen4StateFormat {
            qsa: QsaKvFormat::Fp8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        }
    }

    /// The shared-expert INT8 shadows are their own census term: present (48 layers x
    /// (20 x 1280 + 5 x 2560) blocks of 136 B) when the route is enabled at a route-rows
    /// threshold chunk, absent with the route off (`HIPFIRE_QWEN4_SHARED_INT8=0`, or a non-gfx1201
    /// device) or below the threshold, and part of the census total and its log line.
    #[test]
    fn shared_i8_shadows_are_a_census_term_only_when_the_route_is_on() {
        let weights = target_f16_manifest();
        let per_layer = (20 * 1280 + 5 * 2560) * 136u64;
        let on = shared_i8_charge(true, 16384, &weights, 48).unwrap();
        assert_eq!(on, 48 * per_layer);
        assert_eq!(shared_i8_shadow_bytes(&weights, 48).unwrap(), on);
        assert_eq!(shared_i8_charge(false, 16384, &weights, 48).unwrap(), 0);
        assert_eq!(shared_i8_charge(true, 511, &weights, 48).unwrap(), 0);
        assert!(rdna_compute::shared_i8::shared_i8_route_enabled(
            true, "gfx1201"
        ));
        assert!(!rdna_compute::shared_i8::shared_i8_route_enabled(
            false, "gfx1201"
        ));
        assert!(!rdna_compute::shared_i8::shared_i8_route_enabled(
            true, "gfx1100"
        ));
        let mut census = census_with(1, 2, 3, 4, 5);
        let base = census.total().unwrap();
        census.shared_i8_shadows = on;
        assert_eq!(census.total().unwrap(), base + on);
        assert!(census.describe().contains("INT8 shared-expert shadows"));
    }

    /// The 48-layer target's F16-shadowed owners (BF16, rank 2): the final
    /// and every layer's two hyper-connection down mixes (97), each router
    /// and shared gate/up/down (48 + 144), and the PLE layer's key/value.
    fn target_f16_manifest() -> Vec<WeightEntry> {
        let layer = |i: usize, suffix: &str| format!("model.language_model.layers.{i}.{suffix}");
        let mut weights = vec![bf16(
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            &[320, 10240],
        )];
        for i in 0..48 {
            weights.push(bf16(
                &layer(i, "attn_hyper_connection.input_mix_weight_down.weight"),
                &[320, 10240],
            ));
            weights.push(bf16(
                &layer(i, "mlp_hyper_connection.input_mix_weight_down.weight"),
                &[320, 10240],
            ));
            weights.push(bf16(&layer(i, "mlp.gate.weight"), &[512, 2560]));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.gate_proj.weight"),
                &[640, 2560],
            ));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.up_proj.weight"),
                &[640, 2560],
            ));
            weights.push(bf16(
                &layer(i, "mlp.shared_expert.down_proj.weight"),
                &[2560, 640],
            ));
        }
        weights.push(bf16(&layer(2, "ple.key_proj.weight"), &[6400, 2560]));
        weights.push(bf16(&layer(2, "ple.value_proj.weight"), &[6400, 2560]));
        weights
    }

    /// The full-context census of the target profile on a VMM discrete card
    /// at `max_seq` (native MTP off), every term from its owning query, for a
    /// prefill `chunk` of rows, the forward `mode` it allocates and a
    /// `runtime` overhead term. The F16 shadows are charged only at chunks of
    /// at least [`rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`] rows, as
    /// the loader does.
    fn census_at(
        config: &Qwen4Config,
        max_seq: usize,
        chunk: usize,
        mode: Qwen4ForwardScratchMode,
        runtime: u64,
    ) -> ReserveCensus {
        let format = target_state_format();
        let context =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, max_seq, 2 << 20);
        let state = state_census(config, format, &context, chunk, mode).unwrap();
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            gdn_format: Some(rdna_compute::tensor_ops::GdnStateFormat::Q8),
            routed_i4: true,
            include_short_tails: true,
        };
        let f16_shadows = if chunk >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS {
            f16_shadow_bytes(&target_f16_manifest(), config.num_hidden_layers).unwrap()
        } else {
            0
        };
        ReserveCensus {
            qsa_context: state.qsa_context,
            state: state.state,
            forward: state.forward,
            f16_shadows,
            shared_i8_shadows: 0,
            workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap(),
            qsa_selector: rdna_compute::tensor_ops::qsa_bf16_split_capacity_bytes() as u64,
            mtp: 0,
            gather: rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                max_seq,
            )
            .unwrap() as u64,
            runtime,
        }
    }

    /// The full-context census at the default 4096-row chunk, `Full` forward,
    /// no runtime term: what the census was before the runtime overhead.
    fn real_census(config: &Qwen4Config, max_seq: usize) -> ReserveCensus {
        census_at(
            config,
            max_seq,
            crate::gpu_forward::qwen4_prefill_chunk_default("gfx1201", Full, || {
                crate::host_link::HostLink::Unmeasured
            }),
            Full,
            0,
        )
    }

    /// The 7900 XTX `qwen4_kld eval` pilot (hipx `iu4kldP-1-wt2-iu4-p1`): `n_ctx` 2048 gives a 2048-row
    /// chunk and `auto` placed layers 0..10. The tool then allocates a 2048 x 248320 F32 logits buffer
    /// (1940 MiB) the product never holds; the first prefill ran out of device memory with only 210 MiB
    /// left beyond the reserve and the safety margin. The placement must leave that buffer free.
    #[test]
    fn kld_all_row_logits_are_left_free_by_the_auto_placement() {
        const MIB: u64 = 1 << 20;
        let config = crate::config::compact_test_config();
        let (max_seq, chunk) = (2048usize, 2048usize);
        let base = auto_vram_reserve(
            &config,
            &crate::Qwen4ContextCommit::legacy(max_seq),
            chunk,
            Full,
            target_state_format().layout(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            base / MIB,
            5788,
            "the pilot's historical reserve at a 2048-row chunk"
        );
        // The pilot's measured inputs (the same card's daemon placement lines).
        let (free, non_expert, layer, routed) = (24376 * MIB, 5115 * MIB, 1275 * MIB, 62475 * MIB);
        let place = |lab: u64| {
            incumbent_placement_layers(
                free,
                non_expert,
                routed,
                layer,
                48,
                reserve_with_lab_bytes(base, lab).unwrap(),
                false,
            )
            .unwrap()
        };
        let logits = all_row_logits_bytes(2048, config.vocab_size).unwrap();
        assert_eq!(logits, 2_034_237_440);
        assert_eq!(logits / MIB, 1940);
        // Without the term: ten layers, leaving only 210 MiB over the reserve and margin.
        let uncharged = place(0).unwrap();
        assert_eq!(uncharged, 10);
        let left = |layers: usize| free - non_expert - layers as u64 * layer - base - 512 * MIB;
        assert_eq!(left(uncharged) / MIB, 210);
        assert!(
            left(uncharged) < logits,
            "the pilot placement cannot hold the logits"
        );
        // With it: eight layers, and every layer count the placement returns leaves the buffer free.
        let charged = place(logits).unwrap();
        assert_eq!(charged, 8);
        assert!(left(charged) >= logits);
        assert!(
            left(charged + 1) < logits,
            "the placement keeps the largest fitting count"
        );
        // The code24 corpus scores 512-row chunks: its logits are a quarter of that.
        assert_eq!(
            all_row_logits_bytes(512, config.vocab_size).unwrap(),
            logits / 4
        );
        assert_eq!(reserve_with_lab_bytes(base, 0).unwrap(), base);
        assert!(reserve_with_lab_bytes(u64::MAX, 1).is_err());
        assert!(all_row_logits_bytes(usize::MAX, 2).is_err());
    }

    /// Both reserve formulas read `qwen4_forward_device_bytes` as their forward term, once and
    /// additively; swapping it for the dedicated bank lowers the reserve by `F - D_bytes`.
    #[test]
    fn decode_share_dedicated_forward_lowers_both_reserves_by_exactly_the_delta() {
        let config = crate::config::compact_test_config();
        let (chunk, mode) = (16384usize, FnSymGfx1201);
        let forward = |rows, mode| {
            crate::gpu_forward::qwen4_forward_device_bytes(&config, rows, mode).unwrap()
        };
        let f = forward(chunk, mode);
        let d =
            crate::gpu_forward::qwen4_decode_share_dedicated_bytes(&config, chunk, mode).unwrap();
        let s = crate::gpu_forward::qwen4_decode_share_borrow_bytes(&config, chunk, mode).unwrap();
        let charge = DecodeShareCharge {
            borrow_bytes: s,
            dedicated_bytes: d,
            baseline_forward: f,
            overhead: 12_345,
        };
        assert!(d < f);
        assert_eq!(charge.reserve_delta(), f - d);
        assert_eq!(charge.expert_headroom(), f - d - 12_345);
        assert_eq!(
            DecodeShareCharge {
                overhead: u64::MAX,
                ..charge
            }
            .expert_headroom(),
            0
        );

        // auto_vram_reserve: reserve = base + forward(rows, mode), base independent of both.
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let reserve = |rows, mode| {
            auto_vram_reserve(&config, &measured, rows, mode, F32, None, None).unwrap()
        };
        let base = reserve(chunk, mode) - f;
        for (rows, mode) in [(4096, Full), (8192, FnSymGfx1201), (2048, FnSymGfx1201)] {
            assert_eq!(
                reserve(rows, mode) - forward(rows, mode),
                base,
                "{rows} rows"
            );
        }
        assert_eq!(reserve(chunk, mode) - (base + d), charge.reserve_delta());

        // state_census: its forward term is that same query.
        let ctx = crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, 16384, 16384, 2 << 20);
        let state = state_census(&config, target_state_format(), &ctx, chunk, mode).unwrap();
        assert_eq!(state.forward, f);

        // Census-only path (historical retired) and the historical path where the census wins.
        let census = census_at(&config, 16384, chunk, mode, 700 << 20);
        assert_eq!(census.forward, f);
        let dedicated = ReserveCensus {
            forward: d,
            ..census
        };
        for historical in [None, Some(1u64)] {
            let before = AutoReserve::new(historical, census).unwrap();
            let after = AutoReserve::new(historical, dedicated).unwrap();
            if historical.is_some() {
                assert_eq!(before.winner(), "census");
                assert_eq!(after.winner(), "census");
            }
            assert_eq!(
                before.bytes() - after.bytes(),
                charge.reserve_delta(),
                "historical {historical:?}"
            );
        }
    }

    /// The `*_with_forward` overrides substitute exactly the forward term:
    /// `None` is byte-identical to the ordinary functions; `Some(D)` lowers
    /// the historical floor and the census forward term by `F - D`, and the
    /// reserve's description names the dedicated charge.
    #[test]
    fn decode_share_forward_override_replaces_exactly_the_forward_term() {
        let config = crate::config::compact_test_config();
        let (chunk, mode) = (16384usize, FnSymGfx1201);
        let f = crate::gpu_forward::qwen4_forward_device_bytes(&config, chunk, mode).unwrap();
        let d =
            crate::gpu_forward::qwen4_decode_share_dedicated_bytes(&config, chunk, mode).unwrap();
        let s = crate::gpu_forward::qwen4_decode_share_borrow_bytes(&config, chunk, mode).unwrap();
        assert!(d < f);

        // Historical floor: `None` is the ordinary reserve, `Some(D)` is lower by F - D.
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let historical = |forward| {
            auto_vram_reserve_with_forward(
                &config, &measured, chunk, mode, F32, None, None, forward,
            )
            .unwrap()
        };
        let plain = auto_vram_reserve(&config, &measured, chunk, mode, F32, None, None).unwrap();
        assert_eq!(historical(None), plain);
        assert_eq!(plain - historical(Some(d)), f - d);

        // Census forward term.
        let ctx = crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, 16384, 16384, 2 << 20);
        let ordinary = state_census(&config, target_state_format(), &ctx, chunk, mode).unwrap();
        let none =
            state_census_with_forward(&config, target_state_format(), &ctx, chunk, mode, None)
                .unwrap();
        let some =
            state_census_with_forward(&config, target_state_format(), &ctx, chunk, mode, Some(d))
                .unwrap();
        assert_eq!(none, ordinary);
        assert_eq!((ordinary.forward, some.forward), (f, d));
        assert_eq!(
            (some.qsa_context, some.state),
            (ordinary.qsa_context, ordinary.state)
        );

        // Both expressions of the AutoReserve fall by exactly F - D (census-only
        // path and the historical path where the census wins); the description of
        // a decode-share reserve names the dedicated charge and the borrowed span.
        let census = census_at(&config, 16384, chunk, mode, 700 << 20);
        let dedicated = ReserveCensus {
            forward: d,
            ..census
        };
        for historical in [None, Some(1u64)] {
            let off = AutoReserve::new(historical, census).unwrap();
            let on = AutoReserve::new(historical, dedicated)
                .unwrap()
                .with_decode_share(s);
            assert_eq!(off.bytes() - on.bytes(), f - d, "historical {historical:?}");
            assert_eq!(off.decode_share_borrowed, None);
            assert_eq!(on.decode_share_borrowed, Some(s));
            assert_eq!(on.census.forward, d);
            let text = on.describe();
            assert!(
                text.contains(&format!(
                    "forward {} MiB (decode-share dedicated; {} MiB prefill scratch borrowed from the donor tail)",
                    d >> 20,
                    s >> 20
                )),
                "{text}"
            );
            // `off` reads exactly as before.
            assert!(!off.describe().contains("decode-share"));
            assert_eq!(
                off.describe(),
                AutoReserve::new(historical, census).unwrap().describe()
            );
        }
    }

    /// R9700 receipt (beta load log, arm a) replayed through the real Flash-Next plan sizes.
    #[test]
    fn decode_share_r9700_receipt_fixture() {
        use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
        use crate::expert_slabs::ExpertFormat;
        use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, Placement};

        const MIB: u64 = 1 << 20;
        const EXPERT: u64 = 2_611_200;
        let (chunk, mode) = (16384usize, FnSymGfx1201);
        let config = crate::config::compact_test_config();

        // Receipt.
        let free = 32_550 * MIB;
        let non_expert = 5_115 * MIB;
        let census_reserve = 10_393 * MIB;
        let receipt_forward_mib = 4_836u64;
        let base_tables = 417_792u64;
        let safety = 512 * MIB;
        let stage = 2_002 * MIB;
        let extra_tables = 393_216u64;
        let hot_budget = 14_525 * MIB;
        let hot_experts = 5_833u64;

        let f = crate::gpu_forward::qwen4_forward_device_bytes(&config, chunk, mode).unwrap();
        let f_mib = (f + MIB / 2) >> 20;
        eprintln!("F = {f} B ({f_mib} MiB rounded; receipt {receipt_forward_mib} MiB)");
        if f_mib != receipt_forward_mib {
            eprintln!("RECEIPT MISMATCH: forward term is {f_mib} MiB, not {receipt_forward_mib}");
        }
        // The receipt's budget chain: free - non-expert - census - safety - stage - tables.
        let chain =
            free - non_expert - census_reserve - safety - stage - base_tables - extra_tables;
        eprintln!(
            "receipt chain budget = {} B ({:.3} MiB) -> floor {} experts; receipt hot budget {} MiB = {hot_experts} experts ({} B)",
            chain,
            chain as f64 / MIB as f64,
            chain / EXPERT,
            hot_budget / MIB,
            hot_experts * EXPERT
        );
        assert_eq!(
            chain / EXPERT,
            hot_experts,
            "receipt chain disagrees with the receipt's expert count"
        );

        let d =
            crate::gpu_forward::qwen4_decode_share_dedicated_bytes(&config, chunk, mode).unwrap();
        let s = crate::gpu_forward::qwen4_decode_share_borrow_bytes(&config, chunk, mode).unwrap();
        let guard = ((crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64).max(256) + 255) & !255;
        let pointer_tables = 24_576u64 * 2 * 8;
        let o = guard + pointer_tables;
        let k = s.div_ceil(EXPERT);
        let charge = DecodeShareCharge {
            borrow_bytes: s,
            dedicated_bytes: d,
            baseline_forward: f,
            overhead: o,
        };
        let new_budget = hot_budget + charge.reserve_delta() - o;
        let delta_k = new_budget / EXPERT - hot_experts;
        eprintln!("D_bytes = {d} B ({:.3} MiB)", d as f64 / MIB as f64);
        eprintln!("S = {s} B ({:.3} MiB)", s as f64 / MIB as f64);
        eprintln!(
            "F - D_bytes = {} B ({:.3} MiB)",
            charge.reserve_delta(),
            charge.reserve_delta() as f64 / MIB as f64
        );
        eprintln!("O = {o} B (guard {guard} + pointer tables {pointer_tables})");
        eprintln!("K = ceil(S / {EXPERT}) = {k}");
        eprintln!(
            "new budget = {new_budget} B ({:.3} MiB); dK = {delta_k}; dK/K = {:.4}",
            new_budget as f64 / MIB as f64,
            delta_k as f64 / k as f64
        );
        let chain_new = chain + charge.reserve_delta() - o;
        eprintln!(
            "  (chain-derived budget instead of the rounded 14,525 MiB: dK = {})",
            chain_new / EXPERT - hot_experts
        );

        // Second scenario: F16 weight shadows 1,238 + INT8 shared-expert shadows 239 MiB.
        let shadows = 1_477 * MIB;
        let s2 = s + shadows;
        let k2 = s2.div_ceil(EXPERT);
        let new_budget2 = hot_budget + charge.reserve_delta() + shadows - o;
        let delta_k2 = new_budget2 / EXPERT - hot_experts;
        eprintln!(
            "with shadows: S = {s2} B ({:.3} MiB), K = {k2}, reserve delta = {} B, new budget = {new_budget2} B ({:.3} MiB), dK = {delta_k2}, dK/K = {:.4}",
            s2 as f64 / MIB as f64,
            charge.reserve_delta() + shadows,
            new_budget2 as f64 / MIB as f64,
            delta_k2 as f64 / k2 as f64
        );

        // Capacity classes and the ideal restore time at 56.9 GB/s (decimal).
        for class in crate::gpu_forward::qwen4_decode_share_classes(chunk) {
            let span =
                crate::gpu_forward::qwen4_decode_share_borrow_bytes(&config, class, mode).unwrap();
            eprintln!(
                "class {class:>5} rows: span_bytes = {span} B ({:.3} MiB), ideal restore at 56.9 GB/s = {:.3} ms",
                span as f64 / MIB as f64,
                span as f64 / 56.9e9 * 1e3
            );
        }

        // Kill assertions (verdict change 4).
        let reserve_fall = {
            let census = census_at(&config, 16384, chunk, mode, 0);
            let dedicated = ReserveCensus {
                forward: d,
                ..census
            };
            AutoReserve::new(None, census).unwrap().bytes()
                - AutoReserve::new(None, dedicated).unwrap().bytes()
        };
        assert_eq!(
            reserve_fall,
            f - d,
            "KILL: reserve falls by {reserve_fall} B, expected F - D_bytes = {} B",
            f - d
        );
        assert!(
            delta_k as f64 >= 0.8 * k as f64,
            "KILL: dK = {delta_k} < 0.8 * K = {:.1} (K = {k}, S = {s} B, F = {f} B, D_bytes = {d} B, O = {o} B)",
            0.8 * k as f64
        );
        assert!(
            delta_k2 as f64 >= 0.8 * k2 as f64,
            "KILL (shadows): dK = {delta_k2} < 0.8 * K = {:.1} (K = {k2})",
            0.8 * k2 as f64
        );

        // A real plan over uniform slabs at the new budget donates exactly K experts.
        let slabs: Vec<ExpertSlab> = (0..24_576usize)
            .map(|i| {
                let base = i * EXPERT as usize;
                ExpertSlab {
                    format: ExpertFormat::Mq4Xts,
                    gate_up: ExpertSlabSpan {
                        offset: base,
                        len: 1_740_800,
                    },
                    down: ExpertSlabSpan {
                        offset: base + 1_740_800,
                        len: 870_400,
                    },
                }
            })
            .collect();
        let rank: Vec<u32> = (0..24_576u64)
            .map(|k| ((k * 7919 + 13) % 24_576) as u32)
            .collect();
        let plan = plan_expert_tiers_from_slabs(&slabs, &rank, new_budget).unwrap();
        let resident = plan.device_experts() as u64;
        assert_eq!(resident, (new_budget - guard) / EXPERT);
        assert!(
            resident > k,
            "{resident} resident experts cannot donate {k}"
        );
        let (shared, tail) = plan.clone().with_decode_share(&rank, s).unwrap();
        assert_eq!(tail.donors.len() as u64, k);
        assert_eq!(tail.donor_bytes, k * EXPERT);
        assert!(tail.donor_bytes >= s && tail.donor_bytes - EXPERT < s);
        assert_eq!(shared.device_experts() as u64, resident);
        assert_eq!(tail.donor_offset, tail.guard_offset + guard);
        assert_eq!(
            shared.owner_bytes(Placement::Device),
            tail.donor_offset + tail.donor_bytes + guard
        );
        let window = shared.host_windows().last().unwrap();
        assert_eq!(window.spans[0].1, tail.mirror_offset);
        for (i, &id) in tail.donors.iter().enumerate() {
            let e = shared.entry(id as usize / 512, id as usize % 512).unwrap();
            assert_eq!(e.tier, Placement::Device);
            // Contiguous, back to back.
            assert_eq!(e.gate_up_offset, tail.donor_offset + i as u64 * EXPERT);
            assert_eq!(e.down_offset, e.gate_up_offset + 1_740_800);
            // Mirror equality: one copy of donor_bytes restores the tail.
            assert_eq!(
                window.spans[2 * i].1 - tail.mirror_offset,
                e.gate_up_offset - tail.donor_offset
            );
            assert_eq!(
                window.spans[2 * i + 1].1 - tail.mirror_offset,
                e.down_offset - tail.donor_offset
            );
            assert_eq!(tail.borrow_entries[id as usize].tier, Placement::HostMapped);
        }
        // Nothing else lives in or past the donor tail.
        for id in 0..24_576usize {
            let e = shared.entry(id / 512, id % 512).unwrap();
            if e.tier == Placement::Device && !tail.donors.contains(&(id as u32)) {
                assert!(e.down_offset + e.down_len <= tail.guard_offset);
            }
        }
        eprintln!(
            "plan: {resident} resident experts at the new budget, {} donors, guard at {}, donor tail [{}, {}), mirror at {}",
            tail.donors.len(),
            tail.guard_offset,
            tail.donor_offset,
            tail.donor_offset + tail.donor_bytes,
            tail.mirror_offset
        );
    }

    /// S3b: the donors each prefill class dirties on the real Flash-Next
    /// plan sizes at the R9700 chunk (16384 rows), printed as the
    /// `classes [rows:donors, ...]` load log reads them.
    #[test]
    fn decode_share_r9700_class_donor_table() {
        use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
        use crate::expert_slabs::ExpertFormat;
        use crate::expert_tier_plan::plan_expert_tiers_from_slabs;

        const MIB: u64 = 1 << 20;
        const EXPERT: u64 = 2_611_200;
        let (chunk, mode) = (16384usize, FnSymGfx1201);
        let config = crate::config::compact_test_config();
        let rows = crate::gpu_forward::qwen4_decode_share_classes(chunk);
        let spans =
            crate::gpu_forward::qwen4_decode_share_class_spans(&config, chunk, mode, 0).unwrap();
        assert_eq!(rows.len(), spans.len());
        assert_eq!(
            spans.last().copied(),
            crate::gpu_forward::qwen4_decode_share_borrow_bytes(&config, chunk, mode)
        );
        assert!(spans.windows(2).all(|pair| pair[0] < pair[1]));

        let slabs: Vec<ExpertSlab> = (0..24_576usize)
            .map(|i| {
                let base = i * EXPERT as usize;
                ExpertSlab {
                    format: ExpertFormat::Mq4Xts,
                    gate_up: ExpertSlabSpan {
                        offset: base,
                        len: 1_740_800,
                    },
                    down: ExpertSlabSpan {
                        offset: base + 1_740_800,
                        len: 870_400,
                    },
                }
            })
            .collect();
        let rank: Vec<u32> = (0..24_576u64)
            .map(|k| ((k * 7919 + 13) % 24_576) as u32)
            .collect();
        let plan = plan_expert_tiers_from_slabs(&slabs, &rank, 7_600 * EXPERT + (1 << 20)).unwrap();
        let need = *spans.last().unwrap();
        let (shared, tail) = plan.with_decode_share(&rank, need).unwrap();
        let k = tail.donors.len();
        assert_eq!(k as u64, need.div_ceil(EXPERT));
        let classed = shared.with_decode_share_classes(&spans).unwrap();
        let tail = classed.decode_share_tail().unwrap();
        // Uniform experts laid back to back from the tail start: class c dirties
        // ceil(span / EXPERT) donors, a partly overlapped donor included.
        for (class, &span) in spans.iter().enumerate() {
            assert_eq!(
                tail.class_donors[class] as u64,
                span.div_ceil(EXPERT),
                "class {} rows",
                rows[class]
            );
        }
        assert!(tail.class_donors.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(*tail.class_donors.last().unwrap(), k);
        let table: Vec<String> = rows
            .iter()
            .zip(&spans)
            .zip(&tail.class_donors)
            .map(|((rows, span), donors)| {
                format!("{rows}:{donors} ({:.0} MiB)", *span as f64 / MIB as f64)
            })
            .collect();
        eprintln!("classes [{}] donors dirty, K = {k}", table.join(", "));
    }

    /// The real Flash-Next weight shadows (291 F16 owners, 48 INT8 shared
    /// experts) borrowed at the start of the donor tail: the census terms are
    /// zero, the reserve falls by exactly their sum, every class span grows by
    /// the region, and every class dirties at least the donors the region spans.
    #[test]
    fn decode_share_shadows_borrowed_in_the_donor_tail() {
        use crate::decode_share_shadows::{plan_shadow_region, SHADOW_REGION_ALIGN};
        use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
        use crate::expert_slabs::ExpertFormat;
        use crate::expert_tier_plan::plan_expert_tiers_from_slabs;

        const MIB: u64 = 1 << 20;
        const EXPERT: u64 = 2_611_200;
        let (chunk, mode) = (16384usize, FnSymGfx1201);
        let config = crate::config::compact_test_config();
        let manifest = j1778_bf16_manifest();
        let layers = config.num_hidden_layers;

        // Census terms vs region.
        let f16 = f16_shadow_charge(true, chunk, &manifest, layers).unwrap();
        let i8 = shared_i8_charge(true, chunk, &manifest, layers).unwrap();
        let region = plan_shadow_region(&manifest, layers, true, true).unwrap();
        assert_eq!((region.f16_bytes, region.i8_bytes), (f16, i8));
        let shadow = region.total_bytes;
        assert_eq!(shadow % SHADOW_REGION_ALIGN, 0);
        eprintln!(
            "shadows: F16 {} MiB + INT8 {} MiB = {} MiB census ({} MiB region)",
            f16 >> 20,
            i8 >> 20,
            (f16 + i8) >> 20,
            shadow >> 20
        );
        assert!((1_400..1_560).contains(&((f16 + i8) >> 20)));

        // The census terms vanish; the reserve falls by exactly their sum (census-only
        // path and the historical path where the census wins).
        assert_eq!(census_shadow_terms(f16, i8, 0), (f16, i8));
        assert_eq!(census_shadow_terms(f16, i8, shadow), (0, 0));
        let mut census = census_at(&config, 16384, chunk, mode, 700 << 20);
        census.f16_shadows = f16;
        census.shared_i8_shadows = i8;
        let (zf, zi) = census_shadow_terms(f16, i8, shadow);
        let borrowed = ReserveCensus {
            f16_shadows: zf,
            shared_i8_shadows: zi,
            ..census
        };
        for historical in [None, Some(1u64)] {
            let before = AutoReserve::new(historical, census).unwrap();
            let after = AutoReserve::new(historical, borrowed)
                .unwrap()
                .with_decode_share(1 << 30)
                .with_decode_share_shadows(shadow);
            assert_eq!(
                before.bytes() - after.bytes(),
                f16 + i8,
                "historical {historical:?}"
            );
            let text = after.describe();
            assert!(
                text.contains(&format!(
                    "prefill scratch + {} MiB weight shadows borrowed from the donor tail",
                    shadow >> 20
                )),
                "{text}"
            );
            assert!(text.contains("F16 weight shadows 0 MiB"), "{text}");
        }

        // Class spans: scratch spans plus the region; the donors each class dirties.
        let plain =
            crate::gpu_forward::qwen4_decode_share_class_spans(&config, chunk, mode, 0).unwrap();
        let spans =
            crate::gpu_forward::qwen4_decode_share_class_spans(&config, chunk, mode, shadow)
                .unwrap();
        assert_eq!(plain.len(), spans.len());
        for (with, without) in spans.iter().zip(&plain) {
            assert_eq!(*with, without + shadow);
        }
        let slabs: Vec<ExpertSlab> = (0..24_576usize)
            .map(|i| {
                let base = i * EXPERT as usize;
                ExpertSlab {
                    format: ExpertFormat::Mq4Xts,
                    gate_up: ExpertSlabSpan {
                        offset: base,
                        len: 1_740_800,
                    },
                    down: ExpertSlabSpan {
                        offset: base + 1_740_800,
                        len: 870_400,
                    },
                }
            })
            .collect();
        let rank: Vec<u32> = (0..24_576u64)
            .map(|k| ((k * 7919 + 13) % 24_576) as u32)
            .collect();
        let plan = plan_expert_tiers_from_slabs(&slabs, &rank, 7_600 * EXPERT + (1 << 20)).unwrap();
        let plain_k = {
            let (_, tail) = plan
                .clone()
                .with_decode_share(&rank, *plain.last().unwrap())
                .unwrap();
            tail.donors.len() as u64
        };
        let need = *spans.last().unwrap();
        let (shared, tail) = plan.with_decode_share(&rank, need).unwrap();
        let k = tail.donors.len() as u64;
        assert_eq!(k, need.div_ceil(EXPERT));
        let shared = shared
            .with_decode_share_classes(&spans)
            .unwrap()
            .with_decode_share_shadows(shadow)
            .unwrap();
        let tail = shared.decode_share_tail().unwrap();
        assert_eq!(tail.shadow_bytes, shadow);
        let shadow_donors = shadow.div_ceil(EXPERT) as usize;
        for (class, &donors) in tail.class_donors.iter().enumerate() {
            assert!(
                donors >= shadow_donors,
                "every class dirties the shadow region's donors"
            );
            assert_eq!(donors as u64, spans[class].div_ceil(EXPERT));
        }
        // A region larger than any class span, or unaligned, is refused.
        assert!(shared
            .clone()
            .with_decode_share_shadows(spans[0] + SHADOW_REGION_ALIGN)
            .is_err());
        assert!(shared
            .clone()
            .with_decode_share_shadows(shadow + 1)
            .is_err());
        eprintln!(
            "K without shadows {plain_k}, with shadows {k} (+{} donors for {} MiB); classes {:?}",
            k - plain_k,
            shadow / MIB,
            tail.class_donors
        );
    }

    /// Placement of the reproduction card with the real census: the layers
    /// the unfixed planner kept (`historical` alone), what stage and margin
    /// on the historical floor leave, and what the census-backed reserve
    /// leaves.
    fn replay(
        free: u64,
        non_expert: u64,
        layer: u64,
        historical: u64,
        census: ReserveCensus,
    ) -> (usize, usize, usize, AutoReserve, PlacementCharges) {
        let reserve = AutoReserve::new(Some(historical), census).unwrap();
        let charges = PlacementCharges::new(layer, true).unwrap();
        let unfixed = auto_vram_layers(free, non_expert, layer, 49, historical);
        let floor_charged = auto_vram_layers(
            free,
            non_expert,
            layer,
            49,
            charges.reserve(historical).unwrap(),
        );
        let placed =
            auto_placement_layers(free, non_expert, layer, 49, reserve.bytes(), &charges).unwrap();
        (unfixed, floor_charged, placed, reserve, charges)
    }

    #[test]
    fn real_census_terms_come_from_their_owning_queries() {
        let config = crate::config::compact_test_config();
        let census = real_census(&config, 16384);
        // 97 + 144 + 48 + 2 shadowed BF16 owners.
        assert_eq!(census.f16_shadows, 1_298_923_520);
        // The 4096-row pool workspace with every short tail (slab descriptor
        // sized for the 64 slab lanes: max(union 15,368 B, 8 native row tiles
        // x 1,928 B = 15,424 B) = 15,424 B, 7,736 B above the former 32-lane
        // 7,688 B).
        assert_eq!(
            census.workspace.iter().map(|term| term.1).sum::<u64>(),
            153_269_568
        );
        // The split selector's constant score + filter extent.
        assert_eq!(census.qsa_selector, 83_894_272);
        assert!(census.gather > 0 && census.forward > 0 && census.state > 0);
        assert_eq!(census.mtp, 0);
    }

    #[test]
    fn evidence_placement_charges_the_full_context_and_keeps_fewer_layers() {
        let (free, non_expert, layer, historical) = EVIDENCE;
        let config = crate::config::compact_test_config();
        // The p6 run's 16384-token context, VMM fp8 state: 12 trunk QSA
        // layers x 26 MiB, 312 MiB with the 2 MiB granularity.
        let census = real_census(&config, 16384);
        assert_eq!(census.qsa_context, 312 << 20);
        let (unfixed, floor_charged, placed, reserve, charges) =
            replay(free, non_expert, layer, historical, census);
        // The beta that ran out of memory kept the historical reserve alone:
        // (32550 - 5115 - 7927) / 1060 = 18.40 -> 18 resident layers.
        assert_eq!(unfixed, 18);
        // Stage + margin on the floor already cost layers; the census never
        // raises the count.
        assert!(floor_charged < 18);
        assert!(placed <= floor_charged && placed < 18);
        // The whole full-context reserve plus margin and both stage buffers
        // is left free, and one more layer would not leave it.
        assert_eq!(charges.margin, 512 << 20);
        assert_eq!(charges.stage, 2 * ((1060 << 20) + 4096));
        let charged = charges.reserve(reserve.bytes()).unwrap();
        assert!(free - non_expert - placed as u64 * layer >= charged);
        assert!(free - non_expert - (placed as u64 + 1) * layer < charged);
        // Every census term is itemised in the log text.
        let text = reserve.describe();
        for name in [
            "QSA context",
            "state",
            "forward",
            "F16 weight shadows",
            "QSA selector",
            "native MTP",
            "QSA gather",
            "fp16_x",
            "ksplit_det",
            "gdn_pair_counters",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(charges.describe().contains("512 MiB safety margin"));
        assert!(charges
            .describe()
            .contains("2120 MiB long-prefill stage buffers"));
    }

    #[test]
    fn mq2_16k_failure_card_charges_the_full_qsa_context_and_places_fewer_layers() {
        // r9700.speed.json first_failure: free 32550 MiB, non-expert 5115,
        // reserve 7929, 840 MiB per layer, max_seq 17408 on a VMM fp8 / Q8 GDN
        // state: (32550 - 5115 - 7929) / 840 = 23.2 -> layers 0..23 resident,
        // then the 16343-token prefill ran out of memory mapping the QSA
        // context.
        let (free, non_expert, layer, historical) =
            (32550u64 << 20, 5115u64 << 20, 840u64 << 20, 7929u64 << 20);
        let config = crate::config::compact_test_config();
        let census = real_census(&config, 17_408);
        // 12 trunk QSA layers x 28 MiB with the 2 MiB granularity.
        assert_eq!(census.qsa_context, 336 << 20);
        // The actual full map exceeds what the first prefill chunk maps: the
        // growth the historical floor charges as zero.
        let first =
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, 17_408, 1536, 2 << 20);
        let first = state_census(&config, target_state_format(), &first, 4096, Full).unwrap();
        assert!(census.qsa_context > first.qsa_context);
        let (unfixed, floor_charged, placed, reserve, _) =
            replay(free, non_expert, layer, historical, census);
        assert_eq!(unfixed, 23);
        assert!(floor_charged < 23);
        assert!(placed <= floor_charged && placed < 23);
        assert!(reserve.describe().contains("QSA context"));
    }

    #[test]
    fn runtime_overhead_is_charged_only_on_exact_discrete_single_device_gfx1201() {
        assert_eq!(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES, 1408 << 20);
        assert_eq!(
            runtime_overhead_bytes("gfx1201", false, 1),
            Some(QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES)
        );
        assert_eq!(runtime_overhead_bytes("gfx1201", true, 1), None);
        assert_eq!(runtime_overhead_bytes("gfx1201", false, 2), None);
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", "gfx12010", ""] {
            assert_eq!(runtime_overhead_bytes(arch, false, 1), None, "{arch}");
        }
    }

    #[test]
    fn a_retired_floor_makes_the_census_the_reserve_and_says_so() {
        let census = ReserveCensus {
            runtime: 1344 << 20,
            ..census_with(1 << 30, 1 << 30, 1 << 30, 0, 0)
        };
        let retired = AutoReserve::new(None, census).unwrap();
        assert_eq!(retired.bytes(), retired.census_total);
        assert_eq!(retired.winner(), "census (historical floor retired)");
        let text = retired.describe();
        assert!(text.contains("HIP runtime 1344 MiB"), "{text}");
        assert!(
            text.contains(
                "(historical floor retired: the census carries the measured HIP runtime overhead)"
            ),
            "{text}"
        );
        // Even a census below what a floor would have been is not raised.
        let huge_floor = AutoReserve::new(Some(retired.census_total * 2), census).unwrap();
        assert_eq!(huge_floor.bytes(), retired.census_total * 2);
        assert_eq!(huge_floor.winner(), "historical floor");
    }

    /// The shipped mq3-xts R9700 card (free 32550 MiB, non-expert 5115 MiB,
    /// 1060 MiB per slab layer, 49 MoE layers, VMM fp8 / Q8 state, 4096-row
    /// chunk, slab stage + margin): the layers `auto` keeps resident today
    /// (historical floor, `Full` forward, no runtime term) and with the
    /// census carrying the HIP runtime overhead on the compact forward.
    fn shipped_r9700_layers(max_seq: usize) -> (usize, usize) {
        let (free, non_expert, layer) = (32550u64 << 20, 5115u64 << 20, 1060u64 << 20);
        let config = crate::config::compact_test_config();
        let format = target_state_format();
        let chunk = 4096;
        let charges = PlacementCharges::new(layer, true).unwrap();
        // The loader's historical floor: legacy-forced context, `Full`.
        let context = crate::Qwen4ContextCommit::for_expert_reserve(
            crate::Qwen4KvBackend::Vmm,
            false,
            max_seq,
            chunk,
            2 << 20,
        );
        let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
            config.num_key_value_heads,
            context.tokens,
        )
        .unwrap() as u64;
        let historical = auto_vram_reserve(
            &config,
            &context,
            chunk,
            Full,
            format.layout(),
            None,
            Some(gather),
        )
        .unwrap();
        let today = AutoReserve::new(
            Some(historical),
            census_at(&config, max_seq, chunk, Full, 0),
        )
        .unwrap();
        let new = AutoReserve::new(
            None,
            census_at(
                &config,
                max_seq,
                chunk,
                FnSymGfx1201,
                QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
            ),
        )
        .unwrap();
        let layers = |reserve: &AutoReserve| {
            auto_placement_layers(free, non_expert, layer, 49, reserve.bytes(), &charges).unwrap()
        };
        (layers(&today), layers(&new))
    }

    #[test]
    fn shipped_r9700_keeps_more_layers_with_the_compact_census_and_retired_floor() {
        let (today_8k, new_8k) = shipped_r9700_layers(8192);
        let (today_32k, new_32k) = shipped_r9700_layers(32768);
        assert_eq!((today_8k, today_32k), (15, 15));
        assert_eq!((new_8k, new_32k), (19, 18));
    }

    #[test]
    fn pruned_w3_slab_pack_fits_fully_resident_at_32k_only_with_a_small_compact_chunk() {
        // 29,556,669,976 device bytes (non-expert + routed experts), free
        // 32550 MiB: margin only, no stage (fully resident has no host layer).
        let (free, non_expert, total) = (32550u64 << 20, 5115u64 << 20, 29_556_669_976u64);
        let config = crate::config::compact_test_config();
        let charges = PlacementCharges::new(1060 << 20, false).unwrap();
        let fits = |chunk: usize| {
            let census = census_at(
                &config,
                32768,
                chunk,
                FnSymGfx1201,
                QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
            );
            let reserve = AutoReserve::new(None, census).unwrap();
            fits_fully_resident(
                free,
                non_expert,
                total - non_expert,
                charges.reserve(reserve.bytes()).unwrap(),
            )
        };
        assert!(fits(256));
        assert!(!fits(1024));
    }

    #[test]
    fn the_compact_1024_census_covers_the_measured_peak_beyond_weights() {
        let config = crate::config::compact_test_config();
        let census = census_at(
            &config,
            32768,
            1024,
            FnSymGfx1201,
            QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
        );
        // The pre-arena layout measured 4,141 MiB beyond the weights here
        // (census 4,237 MiB); the phase arena shares the step-local buffers,
        // so the census is that minus the arena's saving.
        assert_eq!(census.total().unwrap() >> 20, 3964);
        // For the record: the `Full` 8192-row chunk at 32K with the runtime.
        let full = census_at(
            &config,
            32768,
            8192,
            Full,
            QWEN4_RUNTIME_OVERHEAD_GFX1201_BYTES,
        );
        assert_eq!(full.total().unwrap() >> 20, 12302);
    }

    /// The shipped Strix Halo (gfx1151, unified memory) admission: VMM QSA
    /// F32 K/V, Q8 GDN, BF16 pooled mirror on, F16 arena off, no MTP,
    /// 8192-row chunk, `max_seq` 16384, free 118,441 MiB (XTS) / 118,415 MiB
    /// (mq3), routed 62,475 / 51,375 MiB, non-expert 5,115 MiB, logged
    /// reserve 12,150 MiB. Unified memory fits fully resident when
    /// `non_expert + routed + reserve <= free`; the reserve there is
    /// `AutoReserve::bytes()`, so this pins both formulas and the decision.
    #[test]
    fn halo_unified_memory_admission_reserve_and_census_decision() {
        let config = crate::config::compact_test_config();
        let format = crate::Qwen4StateFormat {
            qsa: QsaKvFormat::F32,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: true,
        };
        let (max_seq, chunk, granularity) = (16384usize, 8192usize, 2usize << 20);
        // Unified memory: the historical formula charges the first chunk's
        // pages (`for_expert_reserve(use_ranges = true)`), the census the
        // whole admitted context.
        let first = crate::Qwen4ContextCommit::for_expert_reserve(
            crate::Qwen4KvBackend::Vmm,
            true,
            max_seq,
            chunk,
            granularity,
        );
        let full = crate::Qwen4ContextCommit::new(
            crate::Qwen4KvBackend::Vmm,
            max_seq,
            max_seq,
            granularity,
        );
        assert!(
            full.committed_layer_bytes(&config, format.layout())
                > first.committed_layer_bytes(&config, format.layout())
        );
        let gather = |tokens: usize| {
            rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                tokens,
            )
            .unwrap() as u64
        };
        let historical = auto_vram_reserve(
            &config,
            &first,
            chunk,
            Full,
            format.layout(),
            None,
            Some(gather(first.tokens)),
        )
        .unwrap();
        // The logged reserve of the shipped runs.
        assert_eq!(historical >> 20, 12150);
        // Census: gfx1151 (shadow-capable, 8192-row chunk) holds the F16
        // weight shadows of the real artifact's BF16 owners; no dense GDN scan
        // or split selector.
        let shadows = f16_shadow_charge(
            true,
            f16_shadow_route_rows_with("gfx1151", chunk, true, 48, false),
            &j1778_bf16_manifest(),
            config.num_hidden_layers,
        )
        .unwrap();
        assert_eq!(shadows, 1_298_923_520);
        let state = state_census(&config, format, &full, chunk, Full).unwrap();
        let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
            hidden_size: config.hidden_size,
            moe_intermediate_size: config.moe_intermediate_size,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            hc_count: config.hc_count,
            hc_lowrank: config.hc_lowrank,
            gdn_format: None,
            routed_i4: true,
            include_short_tails: true,
        };
        let census = ReserveCensus {
            qsa_context: state.qsa_context,
            state: state.state,
            forward: state.forward,
            f16_shadows: 0,
            shared_i8_shadows: 0,
            workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap(),
            qsa_selector: 0,
            mtp: 0,
            gather: gather(full.tokens),
            runtime: 0,
        };
        let without = AutoReserve::new(Some(historical), census).unwrap();
        let reserve = AutoReserve::new(
            Some(historical),
            ReserveCensus {
                f16_shadows: shadows,
                ..census
            },
        )
        .unwrap();
        // Both Halo artifacts fit fully resident under whichever formula wins
        // (exact non-expert bytes 5,364,055,296 = 5,115.6 MiB). The census
        // without the shadow bound is the pinned 9,854 MiB; the corrected
        // census adds exactly the cold bound.
        assert_eq!(without.census_total >> 20, 9854, "{}", without.describe());
        assert_eq!(reserve.census_total, without.census_total + shadows);
        for (free, routed) in [
            (118_441u64 << 20, 62_475u64 << 20),
            (118_415 << 20, 51_375 << 20),
        ] {
            assert!(
                fits_fully_resident(free, 5_364_055_296, routed, reserve.bytes()),
                "{}",
                reserve.describe()
            );
        }
        // The census decision: the historical floor wins on this admission
        // (no flip), so unified-memory behaviour is unchanged. A failure here
        // is the flip: unified memory must then admit on `historical`.
        assert_eq!(
            reserve.winner(),
            "historical floor",
            "{}",
            reserve.describe()
        );
    }

    #[test]
    fn the_census_winner_sets_the_reserve_and_every_term_is_itemised() {
        let (free, non_expert, layer, historical) = EVIDENCE;
        let census = census_with(
            2000 << 20,
            1000 << 20,
            5000 << 20,
            1_298_923_520,
            83_894_272,
        );
        let total = (2000 + 1000 + 5000) * (1u64 << 20) + 1_298_923_520 + 153_256_072 + 83_894_272;
        let reserve = AutoReserve::new(Some(historical), census).unwrap();
        assert_eq!(reserve.census_total, total);
        assert!(total > historical);
        assert_eq!(reserve.bytes(), total);
        assert_eq!(reserve.winner(), "census");
        let text = reserve.describe();
        for name in [
            "QSA context",
            "state",
            "forward",
            "F16 weight shadows",
            "workspace",
            "QSA selector",
            "native MTP",
            "QSA gather",
        ] {
            assert!(text.contains(name), "{name}: {text}");
        }
        assert!(
            text.contains("census wins") && text.contains("7927 MiB historical floor"),
            "{text}"
        );
        // The census, not the floor, sizes the placement.
        let by_floor = auto_vram_layers(free, non_expert, layer, 49, historical);
        let by_census = auto_vram_layers(free, non_expert, layer, 49, reserve.bytes());
        assert!(by_census < by_floor);
        let overflowing = ReserveCensus {
            gather: 1,
            qsa_context: u64::MAX,
            ..census_with(0, 0, 0, 0, 0)
        };
        assert!(AutoReserve::new(Some(0), overflowing).is_err());
    }

    #[test]
    fn state_census_charges_the_actual_backends_full_context_once() {
        let config = crate::config::compact_test_config();
        let max_seq = 17_408;
        let granularity = 2 << 20;
        // The reference F32 state and the target's actual fp8 K/V + Q8 GDN.
        for format in [crate::Qwen4StateFormat::F32, target_state_format()] {
            for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
                // Whole admitted context mapped, as the actual backend maps it.
                let full = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
                let census = state_census(&config, format, &full, 4096, Full).unwrap();
                let context = full
                    .committed_layer_bytes(&config, format.layout())
                    .unwrap()
                    * config.n_full_layers();
                assert_eq!(census.qsa_context, context as u64, "{backend:?} {format:?}");
                // The rest of the state plus the context is exactly the state.
                let state = crate::state::Qwen4State::device_bytes(&config, format, &full).unwrap();
                assert_eq!(
                    census.qsa_context + census.state,
                    state as u64,
                    "{backend:?} {format:?}"
                );
                assert_eq!(
                    census.forward,
                    crate::gpu_forward::qwen4_forward_device_bytes(&config, 4096, Full).unwrap()
                );
                // Not the first chunk's pages: growth past them is charged.
                let first = crate::Qwen4ContextCommit::new(backend, max_seq, 1536, granularity);
                let first = state_census(&config, format, &first, 4096, Full).unwrap();
                if backend == crate::Qwen4KvBackend::Vmm {
                    assert!(census.qsa_context > first.qsa_context, "{format:?}");
                }
            }
        }
    }

    fn bf16(name: &str, shape: &[usize]) -> WeightEntry {
        WeightEntry::model(name, shape.to_vec(), DType::BF16, ShardPolicy::Replicate)
    }

    /// Count and F16 bytes of the shadow owners of `weights`, by role: the
    /// per-layer suffix, or `final_hc_down` for the mixer's root tensor.
    fn shadow_owners_by_role(
        weights: &[WeightEntry],
    ) -> std::collections::BTreeMap<String, (usize, u64)> {
        let mut roles = std::collections::BTreeMap::new();
        for entry in weights {
            let bytes = f16_shadow_bytes(std::slice::from_ref(entry), 48).unwrap();
            if bytes == 0 {
                continue;
            }
            let role = if entry.name == F16_SHADOW_ROOT {
                "final_hc_down".to_string()
            } else {
                entry
                    .name
                    .strip_prefix("model.language_model.layers.")
                    .and_then(|rest| rest.split_once('.'))
                    .map(|(_, suffix)| suffix.to_string())
                    .expect("a shadow owner is the root or a trunk layer tensor")
            };
            let slot: &mut (usize, u64) = roles.entry(role).or_default();
            slot.0 += 1;
            slot.1 += bytes;
        }
        roles
    }

    /// The real j1778 artifact (model-index metadata, no payload): 547 rank-2
    /// BF16 tensors, 291 of which are lazily shadowed in F16, 1,298,923,520 B.
    #[test]
    fn the_real_j1778_artifact_has_291_f16_shadow_owners_of_1298923520_bytes() {
        let weights = j1778_bf16_manifest();
        assert_eq!(weights.len(), 547);
        assert!(weights.iter().all(|entry| entry.dtype == DType::BF16));
        assert!(weights.iter().all(|entry| entry.logical_shape.len() == 2));
        let roles = shadow_owners_by_role(&weights);
        let expected: [(&str, usize, u64); 9] = [
            (
                "attn_hyper_connection.input_mix_weight_down.weight",
                48,
                314_572_800,
            ),
            (
                "mlp_hyper_connection.input_mix_weight_down.weight",
                48,
                314_572_800,
            ),
            ("mlp.gate.weight", 48, 125_829_120),
            ("mlp.shared_expert.gate_proj.weight", 48, 157_286_400),
            ("mlp.shared_expert.up_proj.weight", 48, 157_286_400),
            ("mlp.shared_expert.down_proj.weight", 48, 157_286_400),
            ("ple.key_proj.weight", 1, 52_428_800),
            ("ple.value_proj.weight", 1, 13_107_200),
            ("final_hc_down", 1, 6_553_600),
        ];
        assert_eq!(roles.len(), expected.len(), "{roles:?}");
        for (role, owners, bytes) in expected {
            assert_eq!(roles.get(role), Some(&(owners, bytes)), "{role}");
        }
        let owners: usize = roles.values().map(|(count, _)| count).sum();
        assert_eq!(owners, 291);
        assert_eq!(f16_shadow_bytes(&weights, 48), Ok(1_298_923_520));
        // The other BF16 tensors (the direct `input_mix_weight_up` / shared
        // gate readers, block-inject mixes, the MTP head) are not shadowed.
        assert_eq!(weights.len() - owners, 256);
        // Layers past the trunk own nothing; only the root mix is left.
        assert_eq!(f16_shadow_bytes(&weights, 0), Ok(6_553_600));
    }

    /// The census charges exactly when the runtime predicate can hold: every
    /// capability family, the 511/512-row threshold, the gfx1201 opt-in and
    /// the canonical routing scopes that lift sub-512-row forwards to
    /// `route_rows >= 512`. The expectations are written independently of the
    /// predicates under test.
    #[test]
    fn shadow_census_agrees_with_the_runtime_route_predicate_across_capability_families() {
        use rdna_compute::arch_caps::ArchCaps;
        use rdna_compute::feature_flags::FeatureFlags;
        use rdna_compute::gemm::{qwen4_f16_shadow_capable, qwen4_f16_wmma_applies_for};
        let weights = j1778_bf16_manifest();
        let bound = 1_298_923_520u64;
        assert_eq!(f16_shadow_bytes(&weights, 48), Ok(bound));
        for arch in [
            "gfx1100", "gfx1101", "gfx1151", "gfx1201", "gfx1200", "gfx1030", "gfx942",
        ] {
            let caps = ArchCaps::new(arch, std::sync::Arc::new(FeatureFlags::for_test(arch)));
            for optin in [true, false] {
                for chunk in [47usize, 48, 256, 511, 512, 1536, 8192] {
                    for (append, pm_invariant) in [(true, false), (false, false), (true, true)] {
                        let route_rows =
                            f16_shadow_route_rows_with(arch, chunk, append, 48, pm_invariant);
                        let charge = f16_shadow_charge(
                            qwen4_f16_shadow_capable(&caps, optin),
                            route_rows,
                            &weights,
                            48,
                        )
                        .unwrap();
                        // Independent spec: WMMA-w32 parts, or gfx1201 with its
                        // opt-in; rows reach 512 raw or through gfx1151's scope.
                        let capable = matches!(arch, "gfx1100" | "gfx1101" | "gfx1151")
                            || (arch == "gfx1201" && optin);
                        let scoped = arch == "gfx1151" && (pm_invariant || (append && chunk >= 48));
                        let reaches = chunk >= 512 || scoped;
                        let label = format!(
                            "{arch} optin {optin} chunk {chunk} append {append} pm {pm_invariant}"
                        );
                        assert_eq!(
                            charge,
                            if capable && reaches { bound } else { 0 },
                            "{label}"
                        );
                        // The runtime predicate over the shadowed owners' K
                        // values agrees: it holds for some owner iff the census
                        // charges.
                        let runtime = [2560usize, 640, 10240].into_iter().any(|k| {
                            qwen4_f16_wmma_applies_for(
                                &caps, optin, true, route_rows, k, true, true,
                            )
                        });
                        assert_eq!(runtime, charge > 0, "{label}");
                    }
                }
            }
        }
        // The guards the census inherits stay in the runtime predicate: a
        // non-BF16 weight, K not a multiple of 64, a recorder/capture and the
        // opt-out never apply.
        let caps = ArchCaps::new(
            "gfx1100",
            std::sync::Arc::new(FeatureFlags::for_test("gfx1100")),
        );
        assert!(qwen4_f16_wmma_applies_for(
            &caps, true, true, 512, 640, true, true
        ));
        assert!(!qwen4_f16_wmma_applies_for(
            &caps, true, false, 512, 640, true, true
        ));
        assert!(!qwen4_f16_wmma_applies_for(
            &caps, true, true, 512, 8, true, true
        ));
        assert!(!qwen4_f16_wmma_applies_for(
            &caps, true, true, 511, 640, true, true
        ));
        assert!(!qwen4_f16_wmma_applies_for(
            &caps, true, true, 512, 640, false, true
        ));
        assert!(!qwen4_f16_wmma_applies_for(
            &caps, true, true, 512, 640, true, false
        ));
    }

    /// The shadows are physical owners DUPLICATING the original BF16 weights:
    /// the weights base (`resident_split`'s non-expert bytes) already holds the
    /// originals once; the cold bound adds only the F16 copies, once, in the
    /// census.
    #[test]
    fn shadow_charge_adds_the_f16_copies_once_and_never_the_original_bf16_weights() {
        let weights = j1778_bf16_manifest();
        let original_bytes =
            |entry: &WeightEntry| Some(entry.logical_shape.iter().product::<usize>() as u64 * 2);
        let (non_expert, layer_experts) = resident_split(&weights, original_bytes).unwrap();
        // No routed experts in this metadata slice: every original BF16 byte
        // is a non-expert byte, counted once.
        assert_eq!(layer_experts, 0);
        assert_eq!(non_expert, 2_020_889_600);
        let shadows = f16_shadow_bytes(&weights, 48).unwrap();
        let shadowed_originals: u64 = weights
            .iter()
            .filter(|entry| f16_shadow_bytes(std::slice::from_ref(entry), 48).unwrap() > 0)
            .map(|entry| original_bytes(entry).unwrap())
            .sum();
        // BF16 and F16 are both 2 bytes per element: the copy is as large as
        // the original it shadows, and charged separately.
        assert_eq!(shadows, shadowed_originals);
        // Adding the entries to a manifest moves the weights base by the
        // originals only; the reserve census moves by the F16 copies only,
        // through its one shadow term.
        let before = AutoReserve::new(None, census_with(0, 0, 0, 0, 0)).unwrap();
        let after = AutoReserve::new(None, census_with(0, 0, 0, shadows, 0)).unwrap();
        assert_eq!(
            after.census_total - before.census_total,
            shadows,
            "the census moves by the shadow term alone"
        );
        assert_eq!(after.placement_bytes(true), after.census_total);
    }

    /// The corrected census does not change the incumbent's reserve-selection
    /// POLICY (`max(historical floor, census)`; floor retired where the census
    /// carries the runtime): it only feeds the census the capability the
    /// runtime already has. Prints the actual reserve impact per chunk on the
    /// real artifact's gfx1100 discrete (Legacy KV) load.
    #[test]
    fn shadow_census_keeps_the_incumbent_reserve_policy_and_reports_its_impact() {
        let config = crate::config::compact_test_config();
        // The gate's actual state formats: Q8 QSA K/V and Q8 GDN.
        let format = crate::Qwen4StateFormat {
            qsa: QsaKvFormat::Q8,
            gdn: crate::GdnStateFormat::Q8,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        };
        let weights = j1778_bf16_manifest();
        let max_seq = 16_640usize;
        let mut rows = Vec::new();
        for chunk in [1536usize, 4096, 8192] {
            let context =
                crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Legacy, max_seq, max_seq, 1);
            let reserve_context = crate::Qwen4ContextCommit::for_expert_reserve(
                crate::Qwen4KvBackend::Legacy,
                false,
                max_seq,
                chunk,
                1,
            );
            let historical = auto_vram_reserve(
                &config,
                &reserve_context,
                chunk,
                Full,
                format.layout(),
                None,
                None,
            )
            .unwrap();
            let state = state_census(&config, format, &context, chunk, Full).unwrap();
            let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
                hidden_size: config.hidden_size,
                moe_intermediate_size: config.moe_intermediate_size,
                num_experts: config.num_experts,
                top_k: config.num_experts_per_tok,
                hc_count: config.hc_count,
                hc_lowrank: config.hc_lowrank,
                gdn_format: None,
                routed_i4: true,
                include_short_tails: true,
            };
            let census = |f16_shadows: u64| ReserveCensus {
                qsa_context: state.qsa_context,
                state: state.state,
                forward: state.forward,
                f16_shadows,
                shared_i8_shadows: 0,
                workspace: rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims)
                    .unwrap(),
                qsa_selector: 0,
                mtp: 0,
                gather: 0,
                runtime: 0,
            };
            let shadows = f16_shadow_charge(
                true,
                f16_shadow_route_rows_with("gfx1100", chunk, true, 48, false),
                &weights,
                config.num_hidden_layers,
            )
            .unwrap();
            assert_eq!(shadows, 1_298_923_520);
            let old = AutoReserve::new(Some(historical), census(0)).unwrap();
            let new = AutoReserve::new(Some(historical), census(shadows)).unwrap();
            // The policy is unchanged: max(historical, census) wherever the
            // floor lives, the census alone for S1 (`placement_bytes(true)`).
            assert_eq!(new.bytes(), historical.max(new.census_total));
            assert_eq!(new.placement_bytes(false), new.bytes());
            assert_eq!(new.placement_bytes(true), new.census_total);
            assert_eq!(new.census_total - old.census_total, shadows);
            if chunk == 1536 {
                // The 1,536-row incumbent: the source census base (2,237,111,724 B
                // with the former 32-lane slab union, 7,736 B more with the 64-lane
                // native row-tile descriptor) and the
                // historical floor still win by a wide margin.
                assert_eq!(old.census_total, 2_237_119_460);
                assert_eq!(
                    new.bytes(),
                    old.bytes(),
                    "incumbent reserve unchanged at 1536"
                );
                assert_eq!(new.winner(), "historical floor");
            }
            rows.push(serde_json::json!({
                "chunk_rows": chunk,
                "historical_floor_bytes": historical,
                "census_without_shadows_bytes": old.census_total,
                "census_with_shadows_bytes": new.census_total,
                "f16_shadow_bytes": shadows,
                "incumbent_reserve_before_bytes": old.bytes(),
                "incumbent_reserve_after_bytes": new.bytes(),
                "incumbent_reserve_delta_bytes": new.bytes() - old.bytes(),
                "winner_before": old.winner(),
                "winner_after": new.winner(),
            }));
        }
        println!(
            "SLOWLINK_SHADOW_RESERVE_IMPACT {}",
            serde_json::to_string(&rows).unwrap()
        );
    }

    #[test]
    fn f16_shadow_bytes_query_the_manifest_for_the_shadowed_owners_only() {
        let layer = |i: usize, suffix: &str| format!("model.language_model.layers.{i}.{suffix}");
        let mut weights = vec![
            // Shadowed: full F16 matrices, 2 bytes per element.
            bf16(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                &[4, 8],
            ),
            bf16(
                &layer(0, "attn_hyper_connection.input_mix_weight_down.weight"),
                &[2, 2],
            ),
            bf16(
                &layer(1, "mlp_hyper_connection.input_mix_weight_down.weight"),
                &[2, 2],
            ),
            bf16(&layer(0, "mlp.gate.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.gate_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.up_proj.weight"), &[3, 5]),
            bf16(&layer(0, "mlp.shared_expert.down_proj.weight"), &[5, 3]),
            bf16(&layer(0, "ple.key_proj.weight"), &[1, 7]),
            bf16(&layer(0, "ple.value_proj.weight"), &[1, 7]),
            // Direct BF16 readers and other tensors: no shadow.
            bf16(
                &layer(0, "attn_hyper_connection.input_mix_weight_up.weight"),
                &[2, 2],
            ),
            bf16(&layer(0, "mlp.shared_expert_gate.weight"), &[1, 9]),
            bf16(&layer(0, "input_layernorm.weight"), &[1, 9]),
            // Past the trunk, wrong rank or wrong dtype: no shadow.
            bf16(&layer(2, "mlp.gate.weight"), &[3, 5]),
            bf16(&layer(0, "ple.key_proj.weight.extra"), &[1, 7]),
            WeightEntry::model(
                layer(1, "mlp.gate.weight"),
                vec![3, 5],
                DType::MQ4G256V2,
                ShardPolicy::Replicate,
            ),
            bf16(&layer(1, "mlp.gate.weight"), &[15]),
        ];
        let expected = 2 * (4 * 8 + 2 * 2 + 2 * 2 + 3 * 5 + 3 * 5 + 3 * 5 + 5 * 3 + 7 + 7);
        assert_eq!(f16_shadow_bytes(&weights, 2), Ok(expected as u64));
        // An alias or external row store has no payload of its own.
        weights.push(WeightEntry::model(
            layer(1, "mlp.gate.weight"),
            vec![3, 5],
            DType::BF16,
            ShardPolicy::Tied { source: "x".into() },
        ));
        assert_eq!(f16_shadow_bytes(&weights, 2), Ok(expected as u64));
        // An MQ-quantized trunk (no BF16 owners) charges nothing.
        assert_eq!(f16_shadow_bytes(&[], 48), Ok(0));
        // The 48-layer total the target profile pins: 97 HC down, 144 shared,
        // 48 routers, 2 PLE.
        let target: u64 = 1_298_923_520;
        assert_eq!(target, 1238 * (1 << 20) + (3 << 18));
    }

    #[test]
    fn stage_switch_is_env_first_then_gfx1201_or_narrow_link() {
        use crate::host_link::HostLink::{self, Discrete, Unified, Unmeasured};
        fn no_probe() -> HostLink {
            unreachable!("the link was probed")
        }
        let narrow = || Discrete { gbps: 12.0 };
        let wide = || Discrete { gbps: 27.3 };
        // Unset: gfx1201 on regardless of link (and never probes); every
        // other arch only on a measured narrow discrete link.
        assert!(expert_stage_switch("gfx1201", None, no_probe));
        assert!(expert_stage_switch("gfx1201", None, || Unmeasured));
        assert!(expert_stage_switch("gfx1201", None, wide));
        assert!(expert_stage_switch("gfx1100", None, narrow));
        assert!(!expert_stage_switch("gfx1100", None, wide));
        assert!(!expert_stage_switch("gfx1100", None, || Unmeasured));
        assert!(!expert_stage_switch("gfx1151", None, || Unified));
        // Anything but the strict `1`/`0` is unset.
        assert!(expert_stage_switch("gfx1100", Some("on"), narrow));
        assert!(!expert_stage_switch("gfx1100", Some(""), wide));
        // Explicit values win over arch and link, and never probe.
        assert!(expert_stage_switch("gfx1100", Some("1"), no_probe));
        assert!(expert_stage_switch("gfx1151", Some("1"), || Unified));
        assert!(!expert_stage_switch("gfx1100", Some("0"), no_probe));
        assert!(!expert_stage_switch("gfx1201", Some("0"), wide));
        // Arithmetic of the probing default only: the load charges placement
        // with the probe-free `expert_stage_placement_switch` instead (see
        // `placement_first_*`), so it never sees this narrow-link result.
        let charged = |arch, link: fn() -> HostLink| {
            dedicated_slab_stage(expert_stage_switch(arch, None, link), true, false)
        };
        assert!(charged("gfx1100", narrow));
        assert!(!charged("gfx1100", wide));
        assert!(!charged("gfx1100", || Unmeasured));
        assert!(charged("gfx1201", wide));
    }

    #[test]
    fn stage_charge_only_where_the_slab_stage_allocates() {
        // Needs the switch, a slab recipe and no expert parallelism.
        assert!(dedicated_slab_stage(true, true, false));
        assert!(!dedicated_slab_stage(false, true, false));
        assert!(!dedicated_slab_stage(true, false, false));
        assert!(!dedicated_slab_stage(true, true, true));
        let (free, non_expert, layer, base) = EVIDENCE;
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(
            off,
            PlacementCharges {
                margin: 512 << 20,
                stage: 0,
                slowlink: 0,
            }
        );
        assert_eq!(off.reserve(base).unwrap(), base + (512 << 20));
        // No stage: only the margin costs layers
        // ((32550 - 5115 - 7927 - 512) / 1060 = 17.9).
        let unstaged = auto_placement_layers(free, non_expert, layer, 49, base, &off).unwrap();
        assert_eq!(unstaged, 17);
        let staged = PlacementCharges::new(layer, true).unwrap();
        let staged = auto_placement_layers(free, non_expert, layer, 49, base, &staged).unwrap();
        assert!(staged < unstaged);
        // Fully resident: the unset fallback charges the same reserve and
        // margin (never a stage), so a card that only just holds everything
        // without the margin falls to `auto` instead of loading.
        let experts = 20 * layer;
        let need = non_expert + experts + base;
        assert!(fits_fully_resident(need, non_expert, experts, base));
        assert!(!fits_fully_resident(
            need,
            non_expert,
            experts,
            off.reserve(base).unwrap()
        ));
    }

    #[test]
    fn charged_placement_refuses_clearly_when_zero_layers_do_not_fit() {
        let (_, non_expert, layer, base) = EVIDENCE;
        let charges = PlacementCharges::new(layer, true).unwrap();
        let need = non_expert + charges.reserve(base).unwrap();
        // Exactly the weights and the charged reserve: zero resident layers.
        assert_eq!(
            auto_placement_layers(need, non_expert, layer, 49, base, &charges),
            Ok(0)
        );
        let error =
            auto_placement_layers(need - 1, non_expert, layer, 49, base, &charges).unwrap_err();
        assert!(
            error.contains("cannot hold") && error.contains("every routed expert in host RAM"),
            "{error}"
        );
        assert!(error.contains("512 MiB safety margin"), "{error}");
        assert!(
            error.contains("2120 MiB long-prefill stage buffers"),
            "{error}"
        );
        // Without the stage the same card still loads: the 2120 MiB + 8191 B
        // the stage took now holds two whole layers.
        let off = PlacementCharges::new(layer, false).unwrap();
        assert_eq!(
            auto_placement_layers(need - 1, non_expert, layer, 49, base, &off),
            Ok(2)
        );
    }

    #[test]
    fn placement_charges_refuse_overflow() {
        assert!(PlacementCharges::new(u64::MAX, true).is_err());
        assert!(PlacementCharges::new(u64::MAX / 2, true).is_err());
        assert!(PlacementCharges::new(u64::MAX, false).is_ok());
        let charges = PlacementCharges::new(1 << 20, true).unwrap();
        assert!(charges.reserve(u64::MAX).is_err());
        assert!(auto_placement_layers(u64::MAX, 0, 1 << 20, 49, u64::MAX, &charges).is_err());
        // A non-expert size that overflows the sum is a refusal, not a wrap.
        assert!(auto_placement_layers(u64::MAX, u64::MAX, 1 << 20, 49, 1, &charges).is_err());
    }

    const WF_LAYER: u64 = 1275 << 20;
    const WF_NON_EXPERT: u64 = 6000 << 20;
    const WF_BASE: u64 = 8000 << 20;

    /// Free VRAM that places exactly 14 whole 1275 MiB layers after the
    /// non-expert weights, the base reserve and the 512 MiB margin (the
    /// receipt's 14-layer geometry; no stage charge).
    fn wf_free_for_14() -> u64 {
        WF_NON_EXPERT + WF_BASE + (512 << 20) + 14 * WF_LAYER + (100 << 20)
    }

    fn wf_auto(
        free: u64,
        charges: &PlacementCharges,
        borrowed: bool,
    ) -> Result<WholeFirstPlan, String> {
        WholeFirstPlan::auto(
            free,
            WF_NON_EXPERT,
            WF_LAYER,
            48,
            49,
            WF_BASE,
            charges,
            borrowed,
        )
    }

    #[test]
    fn whole_first_never_places_fewer_layers_than_the_hotset0_baseline_and_fits() {
        for dedicated in [false, true] {
            let charges = PlacementCharges::new(WF_LAYER, dedicated).unwrap();
            for borrowed in [false, true] {
                for step in 0..=96u64 {
                    let free = (14_000u64 + step * 250) << 20;
                    let baseline =
                        auto_placement_layers(free, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &charges);
                    let plan = wf_auto(free, &charges, borrowed);
                    match (baseline, plan) {
                        (Ok(layers), Ok(plan)) => {
                            assert!(plan.whole_layers >= layers, "free {free}");
                            assert_eq!(plan.hot_experts, 0);
                            assert!(plan.whole_layers <= 48);
                            // Resident layers + charged reserve (dedicated
                            // stage included) + non-expert weights fit.
                            let need = WF_NON_EXPERT
                                + charges.reserve(WF_BASE).unwrap()
                                + plan.whole_layers as u64 * WF_LAYER;
                            assert!(need <= free, "free {free}: need {need}");
                            // One more layer would not have fit.
                            if plan.whole_layers < 48 {
                                assert!(need + WF_LAYER > free);
                            }
                        }
                        (Err(_), Err(_)) => {}
                        (baseline, plan) => panic!("diverged: {baseline:?} vs {plan:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn whole_first_receipt_geometry_is_14_layers_with_2550_mib_borrowed_and_nothing_allocated() {
        let free = wf_free_for_14();
        let off = PlacementCharges::new(WF_LAYER, false).unwrap();
        let plan = wf_auto(free, &off, true).unwrap();
        assert_eq!(plan.whole_layers, 14);
        assert_eq!(
            auto_placement_layers(free, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &off),
            Ok(14)
        );
        assert_eq!(
            plan.stage,
            WholeFirstStage::Borrowed {
                donors: [12, 13],
                capacity: 2550 << 20,
                restore_copies: 2550 << 20,
            }
        );
        assert_eq!(plan.stage.allocated_vram(), 0);
        assert_eq!(plan.stage.borrowed_vram(), 2550 << 20);
        let line = plan.describe(14 * WF_LAYER);
        for part in [
            "policy=whole-first",
            "whole layers 0..14 (14 of 49 MoE layers)",
            "hot experts 0",
            "expert resident 17850 MiB",
            "stage allocated 0 MiB, borrowed 2550 MiB from resident layers 12 and 13",
            "2550 MiB pinned host restore copies",
            "sidecar not consulted",
        ] {
            assert!(line.contains(part), "{part}: {line}");
        }
    }

    #[test]
    fn whole_first_stage_accounting_dedicated_borrowed_and_disabled() {
        let free = wf_free_for_14() + (4000 << 20);
        // Stage switch off or not a donor layout: nothing allocated or borrowed.
        let off = PlacementCharges::new(WF_LAYER, false).unwrap();
        let disabled = wf_auto(free, &off, false).unwrap();
        assert_eq!(disabled.stage, WholeFirstStage::None);
        assert!(disabled
            .describe(0)
            .contains("stage allocated 0 MiB, borrowed 0 MiB"));
        // Borrowed donors cost no VRAM: same layers as the disabled load.
        let borrowed = wf_auto(free, &off, true).unwrap();
        assert_eq!(borrowed.whole_layers, disabled.whole_layers);
        assert_eq!(borrowed.stage.allocated_vram(), 0);
        assert_eq!(borrowed.stage.borrowed_vram(), 2 * WF_LAYER);
        // A slab load's dedicated stage is allocated and charged: fewer
        // layers than the stage-free load, and the buffers are accounted.
        let on = PlacementCharges::new(WF_LAYER, true).unwrap();
        let dedicated = wf_auto(free, &on, false).unwrap();
        assert!(dedicated.whole_layers < disabled.whole_layers);
        let buffer = WF_LAYER + crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64;
        assert_eq!(
            dedicated.stage,
            WholeFirstStage::Dedicated {
                buffer,
                allocated: 2 * buffer
            }
        );
        assert_eq!(dedicated.stage.allocated_vram(), on.stage);
        assert_eq!(dedicated.stage.borrowed_vram(), 0);
        let line = dedicated.describe(0);
        assert!(
            line.contains("borrowed 0 MiB") && line.contains("2 dedicated buffers of"),
            "{line}"
        );
        // The dedicated stage wins over borrowing (slab recipes never borrow).
        assert_eq!(wf_auto(free, &on, true).unwrap().stage, dedicated.stage);
        // Fewer than two resident layers cannot lend donors.
        assert_eq!(
            WholeFirstPlan::stage_for(1, 49, WF_LAYER, &off, true),
            WholeFirstStage::None
        );
        assert!(matches!(
            WholeFirstPlan::stage_for(2, 49, WF_LAYER, &off, true),
            WholeFirstStage::Borrowed { donors: [0, 1], .. }
        ));
        // No host layer, no stage, even with a charge.
        assert_eq!(
            WholeFirstPlan::stage_for(49, 49, WF_LAYER, &on, true),
            WholeFirstStage::None
        );
        let resident = WholeFirstPlan::fully_resident(49, WF_LAYER);
        assert_eq!((resident.whole_layers, resident.hot_experts), (49, 0));
        assert_eq!(resident.stage, WholeFirstStage::None);
        assert!(resident
            .describe(0)
            .contains("whole layers 0..49 (49 of 49"));
    }

    #[test]
    fn whole_first_refuses_exactly_like_the_baseline_when_nothing_fits() {
        let on = PlacementCharges::new(WF_LAYER, true).unwrap();
        let need = WF_NON_EXPERT + on.reserve(WF_BASE).unwrap();
        assert_eq!(wf_auto(need, &on, false).unwrap().whole_layers, 0);
        let plan_error = wf_auto(need - 1, &on, false).unwrap_err();
        let base_error =
            auto_placement_layers(need - 1, WF_NON_EXPERT, WF_LAYER, 48, WF_BASE, &on).unwrap_err();
        assert_eq!(plan_error, base_error);
    }

    #[test]
    fn resident_routed_expert_bytes_counts_only_what_stays_on_device() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.0.mlp.experts.down_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.gate_up_proj", 1),
            entry("model.language_model.layers.0.self_attn.q_proj.weight", 0),
            entry("mtp.layers.0.mlp.experts.gate_up_proj", 0),
        ];
        let bytes_of =
            |entry: &WeightEntry| Some(entry.logical_shape.iter().product::<usize>() as u64);
        let each = 4 * 4 * 256;
        assert_eq!(
            resident_routed_expert_bytes(&weights, bytes_of),
            Ok(4 * each)
        );
        place_routed_experts(&mut weights, 1);
        // Trunk layer 0's two projections stay; layer 1 and the MTP layer move.
        assert_eq!(
            resident_routed_expert_bytes(&weights, bytes_of),
            Ok(2 * each)
        );
        assert!(resident_routed_expert_bytes(&weights, |_| None).is_err());
    }

    #[test]
    fn compact_scratch_lowers_the_reserve_by_exactly_its_saving() {
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let reserve = |rows, mode| {
            auto_vram_reserve(&config, &measured, rows, mode, F32, None, None).unwrap()
        };
        let forward = |rows, mode| {
            crate::gpu_forward::qwen4_forward_device_bytes(&config, rows, mode).unwrap()
        };
        // The 8192-row chunk's compact plan is 6144.59375 MiB smaller, and the
        // whole of it reaches the reserve: the historical baseline stays on
        // the Full layout, so the saving is not subtracted from both sides.
        let saving = forward(8192, Full) - forward(8192, FnSymGfx1201);
        assert_eq!(saving, 6144 * (1 << 20) + 622_592);
        assert_eq!(reserve(8192, Full) - reserve(8192, FnSymGfx1201), saving);
        // The same holds at every chunk, including the measured layout's own
        // 1536 rows.
        for rows in [AUTO_VRAM_RESERVE_CHUNK, 2048, 4096] {
            let delta = forward(rows, Full) - forward(rows, FnSymGfx1201);
            assert!(delta > 0, "{rows} rows");
            assert_eq!(
                reserve(rows, Full) - reserve(rows, FnSymGfx1201),
                delta,
                "{rows} rows"
            );
        }
        // A compact 8192-row chunk costs no more than today's 4096-row chunk,
        // so `auto` keeps at least the layers it keeps today.
        assert!(reserve(8192, FnSymGfx1201) <= reserve(4096, Full));
        let (free, weights, layer) = (32548u64 << 20, 5_364_000_000u64, 1_336_900_000u64);
        let layers = |reserve| auto_vram_layers(free, weights, layer, 48, reserve);
        assert!(layers(reserve(8192, FnSymGfx1201)) >= layers(reserve(4096, Full)));
        assert!(layers(reserve(8192, FnSymGfx1201)) > layers(reserve(8192, Full)));
    }

    #[test]
    fn native_mtp_and_longer_context_grow_the_reserve() {
        let config = crate::config::compact_test_config();
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let base = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            None,
            None,
        )
        .unwrap();
        let native = |context: &crate::Qwen4ContextCommit, max_k, row_capture| {
            crate::mtp_spec::native_mtp_device_bytes(
                &config,
                context,
                AUTO_VRAM_RESERVE_CHUNK,
                max_k,
                DType::MQ6G256V2,
                row_capture,
                false,
            )
            .unwrap()
        };
        let mtp = native(&measured, 3, Some((crate::GdnStateFormat::Q8, 4)));
        let with_mtp = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            Some(mtp),
            None,
        )
        .unwrap();
        assert_eq!(with_mtp - base, mtp);
        // The gathered QSA attention's workspace is charged on top.
        let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
            config.num_key_value_heads,
            AUTO_VRAM_RESERVE_MAX_SEQ,
        )
        .unwrap() as u64;
        let with_gather = auto_vram_reserve(
            &config,
            &measured,
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            F32,
            None,
            Some(gather),
        )
        .unwrap();
        assert_eq!(with_gather - base, gather);
        // The draft head's F32 requant scratch is row-chunked (at most
        // `REQUANT_G256_CHUNK_ROWS` rows, never vocab x hidden x 4) and goes
        // back to the device at attach, before the first request allocates
        // its verify rows and GDN capture: the reserve holds the larger phase
        // on top of what the head keeps, not both.
        let (resident, scratch) =
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, &measured, DType::MQ6G256V2, false)
                .unwrap();
        let (resident, scratch) = (resident as u64, scratch as u64);
        assert_eq!(
            scratch,
            (rdna_compute::Gpu::REQUANT_G256_CHUNK_ROWS * config.hidden_size * 4) as u64
        );
        // The request now outweighs the build scratch at every K.
        let hidden_row = (config.hc_count * config.hidden_size * 4) as u64;
        // Verify rows plus the pending, row and retained-pending hidden carries.
        let verify = (AUTO_VRAM_RESERVE_CHUNK as u64 + 3) * hidden_row;
        assert!(verify > scratch);
        let capture4 =
            crate::state::Qwen4State::row_capture_bytes(&config, crate::GdnStateFormat::Q8, 4)
                .unwrap() as u64;
        assert_eq!(mtp, resident + verify + capture4);
        assert_eq!(native(&measured, 3, None), resident + verify);
        assert_eq!(native(&measured, 10, None), resident + verify);
        // A row-exact verify limit wider than the MTP window (gfx1151: 32)
        // charges its whole ring; a narrower one never shrinks the window's.
        let capture32 =
            crate::state::Qwen4State::row_capture_bytes(&config, crate::GdnStateFormat::Q8, 32)
                .unwrap() as u64;
        assert_eq!(
            native(&measured, 3, Some((crate::GdnStateFormat::Q8, 32))),
            resident + verify + capture32
        );
        assert_eq!(
            native(&measured, 3, Some((crate::GdnStateFormat::Q8, 2))),
            mtp
        );
        // The 11-row GDN capture is charged in the state's real GDN format.
        for gdn in [crate::GdnStateFormat::Q8, crate::GdnStateFormat::F32] {
            let capture =
                crate::state::Qwen4State::row_capture_bytes(&config, gdn, 11).unwrap() as u64;
            let request = (AUTO_VRAM_RESERVE_CHUNK as u64 + 3) * hidden_row + capture;
            assert_eq!(
                native(&measured, 10, Some((gdn, 11))),
                resident + scratch.max(request)
            );
            // The gfx1151 batched prompt fill adds its row scratch (sub-chunks
            // of at most MTP_FILL_ROWS rows) and one chunk of prompt token ids
            // to the request.
            let fill = crate::mtp_gpu::MtpAppendScratch::device_bytes(
                &config,
                crate::mtp_gpu::MTP_FILL_ROWS.min(AUTO_VRAM_RESERVE_CHUNK),
                AUTO_VRAM_RESERVE_CHUNK,
            )
            .unwrap() as u64;
            assert!(fill > 0);
            let batched = crate::mtp_spec::native_mtp_device_bytes(
                &config,
                &measured,
                AUTO_VRAM_RESERVE_CHUNK,
                10,
                DType::MQ6G256V2,
                Some((gdn, 11)),
                true,
            )
            .unwrap();
            assert_eq!(batched, resident + scratch.max(request + fill));
        }
        // On a card whose free VRAM sits exactly on a layer boundary of the
        // base reserve (non-expert weights + base + 16 layers; not a
        // measured card), any positive MTP charge must cost a layer; the
        // MTP bytes then fit beside the chosen layers, and one layer more
        // would not have left them.
        let (weights, layer) = (5_364_000_000u64, 1_336_900_000u64);
        let free = weights + base + 16 * layer;
        assert_eq!(auto_vram_layers(free, weights, layer, 48, base), 16);
        let with = auto_vram_layers(free, weights, layer, 48, with_mtp) as u64;
        assert!(with < auto_vram_layers(free, weights, layer, 48, base) as u64);
        assert!(free - weights - with * layer >= with_mtp);
        assert!(free - weights - (with + 1) * layer < with_mtp);
        // A legacy context past the measured one adds the trunk QSA arenas'
        // growth in the load's QSA format: fp8 K/V grows less than F32.
        let long_seq = 4 * AUTO_VRAM_RESERVE_MAX_SEQ;
        let growth = |format| {
            let long = auto_vram_reserve(
                &config,
                &crate::Qwen4ContextCommit::legacy(long_seq),
                AUTO_VRAM_RESERVE_CHUNK,
                Full,
                format,
                None,
                None,
            )
            .unwrap();
            let arena = (config.qsa_context_arena_bytes(long_seq, format).unwrap()
                - config
                    .qsa_context_arena_bytes(AUTO_VRAM_RESERVE_MAX_SEQ, format)
                    .unwrap())
                * config.n_full_layers();
            assert_eq!(long - base, arena as u64);
            long - base
        };
        assert!(growth(QsaKvFormat::Fp8) < growth(F32));
        // A shorter context keeps the measured reserve.
        let short = auto_vram_reserve(
            &config,
            &crate::Qwen4ContextCommit::legacy(2048),
            AUTO_VRAM_RESERVE_CHUNK,
            Full,
            QsaKvFormat::Fp8,
            None,
            None,
        );
        assert_eq!(short.unwrap(), base);
    }

    #[test]
    fn vmm_context_charges_only_committed_pages() {
        let config = crate::config::compact_test_config();
        let chunk = AUTO_VRAM_RESERVE_CHUNK;
        let measured = crate::Qwen4ContextCommit::legacy(AUTO_VRAM_RESERVE_MAX_SEQ);
        let base = auto_vram_reserve(&config, &measured, chunk, Full, F32, None, None).unwrap();
        let vmm = |max_seq| {
            crate::Qwen4ContextCommit::new(crate::Qwen4KvBackend::Vmm, max_seq, chunk, 2 << 20)
        };
        // The native logical context, virtual under VMM, does not size the
        // reserve; legacy storage at that context commits every arena.
        let native = crate::QWEN4_MAX_CONTEXT;
        let vmm_native =
            auto_vram_reserve(&config, &vmm(native), chunk, Full, F32, None, None).unwrap();
        assert_eq!(vmm_native, base);
        let legacy_native = auto_vram_reserve(
            &config,
            &crate::Qwen4ContextCommit::legacy(native),
            chunk,
            Full,
            F32,
            None,
            None,
        )
        .unwrap();
        assert!(legacy_native > vmm_native);
        // Native MTP's one F32 QSA layer: the VMM head charges its first
        // chunk's pages instead of the full legacy arenas.
        let mtp = |context: &crate::Qwen4ContextCommit| {
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, context, DType::MQ6G256V2, false)
                .unwrap()
                .0
        };
        let legacy_mtp = mtp(&crate::Qwen4ContextCommit::legacy(native));
        let vmm_mtp = mtp(&vmm(native));
        assert_eq!(
            legacy_mtp - vmm_mtp,
            config.qsa_context_arena_bytes(native, F32).unwrap()
                - vmm(native).committed_layer_bytes(&config, F32).unwrap()
        );
    }

    #[test]
    fn discrete_reserve_matches_legacy_and_unified_memory_charges_committed() {
        use crate::{Qwen4ContextCommit as Commit, Qwen4KvBackend as Backend};
        let config = crate::config::compact_test_config();
        let chunk = AUTO_VRAM_RESERVE_CHUNK;
        let granularity = 2 << 20;
        let mtp = |context: &Commit| {
            crate::mtp_gpu::Qwen4MtpGpu::device_bytes(&config, context, DType::MQ6G256V2, false)
                .unwrap()
                .0
        };
        // The expert layers `auto` places for a load whose reserve charges
        // `context`, with the native MTP and QSA gather parts as the loader
        // computes them from the same context, on the measured R9700 fixture.
        let placement = |context: &Commit, format| {
            let gather = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(
                config.num_key_value_heads,
                context.tokens,
            )
            .unwrap() as u64;
            let reserve = auto_vram_reserve(
                &config,
                context,
                chunk,
                Full,
                format,
                Some(mtp(context) as u64),
                Some(gather),
            )
            .unwrap();
            let free = 32548u64 << 20;
            let (weights, layer) = (5_364_000_000u64, 1_336_900_000u64);
            (reserve, auto_vram_layers(free, weights, layer, 48, reserve))
        };
        for max_seq in [65_536, 262_144] {
            let legacy = Commit::legacy(max_seq);
            for format in [QsaKvFormat::Fp8, F32] {
                // Discrete: a VMM backend's reserve and placement are legacy
                // storage's at this `max_seq`.
                let discrete =
                    Commit::for_expert_reserve(Backend::Vmm, false, max_seq, chunk, granularity);
                assert_eq!(discrete, legacy);
                assert_eq!(placement(&discrete, format), placement(&legacy, format));
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, false, max_seq, chunk, granularity),
                    legacy
                );
                // Unified memory is unchanged: VMM charges the first chunk's
                // pages, legacy all of `max_seq`.
                let vmm = Commit::new(Backend::Vmm, max_seq, chunk, granularity);
                let uma_vmm =
                    Commit::for_expert_reserve(Backend::Vmm, true, max_seq, chunk, granularity);
                assert_eq!(uma_vmm, vmm);
                assert_eq!(placement(&uma_vmm, format), placement(&vmm, format));
                assert!(placement(&uma_vmm, format).0 < placement(&legacy, format).0);
                assert!(placement(&uma_vmm, format).1 >= placement(&legacy, format).1);
                assert_eq!(
                    Commit::for_expert_reserve(Backend::Legacy, true, max_seq, chunk, granularity),
                    legacy
                );
            }
            assert!(
                mtp(&Commit::for_expert_reserve(
                    Backend::Vmm,
                    true,
                    max_seq,
                    chunk,
                    granularity
                )) < mtp(&legacy)
            );
        }
    }

    #[test]
    fn host_ram_check_refuses_below_headroom() {
        let host = 60u64 << 30;
        assert!(check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        let error = check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES - 1), 0).unwrap_err();
        assert!(error.contains("MemAvailable"), "{error}");
        assert!(check_host_ram(host, None, 0).is_err());
        assert!(check_host_ram(host, None, u64::MAX).is_err());
        assert!(check_host_ram(0, None, 0).is_ok());
        // The pool counts toward the room, byte for byte.
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES).is_ok());
        assert!(check_host_ram(host, Some(host), HOST_RAM_HEADROOM_BYTES - 1).is_err());
    }

    #[test]
    fn stage_mirrors_charge_every_rank_on_the_shared_host() {
        let full = 512 * (1_740_800 + 870_400);
        assert_eq!(
            expert_stage_mirror_bytes(&[full]).unwrap(),
            2 * full as u64 + 4 * HOST_MAPPED_PAD_BYTES
        );
        assert_eq!(
            expert_stage_mirror_bytes(&[full / 2; 2]).unwrap(),
            2 * full as u64 + 8 * HOST_MAPPED_PAD_BYTES
        );
        // An uneven seal charges each rank its own share.
        assert_eq!(
            expert_stage_mirror_bytes(&[300, 212]).unwrap(),
            2 * 512 + 8 * HOST_MAPPED_PAD_BYTES
        );
        assert!(expert_stage_mirror_bytes(&[]).is_err());
        assert!(expert_stage_mirror_bytes(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn slab_stage_mirrors_are_the_two_overwritten_arena_prefixes_plus_their_pads() {
        // mq2-xts: 512 E8P experts of 1,305,600 B per arena, plus the 4096 B
        // tail pad, one pinned restore copy per layer parity.
        let need = 512 * 1_305_600 + crate::expert_arena::ARENA_TAIL_PAD_BYTES;
        assert_eq!(
            slab_stage_mirror_bytes([need; 2]).unwrap(),
            2 * need as u64 + 2 * HOST_MAPPED_PAD_BYTES
        );
        // Two 637.5 MiB arenas: 1275 MiB of host copies, against the packed
        // stage's 2550 MiB, and never any VRAM.
        assert_eq!(
            (slab_stage_mirror_bytes([need; 2]).unwrap() - 2 * HOST_MAPPED_PAD_BYTES) >> 20,
            1275
        );
        // Uneven parities and overflow.
        assert_eq!(
            slab_stage_mirror_bytes([1000, 24]).unwrap(),
            1024 + 2 * HOST_MAPPED_PAD_BYTES
        );
        assert!(slab_stage_mirror_bytes([usize::MAX, 1]).is_err());
    }

    #[test]
    fn narrow_link_slab_stage_is_borrowed_charged_nothing_and_keeps_every_layer() {
        use crate::host_link::HostLink::Discrete;
        let narrow = Discrete { gbps: 7.1 };
        // The measured narrow link turns the stage on, but only the probe-free
        // switch charges placement, so a slab load is charged no dedicated
        // stage here: the stage it builds must borrow resident layers.
        assert!(expert_stage_switch("gfx1100", None, || narrow));
        assert!(!expert_stage_placement_switch("gfx1100", None));
        assert!(!dedicated_slab_stage(
            expert_stage_placement_switch("gfx1100", None),
            true,
            false
        ));
        // Same free VRAM: identical placement to the packed (borrowed) load
        // and to stage=0, and the chunk admission charges no stage either.
        let slab = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, narrow, true);
        let packed = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, narrow, false);
        let off = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("0"), narrow, true);
        for run in [&slab, &off] {
            assert_eq!(run.resident_layers, packed.resident_layers);
            assert_eq!(run.host_tensors, packed.host_tensors);
            assert_eq!(run.host_bytes, packed.host_bytes);
            assert_eq!(run.resident_bytes, packed.resident_bytes);
        }
        assert_eq!(slab.placement_rows, packed.placement_rows);
        assert_eq!(slab.placement_rows, off.placement_rows);
        // Both donors are the last two resident layers of that frozen
        // placement (whatever chunk policy sized it), parity ordered.
        let last = slab.resident_layers;
        let mut donors = [last - 2, last - 1];
        donors.sort_by_key(|layer| layer % 2);
        assert_eq!(slab.donors, Some(donors));
        // Explicit `1` keeps the charged dedicated stage (fewer layers), and
        // gfx1201 keeps its charged default, exactly as before.
        let explicit = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("1"), narrow, true);
        assert!(explicit.resident_layers < slab.resident_layers);
        assert!(dedicated_slab_stage(
            expert_stage_placement_switch("gfx1201", None),
            true,
            false
        ));
    }

    #[test]
    fn ep4_stage_mirrors_charge_all_four_ranks_against_host_and_gtt() {
        let full = 512 * (1_740_800 + 870_400);
        let charge = expert_stage_mirror_bytes(&[full / 4; 4]).unwrap();
        assert_eq!(charge, 2 * full as u64 + 16 * HOST_MAPPED_PAD_BYTES);
        // Same payload as EP2 (the ranks split it); every extra rank adds its own pads.
        let ep2 = expert_stage_mirror_bytes(&[full / 2; 2]).unwrap();
        assert_eq!(charge - ep2, 8 * HOST_MAPPED_PAD_BYTES);
        // Host RAM admission takes the total over all four ranks, byte for byte.
        assert!(check_host_ram(charge, Some(charge + HOST_RAM_HEADROOM_BYTES), 0).is_ok());
        assert!(check_host_ram(charge, Some(charge + HOST_RAM_HEADROOM_BYTES - 1), 0).is_err());
        // So does the GTT cap: one rank's quarter fits where the four-rank total does not.
        let budget = |limit_bytes| {
            Some(GttBudget {
                limit_bytes,
                used_bytes: 0,
            })
        };
        assert!(check_gtt_cap(charge, budget(charge)).is_ok());
        assert!(check_gtt_cap(charge, budget(charge - 1)).is_err());
        assert!(check_gtt_cap(charge / 4, budget(charge - 1)).is_ok());
    }

    /// `/proc/meminfo` text from `(field, MiB)` pairs.
    fn meminfo(fields: &[(&str, u64)]) -> String {
        fields
            .iter()
            .map(|(key, mib)| format!("{key}:{:>16} kB\n", mib << 10))
            .collect()
    }

    /// The measured 5-card gfx1201 host (MemTotal 128865156 kB, pool cap
    /// 16108144 pages), 5 x 16 MiB of idle GTT, and `pool_mib` of pages outside
    /// every counter on top of `baseline_mib`.
    fn gfx1201_host(pool_mib: u64, baseline_mib: u64) -> String {
        let total = 128_865_156 >> 10;
        let (anon, slab, sreclaim) = (14 << 10, 3 << 10, 2 << 10);
        let cached = total - pool_mib - baseline_mib - 80 - anon - slab - 1024;
        meminfo(&[
            ("MemTotal", total),
            ("MemFree", 1024),
            ("MemAvailable", 45_800),
            ("Cached", cached),
            ("SwapCached", 0),
            ("AnonPages", anon),
            ("Shmem", 2048),
            ("KReclaimable", sreclaim),
            ("Slab", slab),
            ("SReclaimable", sreclaim),
        ])
    }

    #[test]
    fn leftover_pool_admits_the_reload_and_a_real_shortage_still_refuses() {
        // N=12 host-maps 46.1 GiB; the reported refusal read MemAvailable 45.8 GiB.
        let host = 47_206u64 << 20;
        let available = Some(46_899u64 << 20);
        let gtt = 5 * (16 << 20);
        assert!(check_host_ram(host, available, 0).is_err());
        let leftover = ttm_pool_estimate_from(&gfx1201_host(59_737, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, leftover.unwrap()).is_ok());
        // The same MemAvailable with nothing parked in the pool.
        let empty = ttm_pool_estimate_from(&gfx1201_host(0, 2826), gtt, 16_108_144);
        let error = check_host_ram(host, available, empty.unwrap()).unwrap_err();
        assert!(
            error.contains("MemAvailable - CMA is 45.8 GiB (plus 0.0 GiB"),
            "{error}"
        );
        // A pool smaller than the 4.3 GiB gap still refuses.
        let small = ttm_pool_estimate_from(&gfx1201_host(3 << 10, 2826), gtt, 16_108_144);
        assert!(check_host_ram(host, available, small.unwrap()).is_err());
    }

    #[test]
    fn gtt_cap_counts_what_devices_already_hold() {
        // This host: pages_limit 16108144 (61.4 GiB); N=12 host-maps 46.1 GiB.
        let limit_bytes = 16_108_144 * TTM_PAGE_BYTES;
        let host = 46u64 << 30;
        let fresh = GttBudget {
            limit_bytes,
            used_bytes: 16 << 20,
        };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(
            error.contains("pages_limit") && error.contains("16108144 pages"),
            "{error}"
        );
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget {
            limit_bytes,
            used_bytes: 46 << 30,
        };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn parse_accepts_auto_and_counts_only() {
        assert_eq!(
            parse_expert_vram_layers(" auto "),
            Ok(ExpertVramLayers::Auto)
        );
        assert_eq!(
            parse_expert_vram_layers("16"),
            Ok(ExpertVramLayers::Layers(16))
        );
        assert!(parse_expert_vram_layers("16GB").is_err());
    }

    #[test]
    fn unset_is_resident_only_where_everything_fits_and_explicit_wins() {
        const GIB: u64 = 1 << 30;
        // Flash-Next: ~5 GiB non-expert, ~64 GiB experts, 6.5 GiB reserve.
        let r9700 = fits_fully_resident(32 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        let halo = fits_fully_resident(120 * GIB, 5 * GIB, 64 * GIB, 6 * GIB + GIB / 2);
        assert!(!r9700 && halo);
        assert!(fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB));
        assert!(!fits_fully_resident(10 * GIB, 2 * GIB, 7 * GIB, GIB + 1));
        assert!(!fits_fully_resident(u64::MAX, 1, u64::MAX, 0));
        assert_eq!(resolve_expert_vram_layers(None, || Ok(true)), Ok(None));
        assert_eq!(
            resolve_expert_vram_layers(None, || Ok(false)),
            Ok(Some(ExpertVramLayers::Auto))
        );
        assert!(resolve_expert_vram_layers(None, || Err("vram".into())).is_err());
        // Explicit values never consult the fit.
        for policy in [
            ExpertVramLayers::Layers(12),
            ExpertVramLayers::Layers(99),
            ExpertVramLayers::Auto,
        ] {
            let resolved =
                resolve_expert_vram_layers(Some(policy), || panic!("explicit consulted the fit"));
            assert_eq!(resolved, Ok(Some(policy)));
        }
    }

    #[test]
    fn only_qwen4_on_discrete_gpus_keeps_host_memory_out_of_reclaim() {
        assert!(keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1201"]
        ));
        assert!(keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1100", "gfx942"]
        ));
        // Another model on the same card (arch 5: the Qwen3.5-family trunk).
        assert!(!keeps_host_memory_out_of_reclaim(5, &["gfx1201"]));
        // Unified memory, alone or beside a discrete card ROCr also exposes.
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1151"]
        ));
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx1100", "gfx1151"]
        ));
        // No card known, or one whose memory kind is not known.
        assert!(!keeps_host_memory_out_of_reclaim(crate::ARCH_ID, &[]));
        assert!(!keeps_host_memory_out_of_reclaim(
            crate::ARCH_ID,
            &["gfx90a"]
        ));
    }

    /// Total payload bytes of a manifest entry at 2 bytes per element.
    fn element_bytes(entry: &WeightEntry) -> Option<u64> {
        Some(entry.logical_shape.iter().product::<usize>() as u64 * 2)
    }

    #[test]
    fn dense_profile_has_no_routed_experts_and_places_nothing() {
        use crate::config::Qwen4FfnConfig;
        let config = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        );
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("dense manifest");
        assert!(config.ffn.is_dense());
        assert_eq!(routed_expert_bytes(&manifest.weights, element_bytes), Ok(0));
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        assert_eq!(layer_experts, 0);
        // The dense gate/up/down are resident non-expert weights.
        let dense_ffn: u64 = manifest
            .weights
            .iter()
            .filter(|entry| {
                entry.name.ends_with(".mlp.gate_proj.weight")
                    || entry.name.ends_with(".mlp.up_proj.weight")
                    || entry.name.ends_with(".mlp.down_proj.weight")
            })
            .map(|entry| {
                assert_eq!(entry.residency, WeightResidency::Resident, "{}", entry.name);
                element_bytes(entry).unwrap()
            })
            .sum();
        assert!(dense_ffn > 0 && non_expert >= dense_ffn);
        // Fully resident, no division by the zero per-layer expert bytes,
        // nothing host-mapped under any placement.
        assert!(fits_fully_resident(
            non_expert + (1 << 20),
            non_expert,
            0,
            1 << 20
        ));
        assert!(!fits_fully_resident(
            non_expert + (1 << 20) - 1,
            non_expert,
            0,
            1 << 20
        ));
        assert_eq!(auto_vram_layers(1 << 40, non_expert, 0, 4, 1 << 20), 4);
        assert_eq!(auto_vram_layers(0, non_expert, 0, 4, 1 << 20), 4);
        for vram_layers in [0, 2, 4] {
            assert_eq!(place_routed_experts(&mut manifest.weights, vram_layers), 0);
        }
        assert_eq!(host_mapped_bytes(&manifest.weights, element_bytes), Ok(0));
        assert!(manifest
            .weights
            .iter()
            .all(|entry| entry.residency != WeightResidency::HostMapped));
    }

    #[test]
    fn host_expert_requests_are_refused_for_a_dense_profile_only_when_explicit() {
        assert_eq!(refuse_host_expert_request(None, "dense-dev"), Ok(()));
        for policy in [
            ExpertVramLayers::Auto,
            ExpertVramLayers::Layers(0),
            ExpertVramLayers::Layers(12),
        ] {
            let error = refuse_host_expert_request(Some(policy), "dense-dev").unwrap_err();
            assert!(
                error.contains(EXPERT_VRAM_LAYERS_ENV)
                    && error.contains("dense-dev")
                    && error.contains("no routed experts"),
                "{error}"
            );
        }
    }

    #[test]
    fn flash_next_placement_is_unchanged_by_the_dense_path() {
        // The pinned MoE manifest still counts every routed expert (trunk and
        // MTP) as an expert, none as non-expert, and host-maps exactly them.
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        assert!(config.ffn.is_moe());
        let routed = manifest
            .weights
            .iter()
            .filter(|entry| is_routed_expert(&entry.name))
            .count();
        assert_eq!(routed, 2 * (config.num_hidden_layers + 1));
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        assert!(layer_experts > 0);
        let experts = routed_expert_bytes(&manifest.weights, element_bytes).unwrap();
        assert!(experts > layer_experts && non_expert > 0);
        assert_eq!(place_routed_experts(&mut manifest.weights, 0), routed);
        assert!(host_mapped_bytes(&manifest.weights, element_bytes).unwrap() > 0);
        assert_eq!(
            auto_vram_layers(
                u64::MAX / 2,
                non_expert,
                layer_experts,
                config.num_hidden_layers,
                0
            ),
            config.num_hidden_layers
        );
    }

    #[test]
    fn ep_ranks_split_only_the_routed_expert_bytes() {
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let (non_expert, layer_experts) =
            resident_split(&manifest.weights, element_bytes).expect("split");
        let experts = routed_expert_bytes(&manifest.weights, element_bytes).unwrap();
        // A single-rank span is the single-device arithmetic.
        let single_spans = EpExpertSpans::single();
        assert_eq!(
            resident_split_ranks(&manifest.weights, element_bytes, &single_spans),
            Ok((non_expert, layer_experts))
        );
        assert_eq!(
            routed_expert_bytes_ranks(&manifest.weights, element_bytes, &single_spans),
            Ok(experts)
        );
        for r in [2usize, 3, 4, 5, 6, 8] {
            // The trunk is replicated; every rank budgets the largest share:
            // the per-expert bytes times `max_local_count` experts.
            let max = Qwen4EpPlacement { rank: 0, ranks: r }.max_local_count(512) as u64;
            let spans = stride_spans(512, r);
            assert_eq!(
                resident_split_ranks(&manifest.weights, element_bytes, &spans),
                Ok((non_expert, layer_experts / 512 * max)),
                "ranks={r}"
            );
            assert_eq!(
                routed_expert_bytes_ranks(&manifest.weights, element_bytes, &spans),
                Ok(experts / 512 * max),
                "ranks={r}"
            );
        }
        // Every rank host-maps its own share; the ranks' charges sum to the
        // exact payload (plus one pad per tensor per rank), and one rank
        // equals host_mapped_bytes.
        let routed = manifest
            .weights
            .iter()
            .filter(|entry| is_routed_expert(&entry.name))
            .count();
        assert_eq!(place_routed_experts(&mut manifest.weights, 0), routed);
        let single = host_mapped_bytes(&manifest.weights, element_bytes).unwrap();
        assert_eq!(
            host_mapped_bytes_ranks(&manifest.weights, element_bytes, &single_spans),
            Ok(vec![single])
        );
        let pad = routed as u64 * HOST_MAPPED_PAD_BYTES;
        for ranks in [2usize, 3, 4, 5, 6, 8] {
            let charges = host_mapped_bytes_ranks(
                &manifest.weights,
                element_bytes,
                &stride_spans(512, ranks),
            )
            .unwrap();
            assert_eq!(charges.len(), ranks);
            for (rank, &charge) in charges.iter().enumerate() {
                let local = Qwen4EpPlacement { rank, ranks }.local_count(512) as u64;
                assert_eq!(
                    charge,
                    (single - pad) / 512 * local + pad,
                    "ranks={ranks} rank={rank}"
                );
            }
            // The host charge equals the full payload: nothing lost or doubled.
            assert_eq!(
                charges.iter().sum::<u64>(),
                (single - pad) + pad * ranks as u64,
                "ranks={ranks}"
            );
        }
        // A seal that is not the stride split charges what it seals: the
        // largest rank budgets residency, each rank pins its own share.
        let uneven = EpExpertSpans::new(512, vec![300, 212]).unwrap();
        assert_eq!(uneven.largest_rank(), 0);
        assert_eq!(
            routed_expert_bytes_ranks(&manifest.weights, element_bytes, &uneven),
            Ok(experts / 512 * 300)
        );
        assert_eq!(
            host_mapped_bytes_ranks(&manifest.weights, element_bytes, &uneven),
            Ok(vec![
                (single - pad) / 512 * 300 + pad,
                (single - pad) / 512 * 212 + pad
            ])
        );
        // Spans that are not a cover of every expert by two or more ranks,
        // or that split another expert count, are refused.
        assert!(EpExpertSpans::new(512, vec![512]).is_err());
        assert!(EpExpertSpans::new(512, vec![512, 0]).is_err());
        assert!(EpExpertSpans::new(512, vec![256, 255]).is_err());
        let other = EpExpertSpans::new(256, vec![128, 128]).unwrap();
        assert!(host_mapped_bytes_ranks(&manifest.weights, element_bytes, &other).is_err());
        // A stack whose bytes are not a multiple of its expert count is refused.
        assert!(
            resident_split_ranks(&manifest.weights, |_| Some(1), &stride_spans(512, 3)).is_err()
        );
    }

    /// The `e % ranks` spans of `experts` experts (what the seal gives today).
    fn stride_spans(experts: usize, ranks: usize) -> EpExpertSpans {
        EpExpertSpans::new(
            experts,
            (0..ranks)
                .map(|rank| Qwen4EpPlacement { rank, ranks }.local_count(experts))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn uneven_ep_split_uses_the_largest_rank_share() {
        let config = crate::config::compact_test_config();
        assert_eq!(moe_layer_count(&config), 49);
        let (free, non_expert, reserve) = (24_520u64 << 20, 5_115u64 << 20, 6_656u64 << 20);
        let per_expert = 1_336_934_400u64 / 512;
        let split = |ranks: usize| {
            let max = Qwen4EpPlacement { rank: 0, ranks }.max_local_count(512) as u64;
            ep_vram_layers(
                &config,
                &[free; 8][..ranks],
                non_expert,
                per_expert * max,
                reserve,
            )
        };
        assert_eq!(split(2), 19);
        assert_eq!(split(4), 39);
        assert_eq!(split(6), 49);
        // N = 3 keeps 29 layers: an uneven 171/171/170 split budgets 171 experts.
        assert_eq!(split(3), 29);
    }

    #[test]
    fn ep_auto_layers_use_the_per_rank_expert_bytes() {
        let free = 30u64 << 30;
        let non_expert = 5_990_000_000u64;
        let layer = 1_336_900_000u64;
        let reserve = 6_656u64 << 20;
        let single = auto_vram_layers(free, non_expert, layer, 48, reserve);
        let ep4 = auto_vram_layers(free, non_expert, layer / 4, 48, reserve);
        let ep2 = auto_vram_layers(free, non_expert, layer / 2, 48, reserve);
        assert!(single < ep2 && ep2 < ep4);
        // EP4 at 30 GiB keeps every trunk layer resident; EP2 does not.
        assert_eq!(ep4, 48);
        assert!(ep2 < 48);
    }

    #[test]
    fn ep_fit_report_prints_one_line_per_rank() {
        let config = crate::config::compact_test_config();
        let free = [30u64 << 30, 29 << 30, 30 << 30, 28 << 30];
        let report = ep_fit_report(
            &config,
            4,
            &free,
            5_990_000_000,
            334_225_000,
            6_656u64 << 20,
            32768,
            1536,
        );
        let lines: Vec<_> = report.lines().collect();
        assert_eq!(lines.len(), 5, "{report}");
        assert!(
            lines[0].contains("4 ranks") && lines[0].contains("32768") && lines[0].contains("1536")
        );
        for (rank, line) in lines[1..].iter().enumerate() {
            assert!(
                line.starts_with(&format!("qwen4 EP rank {rank}/4:")),
                "{line}"
            );
            for part in [
                "free",
                "trunk",
                "experts",
                "resident expert layers",
                "reserve",
                "headroom",
            ] {
                assert!(line.contains(part), "{part} missing from {line}");
            }
            // 49 MoE layers incl. the MTP head.
            assert!(line.contains("of 49"), "{line}");
        }
        // EP4 at 28..30 GiB free keeps all 49 MoE layers (the MTP layer too) resident.
        assert!(lines[0].contains("VRAM split 49 of 49"), "{}", lines[0]);
        assert!(
            lines[1].contains("resident expert layers 49 of 49 (0 host-mapped"),
            "{}",
            lines[1]
        );
        // Headroom: 30 GiB - 5.99 GB trunk - 49 x 0.334 GB - 6.5 GiB reserve > 0 on rank 0.
        assert!(lines[1].contains("headroom +"), "{}", lines[1]);
        // A rank with too little VRAM for the reserve reports negative headroom.
        let tight = ep_fit_rank_line(
            &config,
            1,
            2,
            8 << 30,
            5_990_000_000,
            668_450_000,
            6_656u64 << 20,
            0,
        );
        assert!(tight.contains("headroom -"), "{tight}");
        assert!(tight.contains("resident expert layers 0 of 49"), "{tight}");
    }

    #[test]
    fn ep_split_is_the_minimum_over_ranks_and_identical_everywhere() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let rank = |free| ep_rank_layers(&config, free, non_expert, layer, reserve);
        // EP2-sized layers: 49 x 0.668 GB does not fit 30 GiB with the reserve.
        let roomy = rank(30 << 30);
        let tight = rank(24 << 30);
        assert!(tight < roomy && roomy < 49 && tight > 0, "{tight} {roomy}");
        assert_eq!(rank(64 << 30), 49, "everything fits");
        assert_eq!(rank(1 << 30), 0, "reserve alone does not fit");
        let free = [30u64 << 30, 24 << 30, 30 << 30, 29 << 30];
        assert_eq!(
            ep_vram_layers(&config, &free, non_expert, layer, reserve),
            tight
        );
        assert_eq!(
            ep_vram_layers(&config, &free[..1], non_expert, layer, reserve),
            roomy
        );
        assert_eq!(ep_vram_layers(&config, &[], non_expert, layer, reserve), 0);
        // The report applies the minimum to every rank.
        let report = ep_fit_report(&config, 4, &free, non_expert, layer, reserve, 32768, 1536);
        let lines: Vec<_> = report.lines().collect();
        assert!(
            lines[0].contains(&format!("VRAM split {tight} of 49")),
            "{}",
            lines[0]
        );
        for line in &lines[1..] {
            assert!(
                line.contains(&format!("resident expert layers {tight} of 49")),
                "{line}"
            );
        }
        assert!(
            lines[2].contains(&format!("this rank alone fits {tight}")),
            "{}",
            lines[2]
        );
        assert!(
            lines[1].contains(&format!("this rank alone fits {roomy}")),
            "{}",
            lines[1]
        );
    }

    #[test]
    fn ep_lane_reservation_subtracts_exactly_the_lane_bytes_from_every_rank() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let free = [30u64 << 30, 29 << 30, 30 << 30, 30 << 30];
        let baseline = ep_vram_layers(&config, &free, non_expert, layer, reserve);
        assert!(baseline > 0 && baseline < 49, "{baseline}");
        // No lane bytes: the baseline twice, identical plan.
        let off = ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, 0).unwrap();
        assert_eq!(
            off,
            EpLaneReservedSplit {
                lane_bytes: 0,
                baseline_layers: baseline,
                reserved_layers: baseline,
            }
        );
        // Lane bytes equal to whole layers cost exactly that many layers on
        // the tightest rank, no more and no fewer.
        let one =
            ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, layer).unwrap();
        assert_eq!(one.baseline_layers, baseline);
        assert_eq!(one.reserved_layers, baseline - 1);
        let four = ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, 4 * layer)
            .unwrap();
        assert_eq!(four.reserved_layers, baseline - 4);
        // The reservation is the sum added to the reserve, byte for byte.
        let lane = 3_221_225_472u64;
        let split =
            ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, lane).unwrap();
        assert_eq!(
            split.reserved_layers,
            ep_vram_layers(&config, &free, non_expert, layer, reserve + lane)
        );
        assert!(split.reserved_layers < split.baseline_layers);
        // Never above the baseline, and monotone in the reserved bytes.
        let mut last = baseline;
        for lane in [1u64 << 20, 1 << 28, 1 << 30, 2 << 30, 4 << 30] {
            let split = ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, lane)
                .unwrap();
            assert!(split.reserved_layers <= last, "{lane}");
            last = split.reserved_layers;
        }
    }

    #[test]
    fn ep_lane_reservation_refuses_overflow_and_impossible_reserve() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let free = [30u64 << 30; 4];
        let overflow =
            ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, u64::MAX)
                .unwrap_err();
        assert!(overflow.contains("overflows"), "{overflow}");
        let overflow =
            ep_vram_layers_with_lanes(&config, &free, u64::MAX, layer, 0, 1).unwrap_err();
        assert!(overflow.contains("overflows"), "{overflow}");
        // Even with every expert in host RAM the lane store does not fit
        // the tightest rank: the error names that rank.
        let free = [30u64 << 30, 30 << 30, 10 << 30, 30 << 30];
        let error = ep_vram_layers_with_lanes(&config, &free, non_expert, layer, reserve, 4 << 30)
            .unwrap_err();
        assert!(
            error.contains("rank 2") && error.contains("lane store"),
            "{error}"
        );
        // The boundary: exactly the floor fits (zero resident layers), one
        // byte less is refused.
        let lane = 2u64 << 30;
        let floor = non_expert + reserve + lane;
        let at =
            ep_vram_layers_with_lanes(&config, &[floor], non_expert, layer, reserve, lane).unwrap();
        assert_eq!(at.reserved_layers, 0);
        assert!(
            ep_vram_layers_with_lanes(&config, &[floor - 1], non_expert, layer, reserve, lane)
                .is_err()
        );
    }

    #[test]
    fn ep_lane_reservation_plan_is_the_baseline_without_a_request() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let free = [30u64 << 30, 29 << 30, 30 << 30, 30 << 30];
        let plan = ep_vram_plan(&config, &free, non_expert, layer, reserve, None);
        assert_eq!(
            plan,
            EpVramPlan {
                layers: ep_vram_layers(&config, &free, non_expert, layer, reserve),
                reservation: None,
                unreserved: None,
            }
        );
        assert_eq!(ep_lane_reservation_line(32768, &plan), None);
    }

    #[test]
    fn ep_lane_reservation_plan_trades_layers_for_lane_bytes_and_logs_once() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let free = [30u64 << 30, 29 << 30, 30 << 30, 30 << 30];
        let baseline = ep_vram_layers(&config, &free, non_expert, layer, reserve);
        let lane = EpLaneRequest {
            lanes: 4,
            row_budget: 4096,
            bytes: Ok(2 * layer),
        };
        let plan = ep_vram_plan(&config, &free, non_expert, layer, reserve, Some(lane));
        assert_eq!(plan.layers, baseline - 2);
        assert_eq!(plan.unreserved, None);
        let reservation = plan.reservation.clone().unwrap();
        assert_eq!(
            (
                reservation.lanes,
                reservation.row_budget,
                reservation.lane_bytes,
                reservation.baseline_layers
            ),
            (4, 4096, 2 * layer, baseline)
        );
        let line = ep_lane_reservation_line(32768, &plan).unwrap();
        assert_eq!(
            line,
            format!(
                "[daemon] qwen4 EP lane reservation: lanes=4 max_seq=32768 -> {:.3} GiB/rank; resident expert layers {baseline} -> {}",
                (2 * layer) as f64 / (1u64 << 30) as f64,
                baseline - 2
            )
        );
        assert!(!line.contains('\n'));
    }

    #[test]
    fn ep_lane_reservation_plan_keeps_the_baseline_when_impossible() {
        let config = crate::config::compact_test_config();
        let (non_expert, layer, reserve) = (5_990_000_000u64, 668_450_000u64, 6_656u64 << 20);
        let free = [30u64 << 30, 30 << 30, 10 << 30, 30 << 30];
        let baseline = ep_vram_layers(&config, &free, non_expert, layer, reserve);
        for bytes in [
            Ok(4u64 << 30),
            Err("lane store device bytes overflow".to_string()),
        ] {
            let lane = EpLaneRequest {
                lanes: 8,
                row_budget: 8,
                bytes,
            };
            let plan = ep_vram_plan(&config, &free, non_expert, layer, reserve, Some(lane));
            assert_eq!(plan.layers, baseline);
            assert_eq!(plan.reservation, None);
            let line = ep_lane_reservation_line(32768, &plan).unwrap();
            assert!(
                line.starts_with("[daemon] qwen4 EP lane reservation skipped:")
                    && line.contains(&format!("resident expert layers {baseline}")),
                "{line}"
            );
        }
    }

    fn band_scopes(config: &Qwen4Config, counts: &[usize]) -> Vec<Qwen4BandScope> {
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

    fn census_of(
        config: &Qwen4Config,
        weights: &[WeightEntry],
        scope: &Qwen4BandScope,
    ) -> BandCensus {
        let context = crate::Qwen4ContextCommit::legacy(1024);
        band_census(
            config,
            weights,
            element_bytes,
            scope,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .expect("band census")
    }

    #[test]
    fn single_band_census_equals_todays_whole_model_queries() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let single = Qwen4BandScope::single(&config);
        let census = census_of(&config, weights, &single);
        let (non_expert, _) = resident_split(weights, element_bytes).unwrap();
        let routed = routed_expert_bytes(weights, element_bytes).unwrap();
        assert!(routed > 0 && non_expert > 0);
        assert_eq!(census.non_expert_bytes, non_expert);
        assert_eq!(census.routed_bytes(), Ok(routed));
        assert_eq!(
            routed_expert_bytes_scope(weights, element_bytes, &single),
            Ok(routed)
        );
        assert_eq!(census.endpoint_duplicate_bytes, 0);
        assert_eq!(census.moe_layers, moe_layer_count(&config));
        assert_eq!(
            moe_layer_count_scope(&config, &single),
            moe_layer_count(&config)
        );
        assert_eq!(
            census.f16_shadow_bytes,
            f16_shadow_bytes(weights, config.num_hidden_layers).unwrap()
        );
        let context = crate::Qwen4ContextCommit::legacy(1024);
        assert_eq!(
            census.state,
            state_census(&config, target_state_format(), &context, 256, Full).unwrap()
        );
        assert_eq!(
            refuse_band_with_expert_parallel(&config, &single, 4),
            Ok(())
        );
    }

    #[test]
    fn pp_band_censuses_sum_to_single_plus_only_the_declared_endpoint_duplicates() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let single = census_of(&config, weights, &Qwen4BandScope::single(&config));
        let embed = weights
            .iter()
            .find(|entry| entry.name == "model.language_model.embed_tokens.weight")
            .map(|entry| element_bytes(entry).unwrap())
            .expect("embedding");
        let layer_bytes = single.routed_trunk_bytes / config.num_hidden_layers as u64;
        assert!(layer_bytes > 0 && single.routed_mtp_bytes > 0);
        assert_eq!(single.max_layer_routed_bytes, layer_bytes);

        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let scopes = band_scopes(&config, counts);
            let ranks: Vec<_> = scopes
                .iter()
                .map(|scope| census_of(&config, weights, scope))
                .collect();
            let sum = |field: fn(&BandCensus) -> u64| ranks.iter().map(field).sum::<u64>();
            // Trunk experts: every owned layer exactly once, all experts.
            assert_eq!(
                sum(|c| c.routed_trunk_bytes),
                single.routed_trunk_bytes,
                "{counts:?}"
            );
            for (census, count) in ranks.iter().zip(counts) {
                assert_eq!(census.routed_trunk_bytes, layer_bytes * *count as u64);
                assert_eq!(census.max_layer_routed_bytes, layer_bytes);
                assert_eq!(census.band.len(), *count);
            }
            // The MTP head's experts and layer only on the output rank.
            let last = ranks.len() - 1;
            for (rank, census) in ranks.iter().enumerate() {
                let mtp = rank == last;
                assert_eq!(
                    census.routed_mtp_bytes,
                    if mtp { single.routed_mtp_bytes } else { 0 }
                );
                assert_eq!(census.moe_layers, counts[rank] + usize::from(mtp));
            }
            assert_eq!(
                ranks.iter().map(|c| c.moe_layers).sum::<usize>(),
                single.moe_layers
            );
            assert_eq!(sum(|c| c.routed_mtp_bytes), single.routed_mtp_bytes);
            assert_eq!(sum(|c| c.non_expert_bytes), single.non_expert_bytes);
            assert_eq!(sum(|c| c.f16_shadow_bytes), single.f16_shadow_bytes);
            // The only duplicate is the tied MTP embedding at the output endpoint.
            assert_eq!(sum(|c| c.endpoint_duplicate_bytes), embed);
            assert_eq!(ranks[last].endpoint_duplicate_bytes, embed);
            assert_eq!(
                ranks.iter().map(|c| c.weight_bytes().unwrap()).sum::<u64>(),
                single.weight_bytes().unwrap() + embed
            );
            // State: owned layers only, PLE and feedback only on the input band.
            assert_eq!(sum(|c| c.state.qsa_context), single.state.qsa_context);
            assert_eq!(sum(|c| c.state.state), single.state.state);
            // Forward scratch: input-only and output-only fields exactly once,
            // the per-stage shared copy and the GDN/QSA fields on every band
            // that runs them (every band here has both layer kinds).
            assert_eq!(sum(|c| c.forward.input_only), single.forward.input_only);
            assert_eq!(sum(|c| c.forward.output_only), single.forward.output_only);
            for (rank, census) in ranks.iter().enumerate() {
                assert_eq!(census.forward.stage_shared, single.forward.stage_shared);
                assert_eq!(census.forward.band_local, single.forward.band_local);
                assert_eq!(census.state.forward, census.forward.total().unwrap());
                assert_eq!(census.forward.input_only == 0, rank != 0);
                assert_eq!(census.forward.output_only == 0, rank != last);
            }
            // Input holds the embedding, output the head: heavier than a middle rank.
            if ranks.len() > 2 {
                let middle = &ranks[1];
                assert!(ranks[0].non_expert_bytes > middle.non_expert_bytes);
                assert!(ranks[last].non_expert_bytes > middle.non_expert_bytes);
            }
        }
    }

    #[test]
    fn native_mtp_single_scope_census_is_the_single_load_formula() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let head = language_head_dtype(weights).expect("language head");
        let single = Qwen4BandScope::single(&config);
        let context = crate::Qwen4ContextCommit::legacy(1024);
        for gdn in [
            None,
            Some((crate::GdnStateFormat::Q8, 32)),
            Some((crate::GdnStateFormat::F32, 32)),
        ] {
            for batched_fill in [false, true] {
                for max_k in [1usize, 3, 10, 64] {
                    let inputs = NativeMtpCensusInputs {
                        max_k,
                        row_capture: gdn,
                        batched_fill,
                    };
                    let census =
                        native_mtp_census_scope(&config, weights, &context, 256, &inputs, &single)
                            .expect("single census");
                    assert_eq!(
                        census.reserve().ok(),
                        crate::mtp_spec::native_mtp_device_bytes(
                            &config,
                            &context,
                            256,
                            max_k,
                            head,
                            gdn,
                            batched_fill,
                        ),
                        "gdn {gdn:?} fill {batched_fill} K {max_k}"
                    );
                    assert!(census.head_resident > 0 && census.request > 0);
                    assert_eq!(census.row_capture > 0, gdn.is_some());
                }
            }
        }
    }

    #[test]
    fn native_mtp_pp_bands_charge_their_own_capture_and_only_the_output_the_head() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let gdn = crate::GdnStateFormat::Q8;
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((gdn, 4)),
            batched_fill: true,
        };
        let whole = native_mtp_census_scope(
            &config,
            weights,
            &context,
            256,
            &inputs,
            &Qwen4BandScope::single(&config),
        )
        .unwrap();
        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let scopes = band_scopes(&config, counts);
            for scope in &scopes {
                let census =
                    native_mtp_census_scope(&config, weights, &context, 256, &inputs, scope)
                        .unwrap();
                let gdn_layers = scope
                    .band
                    .clone()
                    .filter(|&layer| {
                        config.layer_types[layer] == crate::config::LayerType::LinearAttention
                    })
                    .count();
                // Every band: the capture of its own GDN layers, four rows.
                assert!(gdn_layers > 0);
                assert_eq!(
                    census.row_capture,
                    crate::state::Qwen4State::row_capture_bytes_scoped(&config, gdn, 4, gdn_layers)
                        .unwrap() as u64,
                    "{counts:?} {:?}",
                    scope.band
                );
                // No band is charged the whole model's capture.
                assert!(census.row_capture < whole.row_capture, "{counts:?}");
                if scope.output {
                    // The head and the request buffers once, on the output band.
                    assert_eq!(census.head_resident, whole.head_resident);
                    assert_eq!(census.head_build_scratch, whole.head_build_scratch);
                    assert_eq!(census.request, whole.request);
                    assert_eq!(
                        census.reserve().unwrap(),
                        census.head_resident
                            + census
                                .head_build_scratch
                                .max(census.request + census.row_capture)
                    );
                } else {
                    assert_eq!(
                        (
                            census.head_resident,
                            census.head_build_scratch,
                            census.request
                        ),
                        (0, 0, 0)
                    );
                    assert_eq!(census.reserve().unwrap(), census.row_capture);
                }
            }
            // Row capture off (a GPU without the few-row GDN route): no band
            // charges it, the output band keeps the head and the request.
            let off = NativeMtpCensusInputs {
                row_capture: None,
                ..inputs
            };
            for scope in &scopes {
                let census =
                    native_mtp_census_scope(&config, weights, &context, 256, &off, scope).unwrap();
                assert_eq!(census.row_capture, 0);
                assert_eq!(census.reserve().unwrap() > 0, scope.output);
            }
        }
    }

    #[test]
    fn band_census_adds_only_the_mtp_terms_and_ar_has_none() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((crate::GdnStateFormat::Q8, 4)),
            batched_fill: false,
        };
        for scope in band_scopes(&config, &[24, 24]) {
            let ar = census_of(&config, weights, &scope);
            assert_eq!(ar.mtp, NativeMtpCensus::default());
            // The head's weights are loaded (and counted) on the output band
            // whether or not the head attaches.
            assert_eq!(ar.routed_mtp_bytes > 0, scope.output);
            let with = band_census_native_mtp(
                &config,
                weights,
                element_bytes,
                &scope,
                target_state_format(),
                &context,
                256,
                Full,
                Some(&inputs),
            )
            .unwrap();
            assert_eq!(
                with.mtp,
                native_mtp_census_scope(&config, weights, &context, 256, &inputs, &scope).unwrap()
            );
            assert!(with.mtp.reserve().unwrap() > 0);
            assert_eq!(
                BandCensus {
                    mtp: NativeMtpCensus::default(),
                    ..with
                },
                ar
            );
        }
    }

    #[test]
    fn native_mtp_census_refuses_a_profile_without_a_head() {
        use crate::config::Qwen4FfnConfig;
        let headless = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            false,
        );
        let inputs = NativeMtpCensusInputs {
            max_k: 3,
            row_capture: None,
            batched_fill: false,
        };
        let error = native_mtp_census_scope(
            &headless,
            &[],
            &crate::Qwen4ContextCommit::legacy(64),
            8,
            &inputs,
            &Qwen4BandScope::single(&headless),
        )
        .unwrap_err();
        assert!(error.contains("no MTP head"), "{error}");
    }

    #[test]
    fn single_forward_census_equals_the_whole_model_forward_bytes() {
        use crate::config::Qwen4FfnConfig;
        let dense = crate::config::synthetic_test_config(
            Qwen4FfnConfig::Dense {
                intermediate_size: 8,
            },
            16,
            4,
            true,
            true,
        );
        let mut full_seen = 0;
        for config in [crate::config::compact_test_config(), dense] {
            let single = Qwen4BandScope::single(&config);
            for rows in [1usize, 7, 64, 256, 513, 2048, 8192] {
                for mode in [Full, FnSymGfx1201] {
                    let scoped = forward_census_scope(&config, rows, mode, &single)
                        .and_then(|census| census.total())
                        .ok();
                    let whole = crate::gpu_forward::qwen4_forward_device_bytes(&config, rows, mode);
                    assert_eq!(scoped, whole, "rows {rows} {mode:?}");
                    full_seen += usize::from(scoped.is_some() && mode == FnSymGfx1201);
                }
            }
        }
        assert!(
            full_seen > 0,
            "the compact plan must be compared at least once"
        );
    }

    #[test]
    fn compact_forward_census_plans_a_gdn_band_without_a_qsa_layer() {
        let config = crate::config::compact_test_config();
        // Layers 0..3 are GDN, the first QSA layer is 3.
        let gdn_only = Qwen4BandScope {
            band: 0..3,
            input: true,
            output: false,
        };
        let compact = forward_census_scope(&config, 256, FnSymGfx1201, &gdn_only)
            .expect("compact plan of a GDN band");
        assert!(compact.stage_shared > 0 && compact.output_only == 0);
        let full = forward_census_scope(&config, 256, Full, &gdn_only).expect("full plan");
        assert!(full.band_local > 0 && full.output_only == 0 && full.input_only > 0);
        let single = forward_census_scope(&config, 256, Full, &Qwen4BandScope::single(&config))
            .expect("single");
        // No QSA scratch and a narrower projection than the whole model.
        assert!(full.band_local < single.band_local && full.stage_shared < single.stage_shared);
    }

    #[test]
    fn band_census_of_a_scope_filtered_manifest_equals_the_full_manifests() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let embed = manifest
            .weights
            .iter()
            .find(|entry| entry.name == "model.language_model.embed_tokens.weight")
            .map(|entry| element_bytes(entry).unwrap())
            .expect("embedding");
        for counts in [&[48usize][..], &[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            for scope in band_scopes(&config, counts) {
                let scoped = manifest
                    .clone()
                    .with_band_scope(&config, &scope)
                    .expect("scope-filtered manifest");
                if !scope.input {
                    // The tied source is gone from the filtered manifest.
                    assert!(scoped
                        .weights
                        .iter()
                        .all(|entry| entry.name != "model.language_model.embed_tokens.weight"));
                }
                let full = census_of(&config, &manifest.weights, &scope);
                let filtered = census_of(&config, &scoped.weights, &scope);
                assert_eq!(full, filtered, "{counts:?} band {:?}", scope.band);
                let duplicate = if scope.output && !scope.input {
                    embed
                } else {
                    0
                };
                assert_eq!(
                    filtered.endpoint_duplicate_bytes, duplicate,
                    "{:?}",
                    scope.band
                );
            }
        }
    }

    #[test]
    fn slab_arena_band_census_charges_each_band_its_own_arenas() {
        use crate::expert_slabs::ExpertFormat;
        // Non-uniform arenas: layer `l` holds `(5 * l) % 64` MQ4 experts and W3 for the rest.
        let tensors = crate::expert_slab_table::full_fn_headers(|layer, expert| {
            if expert < (layer * 5) % 64 {
                ExpertFormat::Mq4Xts
            } else {
                ExpertFormat::Mq3I4
            }
        });
        let table = crate::expert_slab_table::build_expert_slab_table(
            &crate::expert_slabs::ExpertSlabsV1::fn_v1(),
            &tensors,
        )
        .unwrap();
        let plan =
            std::sync::Arc::new(crate::expert_arena::ExpertArenaPlan::from_table(&table).unwrap());
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build_with_recipe(
            &config,
            crate::weights::Qwen4TierRecipe::FlashNextSlab {
                trunk_tier: DType::Q8_0,
                mtp_source: false,
                arenas: plan.clone(),
            },
        )
        .expect("slab manifest");
        // An arena's bytes are its raw length; everything else is 2 B/element.
        let bytes_of = |entry: &WeightEntry| {
            if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
                Some(entry.logical_shape[0] as u64)
            } else {
                element_bytes(entry)
            }
        };
        let arena = |layer: usize| plan.layer(layer).expect("layer").bytes as u64;
        let mtp_bytes: u64 = manifest
            .weights
            .iter()
            .filter(|entry| entry.name.starts_with("mtp.") && entry.name.contains(".mlp.experts."))
            .map(|entry| element_bytes(entry).expect("mtp aggregate"))
            .sum();
        assert!(mtp_bytes > 0, "the manifest declares the MTP aggregates");
        let distinct: std::collections::BTreeSet<_> = (0..48).map(|layer| arena(layer)).collect();
        assert_eq!(distinct.len(), 48, "every layer's arena differs");

        let context = crate::Qwen4ContextCommit::legacy(1024);
        let census = |weights: &[WeightEntry], scope: &Qwen4BandScope| {
            band_census(
                &config,
                weights,
                bytes_of,
                scope,
                target_state_format(),
                &context,
                256,
                Full,
            )
            .expect("band census")
        };
        let single = census(&manifest.weights, &Qwen4BandScope::single(&config));
        assert_eq!(
            single.routed_trunk_bytes,
            (0..48).map(arena).sum::<u64>(),
            "the single scope charges every arena exactly"
        );
        assert_eq!(single.routed_mtp_bytes, mtp_bytes);
        assert_eq!(
            single.max_layer_routed_bytes,
            (0..48).map(arena).max().unwrap()
        );

        for counts in [&[24usize, 24][..], &[16, 16, 16][..]] {
            let mut trunk_sum = 0u64;
            let mut mtp_sum = 0u64;
            for scope in band_scopes(&config, counts) {
                let own: Vec<u64> = scope.band.clone().map(arena).collect();
                let scoped = manifest
                    .clone()
                    .with_band_scope(&config, &scope)
                    .expect("scope-filtered manifest");
                // The full and the scope-filtered manifest give the same census.
                let from_full = census(&manifest.weights, &scope);
                let from_scoped = census(&scoped.weights, &scope);
                assert_eq!(from_full, from_scoped, "{counts:?} band {:?}", scope.band);
                assert_eq!(
                    from_scoped.routed_trunk_bytes,
                    own.iter().sum::<u64>(),
                    "{counts:?} band {:?}",
                    scope.band
                );
                assert_eq!(
                    from_scoped.max_layer_routed_bytes,
                    own.iter().copied().max().unwrap(),
                    "{counts:?} band {:?}",
                    scope.band
                );
                assert_eq!(
                    from_scoped.routed_mtp_bytes,
                    if scope.output { mtp_bytes } else { 0 },
                    "{counts:?} band {:?}",
                    scope.band
                );
                trunk_sum += from_scoped.routed_trunk_bytes;
                mtp_sum += from_scoped.routed_mtp_bytes;
            }
            assert_eq!(trunk_sum, single.routed_trunk_bytes, "{counts:?}");
            assert_eq!(mtp_sum, single.routed_mtp_bytes, "{counts:?}");
        }
    }

    #[test]
    fn band_census_refuses_expert_sharded_entries_for_a_band_but_not_single() {
        use hipfire_runtime::tp_shard::ExpertAssign;
        let config = crate::config::compact_test_config();
        let mut manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let sharded = manifest
            .weights
            .iter_mut()
            .find(|entry| is_routed_expert(&entry.name) && entry.layer == Some(0))
            .expect("routed expert entry");
        sharded.policy = ShardPolicy::ExpertSharded {
            n_experts: config.num_experts,
            assign: ExpertAssign::Stride,
        };
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let calls = std::cell::Cell::new(0usize);
        let counted = |entry: &WeightEntry| {
            calls.set(calls.get() + 1);
            element_bytes(entry)
        };
        for scope in band_scopes(&config, &[24, 24]) {
            let error = band_census(
                &config,
                &manifest.weights,
                counted,
                &scope,
                target_state_format(),
                &context,
                256,
                Full,
            )
            .unwrap_err();
            assert!(error.contains("expert parallelism"), "{error}");
            assert!(
                routed_expert_bytes_scope(&manifest.weights, counted, &scope)
                    .unwrap_err()
                    .contains("expert parallelism")
            );
        }
        assert_eq!(calls.get(), 0, "refused before any payload callback");
        let single = Qwen4BandScope::single(&config);
        assert!(band_census(
            &config,
            &manifest.weights,
            element_bytes,
            &single,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .is_ok());
    }

    #[test]
    fn band_census_refuses_a_band_without_the_ple_layer_and_pp_with_ep() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let ple_layer = config.ple.as_ref().expect("PLE profile").layer;
        let bad = Qwen4BandScope {
            band: 0..ple_layer,
            input: true,
            output: false,
        };
        let error = band_census(
            &config,
            &manifest.weights,
            element_bytes,
            &bad,
            target_state_format(),
            &context,
            256,
            Full,
        )
        .unwrap_err();
        assert!(error.contains("PLE"), "{error}");
        let scopes = band_scopes(&config, &[24, 24]);
        for scope in &scopes {
            assert_eq!(refuse_band_with_expert_parallel(&config, scope, 1), Ok(()));
            let error = refuse_band_with_expert_parallel(&config, scope, 2).unwrap_err();
            assert!(error.contains("expert parallelism"), "{error}");
        }
    }

    /// One recurrent layer's F32 storage minus its Q8 storage (3145728 - 811008).
    const GDN_SLOT_DELTA: u64 = 2_334_720;

    fn precision_format(qsa: QsaKvFormat, gdn: crate::GdnStateFormat) -> crate::Qwen4StateFormat {
        crate::Qwen4StateFormat {
            qsa,
            gdn,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
        }
    }

    #[test]
    fn state_census_charges_f32_gdn_as_live_plus_mirror_over_q8() {
        let config = crate::config::compact_test_config();
        let gdn_layers = config.n_linear_layers() as u64;
        assert_eq!(gdn_layers, 36);
        let max_seq = 17_408;
        let granularity = 2 << 20;
        for backend in [crate::Qwen4KvBackend::Legacy, crate::Qwen4KvBackend::Vmm] {
            let full = crate::Qwen4ContextCommit::new(backend, max_seq, max_seq, granularity);
            for qsa in [QsaKvFormat::Fp8, F32, QsaKvFormat::Q8] {
                let q8 = state_census(
                    &config,
                    precision_format(qsa, crate::GdnStateFormat::Q8),
                    &full,
                    4096,
                    Full,
                )
                .unwrap();
                let f32 = state_census(
                    &config,
                    precision_format(qsa, crate::GdnStateFormat::F32),
                    &full,
                    4096,
                    Full,
                )
                .unwrap();
                // Exactly (F32 - Q8) x (live + mirror) x the 36 GDN layers:
                // 2 x 80.15625 MiB. The context, forward scratch and every
                // other term are the same inputs: nothing is lowered.
                assert_eq!(
                    f32.state - q8.state,
                    2 * gdn_layers * GDN_SLOT_DELTA,
                    "{backend:?} {qsa:?}"
                );
                assert_eq!(f32.state - q8.state, 168_099_840);
                assert_eq!(f32.qsa_context, q8.qsa_context, "{backend:?} {qsa:?}");
                assert_eq!(f32.forward, q8.forward, "{backend:?} {qsa:?}");
                // The reserve's total moves by the same bytes and no more.
                let total = |census: StateCensus| {
                    census_with(census.qsa_context, census.state, census.forward, 0, 0)
                        .total()
                        .unwrap()
                };
                assert_eq!(total(f32) - total(q8), 168_099_840);
            }
        }
    }

    #[test]
    fn state_census_scope_charges_local_gdn_layers_and_ep_replicates() {
        let config = crate::config::compact_test_config();
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let q8 = precision_format(QsaKvFormat::Fp8, crate::GdnStateFormat::Q8);
        let f32 = precision_format(QsaKvFormat::Fp8, crate::GdnStateFormat::F32);
        let delta = |scope: &Qwen4BandScope| {
            let q8 = state_census_scope(&config, q8, &context, 256, Full, scope).unwrap();
            let f32 = state_census_scope(&config, f32, &context, 256, Full, scope).unwrap();
            assert_eq!(q8.qsa_context, f32.qsa_context);
            assert_eq!(q8.forward, f32.forward);
            f32.state - q8.state
        };
        let whole = delta(&Qwen4BandScope::single(&config));
        assert_eq!(whole, 2 * 36 * GDN_SLOT_DELTA);
        // PP: each band is charged its own GDN layers (live + mirror); the
        // bands sum to the whole model, nothing is divided or duplicated.
        for counts in [&[24usize, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
            let mut sum = 0;
            for scope in band_scopes(&config, counts) {
                let (gdn_layers, _) = crate::state::scope_layer_counts(&config, &scope);
                assert!(gdn_layers > 0 && gdn_layers < 36);
                let band = delta(&scope);
                assert_eq!(band, 2 * gdn_layers as u64 * GDN_SLOT_DELTA, "{counts:?}");
                sum += band;
            }
            assert_eq!(sum, whole, "{counts:?}");
        }
        // EP: every rank replicates the recurrent state, so the census of a
        // whole-model rank is the single census at any rank count, and a band
        // never combines with EP.
        let single = Qwen4BandScope::single(&config);
        for ranks in 1..=8 {
            assert_eq!(
                refuse_band_with_expert_parallel(&config, &single, ranks),
                Ok(())
            );
            assert_eq!(
                state_census_scope(&config, f32, &context, 256, Full, &single).unwrap(),
                state_census(&config, f32, &context, 256, Full).unwrap(),
                "{ranks} ranks"
            );
        }
        for scope in band_scopes(&config, &[24, 24]) {
            assert!(refuse_band_with_expert_parallel(&config, &scope, 2).is_err());
        }
    }

    #[test]
    fn state_census_native_mtp_reserve_charges_the_f32_ring_per_band() {
        let config = crate::config::compact_test_config();
        let manifest = crate::weights::Qwen4Manifest::build(&config).expect("FN manifest");
        let weights = &manifest.weights;
        let context = crate::Qwen4ContextCommit::legacy(1024);
        let inputs = |gdn, ring| NativeMtpCensusInputs {
            max_k: 3,
            row_capture: Some((gdn, ring)),
            batched_fill: false,
        };
        for ring in [4usize, 8, 32] {
            let census = |gdn, scope: &Qwen4BandScope| {
                native_mtp_census_scope(&config, weights, &context, 256, &inputs(gdn, ring), scope)
                    .unwrap()
            };
            for counts in [&[48usize][..], &[24, 24][..], &[9, 8, 8, 8, 8, 7][..]] {
                for scope in band_scopes(&config, counts) {
                    let (gdn_layers, _) = crate::state::scope_layer_counts(&config, &scope);
                    let q8 = census(crate::GdnStateFormat::Q8, &scope);
                    let f32 = census(crate::GdnStateFormat::F32, &scope);
                    // The band's own 2R ring slots per local GDN layer.
                    assert_eq!(
                        f32.row_capture - q8.row_capture,
                        gdn_layers as u64 * 2 * ring as u64 * GDN_SLOT_DELTA,
                        "{counts:?} {:?} R={ring}",
                        scope.band
                    );
                    // The head and request terms do not depend on the format.
                    assert_eq!(f32.head_resident, q8.head_resident);
                    assert_eq!(f32.request, q8.request);
                    assert_eq!(
                        f32.reserve().unwrap() - q8.reserve().unwrap(),
                        f32.row_capture - q8.row_capture,
                        "{counts:?} {:?} R={ring}",
                        scope.band
                    );
                }
            }
        }
    }

    // ---- XTX placement: receipt-driven CPU dryrun --------------------------
    //
    // Inputs are the real gfx1100 XTX hold receipts
    // (`g2-xtx-default/hold-20261017T102413Z/pf-{base,cand}/stderr.txt`):
    // Flash-Next xts, `max_seq` 8224, legacy QSA context (q8 K/V), packed
    // (aggregate) experts so no dedicated stage is charged, 1275 MiB per
    // expert layer (512 experts x (1,740,800 + 870,400) B, exact), 5115 MiB
    // of non-expert weights and the logged free VRAM. The logs print free
    // VRAM and non-expert bytes floored to MiB, so those two are exact only
    // to the MiB; the reserve is recomputed by the repo's own
    // `auto_vram_reserve` for each chunk and cross-checked against the
    // receipt's logged MiB below.
    const XTX_MAX_SEQ: usize = 8224;
    const XTX_NON_EXPERT: u64 = 5115 << 20;
    const XTX_GATE_UP: u64 = 512 * 1_740_800;
    const XTX_DOWN: u64 = 512 * 870_400;
    /// Free VRAM the pf-base and pf-cand placement logs print.
    const XTX_FREE_BASE: u64 = 24522 << 20;
    const XTX_FREE_CAND: u64 = 24376 << 20;

    fn xtx_reserve(chunk: usize) -> AutoReserve {
        let config = crate::config::compact_test_config();
        let context = crate::Qwen4ContextCommit::legacy(XTX_MAX_SEQ);
        let historical =
            auto_vram_reserve(&config, &context, chunk, Full, QsaKvFormat::Q8, None, None).unwrap();
        // The census terms the receipts print (QSA context 132 MiB, state
        // 65 MiB, forward at this chunk); it stays below the floor at every
        // rung, as in both receipts, so the historical floor sets the reserve.
        let forward = crate::gpu_forward::qwen4_forward_device_bytes(&config, chunk, Full).unwrap();
        AutoReserve::new(
            Some(historical),
            census_with(132 << 20, 65 << 20, forward, 0, 0),
        )
        .unwrap()
    }

    fn xtx_manifest() -> Vec<WeightEntry> {
        let mut weights = vec![entry("model.language_model.embed_tokens.weight", 0)];
        weights[0].layer = None;
        for layer in 0..48 {
            for suffix in ["gate_up_proj", "down_proj"] {
                weights.push(entry(
                    &format!("model.language_model.layers.{layer}.mlp.experts.{suffix}"),
                    layer,
                ));
            }
        }
        for suffix in ["gate_up_proj", "down_proj"] {
            let mut mtp = entry(&format!("mtp.layers.0.mlp.experts.{suffix}"), 0);
            mtp.layer = Some(0);
            weights.push(mtp);
        }
        weights
    }

    fn xtx_bytes(entry: &WeightEntry) -> Option<u64> {
        Some(if entry.name.ends_with("experts.gate_up_proj") {
            XTX_GATE_UP
        } else if entry.name.ends_with("experts.down_proj") {
            XTX_DOWN
        } else {
            XTX_NON_EXPERT
        })
    }

    /// What one load resolves, in the loader's order: the requested chunk
    /// (the link is read only on gfx1100 without an explicit value), then the
    /// placement sized from it with the placement stage switch.
    struct XtxDryrun {
        placement_rows: usize,
        resident_layers: usize,
        host_tensors: usize,
        host_bytes: u64,
        resident_bytes: u64,
        donors: Option<[usize; 2]>,
        mirror_bytes: u64,
    }

    fn xtx_dryrun(
        arch: &str,
        free: u64,
        chunk_env: Option<&str>,
        stage_env: Option<&str>,
        link: crate::host_link::HostLink,
        slab: bool,
    ) -> XtxDryrun {
        let mut weights = xtx_manifest();
        let (non_expert, layer) = resident_split(&weights, xtx_bytes).unwrap();
        assert_eq!(
            (non_expert, layer),
            (XTX_NON_EXPERT, XTX_GATE_UP + XTX_DOWN)
        );
        // Placement: the requested chunk, the placement stage switch.
        let placement_rows = crate::gpu_forward::prefill_chunk_requested_from(
            arch,
            chunk_env,
            XTX_MAX_SEQ,
            Full,
            || link,
        );
        let charges = PlacementCharges::new(
            layer,
            dedicated_slab_stage(expert_stage_placement_switch(arch, stage_env), slab, false),
        )
        .unwrap();
        let reserve = xtx_reserve(placement_rows);
        let resident_layers =
            auto_placement_layers(free, non_expert, layer, 48, reserve.bytes(), &charges).unwrap();
        let host_tensors = place_routed_experts(&mut weights, resident_layers);
        let host_bytes = host_mapped_bytes(&weights, xtx_bytes).unwrap();
        let resident_bytes = resident_routed_expert_bytes(&weights, xtx_bytes).unwrap();
        // The borrowed packed stage lends the last two resident layers; its
        // restore copies are separate pinned host RAM.
        let donors = (resident_layers >= 2).then(|| {
            let mut donors = [resident_layers - 2, resident_layers - 1];
            donors.sort_by_key(|layer| layer % 2);
            donors
        });
        let mirror_bytes = expert_stage_mirror_bytes(&[layer as usize]).unwrap();
        XtxDryrun {
            placement_rows,
            resident_layers,
            host_tensors,
            host_bytes,
            resident_bytes,
            donors,
            mirror_bytes,
        }
    }

    impl XtxDryrun {
        fn print(&self, label: &str) {
            const MIB: u64 = 1 << 20;
            let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
            println!(
                "[dryrun {label}] placement chunk {} rows -> {} resident layers ({} MiB experts), \
                 {} host expert tensors ({:.1} GiB pinned, placement only); \
                 G2 borrowed stage: donors {:?}, dedicated VRAM 0 MiB, restore mirrors {} MiB \
                 pinned host RAM charged separately",
                self.placement_rows,
                self.resident_layers,
                self.resident_bytes / MIB,
                self.host_tensors,
                gib(self.host_bytes),
                self.donors,
                (self.mirror_bytes - 4 * HOST_MAPPED_PAD_BYTES) / MIB,
            );
        }
    }

    #[test]
    fn xtx_placement_reserves_match_the_receipts() {
        // The receipts' logged reserves (floored MiB): base 5261 at the
        // 1536-row chunk, 12110 at 8192; the mid rungs grow by the
        // forward-resource deltas of the same formula.
        let at = |rows| xtx_reserve(rows).bytes() >> 20;
        assert_eq!(at(1536), 5261);
        assert_eq!(at(8192), 12110);
        assert!(at(2048) > at(1536) && at(4096) > at(2048) && at(8192) > at(4096));
        assert_eq!(xtx_reserve(1536).winner(), "historical floor");
        assert_eq!(xtx_reserve(8192).winner(), "historical floor");
        // With the margin the receipts print 5773 and 12622 MiB.
        let charges = PlacementCharges::new(XTX_GATE_UP + XTX_DOWN, false).unwrap();
        assert_eq!(
            charges.reserve(xtx_reserve(1536).bytes()).unwrap() >> 20,
            5773
        );
        assert_eq!(
            charges.reserve(xtx_reserve(8192).bytes()).unwrap() >> 20,
            12622
        );
    }

    #[test]
    fn xtx_placement_default_cedes_layers_only_on_a_measured_narrow_link() {
        use crate::host_link::HostLink::{Discrete, Unified, Unmeasured};
        // pf-base: no link policy, beta chunk and stage defaults.
        let beta = xtx_dryrun("gfx1100", XTX_FREE_BASE, None, None, Unmeasured, false);
        // A fast link keeps the beta 1536-row placement on the candidate's
        // free VRAM, as do unified and unmeasured links.
        let fast = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            None,
            None,
            Discrete { gbps: 26.0 },
            false,
        );
        let unified = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, Unified, false);
        // The failed candidate's environment: a measured 7.1 GB/s narrow link
        // and 146 MiB less free VRAM, nothing set explicitly.
        let narrow = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            None,
            None,
            Discrete { gbps: 7.1 },
            false,
        );
        beta.print("beta (pf-base inputs)");
        fast.print("fast link (pf-cand inputs)");
        narrow.print("narrow link default (pf-cand inputs)");
        for run in [&beta, &fast, &unified] {
            assert_eq!(run.placement_rows, 1536);
            assert_eq!(run.resident_layers, 10);
            assert_eq!(run.host_tensors, 78);
            // 38 trunk layers + the MTP layer, each tensor with its pinned pad.
            assert_eq!(
                run.host_bytes,
                39 * (XTX_GATE_UP + XTX_DOWN) + 78 * HOST_MAPPED_PAD_BYTES
            );
            assert_eq!(
                format!("{:.1}", run.host_bytes as f64 / (1u64 << 30) as f64),
                "48.6"
            );
            assert_eq!(run.resident_bytes, 10 * (XTX_GATE_UP + XTX_DOWN));
            // The borrowed donors of ten resident layers, parity ordered.
            assert_eq!(run.donors, Some([8, 9]));
        }
        // The narrow-link default places for the 8192-row reserve: it cedes
        // layers (the receipt's five-layer split) for the wider chunk.
        assert_eq!(narrow.placement_rows, 8192);
        assert_eq!(narrow.resident_layers, 5);
        assert_eq!(narrow.host_tensors, 88);
        assert_eq!(narrow.donors, Some([4, 3]));
        assert!(narrow.resident_bytes < beta.resident_bytes);
        assert!(narrow.host_bytes > beta.host_bytes);
        // The restore mirrors are a separate host charge: two donor layers,
        // 2550 MiB of copies (plus four tensor pads in the host check).
        assert_eq!(narrow.mirror_bytes - 4 * HOST_MAPPED_PAD_BYTES, 2550 << 20);
        assert_eq!(narrow.mirror_bytes, beta.mirror_bytes);
    }

    #[test]
    fn xtx_placement_explicit_chunk_is_authoritative_whatever_the_link() {
        use crate::host_link::HostLink::{Discrete, Unmeasured};
        // HIPFIRE_PREFILL_CHUNK_ROWS=8192 sizes the placement exactly as the
        // narrow-link default does (the failed candidate's five-layer split:
        // the receipt placement, 88 host tensors, donors 4 and 3), and never
        // reads the link: a fast or unmeasured link places the same.
        for link in [Discrete { gbps: 7.1 }, Discrete { gbps: 26.0 }, Unmeasured] {
            let opt_in = xtx_dryrun("gfx1100", XTX_FREE_CAND, Some("8192"), None, link, false);
            opt_in.print("explicit HIPFIRE_PREFILL_CHUNK_ROWS=8192");
            assert_eq!(opt_in.placement_rows, 8192);
            assert_eq!(opt_in.resident_layers, 5);
            assert_eq!(opt_in.host_tensors, 88);
            assert_eq!(opt_in.donors, Some([4, 3]));
        }
        // An explicit smaller value (the 1536 opt-out) is honoured exactly,
        // even on a narrow link.
        let small = xtx_dryrun(
            "gfx1100",
            XTX_FREE_CAND,
            Some("1536"),
            None,
            Discrete { gbps: 7.1 },
            false,
        );
        assert_eq!(small.placement_rows, 1536);
        assert_eq!(small.resident_layers, 10);
    }

    #[test]
    fn xtx_placement_wider_default_chunk_cedes_layers_and_abundant_vram_keeps_all() {
        use crate::host_link::HostLink::Discrete;
        let narrow = Discrete { gbps: 7.1 };
        // The narrow-link placement is sized for the full 8192-row reserve.
        let ceded = xtx_dryrun("gfx1100", XTX_FREE_BASE, None, None, narrow, false);
        let beta = xtx_dryrun(
            "gfx1100",
            XTX_FREE_BASE,
            None,
            None,
            Discrete { gbps: 26.0 },
            false,
        );
        ceded.print("narrow link, base free VRAM");
        assert_eq!(ceded.placement_rows, 8192);
        assert!(ceded.resident_layers < beta.resident_layers);
        assert!(ceded.host_bytes > beta.host_bytes);
        // Abundant VRAM keeps every layer resident at the 8192 reserve.
        let rich = xtx_dryrun("gfx1100", 100_000 << 20, None, None, narrow, false);
        rich.print("abundant VRAM");
        assert_eq!(rich.placement_rows, 8192);
        assert_eq!(rich.resident_layers, 48);
    }

    #[test]
    fn xtx_placement_stage_charge_is_probe_free_and_keeps_explicit_and_gfx1201_semantics() {
        use crate::host_link::HostLink::Discrete;
        // Never reads the link: the placement switch is env-first, then
        // gfx1201, so a narrow-link gfx1100 is not charged a stage.
        assert!(!expert_stage_placement_switch("gfx1100", None));
        assert!(!expert_stage_placement_switch("gfx1151", None));
        assert!(expert_stage_placement_switch("gfx1201", None));
        assert!(expert_stage_placement_switch("gfx1100", Some("1")));
        assert!(!expert_stage_placement_switch("gfx1201", Some("0")));
        assert!(expert_stage_placement_switch("gfx1201", Some("junk")));
        assert!(!expert_stage_placement_switch("gfx1100", Some("on")));
        // The probing default still turns the borrowed stage on afterwards.
        assert!(expert_stage_switch("gfx1100", None, || Discrete {
            gbps: 7.1
        }));
        // A dedicated slab stage is charged only where the placement switch
        // is on: same free VRAM, same layers as the stage-off beta on gfx1100,
        // fewer where the stage is explicit.
        let narrow = Discrete { gbps: 7.1 };
        let default = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, None, narrow, true);
        let off = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("0"), narrow, true);
        let explicit = xtx_dryrun("gfx1100", XTX_FREE_CAND, None, Some("1"), narrow, true);
        assert_eq!(default.resident_layers, off.resident_layers);
        assert!(explicit.resident_layers < default.resident_layers);
        // gfx1151 (unified memory), gfx1201 and gfx942 keep their arch chunk
        // defaults, and the link is never evaluated for them.
        for (arch, rows) in [("gfx1151", 8192), ("gfx1201", 4096), ("gfx942", 1536)] {
            assert_eq!(
                crate::gpu_forward::prefill_chunk_requested_from(
                    arch,
                    None,
                    usize::MAX,
                    Full,
                    || { unreachable!("probed") }
                ),
                rows
            );
        }
    }

    // ---- mq2-xts (mixed-codec slab) XTX placement + borrowed-stage dryrun ----
    //
    // The 48 trunk layers of `qwen3.8-flash-next.mq2-xts` come from the
    // checked-in layout fixture (`tests/fixtures/mq2_xts_expert_layout.txt`:
    // each layer's 512 experts' codec, read from the artifact's tensor index;
    // its header records the artifact's stat and SHA-256): a layer's physical
    // arena is the sum of its experts' gate/up + down payloads. Mixed MQ4/E8P
    // layers: 782-840 MiB each, the largest layer 38 (840.45 MiB) being the
    // placement's per-layer charge (the receipt's "840 MiB per expert layer").
    // The MTP head keeps the legacy aggregates: 891,289,600 B gate/up +
    // 445,644,800 B down.
    use crate::gpu_forward::slab_stage::mq2_xts_layout as layout;
    const MQ2_XTS_MTP_GATE_UP: u64 = 891_289_600;
    const MQ2_XTS_MTP_DOWN: u64 = 445_644_800;

    fn mq2_xts_manifest() -> Vec<WeightEntry> {
        let mut weights = vec![entry("model.language_model.embed_tokens.weight", 0)];
        weights[0].layer = None;
        for layer in 0..48 {
            weights.push(arena_entry(layer, layout::arena_bytes(layer)));
        }
        for suffix in ["gate_up_proj", "down_proj"] {
            let mut mtp = entry(&format!("mtp.layers.0.mlp.experts.{suffix}"), 0);
            mtp.layer = Some(0);
            weights.push(mtp);
        }
        weights
    }

    fn mq2_xts_bytes(entry: &WeightEntry) -> Option<u64> {
        Some(if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
            entry.logical_shape[0] as u64
        } else if entry.name.ends_with("experts.gate_up_proj") {
            MQ2_XTS_MTP_GATE_UP
        } else if entry.name.ends_with("experts.down_proj") {
            MQ2_XTS_MTP_DOWN
        } else {
            XTX_NON_EXPERT
        })
    }

    /// `(resident layers, host expert tensors, pinned host bytes, per-layer
    /// charge)` of one gfx1100 mq2-xts load at the receipt's free VRAM, the
    /// slab recipe with the given stage env, on a measured narrow link.
    fn mq2_xts_placement(stage_env: Option<&str>) -> (usize, usize, u64, u64) {
        let narrow = crate::host_link::HostLink::Discrete { gbps: 7.1 };
        let mut weights = mq2_xts_manifest();
        let (non_expert, layer) = resident_split(&weights, mq2_xts_bytes).unwrap();
        assert_eq!(non_expert, XTX_NON_EXPERT);
        let rows = crate::gpu_forward::prefill_chunk_requested_from(
            "gfx1100",
            None,
            XTX_MAX_SEQ,
            Full,
            || narrow,
        );
        assert_eq!(rows, 8192, "the shipped narrow-link default chunk");
        let charges = PlacementCharges::new(
            layer,
            dedicated_slab_stage(
                expert_stage_placement_switch("gfx1100", stage_env),
                true,
                false,
            ),
        )
        .unwrap();
        let reserve = xtx_reserve(rows);
        let resident_layers = auto_placement_layers(
            XTX_FREE_CAND,
            non_expert,
            layer,
            48,
            reserve.bytes(),
            &charges,
        )
        .unwrap();
        let host_tensors = place_routed_experts(&mut weights, resident_layers);
        let host_bytes = host_mapped_bytes(&weights, mq2_xts_bytes).unwrap();
        (resident_layers, host_tensors, host_bytes, layer)
    }

    #[test]
    fn mq2_xts_narrow_link_stage_overflow_keeps_7_resident_layers_and_33_4_gib_host() {
        use crate::expert_arena::ARENA_TAIL_PAD_BYTES as PAD;
        use crate::gpu_forward::slab_stage::{
            borrowed_stage_need, layer_stage_cut, plan_slab_donors, stage_layer_report,
            stage_split_counts,
        };
        // The shipped placement (8192-row chunk, 12622 MiB reserve, no stage
        // charge): 7 resident layers, 43 host tensors, 33.4 GiB pinned. The
        // stage-off load places identically, so the borrowed stage is free.
        let (resident_layers, host_tensors, host_bytes, layer) = mq2_xts_placement(None);
        assert_eq!(layer, 881_280_000, "layer 38, 840.45 MiB, is the charge");
        assert_eq!(layer >> 20, 840);
        assert_eq!(
            (resident_layers, host_tensors),
            (7, 43),
            "layers 0..7 in VRAM, 43 expert tensors in pinned host RAM"
        );
        assert_eq!(
            format!("{:.1}", host_bytes as f64 / (1u64 << 30) as f64),
            "33.4"
        );
        let off = mq2_xts_placement(Some("0"));
        assert_eq!(
            (off.0, off.1, off.2),
            (resident_layers, host_tensors, host_bytes)
        );
        // Explicit `1` charges the dedicated buffers and cedes layers.
        assert!(mq2_xts_placement(Some("1")).0 < resident_layers);

        // Borrowed stage: the largest cold arena of each parity over the
        // host layers 7..48, the resident layers 0..7 as donors.
        let need = borrowed_stage_need(
            (resident_layers..48).map(|layer| (layer, layout::arena_bytes(layer))),
        )
        .unwrap();
        // Even host layer 38 is 840.45 MiB; the largest odd one is layer 35.
        assert_eq!(need, [881_280_000 + PAD, 853_862_400 + PAD]);
        let resident: Vec<Option<usize>> = (0..resident_layers)
            .map(|layer| Some(layout::arena_bytes(layer) + PAD))
            .collect();
        // Before: no even resident layer holds 840.45 MiB (the largest, layer 2,
        // is 806.84 MiB), the `no resident slab layer of parity 0` refusal.
        assert!(resident
            .iter()
            .step_by(2)
            .all(|bytes| bytes.unwrap() < need[0]));
        // After: parity 0 lends layer 2 (its whole allocation is the window,
        // 33.6 MiB short of the stage), parity 1 lends layer 1, which holds
        // the 814.31 MiB layer 35 whole.
        let (donors, window) = plan_slab_donors(need, &resident).unwrap();
        assert_eq!(donors, [2, 1]);
        assert_eq!(window, [846_028_800 + PAD, need[1]]);
        assert!(donors.iter().all(|&donor| donor < resident_layers));
        // Residency is untouched: no VRAM is added, no resident layer is
        // given up, and the only extra is the two pinned restore copies.
        assert_eq!(
            slab_stage_mirror_bytes(window).unwrap() - 2 * HOST_MAPPED_PAD_BYTES,
            (window[0] + window[1]) as u64
        );

        // Per host layer, with the real expert layout: which layers overflow
        // and how much of their arena stays reading host memory in place. The
        // cut is always an expert boundary (an expert is staged or read in
        // place whole), and each layer reports once, as at load.
        let (arena_base, mut overflowing, mut worst) = (0x7f00_0000_0000usize, Vec::new(), 0usize);
        let mut reports = Vec::new();
        for layer in resident_layers..48 {
            let (experts, arena) = layout::experts(layer);
            let (gate_up, down) = layout::tables(&experts, arena_base);
            let cut = layer_stage_cut(&gate_up, &down, arena_base, arena, window[layer % 2]);
            assert!(cut + PAD <= window[layer % 2]);
            assert!(
                cut == arena || experts.iter().any(|e| e[0] == cut),
                "layer {layer} cut {cut} is not an expert boundary"
            );
            let in_place = arena - cut;
            reports.push(stage_layer_report(
                layer,
                donors[layer % 2],
                arena,
                cut,
                stage_split_counts([&gate_up, &down], arena_base, cut),
            ));
            if in_place > 0 {
                overflowing.push(layer);
                worst = worst.max(in_place);
                // Never more than the overflow plus one MQ4 expert, and never
                // above 5% of the layer.
                assert!(in_place <= arena + PAD - window[layer % 2] + 2_611_200);
                assert!(in_place * 20 <= arena);
            }
        }
        assert_eq!(overflowing, [10, 12, 18, 22, 26, 38, 42]);
        assert_eq!(reports.len(), 41);
        for report in &reports {
            println!("{report}");
        }
        assert!(worst < 40 << 20, "worst layer reads {worst} B in place");
        println!(
            "[dryrun mq2-xts] placement chunk 8192 rows -> {resident_layers} resident layers, \
             {host_tensors} host expert tensors ({:.1} GiB pinned); G2 borrowed stage: donors \
             {donors:?}, windows {} + {} MiB (pinned restore copies, dedicated VRAM 0 MiB), \
             {} of 41 host layers overflow, at most {} MiB of one layer reads in place",
            host_bytes as f64 / (1u64 << 30) as f64,
            window[0] >> 20,
            window[1] >> 20,
            overflowing.len(),
            worst >> 20,
        );
    }

    /// The product loader's inline composition of the placement primitives for an
    /// unset layer count (what `carriers.rs` computed before
    /// [`resident_fit`]/[`incumbent_placement_layers`] named it): the
    /// reference the shared resolver must equal exactly.
    fn inline_unset_placement(
        free: u64,
        non_expert: u64,
        experts: u64,
        layer: u64,
        layers: usize,
        base: u64,
        stage: bool,
    ) -> Result<Option<usize>, String> {
        let charges = PlacementCharges::new(layer, false)?;
        let reserve = charges.reserve(base)?;
        if fits_fully_resident(free, non_expert, experts, reserve) {
            return Ok(None);
        }
        let charges = PlacementCharges::new(layer, stage)?;
        auto_placement_layers(free, non_expert, layer, layers, base, &charges).map(Some)
    }

    #[test]
    fn incumbent_placement_equals_the_inline_composition_on_the_shipped_layout() {
        use crate::host_link::HostLink::Unmeasured;
        let weights = xtx_manifest();
        let (non_expert, layer) = resident_split(&weights, xtx_bytes).unwrap();
        let experts = routed_expert_bytes(&weights, xtx_bytes).unwrap();
        // The 48-layer aggregate layout of the shipped split base plus the MTP
        // head's two expert tensors.
        assert_eq!(experts, 49 * (XTX_GATE_UP + XTX_DOWN));
        let frees = [
            0,
            non_expert,
            XTX_FREE_BASE,
            XTX_FREE_CAND,
            experts + non_expert + (64 << 30),
            u64::MAX / 4,
        ];
        for free in frees {
            for chunk in [512, 1536, 2048, 4096, 8192] {
                for stage in [false, true] {
                    let base = xtx_reserve(chunk).bytes();
                    let shared = incumbent_placement_layers(
                        free, non_expert, experts, layer, 48, base, stage,
                    );
                    let inline =
                        inline_unset_placement(free, non_expert, experts, layer, 48, base, stage);
                    assert_eq!(shared, inline, "free {free} chunk {chunk} stage {stage}");
                }
            }
        }
        // The receipts' placements: base 1536 rows and candidate 8192 rows.
        let base = xtx_dryrun("gfx1100", XTX_FREE_BASE, None, None, Unmeasured, false);
        let resolved = incumbent_placement_layers(
            XTX_FREE_BASE,
            non_expert,
            experts,
            layer,
            48,
            xtx_reserve(base.placement_rows).bytes(),
            false,
        );
        assert_eq!(resolved, Ok(Some(base.resident_layers)));
        // Everything fits: no placement at all (the product keeps it resident).
        assert_eq!(
            incumbent_placement_layers(
                experts + non_expert + (64 << 30),
                non_expert,
                experts,
                layer,
                48,
                xtx_reserve(1536).bytes(),
                false
            ),
            Ok(None)
        );
    }

    /// The `auto` reserve of the shipped layout under `gdn`, from the CURRENT
    /// state census (`state_census` -> `Qwen4State::device_bytes`, which adds
    /// the Q8Ef live residual plane once). `floor` keeps the historical
    /// formula (gfx1100: it still applies); without it the census alone is the
    /// reserve (the exact-gfx1201 route that carries the runtime overhead).
    fn gdn_reserve(chunk: usize, gdn: crate::GdnStateFormat, floor: bool) -> AutoReserve {
        let config = crate::config::compact_test_config();
        let format = precision_format(QsaKvFormat::Q8, gdn);
        let context = crate::Qwen4ContextCommit::legacy(XTX_MAX_SEQ);
        let census = state_census(&config, format, &context, chunk, Full).unwrap();
        let historical = floor.then(|| {
            auto_vram_reserve(&config, &context, chunk, Full, format.layout(), None, None).unwrap()
        });
        AutoReserve::new(
            historical,
            census_with(census.qsa_context, census.state, census.forward, 0, 0),
        )
        .unwrap()
    }

    #[test]
    fn q8ef_live_residual_is_charged_once_by_the_state_census_and_flows_into_the_placement() {
        let config = crate::config::compact_test_config();
        // One F16 residual plane per GDN layer, not mirrored, not per ring row.
        let plane = 2
            * (config.linear_num_value_heads
                * config.linear_key_head_dim
                * config.linear_value_head_dim) as u64;
        let live = plane * config.n_linear_layers() as u64;
        assert_eq!(live, 54 << 20);
        let weights = xtx_manifest();
        let (non_expert, layer) = resident_split(&weights, xtx_bytes).unwrap();
        let experts = routed_expert_bytes(&weights, xtx_bytes).unwrap();
        for chunk in [512, 1536, 8192] {
            let q8 = gdn_reserve(chunk, crate::GdnStateFormat::Q8, false);
            let ef = gdn_reserve(chunk, crate::GdnStateFormat::Q8Ef, false);
            assert_eq!(ef.census.state - q8.census.state, live, "chunk {chunk}");
            assert_eq!(ef.census.qsa_context, q8.census.qsa_context);
            assert_eq!(ef.census.forward, q8.census.forward);
            assert_eq!(ef.census_total - q8.census_total, live, "chunk {chunk}");
            // The floor never sees the GDN format: a floor-led reserve is equal.
            let q8_floor = gdn_reserve(chunk, crate::GdnStateFormat::Q8, true);
            let ef_floor = gdn_reserve(chunk, crate::GdnStateFormat::Q8Ef, true);
            assert_eq!(q8_floor.historical, ef_floor.historical);
            // Where the dense GDN scan's scratch is charged (exact gfx1201) the formats
            // also differ in exactly one workspace term: plain Q8 charges the scan's
            // thread-local zero-EF plane, Q8Ef (whose residual is the live state above)
            // charges none. The total census delta is the live plane minus that scratch.
            let workspace = |gdn| {
                let dims = rdna_compute::scratch::Qwen4PrefillWorkspaceDims {
                    hidden_size: config.hidden_size,
                    moe_intermediate_size: config.moe_intermediate_size,
                    num_experts: config.num_experts,
                    top_k: config.num_experts_per_tok,
                    hc_count: config.hc_count,
                    hc_lowrank: config.hc_lowrank,
                    gdn_format: Some(gdn),
                    routed_i4: true,
                    include_short_tails: true,
                };
                rdna_compute::scratch::qwen4_prefill_workspace_terms(chunk, &dims).unwrap()
            };
            let (w8, wef) = (
                workspace(crate::GdnStateFormat::Q8),
                workspace(crate::GdnStateFormat::Q8Ef),
            );
            let scratch = rdna_compute::fn_gdn_dense::workspace_ef_bytes_for(
                crate::GdnStateFormat::Q8,
                chunk,
            ) as u64;
            for ((name, q8_bytes), (_, ef_bytes)) in w8.iter().zip(wef.iter()) {
                if *name == "gdn_dense_ef" {
                    assert_eq!((*q8_bytes, *ef_bytes), (scratch, 0), "chunk {chunk}");
                } else {
                    assert_eq!(q8_bytes, ef_bytes, "{name} chunk {chunk}");
                }
            }
            let total = |terms: &[(&'static str, u64); WORKSPACE_TERM_COUNT]| {
                terms.iter().map(|(_, bytes)| bytes).sum::<u64>()
            };
            assert_eq!(total(&w8) - total(&wef), scratch, "chunk {chunk}");
        }
        // A free-VRAM window that fits exactly three resident layers beside the Q8
        // census loses one under Q8Ef: the 54 MiB live plane is a real reserve.
        let q8 = gdn_reserve(1536, crate::GdnStateFormat::Q8, false).bytes();
        let ef = gdn_reserve(1536, crate::GdnStateFormat::Q8Ef, false).bytes();
        let charges = PlacementCharges::new(layer, false).unwrap();
        let free = non_expert + charges.reserve(q8).unwrap() + 3 * layer + (10 << 20);
        assert_eq!(
            incumbent_placement_layers(free, non_expert, experts, layer, 48, q8, false),
            Ok(Some(3))
        );
        assert_eq!(
            incumbent_placement_layers(free, non_expert, experts, layer, 48, ef, false),
            Ok(Some(2))
        );
    }

    #[test]
    fn incumbent_placement_equals_the_inline_composition_for_q8_and_q8_ef_reserves() {
        let weights = xtx_manifest();
        let (non_expert, layer) = resident_split(&weights, xtx_bytes).unwrap();
        let experts = routed_expert_bytes(&weights, xtx_bytes).unwrap();
        for gdn in [crate::GdnStateFormat::Q8, crate::GdnStateFormat::Q8Ef] {
            for floor in [true, false] {
                for (free, chunk) in [
                    (XTX_FREE_BASE, 1536),
                    (XTX_FREE_CAND, 8192),
                    (experts + non_expert + (64 << 30), 8192),
                ] {
                    for stage in [false, true] {
                        let base = gdn_reserve(chunk, gdn, floor).bytes();
                        let shared = incumbent_placement_layers(
                            free, non_expert, experts, layer, 48, base, stage,
                        );
                        let inline = inline_unset_placement(
                            free, non_expert, experts, layer, 48, base, stage,
                        );
                        assert_eq!(
                            shared, inline,
                            "{gdn:?} floor {floor} free {free} chunk {chunk} stage {stage}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn host_placed_layers_and_digest_read_back_the_placed_manifest() {
        let mut weights = xtx_manifest();
        assert!(host_placed_trunk_layers(&weights).is_empty());
        place_routed_experts(&mut weights, 40);
        let host = host_placed_trunk_layers(&weights);
        assert_eq!(host, (40..48).collect::<Vec<_>>());
        let digest = placement_digest(48, &host);
        assert_eq!(digest, placement_digest(48, &host));
        assert_ne!(digest, placement_digest(48, &host[1..]));
        assert_ne!(digest, placement_digest(47, &host));
        // The MTP head never counts, even though placement host-maps it.
        let mut all_host = xtx_manifest();
        place_routed_experts(&mut all_host, 48);
        assert!(host_placed_trunk_layers(&all_host).is_empty());
    }

    #[test]
    fn slowlink_payload_split_is_the_largest_trunk_layer_per_projection() {
        let mut big = entry("model.language_model.layers.1.mlp.experts.gate_up_proj", 1);
        big.logical_shape = vec![8, 4, 256];
        let mut mtp = entry("mtp.layers.0.mlp.experts.gate_up_proj", 0);
        mtp.logical_shape = vec![64, 4, 256];
        let weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.0.mlp.experts.down_proj", 0),
            big,
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry("model.language_model.layers.1.self_attn.q_proj.weight", 1),
            mtp,
        ];
        // 2 bytes per element: [4,4,256] = 8192, [8,4,256] = 16384; the MTP
        // layer and non-expert tensors are not a trunk layer's payload.
        assert_eq!(
            routed_layer_payload_split(&weights, element_bytes),
            Ok((16384, 8192))
        );
        assert!(routed_layer_payload_split(&weights[4..5], element_bytes).is_err());
        assert!(routed_layer_payload_split(&[], element_bytes).is_err());
        // A missing payload size is an error, never zero.
        assert!(routed_layer_payload_split(&weights, |_| None).is_err());
    }

    /// The weight slack counts the resident non-routed-expert tensors only:
    /// routed experts, tied aliases and sub-granule tensors add none, an
    /// exact granule multiple adds none, and the total is checked.
    #[test]
    fn slowlink_granule_weight_slack_counts_the_resident_non_expert_tensors_only() {
        let granule = 2 << 20;
        let model = |name: &str, shape: Vec<usize>| {
            WeightEntry::model(name, shape, DType::BF16, ShardPolicy::Replicate)
        };
        let tied = WeightEntry::model(
            "model.tied",
            vec![1000, 1500],
            DType::BF16,
            ShardPolicy::Tied {
                source: "model.language_model.embed_tokens.weight".into(),
            },
        );
        let weights = vec![
            // 3,000,000 B: 1,194,304 B short of the next granule.
            model("model.language_model.embed_tokens.weight", vec![1000, 1500]),
            // Exactly one granule.
            model("lm_head.weight", vec![1024, 1024]),
            // Below a granule.
            model("model.norm.weight", vec![512]),
            // A routed expert tensor and a tied alias are not resident weights.
            model(
                "model.language_model.layers.0.mlp.experts.down_proj",
                vec![1000, 1500],
            ),
            tied,
        ];
        assert_eq!(
            weight_granule_slack(&weights, element_bytes, granule),
            Ok(1_194_304)
        );
        assert_eq!(weight_granule_slack(&weights, element_bytes, 0), Ok(0));
        assert!(weight_granule_slack(&weights, |_| None, granule).is_err());
        let slack = GranuleSlack {
            weights: u64::MAX,
            f16_shadows: 1,
            ..GranuleSlack::default()
        };
        assert!(slack.total().unwrap_err().contains("overflows"));
        assert_eq!(GranuleSlack::default().total(), Ok(0));
    }

    /// The native MTP slack follows the phase with the larger BACKING, not the
    /// larger logical sum: here the build scratch is the logical winner (1000
    /// vs 990) but the request's allocations round up to a larger backing
    /// (1100 vs 1000), so the request sets the peak. Empty (no kept head),
    /// granule 0 and overflow are explicit.
    #[test]
    fn slowlink_granule_native_mtp_slack_follows_the_larger_backing_phase() {
        let g = 100;
        let resident = [250u64];
        let build = [1000u64];
        let request = [450u64, 540];
        // Resident 250 -> 300; build 1000 -> 1000 (logical peak phase);
        // request 990 -> 500 + 600 = 1100 (backing peak phase).
        assert_eq!(granule_slack_total(build, g), Ok(0));
        assert_eq!(granule_slack_total(request, g), Ok(110));
        // backing peak 300 + 1100, logical peak 250 + 1000.
        assert_eq!(
            native_mtp_granule_slack(&resident, &build, &request, g),
            Ok(1400 - 1250)
        );
        // What charging only the logical winner would have produced: 50.
        assert_eq!(
            granule_slack_total(resident.iter().chain(build.iter()).copied(), g),
            Ok(50)
        );
        // The same inputs with the phases swapped agree (alternatives).
        assert_eq!(
            native_mtp_granule_slack(&resident, &request, &build, g),
            Ok(150)
        );
        // The logical winner also winning on backing keeps its own slack.
        assert_eq!(
            native_mtp_granule_slack(&resident, &[1000], &[500], g),
            Ok(50)
        );
        // No kept head (all lists empty) and no granule model add nothing, and
        // the slack is never negative.
        assert_eq!(native_mtp_granule_slack(&[], &[], &[], g), Ok(0));
        assert_eq!(
            native_mtp_granule_slack(&resident, &build, &request, 0),
            Ok(0)
        );
        // Overflow of a list sum or of the backing is a named error.
        for lists in [
            (vec![u64::MAX, 1], vec![1], vec![1]),
            (vec![1], vec![u64::MAX, 1], vec![1]),
            (vec![1], vec![1], vec![u64::MAX, 1]),
            (vec![u64::MAX - 1], vec![u64::MAX - 1], vec![1]),
        ] {
            let error = native_mtp_granule_slack(&lists.0, &lists.1, &lists.2, g).unwrap_err();
            assert!(error.contains("overflows"), "{error}");
        }
    }
}

#[cfg(test)]
mod decode_share_stage_tests {
    use super::*;
    use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
    use crate::expert_slabs::ExpertFormat;
    use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, ExpertTierPlan, Placement};

    const GU: usize = 1000; // pads to 1024
    const DN: usize = 500; // pads to 512
    const PADDED: u64 = 1024 + 512;

    fn uniform_slabs() -> Vec<ExpertSlab> {
        let mut cursor = 0usize;
        (0..48 * 512)
            .map(|_| {
                let gate_up = ExpertSlabSpan {
                    offset: cursor,
                    len: GU,
                };
                cursor += GU;
                let down = ExpertSlabSpan {
                    offset: cursor,
                    len: DN,
                };
                cursor += DN;
                ExpertSlab {
                    format: ExpertFormat::Mq4Xts,
                    gate_up,
                    down,
                }
            })
            .collect()
    }

    fn tail_pad() -> u64 {
        let align = ExpertTierPlan::PAYLOAD_ALIGN;
        ((crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64).max(align) + align - 1) & !(align - 1)
    }

    /// Two resident experts per layer except layer 7 (odd), which keeps one:
    /// layer 7 has the largest host payload of the odd parity. The last
    /// ranked expert (the coldest) is layer 7's.
    fn plan_and_rank() -> (ExpertTierPlan, Vec<u32>) {
        let mut rank = Vec::new();
        for layer in 0..48u32 {
            if layer == 7 {
                continue;
            }
            rank.push(layer * 512);
            rank.push(layer * 512 + 1);
        }
        rank.push(7 * 512);
        let budget = rank.len() as u64 * PADDED + tail_pad();
        let plan = plan_expert_tiers_from_slabs(&uniform_slabs(), &rank, budget).unwrap();
        assert_eq!(plan.device_experts(), rank.len());
        (plan, rank)
    }

    fn largest_per_parity(
        plan: &ExpertTierPlan,
        entries: &[crate::expert_tier_plan::ExpertTierEntry],
    ) -> [u64; 2] {
        let _ = plan;
        let mut largest = [0u64; 2];
        for layer in 0..48usize {
            let host: u64 = entries[layer * 512..(layer + 1) * 512]
                .iter()
                .filter(|e| e.tier == Placement::HostMapped)
                .map(|e| e.gate_up_len.div_ceil(256) * 256 + e.down_len.div_ceil(256) * 256)
                .sum();
            largest[layer % 2] = largest[layer % 2].max(host);
        }
        largest
    }

    #[test]
    fn decode_share_stage_bytes_resident_view_is_unchanged_for_non_sharing_plans() {
        let (plan, _) = plan_and_rank();
        let resident = expert_tier_stage_bytes(&plan);
        let pad = crate::expert_arena::ARENA_TAIL_PAD_BYTES as u64;
        let largest = largest_per_parity(&plan, plan.entries());
        assert_eq!(resident, largest[0] + largest[1] + 2 * pad);
        assert_eq!(
            expert_tier_stage_bytes_for_entries(plan.entries()),
            resident
        );
        assert_eq!(expert_tier_stage_bytes_prefill(&plan), resident);
        // Per-entry arithmetic agrees with the closed form the loader used before.
        assert_eq!(largest, [510 * PADDED, 511 * PADDED]);
        // A need-0 share is an empty tail and keeps the Resident bytes.
        let (plan, rank) = plan_and_rank();
        let (same, _) = plan.with_decode_share(&rank, 0).unwrap();
        assert_eq!(expert_tier_stage_bytes_prefill(&same), resident);
    }

    #[test]
    fn decode_share_stage_bytes_borrow_view_counts_donors_as_host_experts() {
        let (plan, rank) = plan_and_rank();
        let resident = expert_tier_stage_bytes(&plan);
        // One donor: layer 7's expert, in the layer with the largest host payload.
        let (shared, tail) = plan.clone().with_decode_share(&rank, PADDED).unwrap();
        assert_eq!(tail.donors, vec![7 * 512]);
        let borrow = expert_tier_stage_bytes_prefill(&shared);
        assert_eq!(
            borrow,
            resident + PADDED,
            "the odd-parity maximum grows by exactly the donor payload"
        );
        assert!(borrow >= resident);
        // The Resident view of the shared plan is untouched by sharing (donors
        // stay Device in the plan's own entries).
        assert_eq!(expert_tier_stage_bytes(&shared), resident);
        // A second donor (layer 47, odd, 511 host experts after donating) stays
        // below layer 7's 512, so the odd maximum still grows by one payload.
        let (shared, tail) = plan.clone().with_decode_share(&rank, 2 * PADDED).unwrap();
        assert_eq!(tail.donors, vec![7 * 512, 47 * 512 + 1]);
        assert_eq!(expert_tier_stage_bytes_prefill(&shared), resident + PADDED);
    }
}
