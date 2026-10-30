// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Long-prefill expert staging for sealed expert-slab layers.
//!
//! The legacy stage (`Qwen4ExpertStage`) borrows the two last resident trunk
//! layers as donors and copies a host layer's aggregate gate/up and down
//! tensors over them. A slab layer is one arena whose per-expert extents
//! (MQ4 or W3, by codec) differ from layer to layer, so a donor layer cannot
//! hold an arbitrary cold arena. RouteOwner3 therefore refused staging for
//! slab bundles, and every cold layer's grouped K2 GEMMs read host memory in
//! place over PCIe (measured: cold-layer expert kernels 4.8-6.3x the staged
//! legacy ones, rcpt/speed/prefill.attrib.json).
//!
//! Slab staging uses two dedicated VRAM stage buffers (one per layer parity,
//! each the largest cold arena plus its zeroed tail pad) instead of donors:
//! nothing is overwritten, so there is nothing to restore. Each cold layer's
//! pointer tables name the layer's own arena offsets rebased onto its parity's
//! stage buffer; the layer's codec tag tensor (VRAM, per layer) is untouched.
//! The copy moves the arena plus its tail pad, so the kernels read the same
//! bytes (including the W3 down over-read) they would read in place.
//! Forwards below `min_rows` rows, decode and MTP verify keep reading in place.
//!
//! The dedicated buffers are fresh VRAM, so they are built only where the
//! expert placement charged them (the probe-free switch
//! `expert_stage_placement_enabled`: explicit `1`, or gfx1201). A slab load
//! whose stage is enabled by the measured host link alone (a narrow discrete
//! link, e.g. gfx1100) cannot take uncharged VRAM and must not shrink the
//! frozen placement, so it uses a borrowed stage instead
//! ([`slab_stage_kind`], `new_slab(.., borrowed = true)`): the last resident
//! slab layer of each parity that holds the largest cold arena plus its tail
//! pad lends its own arena allocation as that parity's stage buffer, exactly
//! like the legacy donors. The cold layer's pointer tables are rebased onto
//! the donor arena, its codec tag tensor stays its own, and the donor's own
//! pointer tables and tags are untouched. The donor is overwritten from the
//! first byte of the first copy, which waits on the compute stream's `free`
//! event recorded after the donor's own MoE, and is restored from a pinned
//! host copy of the overwritten prefix after the last host layer of its
//! parity (`join` waits on the restores; `slab_restore_blocking` recovers a
//! forward that never reached it). Residency, decode and every forward
//! below `min_rows` are therefore exactly the unstaged ones.
//!
//! A mixed-codec artifact's cold arenas differ per layer by a few percent, so
//! the resident layer of a parity may be smaller than that parity's largest
//! cold arena (mq2-xts on gfx1100: 840.5 MiB cold layer 38 against an 806.8
//! MiB largest even resident layer). The parity then lends its largest
//! resident layer, provided it is within a sixteenth of the stage
//! ([`plan_slab_donors`]), and each cold layer that does not fit stages its
//! leading experts, whole, up to [`layer_stage_cut`]: the layer's pointer
//! tables rebase only the entries below the cut and keep the host arena address for
//! the trailing ones, which read host memory in place exactly as an unstaged
//! forward does. The bytes are the same either way (the copy carries the
//! following host bytes, so even the over-read past the last staged
//! projection matches), residency is the frozen placement's, and no VRAM or
//! resident layer is added to pay for the overflow.
//!
//! Per-expert tier plans (`expert_tier_*`, aggregate MQ4 Flash-Next, one GPU)
//! reuse the same two dedicated stage buffers but stage per expert: a layer
//! with any host-tier expert copies exactly its HostMapped experts' gate/up
//! and down projections (runs of adjacent host bytes coalesced) into its
//! parity's stage buffer, followed by a zeroed tail pad, and its two 512-entry
//! pointer tables rebase only those host entries. Device-tier entries keep the
//! owner's absolute address: the model-level owners are never a copy
//! destination, so nothing is overwritten and nothing is restored. Tables stay
//! indexed by the original local expert id, so the kernels see the same ids,
//! slot order and combine order as in place; the codec table is untouched.
//! Host layers need not be a suffix: each parity's next host layer is found by
//! scanning `layer + 2, layer + 4, ...`.

use super::*;
use crate::expert_arena::ARENA_TAIL_PAD_BYTES;
use crate::expert_tier_owners::{ExpertReadView, ExpertTierLease};
use crate::expert_tier_plan::Placement;

/// Host sources of the staged layers.
pub(super) struct SlabStageMeta {
    /// Per host layer from `Qwen4ExpertStage::first_host`: the host address of
    /// the layer's arena and the bytes to copy (the staged arena prefix, all of
    /// it unless the layer overflows its donor, plus the tail pad).
    sources: Vec<(usize, usize)>,
    /// `Some` for a per-expert tier stage (`sources` is then empty).
    tier: Option<TierStageMeta>,
    /// The stage buffers are two resident slab layers' arenas (the stage's
    /// `donors`/`donor_views`, restored from `mirrors`) instead of dedicated
    /// VRAM buffers (in `mirrors`). Never set for a tier stage.
    borrowed: bool,
}

/// One host-to-stage copy of a tier layer.
struct StageCopy {
    /// CPU address of the pinned host owner's bytes.
    src: usize,
    /// Byte offset inside the parity's stage buffer.
    dst: usize,
    len: usize,
}

/// Staged copies of the layers with a host-tier expert.
struct TierStageMeta {
    /// Per trunk layer: its slot in `copies`/`used` and in the stage's
    /// `tables` (two per slot: gate/up, down), `None` for a layer with no
    /// host-tier expert (never staged, reads owner VRAM).
    slots: Vec<Option<usize>>,
    copies: Vec<Vec<StageCopy>>,
    /// Bytes of each slot's staged payload (the zeroed tail pad follows).
    used: Vec<usize>,
    /// Keeps both model-level owners alive while the stage exists.
    _keepalive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

/// One expert of a tier layer, as the plan and view give it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TierStageExpert {
    pub(super) host: bool,
    /// Absolute device-visible addresses and lengths of the projections.
    pub(super) gate_up: usize,
    pub(super) gate_up_len: usize,
    pub(super) down: usize,
    pub(super) down_len: usize,
}

/// Where a tier layer's host experts sit in its stage buffer.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct TierStageLayout {
    /// `(source address, stage offset, bytes)`, each a run of adjacent or
    /// overlapping host projections copied once.
    pub(super) copies: Vec<(usize, usize, usize)>,
    /// Stage offset of each expert's gate/up and down, `None` for a device
    /// expert (its table entry stays the owner address).
    pub(super) gate_up: Vec<Option<usize>>,
    pub(super) down: Vec<Option<usize>>,
    /// End of the staged payload; the zeroed tail pad starts here.
    pub(super) used: usize,
}

const STAGE_ALIGN: usize = 256;

/// Lay out the host experts of one tier layer: every host projection must lie
/// inside `[host_base, host_base + host_bytes)`. Projections are sorted by
/// source address and merged where they touch, so a run keeps its relative
/// layout; runs are packed back to back at 256-byte alignment.
pub(super) fn tier_stage_layout(
    experts: &[TierStageExpert],
    host_base: usize,
    host_bytes: usize,
) -> Result<TierStageLayout, String> {
    let host_end = host_base
        .checked_add(host_bytes)
        .ok_or_else(|| "host owner range overflows".to_string())?;
    // (source, len, expert, is_down)
    let mut items: Vec<(usize, usize, usize, bool)> = Vec::new();
    for (eid, expert) in experts.iter().enumerate() {
        if !expert.host {
            continue;
        }
        for (src, len, down) in [
            (expert.gate_up, expert.gate_up_len, false),
            (expert.down, expert.down_len, true),
        ] {
            let end = src
                .checked_add(len)
                .ok_or_else(|| format!("expert {eid} projection overflows"))?;
            if len == 0 || src < host_base || end > host_end {
                return Err(format!(
                    "host expert {eid} extent [{src:#x}, +{len}) is outside the host owner [{host_base:#x}, +{host_bytes})"
                ));
            }
            if src % STAGE_ALIGN != 0 {
                return Err(format!(
                    "host expert {eid} source {src:#x} is not {STAGE_ALIGN}-byte aligned"
                ));
            }
            items.push((src, len, eid, down));
        }
    }
    items.sort_unstable();
    // Runs: (source start, source end, stage offset).
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    let mut item_run = Vec::with_capacity(items.len());
    let mut cursor = 0usize;
    for &(src, len, _, _) in &items {
        let end = src + len;
        match runs.last_mut() {
            Some(run) if src <= run.1 => {
                if end > run.1 {
                    run.1 = end;
                }
            }
            _ => {
                let dst = cursor
                    .checked_add(STAGE_ALIGN - 1)
                    .ok_or_else(|| "stage layout overflows".to_string())?
                    & !(STAGE_ALIGN - 1);
                runs.push((src, end, dst));
                // Provisional end; fixed below once the run is complete.
                cursor = dst;
            }
        }
        item_run.push(runs.len() - 1);
        let run = runs.last().expect("run just pushed or extended");
        cursor = run.2 + (run.1 - run.0);
    }
    let mut layout = TierStageLayout {
        copies: runs
            .iter()
            .map(|&(src, end, dst)| (src, dst, end - src))
            .collect(),
        gate_up: vec![None; experts.len()],
        down: vec![None; experts.len()],
        used: cursor,
    };
    for (&(src, _, eid, down), &run) in items.iter().zip(&item_run) {
        let (run_src, _, run_dst) = runs[run];
        let offset = run_dst + (src - run_src);
        if down {
            layout.down[eid] = Some(offset);
        } else {
            layout.gate_up[eid] = Some(offset);
        }
    }
    Ok(layout)
}

/// A tier layer's little-endian device pointer table: host experts rebased to
/// `stage_base + offset`, every other entry unchanged (the owner address).
pub(super) fn tier_stage_table(
    entries: &[usize],
    offsets: &[Option<usize>],
    stage_base: usize,
) -> Result<Vec<u8>, String> {
    if entries.len() != offsets.len() {
        return Err(format!(
            "{} table entries but {} stage offsets",
            entries.len(),
            offsets.len()
        ));
    }
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    for (expert, (&entry, offset)) in entries.iter().zip(offsets).enumerate() {
        let value = match offset {
            None => entry,
            Some(offset) => stage_base
                .checked_add(*offset)
                .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?,
        };
        table.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(table)
}

/// The next layer after `layer` of the same parity with a staged slot.
fn next_staged_layer(slots: &[Option<usize>], layer: usize) -> Option<usize> {
    (layer + 2..slots.len())
        .step_by(2)
        .find(|&next| slots[next].is_some())
}

/// Bytes of one stage buffer: the largest cold arena plus its tail pad.
pub(super) fn stage_buffer_bytes(arena_bytes: impl IntoIterator<Item = usize>) -> Option<usize> {
    arena_bytes
        .into_iter()
        .max()
        .and_then(|largest| largest.checked_add(ARENA_TAIL_PAD_BYTES))
}

/// Which stage a slab-layout load builds. The expert placement is frozen
/// before the host link is measured (`expert_stage_placement_enabled`), so
/// only a stage that placement charged may take fresh VRAM; a stage the
/// measured link alone enables must borrow resident VRAM instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SlabStageKind {
    /// Two dedicated VRAM stage buffers, charged to the placement.
    Dedicated,
    /// Two resident slab layers' arenas lent as the stage, restored from
    /// pinned host copies: no VRAM beyond the placement.
    Borrowed,
    /// No stage: it would need dedicated VRAM the placement did not charge.
    Off,
}

/// `placement_charged`: the probe-free placement switch
/// (`expert_stage_placement_enabled`) is on, so the dedicated buffers were
/// charged. `tier_layout`: per-expert tiers, which only have the dedicated
/// stage.
pub(super) fn slab_stage_kind(placement_charged: bool, tier_layout: bool) -> SlabStageKind {
    match (placement_charged, tier_layout) {
        (true, _) => SlabStageKind::Dedicated,
        (false, false) => SlabStageKind::Borrowed,
        (false, true) => SlabStageKind::Off,
    }
}

/// Bytes a borrowed stage writes over each layer parity's donor, from the
/// `(layer, arena bytes)` of every host layer: the largest cold arena of that
/// parity plus its tail pad. Both parities must stage something.
pub(crate) fn borrowed_stage_need(
    host_arenas: impl IntoIterator<Item = (usize, usize)>,
) -> Result<[usize; 2], String> {
    let mut need = [0usize; 2];
    for (layer, arena_bytes) in host_arenas {
        let bytes = arena_bytes
            .checked_add(ARENA_TAIL_PAD_BYTES)
            .ok_or("borrowed slab stage bytes overflow")?;
        need[layer % 2] = need[layer % 2].max(bytes);
    }
    if need.contains(&0) {
        return Err("a borrowed slab stage needs a host layer of each parity".to_string());
    }
    Ok(need)
}

/// A donor may be at most `1/BORROWED_OVERFLOW_DIVISOR` of its parity's
/// largest cold stage (arena plus tail pad) short of it. The cold arenas of a
/// mixed-codec artifact differ per layer by a few percent, so the resident
/// layers need not all be as large as the biggest host layer; the overflow of
/// the rare larger layers keeps reading host memory in place ([`stage_cut`]).
/// A donor further short than this stages too little of its layers to pay for
/// the restore copy it costs, so that parity refuses staging.
const BORROWED_OVERFLOW_DIVISOR: usize = 16;

/// The resident slab layer lent to each parity and the bytes of its
/// allocation a staged copy may overwrite (the parity's stage window, also
/// the size of its pinned restore copy). `resident[layer]` is the layer's
/// arena allocation, `None` for a layer that is not a resident slab arena;
/// `resident` covers the layers below the host suffix, so each donor's own
/// MoE runs before every host layer of its parity.
///
/// The donor is the latest layer of the parity that holds `need[parity]`
/// bytes, and its window is exactly `need[parity]`. When no resident layer of
/// the parity is that large, the donor is the layer with the largest
/// allocation (the latest on a tie), its window is the whole allocation, and
/// the parity's larger cold layers overflow it, provided the shortfall is at
/// most `1/BORROWED_OVERFLOW_DIVISOR` of `need[parity]`.
pub(crate) fn plan_slab_donors(
    need: [usize; 2],
    resident: &[Option<usize>],
) -> Result<([usize; 2], [usize; 2]), String> {
    let mut donors = [0usize; 2];
    let mut window = [0usize; 2];
    for parity in 0..2 {
        let candidates = || {
            (0..resident.len())
                .rev()
                .filter(move |layer| layer % 2 == parity)
                .filter_map(move |layer| resident[layer].map(|bytes| (layer, bytes)))
        };
        let (layer, bytes) = candidates()
            .find(|&(_, bytes)| bytes >= need[parity])
            .or_else(|| {
                candidates()
                    .max_by_key(|&(layer, bytes)| (bytes, layer))
                    .filter(|&(_, bytes)| {
                        need[parity] - bytes <= need[parity] / BORROWED_OVERFLOW_DIVISOR
                    })
            })
            .ok_or_else(|| {
                format!(
                    "no resident slab layer of parity {parity} holds the {} MiB stage or comes within 1/{BORROWED_OVERFLOW_DIVISOR} of it",
                    need[parity] >> 20
                )
            })?;
        donors[parity] = layer;
        window[parity] = bytes.min(need[parity]);
    }
    Ok((donors, window))
}

/// Bytes of a host layer's arena (a boundary offset, so everything before it
/// is whole) that a stage window of `window` bytes holds next to the arena's
/// `ARENA_TAIL_PAD_BYTES` that follow them. The whole arena when it and its
/// pad fit; otherwise the largest boundary `<= window - pad` (`offsets`: the
/// arena-relative start of every unit that must not be split, an expert for
/// [`layer_stage_cut`]), `0` if there is none. A layer's payloads are laid
/// out back to back, so a unit that starts before the cut ends at or before
/// it. The copy moves `cut + ARENA_TAIL_PAD_BYTES` source bytes: the last
/// staged projection's kernel over-read sees the same following bytes as it
/// does in place.
pub(crate) fn stage_cut(
    offsets: impl IntoIterator<Item = usize>,
    arena_bytes: usize,
    window: usize,
) -> usize {
    if arena_bytes.saturating_add(ARENA_TAIL_PAD_BYTES) <= window {
        return arena_bytes;
    }
    let limit = window.saturating_sub(ARENA_TAIL_PAD_BYTES);
    offsets
        .into_iter()
        .filter(|&offset| offset <= limit && offset < arena_bytes)
        .max()
        .unwrap_or(0)
}

/// [`stage_cut`] of one host layer at expert granularity: the boundaries are
/// each expert's first byte (the smaller of its non-null gate/up and down
/// table entries), so an expert is staged whole (both projections below the
/// cut) or read in place whole (both at or past it); no expert is ever split
/// between the stage and the host arena. `gate_up` and `down` are the layer's
/// per-expert pointer tables (null = pruned expert).
pub(crate) fn layer_stage_cut(
    gate_up: &[usize],
    down: &[usize],
    arena_base: usize,
    arena_bytes: usize,
    window: usize,
) -> usize {
    stage_cut(
        gate_up.iter().zip(down).filter_map(|(&gate_up, &down)| {
            [gate_up, down]
                .into_iter()
                .filter(|&entry| entry != 0)
                .filter_map(|entry| entry.checked_sub(arena_base))
                .min()
        }),
        arena_bytes,
        window,
    )
}

/// `(staged, in place)` counts of a layer's non-null pointer-table entries
/// (gate/up and down) for a cut: an entry at an arena offset below `cut` is
/// rebased onto the stage, any other keeps its host-arena address
/// ([`rebase_stage_entries`]).
pub(crate) fn stage_split_counts(
    tables: [&[usize]; 2],
    arena_base: usize,
    cut: usize,
) -> (usize, usize) {
    let (mut staged, mut in_place) = (0usize, 0usize);
    for &entry in tables.into_iter().flatten().filter(|&&entry| entry != 0) {
        if entry
            .checked_sub(arena_base)
            .is_some_and(|offset| offset < cut)
        {
            staged += 1;
        } else {
            in_place += 1;
        }
    }
    (staged, in_place)
}

/// The once-per-load log line of one borrowed-stage host layer: the arena
/// bytes copied to its parity's donor against the tail bytes that keep
/// reading the host arena in place, and the pointer-table entries each side
/// holds.
pub(crate) fn stage_layer_report(
    layer: usize,
    donor: usize,
    arena_bytes: usize,
    cut: usize,
    (staged, in_place): (usize, usize),
) -> String {
    format!(
        "  qwen4 slab stage layer {layer} (donor {donor}): staged {cut} B of {arena_bytes} B arena ({staged} pointer-table entries rebased), host tail {} B ({in_place} entries read in place)",
        arena_bytes - cut
    )
}

/// What a borrowed stage does after layer `layer`'s MoE call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BorrowedAfterMoe {
    /// Record `free` on the compute stream: a donor's own MoE has read its
    /// arena. (A host layer's routed down records it itself.)
    pub(super) release: bool,
    pub(super) next: BorrowedNext,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BorrowedNext {
    Nothing,
    /// Copy this host layer over the parity's donor.
    Enqueue(usize),
    /// The parity's last host layer is done: restore the donor.
    Restore,
}

/// The borrowed stage's step after `layer`'s MoE: a donor or host layer
/// refills its parity for the next host layer `layer + 2, ...`, or, past the
/// last one, restores the donor if anything overwrote it (`overwritten`).
pub(super) fn borrowed_after_moe(
    donors: [usize; 2],
    layers: usize,
    is_host: impl Fn(usize) -> bool,
    layer: usize,
    overwritten: bool,
) -> BorrowedAfterMoe {
    let donor = donors[layer % 2] == layer;
    if !donor && !is_host(layer) {
        return BorrowedAfterMoe {
            release: false,
            next: BorrowedNext::Nothing,
        };
    }
    let next = match (layer + 2..layers).step_by(2).find(|&next| is_host(next)) {
        Some(next) => BorrowedNext::Enqueue(next),
        None if overwritten => BorrowedNext::Restore,
        None => BorrowedNext::Nothing,
    };
    BorrowedAfterMoe {
        release: donor,
        next,
    }
}

/// A layer's per-expert pointer table moved from the layer's arena to
/// `stage_base`: every entry must lie inside `[arena_base, arena_base +
/// arena_bytes)`; the result is the little-endian device pointer table.
/// Entries at an arena offset below `cut` ([`stage_cut`]; `arena_bytes` for a
/// fully staged layer) are rebased onto the stage; the rest keep their arena
/// address, so those projections read the host arena in place.
/// `kept` is the layer's `qwen4.expert_keep` row (`None` for dense artifacts):
/// a zero entry is accepted only for an expert outside that row.
pub(super) fn rebase_stage_entries(
    entries: &[usize],
    arena_base: usize,
    arena_bytes: usize,
    cut: usize,
    stage_base: usize,
    kept: Option<&[u16]>,
) -> Result<Vec<u8>, String> {
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    for (expert, &entry) in entries.iter().enumerate() {
        if entry == 0
            && kept.is_some_and(|ids| {
                u16::try_from(expert).is_ok_and(|e| ids.binary_search(&e).is_err())
            })
        {
            table.extend_from_slice(&0usize.to_ne_bytes());
            continue;
        }
        let offset = entry
            .checked_sub(arena_base)
            .filter(|offset| *offset < arena_bytes)
            .ok_or_else(|| {
                format!(
                    "expert {expert} entry {entry:#x} is outside the arena [{arena_base:#x}, +{arena_bytes})"
                )
            })?;
        let target = if offset < cut {
            stage_base
                .checked_add(offset)
                .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?
        } else {
            entry
        };
        table.extend_from_slice(&target.to_ne_bytes());
    }
    Ok(table)
}

impl Qwen4ExpertStage {
    /// Slab-arena stage, or `None` (printed) when this bundle cannot stage.
    /// `borrowed` selects the stage buffers: the two dedicated VRAM buffers
    /// (charged to the placement), or two resident slab layers' arenas lent
    /// as the stage (no VRAM beyond the placement; restored from pinned host
    /// copies after each staged forward).
    pub(super) fn new_slab(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        first_host: usize,
        chunk_rows: usize,
        borrowed: bool,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("long-prefill expert staging off ({reason})"),
            );
            Ok(None)
        };
        if let Some(lease) = bundle.weights.expert_tiers() {
            if borrowed {
                return off("expert tiers have no borrowed stage");
            }
            return Self::new_tier(gpu, bundle, moe, first_host, lease, chunk_rows);
        }
        if bundle.weights.ep_placement().is_some() {
            return off("slab arenas have no EP route");
        }
        if !moe[first_host..]
            .iter()
            .all(|layer| layer.as_ref().is_some_and(RoutedExpertWeights::host_mapped))
        {
            return off("host-mapped slab layers are not a suffix");
        }
        let config = &bundle.config;
        let mut sources = Vec::with_capacity(moe.len() - first_host);
        for (layer, runtime) in moe
            .iter()
            .enumerate()
            .skip(first_host)
            .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
        {
            let Some(slab) = runtime.slab.as_ref() else {
                return off("a host layer is not a slab arena");
            };
            // Stage ELIGIBILITY (the `Route::Stage` capability of every codec this layer holds on
            // this arch), separate from the default switch that decided to build a stage at all
            // (`stage_default_enabled` / `HIPFIRE_QWEN4_EXPERT_STAGE`). Every slab layer was
            // already admitted at load by the same cells (`slab_serves`), so this never refuses a
            // loadable layer; it keeps a measured link or an explicit `1` from staging a codec the
            // arch has no K1/K2 for.
            if !rdna_compute::expert_capabilities::stage_eligible(
                &gpu.arch,
                slab.codecs.iter().filter_map(|&codec| {
                    rdna_compute::expert_capabilities::ExpertQt::of_slab_codec(codec)
                }),
            ) || slab.codecs.iter().any(|&codec| {
                rdna_compute::expert_capabilities::ExpertQt::of_slab_codec(codec).is_none()
            }) {
                return off(&format!(
                    "a host slab layer holds a codec without a Stage capability on {}",
                    gpu.arch
                ));
            }
            let arena_ref = bundle
                .weights
                .layer(layer)
                .ok_or_else(|| invalid("slab stage layer is not owned"))?
                .ffn
                .moe()
                .and_then(|weights| weights.expert_arena.as_ref())
                .ok_or_else(|| invalid("slab stage layer has no arena reference"))?;
            let owner = bundle.weights.resident(arena_ref)?;
            let Some(host) = gpu.host_bytes(owner) else {
                return off("a host slab arena has no pinned host bytes");
            };
            if host.len() < slab.arena_bytes {
                return off("a host slab arena is shorter than its plan");
            }
            // Host-mapped allocations carry a zeroed tail pad past `len()`
            // (`HOST_TAIL_PAD_BYTES`, 1 MiB >= `ARENA_TAIL_PAD_BYTES`).
            sources.push((
                host.as_ptr() as usize,
                slab.arena_bytes + ARENA_TAIL_PAD_BYTES,
            ));
        }
        let host_arenas = || {
            moe.iter()
                .enumerate()
                .skip(first_host)
                .filter_map(|(layer, runtime)| {
                    runtime
                        .as_ref()
                        .and_then(|runtime| runtime.slab.as_ref())
                        .map(|slab| (layer, slab.arena_bytes))
                })
        };
        let Some(stage_bytes) = stage_buffer_bytes(host_arenas().map(|(_, bytes)| bytes)) else {
            return off("no slab host layers");
        };
        // Borrowed: the bytes a staged forward writes over each parity's donor
        // (the largest cold arena of that parity plus its tail pad), the
        // resident slab layer lent for each, and the pinned host copies the
        // donors are restored from, admitted before anything is allocated.
        let lent = if borrowed {
            let need = match borrowed_stage_need(host_arenas()) {
                Ok(need) => need,
                Err(reason) => return off(&reason),
            };
            // Arena bytes of each resident slab layer below the host suffix.
            let resident: Vec<Option<usize>> = (0..first_host)
                .map(|layer| {
                    let runtime = moe[layer].as_ref()?;
                    runtime.slab.as_ref()?;
                    let arena_ref = bundle
                        .weights
                        .layer(layer)?
                        .ffn
                        .moe()?
                        .expert_arena
                        .as_ref()?;
                    let owner = bundle.weights.resident(arena_ref).ok()?;
                    (!runtime.host_mapped() && !owner.buf.is_host_mapped())
                        .then(|| owner.buf.size())
                })
                .collect();
            let (donors, window) = match plan_slab_donors(need, &resident) {
                Ok(plan) => plan,
                Err(reason) => return off(&reason),
            };
            let mirror_bytes = match crate::expert_residency::slab_stage_mirror_bytes(window) {
                Ok(bytes) => bytes,
                Err(reason) => return off(&reason),
            };
            let admitted = crate::expert_residency::check_host_ram(
                mirror_bytes,
                rdna_compute::kv_slots::mem_available_bytes(),
                crate::expert_residency::ttm_pool_estimate(),
            )
            .and_then(|()| {
                crate::expert_residency::check_gtt_cap(
                    mirror_bytes,
                    crate::expert_residency::gtt_budget(),
                )
            });
            if let Err(reason) = admitted {
                return off(&format!("restore copies: {reason}"));
            }
            Some((window, donors))
        } else {
            None
        };
        // Per host layer: the arena bytes its parity's stage holds. A donor
        // that holds the whole cold arena stages it all; a parity whose
        // largest cold arena overflows its donor ([`plan_slab_donors`])
        // stages each overflowing layer's leading experts whole
        // ([`layer_stage_cut`]) and leaves the trailing ones reading the host
        // arena in place, as unstaged forwards do. Every borrowed host layer
        // reports its staged and host-tail bytes once.
        let mut cuts = Vec::with_capacity(sources.len());
        let mut layer_reports = Vec::new();
        let (mut in_place_layers, mut in_place_max) = (0usize, 0usize);
        for (layer, runtime) in moe
            .iter()
            .enumerate()
            .skip(first_host)
            .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
        {
            let slab = runtime.slab.as_ref().expect("checked above");
            let cut = match lent {
                Some((window, _)) => layer_stage_cut(
                    &runtime.expert_gate_up_entries,
                    &runtime.expert_down_entries,
                    slab.arena_base,
                    slab.arena_bytes,
                    window[layer % 2],
                ),
                None => slab.arena_bytes,
            };
            if let Some((_, donors)) = lent {
                layer_reports.push(stage_layer_report(
                    layer,
                    donors[layer % 2],
                    slab.arena_bytes,
                    cut,
                    stage_split_counts(
                        [
                            &runtime.expert_gate_up_entries,
                            &runtime.expert_down_entries,
                        ],
                        slab.arena_base,
                        cut,
                    ),
                ));
            }
            if cut < slab.arena_bytes {
                in_place_layers += 1;
                in_place_max = in_place_max.max(slab.arena_bytes - cut);
            }
            cuts.push(cut);
        }
        for (source, &cut) in sources.iter_mut().zip(&cuts) {
            source.1 = cut + ARENA_TAIL_PAD_BYTES;
        }
        // Staging is an optimisation: stage buffers that do not fit VRAM leave
        // every host layer reading its arena in place. A borrowed stage
        // allocates no VRAM.
        let mut buffers = Vec::with_capacity(2);
        let dedicated_buffers = if borrowed { 0 } else { 2 };
        for _ in 0..dedicated_buffers {
            match gpu.alloc_tensor(&[stage_bytes], DType::Raw) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        let _ = gpu.free_tensor(buffer);
                    }
                    return off(&format!(
                        "two {} MiB stage buffers do not fit VRAM: {error}",
                        stage_bytes >> 20
                    ));
                }
            }
        }
        gpu.ensure_capture_stream()?;
        let (min_rows, min_rows_source) = qwen4_expert_stage_min_rows(gpu, chunk_rows);
        let mut stage = Self {
            min_rows,
            min_rows_source,
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors: lent.map_or([usize::MAX; 2], |(_, donors)| donors),
            donor_views: Vec::with_capacity(if borrowed { 2 } else { 0 }),
            mirrors: Vec::with_capacity(2),
            tables: Vec::with_capacity(2 * sources.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            ep_slots: None,
            bytes: lent.map_or([0, 0], |(window, _)| window),
            copy_stream: gpu.hip.stream_create_non_blocking()?,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: Some(SlabStageMeta {
                sources,
                tier: None,
                borrowed,
            }),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            stage.mirrors.extend(buffers);
            if let Some((window, donors)) = lent {
                // Per parity: a view of the donor arena's overwritten prefix,
                // and its bytes read back once, at load, as the restore copy.
                for (parity, layer) in donors.into_iter().enumerate() {
                    let arena_ref = bundle
                        .weights
                        .layer(layer)
                        .and_then(|weights| weights.ffn.moe())
                        .and_then(|weights| weights.expert_arena.as_ref())
                        .ok_or_else(|| invalid("slab stage donor has no arena reference"))?;
                    let view = bundle
                        .weights
                        .resident(arena_ref)?
                        .sub_offset(0, window[parity]);
                    let mut host = vec![0u8; window[parity]];
                    gpu.memcpy_dtoh_auto(&mut host, &view.buf)?;
                    stage
                        .mirrors
                        .push(gpu.upload_raw_host_mapped(&host, &[window[parity]])?);
                    drop(host);
                    stage.donor_views.push(view);
                }
            }
            for (layer, runtime) in moe
                .iter()
                .enumerate()
                .skip(first_host)
                .filter_map(|(layer, runtime)| runtime.as_ref().map(|runtime| (layer, runtime)))
            {
                let slab = runtime.slab.as_ref().expect("checked above");
                let stage_base = stage.stage_buffer(layer % 2).buf.as_ptr() as usize;
                for entries in [
                    &runtime.expert_gate_up_entries,
                    &runtime.expert_down_entries,
                ] {
                    let kept = config.kept_expert_ids(layer, false);
                    let table = rebase_stage_entries(
                        entries,
                        slab.arena_base,
                        slab.arena_bytes,
                        cuts[layer - first_host],
                        stage_base,
                        kept,
                    )
                    .map_err(|reason| {
                        invalid(format!("layer {layer} slab stage table: {reason}"))
                    })?;
                    let tensor = gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?;
                    stage.tables.push(tensor);
                    gpu.memcpy_htod_auto(
                        &stage.tables.last().expect("table just allocated").buf,
                        &table,
                    )?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        for report in &layer_reports {
            eprintln!("{report}");
        }
        if let Some((window, donors)) = lent {
            let in_place = if in_place_layers == 0 {
                String::new()
            } else {
                format!(
                    "; {in_place_layers} layers overflow their donor and read at most {} MiB of their trailing experts in place",
                    in_place_max >> 20
                )
            };
            eprintln!(
                "  qwen4 long-prefill expert staging: forwards of >= {} rows stage slab host layers {first_host}..{} through resident slab layers {} and {} (borrowed donor VRAM, no dedicated stage buffers; {} MiB pinned restore copies, host RAM on top of the placement's pinned experts{in_place}; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows})",
                stage.min_rows,
                moe.len(),
                donors[0],
                donors[1],
                (window[0] + window[1]) >> 20,
                if stage.capture { "stage" } else { "mapped" },
                stage.min_rows_source.label()
            );
        } else {
            eprintln!(
                "  qwen4 long-prefill expert staging: forwards of >= {} rows stage slab host layers {first_host}..{} through two {} MiB VRAM stage buffers (nothing borrowed, nothing to restore; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows})",
                stage.min_rows,
                moe.len(),
                stage_bytes >> 20,
                if stage.capture { "stage" } else { "mapped" },
                stage.min_rows_source.label()
            );
        }
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Prefill,
            format!(
                "long-prefill expert staging on (>= {} rows)",
                stage.min_rows
            ),
        );
        Ok(Some(stage))
    }

    /// Per-expert tier stage, or `None` (printed) when it cannot stage. Only
    /// layers with a host-tier expert are staged; the layers need not be a
    /// suffix. Device-tier entries keep their owner address in the stage
    /// tables and the model-level owners are never written.
    fn new_tier(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        first_host: usize,
        lease: &ExpertTierLease,
        chunk_rows: usize,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("long-prefill expert staging off ({reason})"),
            );
            Ok(None)
        };
        if bundle.weights.ep_placement().is_some() {
            return off("expert tiers have no EP route");
        }
        let (Some((alias_base, alias_bytes)), Some((cpu_base, cpu_bytes))) = (
            lease.owner_range(Placement::HostMapped),
            lease.host_cpu_range(),
        ) else {
            return off("the tier plan has no pinned host owner");
        };
        if alias_bytes != cpu_bytes {
            return Err(invalid(
                "tier host owner alias and CPU ranges differ in size",
            ));
        }
        let config = &bundle.config;
        // A decode-share lease stages from its largest BorrowPrefill class view
        // (every donor at its mirror, a superset of any smaller class's host
        // entries): the stage is only used by tiles of at least `min_rows`
        // (>= 512) rows, which always read a BorrowPrefill runtime, and a
        // staged donor is a copy of its mirror, correct whether or not the
        // tile's own class dirtied it.
        let read_view = match lease.borrow_classes() {
            0 => ExpertReadView::Resident,
            classes => ExpertReadView::BorrowPrefill(classes - 1),
        };
        let mut slots = Vec::with_capacity(moe.len());
        let mut layouts: Vec<(usize, TierStageLayout)> = Vec::new();
        for (layer, runtime) in moe.iter().enumerate() {
            let Some(runtime) = runtime.as_ref() else {
                slots.push(None);
                continue;
            };
            let view = lease
                .layer_view(layer, read_view)
                .ok_or_else(|| invalid(format!("tier stage: layer {layer} has no tier view")))?;
            // The forward's stage selection and `before_moe` wait use the
            // runtime's layer bit; it must be the view's `any_host`.
            if view.any_host() != runtime.host_mapped() {
                return Err(invalid(format!(
                    "tier stage: layer {layer} runtime host_mapped {} disagrees with its tier view {}",
                    runtime.host_mapped(),
                    view.any_host()
                )));
            }
            if !view.any_host() {
                slots.push(None);
                continue;
            }
            let (gate_up, down) = (view.gate_up_ptrs(), view.down_ptrs());
            if gate_up.len() != config.num_experts || down.len() != config.num_experts {
                return Err(invalid(format!(
                    "tier stage: layer {layer} table length differs from num_experts"
                )));
            }
            let len = |bytes: u64| {
                usize::try_from(bytes)
                    .map_err(|_| invalid("tier stage: expert extent overflows usize"))
            };
            let experts = (0..config.num_experts)
                .map(|eid| {
                    let entry = view.entry(eid).ok_or_else(|| {
                        invalid(format!(
                            "tier stage: layer {layer} expert {eid} has no entry"
                        ))
                    })?;
                    Ok(TierStageExpert {
                        host: entry.tier == Placement::HostMapped,
                        gate_up: gate_up[eid],
                        gate_up_len: len(entry.gate_up_len)?,
                        down: down[eid],
                        down_len: len(entry.down_len)?,
                    })
                })
                .collect::<Result<Vec<_>, Qwen4GpuForwardError>>()?;
            let layout = tier_stage_layout(&experts, alias_base, alias_bytes)
                .map_err(|reason| invalid(format!("tier stage: layer {layer}: {reason}")))?;
            slots.push(Some(layouts.len()));
            layouts.push((layer, layout));
        }
        if layouts.is_empty() {
            return off("no tier layer has a host expert");
        }
        // One buffer per layer parity: the largest staged payload of that
        // parity plus the zeroed tail pad.
        let mut sizes = [ARENA_TAIL_PAD_BYTES; 2];
        for (layer, layout) in &layouts {
            let need = layout
                .used
                .checked_add(ARENA_TAIL_PAD_BYTES)
                .ok_or_else(|| invalid("tier stage buffer size overflows"))?;
            sizes[layer % 2] = sizes[layer % 2].max(need);
        }
        // Pure checked translation of every host source from the device alias
        // to the pinned CPU address, before anything is allocated.
        let cpu_end = cpu_base
            .checked_add(cpu_bytes)
            .ok_or_else(|| invalid("tier host owner CPU range overflows"))?;
        let mut copies = Vec::with_capacity(layouts.len());
        for (layer, layout) in &layouts {
            let mut layer_copies = Vec::with_capacity(layout.copies.len());
            for &(src, dst, len) in &layout.copies {
                let cpu = src
                    .checked_sub(alias_base)
                    .and_then(|offset| cpu_base.checked_add(offset))
                    .filter(|cpu| cpu.checked_add(len).is_some_and(|end| end <= cpu_end))
                    .ok_or_else(|| {
                        invalid(format!(
                            "tier stage: layer {layer} source {src:#x} (+{len}) has no CPU address in the host owner"
                        ))
                    })?;
                layer_copies.push(StageCopy { src: cpu, dst, len });
            }
            copies.push(layer_copies);
        }
        let used = layouts
            .iter()
            .map(|(_, layout)| layout.used)
            .collect::<Vec<_>>();
        gpu.ensure_capture_stream()?;
        // Staging is an optimisation: stage buffers that do not fit VRAM leave
        // every host-tier expert read in place.
        let mut buffers = Vec::with_capacity(2);
        for size in sizes {
            match gpu.alloc_tensor(&[size], DType::Raw) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    for buffer in buffers {
                        let _ = gpu.free_tensor(buffer);
                    }
                    return off(&format!(
                        "stage buffers of {} and {} MiB do not fit VRAM: {error}",
                        sizes[0] >> 20,
                        sizes[1] >> 20
                    ));
                }
            }
        }
        // From here the buffers are owned by `stage.mirrors`, except while the
        // copy stream is created.
        let copy_stream = match gpu.hip.stream_create_non_blocking() {
            Ok(stream) => stream,
            Err(error) => {
                for buffer in buffers {
                    let _ = gpu.free_tensor(buffer);
                }
                return Err(error.into());
            }
        };
        let (min_rows, min_rows_source) = qwen4_expert_stage_min_rows(gpu, chunk_rows);
        let mut stage = Self {
            min_rows,
            min_rows_source,
            capture: hipfire_config::developer_bool(QWEN4_EXPERT_STAGE_CAPTURE_ENV, true),
            donors: [usize::MAX; 2],
            donor_views: Vec::new(),
            mirrors: buffers,
            tables: Vec::with_capacity(2 * layouts.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            ep_slots: None,
            bytes: [0, 0],
            copy_stream,
            ready: Vec::with_capacity(2),
            free: Vec::with_capacity(2),
            state: std::cell::Cell::new([DonorState::Resident; 2]),
            slab: Some(SlabStageMeta {
                sources: Vec::new(),
                tier: Some(TierStageMeta {
                    slots,
                    copies,
                    used,
                    _keepalive: lease.keepalive(),
                }),
                borrowed: false,
            }),
        };
        let result = (|| {
            for _ in 0..2 {
                stage.ready.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
                stage.free.push(
                    gpu.hip
                        .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
                );
            }
            // `mirrors` already owns the stage buffers (see `Self { .. }`).
            for (layer, layout) in &layouts {
                let view = lease.layer_view(*layer, read_view).expect("checked above");
                let stage_base = stage.mirrors[layer % 2].buf.as_ptr() as usize;
                for (entries, offsets) in [
                    (view.gate_up_ptrs(), &layout.gate_up),
                    (view.down_ptrs(), &layout.down),
                ] {
                    let table =
                        tier_stage_table(entries, offsets, stage_base).map_err(|reason| {
                            invalid(format!("tier stage: layer {layer} table: {reason}"))
                        })?;
                    let tensor = gpu.alloc_tensor(&[2 * config.num_experts], DType::F32)?;
                    stage.tables.push(tensor);
                    gpu.memcpy_htod_auto(
                        &stage.tables.last().expect("table just allocated").buf,
                        &table,
                    )?;
                }
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            let _ = stage.free_gpu(gpu);
            return Err(error);
        }
        let host_experts = layouts
            .iter()
            .map(|(_, layout)| layout.gate_up.iter().flatten().count())
            .sum::<usize>();
        // Under decode-share the stage tables come from the BorrowPrefill view:
        // the lease's donors are staged copies of their pinned mirror, so the
        // stage never points at a donor's canonical (scratch-borrowed) address.
        let view_note = if matches!(read_view, ExpertReadView::BorrowPrefill(_)) {
            let donors = layouts
                .iter()
                .map(|(layer, layout)| {
                    let resident = lease.layer(*layer).expect("checked above");
                    (0..config.num_experts)
                        .filter(|&eid| {
                            layout.gate_up[eid].is_some()
                                && resident.placement(eid) == Some(Placement::Device)
                        })
                        .count()
                })
                .sum::<usize>();
            format!(
                " [BorrowPrefill view: {} resident-view host + {donors} donor experts; valid for both views]",
                host_experts - donors
            )
        } else {
            String::new()
        };
        eprintln!(
            "  qwen4 long-prefill expert staging: forwards of >= {} rows stage the {host_experts} host-tier experts of {} tier layers through two VRAM stage buffers of {} and {} MiB (device-tier experts keep their owner address; owners are never written, nothing to restore; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows}){view_note}",
            stage.min_rows,
            layouts.len(),
            sizes[0] >> 20,
            sizes[1] >> 20,
            if stage.capture { "stage" } else { "mapped" },
            stage.min_rows_source.label()
        );
        hipfire_config::load_summary::note(
            hipfire_config::load_summary::Facet::Prefill,
            format!(
                "long-prefill expert staging on (>= {} rows)",
                stage.min_rows
            ),
        );
        Ok(Some(stage))
    }

    /// The VRAM a staged host layer's arena (tier: experts) is copied into:
    /// the dedicated stage buffer, or the borrowed donor arena of that parity.
    fn stage_buffer(&self, parity: usize) -> &GpuTensor {
        if self.slab.as_ref().is_some_and(|meta| meta.borrowed) {
            &self.donor_views[parity]
        } else {
            &self.mirrors[parity]
        }
    }

    /// The tables a staged host `layer` reads (slab arena or tier). Replaces
    /// the `layer - first_host` indexing: tier host layers are not a suffix.
    pub(super) fn slab_tables(&self, layer: usize) -> MoeStageTables<'_> {
        let index = match self.slab.as_ref().and_then(|meta| meta.tier.as_ref()) {
            Some(tier) => tier
                .slots
                .get(layer)
                .copied()
                .flatten()
                .expect("tier stage tables requested for a layer with no host-tier expert"),
            None => layer - self.first_host,
        };
        MoeStageTables {
            gate_up: &self.tables[2 * index],
            down: &self.tables[2 * index + 1],
            free: &self.free[layer % 2],
        }
    }

    /// Start of a staged forward: the first two host layers (one per parity)
    /// copy while the resident layers compute. A tier stage primes the first
    /// staged layer of each parity. A borrowed stage primes nothing: each
    /// donor arena holds its own experts until that donor's MoE has run, and
    /// its copy starts from that MoE ([`Self::slab_after_moe`]).
    pub(super) fn slab_prime(&self, gpu: &Gpu, layers: usize) -> Result<(), Qwen4GpuForwardError> {
        let Some(meta) = self.slab.as_ref() else {
            return Ok(());
        };
        if meta.borrowed {
            return Ok(());
        }
        if let Some(tier) = meta.tier.as_ref() {
            for parity in 0..2 {
                let first = tier
                    .slots
                    .iter()
                    .enumerate()
                    .skip(parity)
                    .step_by(2)
                    .find_map(|(layer, slot)| slot.map(|_| layer));
                if let Some(layer) = first.filter(|&layer| layer < layers) {
                    self.slab_enqueue(gpu, layer)?;
                }
            }
            return Ok(());
        }
        for layer in self.first_host..(self.first_host + 2).min(layers) {
            self.slab_enqueue(gpu, layer)?;
        }
        Ok(())
    }

    /// Copy host `layer`'s arena (tier: its host-tier experts) over its
    /// parity's stage buffer once the previous occupant's down released it,
    /// recording `ready`. A borrowed stage marks the donor overwritten from
    /// the first byte issued.
    fn slab_enqueue(&self, gpu: &Gpu, layer: usize) -> Result<(), Qwen4GpuForwardError> {
        let meta = self
            .slab
            .as_ref()
            .ok_or_else(|| invalid("not a slab stage"))?;
        let parity = layer % 2;
        if let Some(tier) = meta.tier.as_ref() {
            let slot = tier
                .slots
                .get(layer)
                .copied()
                .flatten()
                .ok_or_else(|| invalid("tier stage layer has no host-tier expert"))?;
            gpu.hip
                .stream_wait_event(&self.copy_stream, &self.free[parity])?;
            let buffer = &self.mirrors[parity].buf;
            for copy in &tier.copies[slot] {
                // SAFETY: `copy.src` lies inside the lease's pinned host owner
                // (checked against its range in `new_tier`), which `_keepalive`
                // retains and nothing writes after materialization.
                let source = unsafe { std::slice::from_raw_parts(copy.src as *const u8, copy.len) };
                gpu.hip.memcpy_htod_async(
                    &buffer.byte_view(copy.dst, copy.len),
                    source,
                    &self.copy_stream,
                )?;
            }
            // The previous occupant may have been larger: re-zero the pad the
            // kernels' tail over-read can reach past the last staged byte.
            gpu.hip.memset_async(
                &buffer.byte_view(tier.used[slot], ARENA_TAIL_PAD_BYTES),
                0,
                ARENA_TAIL_PAD_BYTES,
                &self.copy_stream,
            )?;
            gpu.hip
                .event_record(&self.ready[parity], Some(&self.copy_stream))?;
            return Ok(());
        }
        let &(address, bytes) = meta
            .sources
            .get(layer - self.first_host)
            .ok_or_else(|| invalid("slab stage layer out of range"))?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        if meta.borrowed {
            // From the first byte issued, the donor no longer holds its arena.
            self.set_state(parity, DonorState::Overwritten);
        }
        // SAFETY: `address` is the registered host allocation of the layer's
        // arena (immutable for the bundle's lifetime) and `bytes` is at most
        // the arena plus `ARENA_TAIL_PAD_BYTES` of its zeroed tail slack (see
        // `new_slab`; a layer that overflows its donor copies a shorter prefix).
        let source = unsafe { std::slice::from_raw_parts(address as *const u8, bytes) };
        gpu.hip
            .memcpy_htod_async(&self.stage_buffer(parity).buf, source, &self.copy_stream)?;
        gpu.hip
            .event_record(&self.ready[parity], Some(&self.copy_stream))?;
        Ok(())
    }

    /// After a host layer's MoE (its down recorded `free`): refill its parity
    /// for the next host layer of the same parity (slab: L+2). A borrowed
    /// stage also hooks its donors: a donor's own MoE frees its arena, then
    /// the first host layer of that parity copies over it; after the last
    /// host layer of a parity the donor is restored from its pinned copy.
    pub(super) fn slab_after_moe(
        &self,
        gpu: &Gpu,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        layer: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        if let Some(tier) = self.slab.as_ref().and_then(|meta| meta.tier.as_ref()) {
            if tier.slots.get(layer).copied().flatten().is_none() {
                return Ok(());
            }
            return match next_staged_layer(&tier.slots, layer) {
                Some(next) => self.slab_enqueue(gpu, next),
                None => Ok(()),
            };
        }
        if self.slab.as_ref().is_some_and(|meta| meta.borrowed) {
            let parity = layer % 2;
            let step = borrowed_after_moe(
                self.donors,
                moe.len(),
                |candidate| {
                    moe[candidate]
                        .as_ref()
                        .is_some_and(RoutedExpertWeights::host_mapped)
                },
                layer,
                self.state.get()[parity] == DonorState::Overwritten,
            );
            if step.release {
                // The donor's own MoE read its arena on the compute stream.
                let compute = gpu
                    .active_stream
                    .as_ref()
                    .ok_or_else(|| invalid("G2 compute stream missing"))?;
                gpu.hip.event_record(&self.free[parity], Some(compute))?;
            }
            return match step.next {
                BorrowedNext::Nothing => Ok(()),
                BorrowedNext::Enqueue(next) => self.slab_enqueue(gpu, next),
                BorrowedNext::Restore => self.slab_enqueue_restore(gpu, parity),
            };
        }
        if !moe[layer]
            .as_ref()
            .is_some_and(RoutedExpertWeights::host_mapped)
        {
            return Ok(());
        }
        match moe.get(layer + 2) {
            Some(Some(next)) if next.host_mapped() => self.slab_enqueue(gpu, layer + 2),
            _ => Ok(()),
        }
    }

    /// Copy the donor arena's own prefix back from its pinned host copy once
    /// the last host layer of its parity is done with it, recording `ready`.
    fn slab_enqueue_restore(&self, gpu: &Gpu, parity: usize) -> Result<(), Qwen4GpuForwardError> {
        let host = gpu
            .host_bytes(&self.mirrors[parity])
            .ok_or_else(|| invalid("G2 slab donor restore copy missing"))?;
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        gpu.hip
            .memcpy_htod_async(&self.donor_views[parity].buf, host, &self.copy_stream)?;
        gpu.hip
            .event_record(&self.ready[parity], Some(&self.copy_stream))?;
        self.set_state(parity, DonorState::Restoring);
        Ok(())
    }

    /// End of a staged forward on a borrowed slab stage: the compute stream
    /// waits on every donor restore, as the legacy [`Self::join`] does.
    pub(super) fn slab_join(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if !self.slab.as_ref().is_some_and(|meta| meta.borrowed) {
            return Ok(());
        }
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Overwritten {
                self.slab_enqueue_restore(gpu, parity)?;
            }
            if self.state.get()[parity] == DonorState::Restoring {
                let compute = gpu
                    .active_stream
                    .as_ref()
                    .ok_or_else(|| invalid("G2 compute stream missing"))?;
                gpu.hip.stream_wait_event(compute, &self.ready[parity])?;
                self.set_state(parity, DonorState::Resident);
            }
        }
        Ok(())
    }

    /// Recovery on a borrowed slab stage when a staged forward did not reach
    /// [`Self::slab_join`]: drain the device, then copy every donor arena not
    /// known resident back synchronously. Dedicated and tier stages never
    /// overwrite an owner, so they have nothing to restore.
    pub(super) fn slab_restore_blocking(&self, gpu: &Gpu) -> Result<(), Qwen4GpuForwardError> {
        if !self.slab.as_ref().is_some_and(|meta| meta.borrowed)
            || self.state.get() == [DonorState::Resident; 2]
        {
            return Ok(());
        }
        if let Some(stream) = gpu.active_stream.as_ref() {
            if gpu.hip.stream_is_capturing(stream)? {
                return Err(invalid(
                    "G2 donor slab arenas need a restore, which a stream capture cannot run",
                ));
            }
        }
        gpu.hip.device_synchronize()?;
        for parity in 0..2 {
            if self.state.get()[parity] == DonorState::Resident {
                continue;
            }
            let host = gpu
                .host_bytes(&self.mirrors[parity])
                .ok_or_else(|| invalid("G2 slab donor restore copy missing"))?;
            gpu.hip.memcpy_htod(&self.donor_views[parity].buf, host)?;
            gpu.hip.device_synchronize()?;
            self.set_state(parity, DonorState::Resident);
        }
        Ok(())
    }
}

/// The per-layer expert layout of the real `mq2-xts` artifact (48 trunk
/// layers x 512 experts, MQ4 or E8P each), read from the checked-in fixture
/// `tests/fixtures/mq2_xts_expert_layout.txt` (provenance, artifact stat and
/// SHA-256 are in its header). CPU tests only.
#[cfg(test)]
pub(crate) mod mq2_xts_layout {
    pub(crate) const FIXTURE: &str = include_str!("../../tests/fixtures/mq2_xts_expert_layout.txt");
    /// `(gate/up, down)` payload bytes of an MQ4 (qt44 + qt53) expert.
    const MQ4: (usize, usize) = (1_740_800, 870_400);
    /// `(gate/up, down)` payload bytes of an E8P (qt135 + qt136) expert.
    const E8P: (usize, usize) = (870_400, 435_200);

    /// Codec characters of every expert of `layer` (`4` = MQ4, `e` = E8P).
    pub(crate) fn codecs(layer: usize) -> &'static str {
        let line = FIXTURE
            .lines()
            .filter(|line| !line.starts_with('#'))
            .nth(layer)
            .expect("fixture has a line for every trunk layer");
        let mut fields = line.split_whitespace();
        assert_eq!(fields.next(), Some("layer"));
        assert_eq!(fields.next().and_then(|n| n.parse().ok()), Some(layer));
        let codecs = fields.next().expect("fixture layer has its codecs");
        assert_eq!(codecs.len(), 512);
        codecs
    }

    /// `[gate/up offset, gate/up bytes, down offset, down bytes]` of every
    /// expert of `layer` (arena-relative, payload order `gate_up, down` per
    /// expert, back to back) and the layer's arena bytes.
    pub(crate) fn experts(layer: usize) -> (Vec<[usize; 4]>, usize) {
        let mut offset = 0;
        let experts = codecs(layer)
            .bytes()
            .map(|codec| {
                let (gate_up, down) = match codec {
                    b'4' => MQ4,
                    b'e' => E8P,
                    other => panic!("unknown codec {:?}", other as char),
                };
                let expert = [offset, gate_up, offset + gate_up, down];
                offset += gate_up + down;
                expert
            })
            .collect();
        (experts, offset)
    }

    pub(crate) fn arena_bytes(layer: usize) -> usize {
        experts(layer).1
    }

    /// The layer's per-expert `(gate/up, down)` pointer tables with the arena
    /// at `arena_base`.
    pub(crate) fn tables(experts: &[[usize; 4]], arena_base: usize) -> (Vec<usize>, Vec<usize>) {
        (
            experts.iter().map(|e| arena_base + e[0]).collect(),
            experts.iter().map(|e| arena_base + e[2]).collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebase_moves_every_entry_by_the_same_delta() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        let entries = [arena, arena + 1280, arena + 99_999];
        let table = rebase_stage_entries(&entries, arena, 100_000, 100_000, stage, None).unwrap();
        let decoded: Vec<usize> = table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![stage, stage + 1280, stage + 99_999]);
    }

    #[test]
    fn rebase_refuses_entries_outside_the_arena_after_an_arena_swap() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        // Below the base, at the end, and a table of a different (swapped) arena.
        assert!(rebase_stage_entries(&[arena - 8], arena, 100_000, 100_000, stage, None).is_err());
        assert!(
            rebase_stage_entries(&[arena + 100_000], arena, 100_000, 100_000, stage, None).is_err()
        );
        let other_arena = 0x9000_0000usize;
        assert!(
            rebase_stage_entries(&[other_arena + 16], arena, 100_000, 100_000, stage, None)
                .is_err()
        );
        assert!(rebase_stage_entries(
            &[arena, usize::MAX],
            arena,
            100_000,
            100_000,
            usize::MAX - 8,
            None
        )
        .is_err());
    }

    #[test]
    fn rebase_rejects_dense_null_and_preserves_pruned_absent_null() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        // Dense (no keep row): a null entry is an error, as before pruning existed.
        assert!(rebase_stage_entries(&[arena, 0], arena, 100_000, 100_000, stage, None).is_err());
        // Pruned: expert 1 is outside the keep row, so its null entry stays null.
        let kept = [0u16, 2];
        let table = rebase_stage_entries(
            &[arena, 0, arena + 8],
            arena,
            100_000,
            100_000,
            stage,
            Some(&kept),
        )
        .unwrap();
        let decoded: Vec<usize> = table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![stage, 0, stage + 8]);
        // A null entry for a kept expert is still an error.
        assert!(rebase_stage_entries(
            &[arena, 0, arena + 8],
            arena,
            100_000,
            100_000,
            stage,
            Some(&[0, 1, 2])
        )
        .is_err());
    }

    #[test]
    fn stage_buffer_holds_the_largest_arena_and_its_tail_pad() {
        assert_eq!(
            stage_buffer_bytes([10, 30, 20]),
            Some(30 + ARENA_TAIL_PAD_BYTES)
        );
        assert_eq!(stage_buffer_bytes(std::iter::empty()), None);
        assert_eq!(stage_buffer_bytes([usize::MAX]), None);
    }

    const GU: usize = 1_740_800;
    const DN: usize = 870_400;
    const HOST: usize = 0x7000_0000;

    fn expert(host: bool, base: usize) -> TierStageExpert {
        TierStageExpert {
            host,
            gate_up: base,
            gate_up_len: GU,
            down: base + GU,
            down_len: DN,
        }
    }

    fn decode(table: &[u8]) -> Vec<usize> {
        table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn tier_layout_copies_only_host_experts_and_keeps_device_entries() {
        let device = 0x2_0000_0000usize;
        // Experts 1 and 3 are host, adjacent in the host owner; 0 and 2 device.
        let experts = [
            expert(false, device),
            expert(true, HOST + 4096),
            expert(false, device + GU + DN),
            expert(true, HOST + 4096 + GU + DN),
        ];
        let layout = tier_stage_layout(&experts, HOST, 64 << 20).unwrap();
        // One coalesced run covering both experts' gate/up and down.
        assert_eq!(layout.copies, vec![(HOST + 4096, 0, 2 * (GU + DN))]);
        assert_eq!(layout.used, 2 * (GU + DN));
        assert_eq!(layout.gate_up, vec![None, Some(0), None, Some(GU + DN)]);
        assert_eq!(layout.down, vec![None, Some(GU), None, Some(GU + DN + GU)]);
        let entries: Vec<usize> = experts.iter().map(|e| e.gate_up).collect();
        let stage = 0x3_0000_0000usize;
        let table = decode(&tier_stage_table(&entries, &layout.gate_up, stage).unwrap());
        // Same length and original local-id order; device entries untouched.
        assert_eq!(
            table,
            vec![device, stage, device + GU + DN, stage + GU + DN]
        );
    }

    /// A BorrowPrefill layer view: the donors (device-resident in the
    /// Resident view, cold) point at the pinned mirror, so they stage as host
    /// entries beside the layer's own host experts, while the non-donor device
    /// experts keep their owner addresses (valid for both views).
    #[test]
    fn tier_layout_from_the_borrow_view_stages_donors_and_keeps_resident_non_donors() {
        let device = 0x2_0000_0000usize;
        let stride = GU + DN;
        // Resident view: 0 and 2 device, 1 host. Experts 2 and 3 are donors:
        // device in the Resident view, at the mirror in the BorrowPrefill view.
        let resident = [
            expert(false, device),
            expert(true, HOST + 4096),
            expert(false, device + 2 * stride),
            expert(false, device + 3 * stride),
        ];
        let mirror = HOST + (8 << 20);
        let borrow = [
            resident[0],
            resident[1],
            expert(true, mirror),
            expert(true, mirror + stride),
        ];
        let res_layout = tier_stage_layout(&resident, HOST, 64 << 20).unwrap();
        let layout = tier_stage_layout(&borrow, HOST, 64 << 20).unwrap();
        assert_eq!(res_layout.gate_up, vec![None, Some(0), None, None]);
        // The donors are staged copies of the mirror; the adjacent mirror
        // slots coalesce into one run.
        assert_eq!(layout.copies.len(), 2);
        assert!(layout
            .copies
            .contains(&(mirror, layout.gate_up[2].unwrap(), 2 * stride)));
        assert!(layout.gate_up[2].is_some() && layout.gate_up[3].is_some());
        assert_eq!(layout.gate_up[0], None);
        // Non-donor device experts keep their owner address in the stage table.
        let entries: Vec<usize> = borrow.iter().map(|e| e.gate_up).collect();
        let stage = 0x3_0000_0000usize;
        let table = decode(&tier_stage_table(&entries, &layout.gate_up, stage).unwrap());
        assert_eq!(table[0], device);
        assert!(table[1] >= stage && table[2] >= stage && table[3] >= stage);
        // No staged entry or copy source is a donor's canonical device address.
        assert!(layout
            .copies
            .iter()
            .all(|&(src, _, _)| src >= HOST && src < HOST + (64 << 20)));
        assert!(table
            .iter()
            .all(|&entry| entry != device + 2 * stride && entry != device + 3 * stride));
    }

    #[test]
    fn tier_layout_packs_separate_runs_aligned_and_never_reads_device_bytes() {
        let experts = [
            expert(true, HOST + 2 * (GU + DN)),
            expert(true, HOST),
            expert(false, 0x2_0000_0000),
        ];
        let layout = tier_stage_layout(&experts, HOST, 64 << 20).unwrap();
        assert_eq!(layout.copies.len(), 2);
        // Sorted by source address; each run 256-aligned in the stage.
        assert_eq!(layout.copies[0], (HOST, 0, GU + DN));
        assert_eq!(layout.copies[1], (HOST + 2 * (GU + DN), GU + DN, GU + DN));
        assert!(layout
            .copies
            .iter()
            .all(|&(src, dst, _)| src % 256 == 0 && dst % 256 == 0));
        // Expert 0 (host, later source) maps to the second run; ids keep their slots.
        assert_eq!(layout.gate_up, vec![Some(GU + DN), Some(0), None]);
    }

    #[test]
    fn tier_layout_refuses_extents_outside_the_host_owner() {
        let in_owner = expert(true, HOST);
        assert!(tier_stage_layout(&[in_owner], HOST, GU + DN - 1).is_err());
        assert!(tier_stage_layout(&[expert(true, HOST - 256)], HOST, 64 << 20).is_err());
        // A host-flagged expert whose pointer names the device owner is refused.
        assert!(tier_stage_layout(&[expert(true, 0x2_0000_0000)], HOST, 64 << 20).is_err());
        let mut zero = in_owner;
        zero.down_len = 0;
        assert!(tier_stage_layout(&[zero], HOST, 64 << 20).is_err());
        let mut unaligned = in_owner;
        unaligned.gate_up += 8;
        assert!(tier_stage_layout(&[unaligned], HOST, 64 << 20).is_err());
    }

    #[test]
    fn tier_layout_with_no_host_experts_stages_nothing() {
        let layout = tier_stage_layout(&[expert(false, 0x2_0000_0000)], HOST, 64 << 20).unwrap();
        assert!(layout.copies.is_empty());
        assert_eq!(layout.used, 0);
        assert_eq!(layout.gate_up, vec![None]);
    }

    #[test]
    fn tier_table_refuses_mismatched_offsets_and_overflow() {
        assert!(tier_stage_table(&[1, 2], &[None], 0).is_err());
        assert!(tier_stage_table(&[0], &[Some(16)], usize::MAX - 8).is_err());
    }

    #[test]
    fn next_staged_layer_walks_same_parity_only() {
        let slots = [None, Some(0), None, Some(1), Some(2), None, Some(3), None];
        assert_eq!(next_staged_layer(&slots, 1), Some(3));
        assert_eq!(next_staged_layer(&slots, 3), None);
        assert_eq!(next_staged_layer(&slots, 4), Some(6));
        assert_eq!(next_staged_layer(&slots, 6), None);
    }

    /// One 512-expert E8P arena (the mq2-xts cold codec): 512 x 1,305,600 B.
    const E8P_ARENA: usize = 512 * 1_305_600;

    #[test]
    fn slab_stage_is_dedicated_only_where_placement_charged_it() {
        // Charged (explicit `1`, or gfx1201): the dedicated buffers, as before.
        assert_eq!(slab_stage_kind(true, false), SlabStageKind::Dedicated);
        assert_eq!(slab_stage_kind(true, true), SlabStageKind::Dedicated);
        // Enabled by the measured link alone (gfx1100 narrow link): nothing
        // was charged, so a slab load borrows resident layers instead of
        // allocating, and keeps every resident layer.
        assert_eq!(slab_stage_kind(false, false), SlabStageKind::Borrowed);
        // A tier layout has only the dedicated stage: still refused uncharged.
        assert_eq!(slab_stage_kind(false, true), SlabStageKind::Off);
    }

    #[test]
    fn borrowed_need_is_the_largest_cold_arena_of_each_parity_plus_pad() {
        let uniform = (10..48).map(|layer| (layer, E8P_ARENA));
        assert_eq!(
            borrowed_stage_need(uniform),
            Ok([E8P_ARENA + ARENA_TAIL_PAD_BYTES; 2])
        );
        // Mixed arenas: each parity takes its own maximum.
        let mixed = [(10, 100), (11, 900), (12, 300), (13, 200)];
        assert_eq!(
            borrowed_stage_need(mixed),
            Ok([300 + ARENA_TAIL_PAD_BYTES, 900 + ARENA_TAIL_PAD_BYTES])
        );
        // One host layer leaves a parity with nothing to stage; overflow is an error.
        assert!(borrowed_stage_need([(10, 100)]).is_err());
        assert!(borrowed_stage_need(std::iter::empty()).is_err());
        assert!(borrowed_stage_need([(10, usize::MAX), (11, 1)]).is_err());
    }

    #[test]
    fn donors_are_the_latest_resident_slab_layers_that_hold_the_stage() {
        let held = Some(E8P_ARENA + ARENA_TAIL_PAD_BYTES);
        let need = [E8P_ARENA + ARENA_TAIL_PAD_BYTES; 2];
        // Ten resident layers (placement unchanged): the last two lend.
        assert_eq!(plan_slab_donors(need, &[held; 10]), Ok(([8, 9], need)));
        // A smaller latest layer of a parity cannot hold the stage: that
        // parity falls back to the previous layer of its parity that does.
        let mut resident = vec![held; 10];
        resident[9] = Some(E8P_ARENA);
        resident[8] = Some(1);
        assert_eq!(plan_slab_donors(need, &resident), Ok(([6, 7], need)));
        // A layer that is not a resident slab arena is never a donor.
        resident[7] = None;
        assert_eq!(plan_slab_donors(need, &resident), Ok(([6, 5], need)));
        // No resident layer of a parity holds it, or comes close: refused,
        // naming the parity.
        let error = plan_slab_donors(need, &[Some(1); 10]).unwrap_err();
        assert!(error.contains("parity 0"), "{error}");
        let error = plan_slab_donors(need, &[held]).unwrap_err();
        assert!(error.contains("parity 1"), "{error}");
        assert!(plan_slab_donors(need, &[]).is_err());
    }

    #[test]
    fn a_parity_without_a_full_donor_lends_its_largest_layer_within_one_sixteenth() {
        let need = [1600usize + ARENA_TAIL_PAD_BYTES; 2];
        let short = |bytes: usize| Some(bytes + ARENA_TAIL_PAD_BYTES);
        // Parity 0 has no layer that holds 1600 B: the largest (layer 2, 1560)
        // lends its whole allocation as the window. Parity 1 has a full layer
        // (layer 3), so it keeps the full-need window and the latest holder
        // even though layer 1 is larger.
        let resident = [short(1500), short(1700), short(1560), short(1600)];
        assert_eq!(
            plan_slab_donors(need, &resident),
            Ok(([2, 3], [1560 + ARENA_TAIL_PAD_BYTES, need[1]]))
        );
        // Equal largest allocations: the latest layer of that parity.
        let tied = [short(1560), short(1600), short(1560), short(1600), None];
        assert_eq!(
            plan_slab_donors(need, &tied),
            Ok(([2, 3], [1560 + ARENA_TAIL_PAD_BYTES, need[1]]))
        );
        // A shortfall of exactly one sixteenth of the stage is still lent;
        // one byte more is refused.
        let limit = need[0] - need[0] / BORROWED_OVERFLOW_DIVISOR;
        assert!(plan_slab_donors(need, &[Some(limit), Some(need[1])]).is_ok());
        let error = plan_slab_donors(need, &[Some(limit - 1), Some(need[1])]).unwrap_err();
        assert!(
            error.contains("parity 0") && error.contains("1/16"),
            "{error}"
        );
    }

    #[test]
    fn stage_cut_is_the_last_projection_boundary_that_fits_next_to_the_pad() {
        let pad = ARENA_TAIL_PAD_BYTES;
        // Four projections of 1000 B starting at 0, 1000, 2000 and 3000.
        let offsets = [0usize, 1000, 2000, 3000];
        // The arena and its pad fit: everything is staged.
        assert_eq!(stage_cut(offsets, 4000, 4000 + pad), 4000);
        assert_eq!(stage_cut(offsets, 4000, 10_000), 4000);
        // One byte short: the last projection reads in place, and the cut
        // is the start of the last projection, never mid-projection.
        assert_eq!(stage_cut(offsets, 4000, 4000 + pad - 1), 3000);
        assert_eq!(stage_cut(offsets, 4000, 3999 + pad - 1), 3000);
        assert_eq!(stage_cut(offsets, 4000, 2999 + pad), 2000);
        assert_eq!(stage_cut(offsets, 4000, 3000 + pad), 3000);
        // Nothing fits (or an unordered offset list): the leading boundary or 0.
        assert_eq!(stage_cut(offsets, 4000, pad), 0);
        assert_eq!(stage_cut(offsets, 4000, 1), 0);
        assert_eq!(
            stage_cut([3000usize, 0, 2000, 1000], 4000, 2500 + pad),
            2000
        );
        // The copy always fits: cut + pad never exceeds the window.
        for window in [pad, 1500 + pad, 3999 + pad] {
            assert!(stage_cut(offsets, 4000, window) + pad <= window);
        }
    }

    #[test]
    fn rebase_keeps_entries_at_or_past_the_cut_on_the_host_arena() {
        let (arena, stage) = (0x7000_0000usize, 0x2_0000_0000usize);
        let entries = [arena, arena + 1000, arena + 2000, arena + 3000];
        let table = rebase_stage_entries(&entries, arena, 4000, 2000, stage, None).unwrap();
        let decoded: Vec<usize> = table
            .chunks_exact(8)
            .map(|chunk| usize::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        // Offsets below the cut rebase; the cut itself and later ones stay.
        assert_eq!(
            decoded,
            vec![stage, stage + 1000, arena + 2000, arena + 3000]
        );
        // A cut of zero stages nothing.
        let table = rebase_stage_entries(&entries, arena, 4000, 0, stage, None).unwrap();
        assert_eq!(
            table,
            entries
                .iter()
                .flat_map(|e| e.to_ne_bytes())
                .collect::<Vec<u8>>()
        );
        // An entry outside the arena is refused whatever the cut.
        assert!(rebase_stage_entries(&[arena + 4000], arena, 4000, 0, stage, None).is_err());
    }

    #[test]
    fn mq2_xts_fixture_is_the_artifacts_layout() {
        use mq2_xts_layout as layout;
        // Counts and sizes read from the artifact's tensor index.
        let codecs: String = (0..48).map(layout::codecs).collect();
        assert_eq!(codecs.matches('4').count(), 6306);
        assert_eq!(codecs.matches('e').count(), 18270);
        assert_eq!(
            (0..48).map(layout::arena_bytes).sum::<usize>(),
            40_319_539_200
        );
        assert_eq!(layout::arena_bytes(38), 881_280_000);
        assert_eq!(layout::arena_bytes(2), 846_028_800);
        assert_eq!(layout::arena_bytes(1), 856_473_600);
        assert_eq!(layout::arena_bytes(35), 853_862_400);
        // Provenance is recorded in the fixture header.
        for needle in [
            "47038936600 bytes",
            "d3d536c040f834699b5a261a5122b67d076461a9493eca16669803757928a949",
            "b1642d2bcdb6c1b299dc66b8be55796745dba341f3b26221c5c4e93c0fe8ea70",
        ] {
            assert!(layout::FIXTURE.contains(needle), "{needle}");
        }
    }

    #[test]
    fn layer_cut_never_splits_an_expert_between_stage_and_host() {
        let arena = 0x7000_0000usize;
        // Experts at 0, 1000, (pruned), 3000; each is gate/up then down.
        let gate_up = [arena, arena + 1000, 0, arena + 3000];
        let down = [arena + 500, arena + 1500, 0, arena + 3500];
        let pad = ARENA_TAIL_PAD_BYTES;
        // Room for 2500 B: the cut is expert 1's first byte (1000), not the
        // down projection of expert 0 (500) or of expert 1 (1500).
        assert_eq!(
            layer_stage_cut(&gate_up, &down, arena, 4000, 2500 + pad),
            1000
        );
        assert_eq!(layer_stage_cut(&gate_up, &down, arena, 4000, 999 + pad), 0);
        assert_eq!(
            layer_stage_cut(&gate_up, &down, arena, 4000, 3000 + pad),
            3000
        );
        assert_eq!(
            layer_stage_cut(&gate_up, &down, arena, 4000, 4000 + pad),
            4000
        );
        // Counts: non-null entries below the cut are staged, the rest in place.
        assert_eq!(stage_split_counts([&gate_up, &down], arena, 1000), (2, 4));
        assert_eq!(stage_split_counts([&gate_up, &down], arena, 4000), (6, 0));
    }

    #[test]
    fn real_mq2_xts_layers_stage_whole_experts_and_tail_pointers_stay_on_the_host_arena() {
        use mq2_xts_layout as layout;
        let pad = ARENA_TAIL_PAD_BYTES;
        let first_host = 7;
        // The shipped placement: layers 0..7 resident, 7..48 host.
        let need =
            borrowed_stage_need((first_host..48).map(|layer| (layer, layout::arena_bytes(layer))))
                .unwrap();
        let resident: Vec<Option<usize>> = (0..first_host)
            .map(|layer| Some(layout::arena_bytes(layer) + pad))
            .collect();
        let (donors, window) = plan_slab_donors(need, &resident).unwrap();
        assert_eq!(donors, [2, 1]);
        assert_eq!(window, [846_028_800 + pad, 853_862_400 + pad]);

        let (arena_base, stage_base) = (0x7f00_0000_0000usize, 0x2_0000_0000usize);
        let mut overflow = Vec::new();
        for layer in first_host..48 {
            let (experts, arena) = layout::experts(layer);
            let (gate_up, down) = layout::tables(&experts, arena_base);
            let window = window[layer % 2];
            let cut = layer_stage_cut(&gate_up, &down, arena_base, arena, window);
            // The copy (cut bytes + the following host pad) fits the window.
            assert!(cut + pad <= window, "layer {layer}");
            let table = |entries: &[usize]| {
                decode(
                    &rebase_stage_entries(entries, arena_base, arena, cut, stage_base, None)
                        .unwrap(),
                )
            };
            let (staged_gate_up, staged_down) = (table(&gate_up), table(&down));
            let (mut staged, mut host) = (0usize, 0usize);
            for (e, &[gate_up_at, gate_up_len, down_at, down_len]) in experts.iter().enumerate() {
                let end = down_at + down_len;
                assert_eq!(down_at, gate_up_at + gate_up_len);
                if end <= cut {
                    // Wholly staged: both projections rebased onto the stage.
                    assert_eq!(staged_gate_up[e], stage_base + gate_up_at);
                    assert_eq!(staged_down[e], stage_base + down_at);
                    staged += 1;
                } else {
                    // Wholly in place: the expert starts at or past the cut,
                    // and both entries are the original host-arena addresses.
                    assert!(
                        gate_up_at >= cut,
                        "layer {layer} expert {e} straddles the cut"
                    );
                    assert_eq!(staged_gate_up[e], arena_base + gate_up_at);
                    assert_eq!(staged_down[e], arena_base + down_at);
                    assert!(end <= arena);
                    host += 1;
                }
            }
            assert_eq!(staged + host, 512);
            let counts = stage_split_counts([&gate_up, &down], arena_base, cut);
            assert_eq!(counts, (2 * staged, 2 * host));
            if host > 0 {
                overflow.push((layer, arena - cut, host));
            }
        }
        // Only the seven even layers larger than layer 2 overflow; the first
        // host expert of each starts exactly at the cut.
        assert_eq!(
            overflow,
            [
                (10, 2_611_200, 1),
                (12, 3_916_800, 2),
                (18, 10_444_800, 6),
                (22, 27_417_600, 15),
                (26, 3_916_800, 2),
                (38, 35_251_200, 19),
                (42, 1_305_600, 1),
            ]
        );
        assert_eq!(
            overflow.iter().map(|&(_, bytes, _)| bytes).sum::<usize>(),
            84_864_000
        );

        // Worst layer (38, 881,280,000 B): cut at expert 493, 19 experts
        // (38 table entries) read in place from the host arena.
        let (experts, arena) = layout::experts(38);
        let (gate_up, down) = layout::tables(&experts, arena_base);
        let cut = layer_stage_cut(&gate_up, &down, arena_base, arena, window[0]);
        assert_eq!(cut, 846_028_800);
        assert_eq!(experts[493][0], cut);
        let rebased = decode(
            &rebase_stage_entries(&gate_up, arena_base, arena, cut, stage_base, None).unwrap(),
        );
        assert_eq!(rebased[492], stage_base + experts[492][0]);
        assert_eq!(rebased[493], arena_base + cut);
        assert_eq!(rebased[511], arena_base + experts[511][0]);
        assert_eq!(
            stage_split_counts([&gate_up, &down], arena_base, cut),
            (986, 38)
        );
        assert_eq!(
            stage_layer_report(38, donors[0], arena, cut, (986, 38)),
            "  qwen4 slab stage layer 38 (donor 2): staged 846028800 B of 881280000 B arena (986 pointer-table entries rebased), host tail 35251200 B (38 entries read in place)"
        );

        // Fully fitting layers (34 fits donor 2 exactly, 35 donor 1 exactly):
        // every entry is rebased onto the stage, no host tail.
        for layer in [34usize, 35, 8, 47] {
            let (experts, arena) = layout::experts(layer);
            let (gate_up, down) = layout::tables(&experts, arena_base);
            let cut = layer_stage_cut(&gate_up, &down, arena_base, arena, window[layer % 2]);
            assert_eq!(cut, arena, "layer {layer}");
            for entries in [&gate_up, &down] {
                let rebased = decode(
                    &rebase_stage_entries(entries, arena_base, arena, cut, stage_base, None)
                        .unwrap(),
                );
                assert!(rebased
                    .iter()
                    .zip(entries.iter())
                    .all(|(&at, &entry)| at == stage_base + (entry - arena_base)));
            }
            assert_eq!(
                stage_split_counts([&gate_up, &down], arena_base, cut),
                (1024, 0)
            );
            assert_eq!(
                stage_layer_report(layer, donors[layer % 2], arena, cut, (1024, 0)),
                format!(
                    "  qwen4 slab stage layer {layer} (donor {}): staged {arena} B of {arena} B arena (1024 pointer-table entries rebased), host tail 0 B (0 entries read in place)",
                    donors[layer % 2]
                )
            );
        }
    }

    #[test]
    fn borrowed_after_moe_frees_donors_copies_forward_and_restores_last() {
        let host = |layers: std::ops::Range<usize>| move |layer: usize| layers.contains(&layer);
        // 12 layers, ten resident (donors 8 and 9), host layers 10 and 11.
        let (donors, layers) = ([8, 9], 12);
        let step = |layer, overwritten| {
            borrowed_after_moe(donors, layers, host(10..12), layer, overwritten)
        };
        // A resident non-donor layer does nothing.
        for layer in [0, 3, 7] {
            assert_eq!(
                step(layer, false),
                BorrowedAfterMoe {
                    release: false,
                    next: BorrowedNext::Nothing
                }
            );
        }
        // A donor's MoE frees its arena, then the first host layer of its
        // parity copies over it.
        assert_eq!(
            step(8, false),
            BorrowedAfterMoe {
                release: true,
                next: BorrowedNext::Enqueue(10)
            }
        );
        assert_eq!(
            step(9, false),
            BorrowedAfterMoe {
                release: true,
                next: BorrowedNext::Enqueue(11)
            }
        );
        // The last host layer of a parity (its down recorded `free` itself)
        // restores the donor it overwrote ...
        for layer in [10, 11] {
            assert_eq!(
                step(layer, true),
                BorrowedAfterMoe {
                    release: false,
                    next: BorrowedNext::Restore
                }
            );
            // ... and has nothing to restore when no copy was issued.
            assert_eq!(step(layer, false).next, BorrowedNext::Nothing);
        }
        // More host layers: L+2 refills, only the final one of each parity restores.
        let step =
            |layer, overwritten| borrowed_after_moe(donors, 14, host(10..14), layer, overwritten);
        assert_eq!(step(10, true).next, BorrowedNext::Enqueue(12));
        assert_eq!(step(11, true).next, BorrowedNext::Enqueue(13));
        assert_eq!(step(12, true).next, BorrowedNext::Restore);
        assert_eq!(step(13, true).next, BorrowedNext::Restore);
        // A donor with no later host layer of its parity only frees.
        let lone = borrowed_after_moe(donors, 11, host(10..11), 9, false);
        assert_eq!(
            lone,
            BorrowedAfterMoe {
                release: true,
                next: BorrowedNext::Nothing
            }
        );
    }
}
