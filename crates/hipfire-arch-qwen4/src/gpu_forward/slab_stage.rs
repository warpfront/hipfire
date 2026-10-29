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
//!
//! EP loads with a per-expert unit plan (`HIPFIRE_QWEN4_EP_EXPERT_HOTSET`, D1)
//! have a mixed layer that is neither a whole resident donor nor a whole host
//! layer, and a plan may leave no resident layer at all, so the donor stage
//! cannot apply. They take the same dedicated-buffer stage, built by
//! [`Qwen4ExpertStage::new_units`] ([`crate::expert_ep_units::ep_unit_stage_kind`]):
//! a layer with host experts copies its whole pinned HOST REGION (a mixed
//! layer's single host arena, a host layer's two aggregates) into its
//! parity's stage buffer, and its compact EP pointer tables rebase only the
//! entries that point inside that region; entries into the VRAM arena (the
//! mixed layer's resident experts) and the zero dummy stay as they are. It
//! shares the tier stage's tables, events, prime/after-MoE protocol and
//! stage buffers; there are no mirrors and nothing to restore. The loader
//! charges the buffers (`place_routed_experts_units_staged`).
//!
//! The BORROWED unit stage (`HIPFIRE_EP_UNIT_STAGE_BORROW=1` with the decoupled reserve,
//! [`crate::expert_ep_units::EpUnitPlan::stage_borrow_layers`]) replaces the two dedicated buffers
//! by the device tensors of two whole-resident layers (trunk layers 0 and 1, which precede every
//! staged layer): parity `p`'s stage is the gate/up tensor (buffer 0) and the down tensor
//! (buffer 1) of layer `p`. A staged layer's regions (a mixed layer's host arena split into its
//! gate/up and down blocks, a host layer's or scratch donor's two aggregates) are packed
//! first-fit into those two tensors ([`unit_stage_layout_bufs`]) and the pointer tables rebase per
//! buffer ([`unit_stage_table_bufs`]). The lent layer's own bytes are mirrored in pinned host RAM
//! at load; the S2 borrowed protocol ([`borrowed_after_moe`]) frees a lent layer after its own MoE,
//! copies the first staged layer of its parity over it, refills L+2 and restores it from the
//! mirror after the last staged layer of the parity; `join` waits on the restores and
//! `restore_blocking` recovers any forward that never reached it (every error path already calls
//! it). Decode and every forward below `min_rows` never overwrite anything. A region that fills a
//! lent tensor flush has no zeroed tail pad behind it ([`tail_pad_room`]): exactly the S2 donor
//! stage's own case (its host-layer aggregates fill the donor flush), whose kernel over-read is the
//! phantom group header documented at `HOST_TAIL_PAD_BYTES`.

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StageCopy {
    /// CPU address of the pinned host owner's bytes.
    src: usize,
    /// Byte offset inside the parity's stage buffer `buf`.
    dst: usize,
    len: usize,
    /// Which of the parity's stage buffers: always 0 for a dedicated stage (one buffer per
    /// parity); 0 = the lent gate/up tensor, 1 = the lent down tensor for a borrowed unit stage.
    buf: usize,
}

/// Staged copies of the layers with a host-tier expert (a per-expert tier
/// stage) or host region (a per-expert-unit EP stage).
struct TierStageMeta {
    /// Per trunk layer: its slot in `copies`/`used` and in the stage's
    /// `tables` (two per slot: gate/up, down), `None` for a layer with no
    /// host-tier expert (never staged, reads owner VRAM).
    slots: Vec<Option<usize>>,
    copies: Vec<Vec<StageCopy>>,
    /// Per slot and stage buffer, the end of the staged payload (the zeroed
    /// tail pad follows, as far as the buffer reaches); `0` for a buffer the
    /// slot does not use. A dedicated stage uses buffer 0 only.
    used: Vec<[usize; 2]>,
    /// The D2 tail's visibility to this unit stage (the tail's prefill flag on and the stage
    /// holds mixed layers): per-forward slot-aware tables and copy runs ([`UnitSlotStage`]).
    /// `None` for every other stage, whose tables and copies are the static ones above.
    unit_slots: Option<UnitSlotStage>,
    /// Keeps both model-level owners alive while the stage exists (tier
    /// stage). The unit stage reads the bundle's pinned arenas, which every
    /// forward holds the bundle for; it keeps a unit `()`.
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

/// One pinned host region a unit-stage layer copies whole into the stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct UnitStageRegion {
    /// Device-visible address of the region's first byte (the address the
    /// layer's pointer-table entries use).
    pub(super) base: usize,
    pub(super) len: usize,
}

/// Where each region of one unit-stage layer sits in its parity's buffer.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct UnitStageLayout {
    /// Stage offset of each region.
    pub(super) dst: Vec<usize>,
    /// End of the staged payload; the zeroed tail pad starts here.
    pub(super) used: usize,
}

/// Refuse an empty region or one that overlaps an earlier region of `regions` (`index`).
fn check_unit_region(regions: &[UnitStageRegion], index: usize) -> Result<(), String> {
    let region = &regions[index];
    let end = region
        .base
        .checked_add(region.len)
        .ok_or_else(|| format!("host region {index} overflows the address space"))?;
    if region.len == 0 {
        return Err(format!("host region {index} is empty"));
    }
    if regions[..index]
        .iter()
        .any(|other| region.base < other.base + other.len && other.base < end)
    {
        return Err(format!("host region {index} overlaps an earlier region"));
    }
    Ok(())
}

/// Pack `regions` back to back at 256-byte alignment. Empty or overlapping
/// regions are refused.
pub(super) fn unit_stage_layout(regions: &[UnitStageRegion]) -> Result<UnitStageLayout, String> {
    let mut cursor = 0usize;
    let mut dst = Vec::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        check_unit_region(regions, index)?;
        let at = cursor
            .checked_add(STAGE_ALIGN - 1)
            .ok_or_else(|| "unit stage layout overflows".to_string())?
            & !(STAGE_ALIGN - 1);
        cursor = at
            .checked_add(region.len)
            .ok_or_else(|| "unit stage layout overflows".to_string())?;
        dst.push(at);
    }
    Ok(UnitStageLayout { dst, used: cursor })
}

/// Where each region of one unit-stage layer sits in its parity's TWO stage buffers (a borrowed
/// stage: the lent layer's gate/up tensor and down tensor).
#[derive(Debug, PartialEq, Eq)]
pub(super) struct UnitStageLayoutBufs {
    /// The buffer (0 or 1) of each region.
    pub(super) buf: Vec<usize>,
    /// Offset of each region inside its buffer.
    pub(super) dst: Vec<usize>,
    /// End of the staged payload of each buffer (`0`: the buffer holds no region); the zeroed
    /// tail pad starts here.
    pub(super) used: [usize; 2],
}

/// First-fit packing of `regions` into two buffers of `caps` bytes: each region goes, back to back
/// at 256-byte alignment, to buffer 0 if it still fits there, else to buffer 1, else the layer
/// cannot be staged. Empty or overlapping regions are refused.
pub(super) fn unit_stage_layout_bufs(
    regions: &[UnitStageRegion],
    caps: [usize; 2],
) -> Result<UnitStageLayoutBufs, String> {
    let mut used = [0usize; 2];
    let mut buf = Vec::with_capacity(regions.len());
    let mut dst = Vec::with_capacity(regions.len());
    for (index, region) in regions.iter().enumerate() {
        check_unit_region(regions, index)?;
        let placed = (0..2).find_map(|candidate| {
            let at = used[candidate].checked_add(STAGE_ALIGN - 1)? & !(STAGE_ALIGN - 1);
            let end = at.checked_add(region.len)?;
            (end <= caps[candidate]).then_some((candidate, at, end))
        });
        let Some((candidate, at, end)) = placed else {
            return Err(format!(
                "host region {index} of {} B fits neither stage buffer ({} B and {} B, {} B and {} B used)",
                region.len, caps[0], caps[1], used[0], used[1]
            ));
        };
        used[candidate] = end;
        buf.push(candidate);
        dst.push(at);
    }
    Ok(UnitStageLayoutBufs { buf, dst, used })
}

/// The two regions of a mixed layer's host arena (`n` host experts of `gate_up` and `down` bytes,
/// gate/up block then down block from `base`): staged as two regions so each lands in its own kind
/// of lent tensor whatever `n` is (a single arena of up to a whole layer fits neither lent tensor).
pub(super) fn split_mixed_host_arena(
    base: usize,
    n: usize,
    gate_up: usize,
    down: usize,
) -> [UnitStageRegion; 2] {
    [
        UnitStageRegion {
            base,
            len: n * gate_up,
        },
        UnitStageRegion {
            base: base + n * gate_up,
            len: n * down,
        },
    ]
}

/// A unit-stage layer's little-endian device pointer table and the number of
/// entries rebased: every entry inside region `i` moves to
/// `stage_base + dst[i] + (entry - region.base)`, every other entry (the VRAM
/// arena, the zero dummy) is unchanged.
#[cfg(test)]
pub(super) fn unit_stage_table(
    entries: &[usize],
    regions: &[UnitStageRegion],
    dst: &[usize],
    stage_base: usize,
) -> Result<(Vec<u8>, usize), String> {
    unit_stage_table_bufs(
        entries,
        regions,
        dst,
        &vec![0; regions.len()],
        [stage_base, 0],
    )
}

/// [`unit_stage_table`] over two stage buffers: an entry inside region `i` moves to
/// `bases[buf[i]] + dst[i] + (entry - region.base)`.
pub(super) fn unit_stage_table_bufs(
    entries: &[usize],
    regions: &[UnitStageRegion],
    dst: &[usize],
    buf: &[usize],
    bases: [usize; 2],
) -> Result<(Vec<u8>, usize), String> {
    if regions.len() != dst.len() || regions.len() != buf.len() {
        return Err(format!(
            "{} host regions but {} stage offsets and {} stage buffers",
            regions.len(),
            dst.len(),
            buf.len()
        ));
    }
    if buf.iter().any(|&b| b > 1) {
        return Err("a stage buffer index is above 1".to_string());
    }
    let mut table = Vec::with_capacity(entries.len() * std::mem::size_of::<u64>());
    let mut rebased = 0usize;
    for (expert, &entry) in entries.iter().enumerate() {
        let value = match regions
            .iter()
            .position(|region| entry >= region.base && entry - region.base < region.len)
        {
            Some(region) => {
                rebased += 1;
                bases[buf[region]]
                    .checked_add(dst[region])
                    .and_then(|at| at.checked_add(entry - regions[region].base))
                    .ok_or_else(|| format!("expert {expert} rebased entry overflows"))?
            }
            None => entry,
        };
        table.extend_from_slice(&value.to_ne_bytes());
    }
    Ok((table, rebased))
}

/// The static sources of one mixed unit-stage layer, as [`Qwen4ExpertStage::new_units`] laid it
/// out: its compact pointer-table entries, its host regions with their CPU addresses and stage
/// placement, and the device addresses of the stage buffers.
struct UnitLayerSources<'a> {
    /// `[gate_up, down]` compact table entries (device-visible addresses).
    entries: [&'a [usize]; 2],
    regions: &'a [UnitStageRegion],
    /// CPU address of each region's first byte (the pinned host arena the copy reads).
    cpu: &'a [usize],
    /// Stage offset and buffer of each region.
    dst: &'a [usize],
    buf: &'a [usize],
    /// Device address of stage buffer 0 / 1 of this layer's parity.
    bases: [usize; 2],
}

/// One mixed layer's stage tables and H2D runs for ONE forward, with the D2 tail's live slots
/// taken into account ([`unit_stage_forward_plan`]).
#[derive(Debug, PartialEq, Eq)]
struct UnitForwardPlan {
    /// Little-endian `[gate_up, down]` pointer tables: a slot-served unit names its slot, every
    /// other entry inside a host region is rebased onto the stage exactly as the static table
    /// does, any other entry is unchanged.
    tables: [Vec<u8>; 2],
    /// The runs to copy H2D: each region's bytes minus the slot-served units' pieces, in region
    /// order and ascending inside a region. With no slot-served unit this is one copy per region,
    /// exactly the static copies.
    copies: Vec<StageCopy>,
    /// Units served from a slot (both pieces).
    served: usize,
    /// Live slots this stage could not honour (the tail's canonical entries are not this
    /// stage's, or a piece is not wholly inside a host region): their units stage as usual.
    rejected: usize,
    /// Slot-live pieces that are copied anyway because they sat in a gap shorter than the merge
    /// gap between two runs ([`unit_stage_forward_plan`]'s `gap_units`): their bytes (a whole
    /// number of pieces) and count. Their table entries still name the slot.
    bridged_bytes: usize,
    bridged_pieces: usize,
}

/// The slot-live pieces of one host region, as `(start, end)` byte ranges inside it, minus the
/// interior gaps the copy bridges (module doc of [`unit_stage_forward_plan`]): the ranges the
/// region's copy runs skip. `holes` is the region's raw piece list (any order). Returns the kept
/// holes, ascending, with touching same-size pieces coalesced, and the `(bytes, pieces)` of the
/// pieces bridged.
///
/// A maximal cluster of touching same-size pieces (one kind: gate/up or down) is bridged, i.e.
/// copied after all, only when it has a copy run on BOTH sides inside the region and is shorter
/// than `gap_units` whole pieces: bridging then merges two runs into one (and never creates a
/// run). `gap_units == 0` keeps every hole. An expert is never split: a bridged cluster is a
/// whole number of whole pieces, and every kept hole is the exact set of its live pieces.
fn bridge_holes(
    holes: &mut [(usize, usize)],
    region_len: usize,
    gap_units: usize,
) -> Result<(Vec<(usize, usize)>, (usize, usize)), String> {
    holes.sort_unstable();
    // Maximal clusters: `(start, end, piece)`, `piece` = the size of one piece in the cluster.
    let mut clusters: Vec<(usize, usize, usize)> = Vec::new();
    let mut cursor = 0usize;
    for &(start, end) in holes.iter() {
        if start < cursor {
            return Err(format!(
                "slot-served pieces overlap in a host region at offset {start}"
            ));
        }
        cursor = end;
        let piece = end - start;
        match clusters.last_mut() {
            Some(last) if last.1 == start && last.2 == piece => last.1 = end,
            _ => clusters.push((start, end, piece)),
        }
    }
    let mut kept: Vec<(usize, usize)> = Vec::with_capacity(clusters.len());
    let (mut bridged_bytes, mut bridged_pieces) = (0usize, 0usize);
    // End of the last KEPT hole: everything between it and a cluster is copied.
    let mut copied_from = 0usize;
    for (i, &(start, end, piece)) in clusters.iter().enumerate() {
        let run_before = start > copied_from;
        let run_after = match clusters.get(i + 1) {
            Some(next) => next.0 > end,
            None => end < region_len,
        };
        let short = piece > 0 && (end - start) / piece < gap_units;
        if gap_units > 0 && run_before && run_after && short {
            bridged_bytes += end - start;
            bridged_pieces += (end - start) / piece;
            continue;
        }
        kept.push((start, end));
        copied_from = end;
    }
    Ok((kept, (bridged_bytes, bridged_pieces)))
}

/// The stage tables and copy runs of one mixed unit-stage layer when the tail's `live` slots hold
/// byte copies of some of its host units.
///
/// A live unit names its slot (VRAM) in BOTH tables and its two pieces (`piece_bytes`: gate/up,
/// down) are cut out of the region copies; every other unit stages as in the static layout (same
/// stage offsets, so a skipped piece leaves its stage bytes untouched and unread). The copy runs
/// are the complement of the skipped pieces inside each region: a unit is cut out whole or not at
/// all, so no expert is split between a run and a slot. A live slot is honoured only if the
/// tail's canonical entries equal this stage's entries and both pieces lie wholly inside a host
/// region; otherwise the unit stages as usual (counted in `rejected`). `live` empty gives exactly
/// the static tables and copies.
///
/// `gap_units` (`HIPFIRE_EP_UNIT_TAIL_STAGE_GAP`): a long prefill with a large tail cuts a copy
/// run per scattered live unit; a maximal cluster of touching live pieces shorter than
/// `gap_units` pieces with a copy run on both sides is copied after all ([`bridge_holes`]), so
/// the two runs merge. Those units stay slot-named in the tables (the copy writes stage bytes
/// nothing reads) and are counted in `bridged_bytes` / `bridged_pieces`; `served` is unchanged.
/// `gap_units == 0` is exactly the complement above.
fn unit_stage_forward_plan(
    src: &UnitLayerSources<'_>,
    piece_bytes: [usize; 2],
    live: &[crate::expert_unit_tail_rank::LiveSlot],
    gap_units: usize,
) -> Result<UnitForwardPlan, String> {
    if src.regions.len() != src.cpu.len() {
        return Err(format!(
            "{} host regions but {} CPU addresses",
            src.regions.len(),
            src.cpu.len()
        ));
    }
    let experts = src.entries[0].len();
    if src.entries[1].len() != experts {
        return Err("the gate/up and down tables differ in length".to_string());
    }
    let mut tables = [Vec::new(), Vec::new()];
    for kind in 0..2 {
        tables[kind] =
            unit_stage_table_bufs(src.entries[kind], src.regions, src.dst, src.buf, src.bases)?.0;
    }
    // `(start, end)` of the skipped pieces, per region.
    let mut holes: Vec<Vec<(usize, usize)>> = vec![Vec::new(); src.regions.len()];
    let place = |kind: usize, entry: usize| -> Option<(usize, usize)> {
        let region = src
            .regions
            .iter()
            .position(|region| entry >= region.base && entry - region.base < region.len)?;
        let start = entry - src.regions[region].base;
        (start + piece_bytes[kind] <= src.regions[region].len).then_some((region, start))
    };
    let mut seen = vec![false; experts];
    let (mut served, mut rejected) = (0usize, 0usize);
    for unit in live {
        let expert = unit.expert;
        if expert >= experts {
            return Err(format!("live slot names expert {expert} of {experts}"));
        }
        if std::mem::replace(&mut seen[expert], true) {
            return Err(format!("expert {expert} is live in two slots"));
        }
        let canon = [src.entries[0][expert], src.entries[1][expert]];
        let places = [place(0, canon[0]), place(1, canon[1])];
        match places {
            [Some(gate_up), Some(down)]
                if unit.canon == canon && unit.slot.iter().all(|&slot| slot != 0) =>
            {
                for (kind, (region, start)) in [gate_up, down].into_iter().enumerate() {
                    holes[region].push((start, start + piece_bytes[kind]));
                    tables[kind][expert * 8..expert * 8 + 8]
                        .copy_from_slice(&unit.slot[kind].to_ne_bytes());
                }
                served += 1;
            }
            _ => rejected += 1,
        }
    }
    let mut copies = Vec::with_capacity(src.regions.len() + served);
    let (mut bridged_bytes, mut bridged_pieces) = (0usize, 0usize);
    for (index, region) in src.regions.iter().enumerate() {
        let run = |from: usize, to: usize| StageCopy {
            src: src.cpu[index] + from,
            dst: src.dst[index] + from,
            len: to - from,
            buf: src.buf[index],
        };
        let (kept, (bytes, pieces)) = bridge_holes(&mut holes[index], region.len, gap_units)
            .map_err(|reason| format!("host region {index}: {reason}"))?;
        bridged_bytes += bytes;
        bridged_pieces += pieces;
        let mut cursor = 0usize;
        for (start, end) in kept {
            if start > cursor {
                copies.push(run(cursor, start));
            }
            cursor = end;
        }
        if cursor < region.len {
            copies.push(run(cursor, region.len));
        }
    }
    Ok(UnitForwardPlan {
        tables,
        copies,
        served,
        rejected,
        bridged_bytes,
        bridged_pieces,
    })
}

/// One mixed layer the tail may serve, with the static sources [`unit_stage_forward_plan`] needs.
struct UnitSlotLayer {
    /// The layer's slot in the stage's `copies` / `tables`.
    slot: usize,
    /// Trunk layer index (the tail's layer label).
    layer: usize,
    entries: [Vec<usize>; 2],
    regions: Vec<UnitStageRegion>,
    cpu: Vec<usize>,
    dst: Vec<usize>,
    buf: Vec<usize>,
    bases: [usize; 2],
}

impl UnitSlotLayer {
    fn sources(&self) -> UnitLayerSources<'_> {
        UnitLayerSources {
            entries: [&self.entries[0], &self.entries[1]],
            regions: &self.regions,
            cpu: &self.cpu,
            dst: &self.dst,
            buf: &self.buf,
            bases: self.bases,
        }
    }
}

/// Per-forward mutable state of [`UnitSlotStage`]; one rank thread drives it (the stage is not
/// `Sync`).
struct UnitSlotState {
    /// Per stage slot: this forward's copy runs, `None` = the static copies.
    copies: Vec<Option<Vec<StageCopy>>>,
    /// Per `UnitSlotStage::layers` index: the `[gate_up, down]` table bytes the device holds
    /// (initially the static tables), which are also the host bytes an enqueued upload reads.
    tables: Vec<[Vec<u8>; 2]>,
    /// A table upload from `tables` was enqueued on the compute stream and the stream has not
    /// been drained since: the host bytes may not change yet.
    upload_pending: bool,
    forwards: u64,
}

/// The D2 tail's pool made visible to the unit stage (the tail's prefill flag; see
/// [`crate::expert_unit_tail::EP_UNIT_TAIL_PREFILL_ENV`]).
///
/// At the start of every staged forward ([`Qwen4ExpertStage::slab_prime`]) the stage asks the
/// rank's tail which owned units of each mixed layer a slot serves as of the last retirement
/// ([`crate::expert_unit_tail_rank::UnitTailRank::live_slots`]) and plans the layer
/// ([`unit_stage_forward_plan`]): the layer's two pointer tables name the slots of those units
/// and the per-layer H2D copy skips their bytes. Until [`Qwen4GpuForward::link_unit_tail_stage`]
/// runs (and always with the tail or its prefill flag off) the stage is byte-for-byte the static
/// one: this value is not even built.
///
/// # Why a named slot cannot be overwritten during the forward
///
/// * A slot is overwritten only by a fill that the tail issues in `step_end` (a decode window's
///   retirement, on the rank thread, between forwards) on its copy stream, AFTER an event
///   recorded on the COMPUTE stream behind that retirement's table upload (the window protocol
///   of `expert_unit_tail_rank`). The compute stream is the stream this stage's table uploads
///   and the forward's kernels run on, so every kernel of a forward that was enqueued before
///   that retirement completes before the fill's first byte lands.
/// * A staged forward runs entirely on the rank thread between two retirements: no retirement
///   (and so no fill, eviction or new publication) happens between the snapshot at its start and
///   the end of its enqueueing, and a retirement after it enqueues its event behind all of the
///   forward's kernels. The snapshot can only name slots whose fill had COMPLETED on the host
///   (published at a retirement), so the bytes are whole when the stage tables are uploaded.
/// * This stage never writes a slot: its own copies read the pinned host arena and write the
///   stage buffers, and slots are only read by the forward's kernels.
struct UnitSlotStage {
    layers: Vec<UnitSlotLayer>,
    /// `[gate_up, down]` bytes of one unit.
    piece_bytes: [usize; 2],
    /// Copy-run merge gap in whole units (`HIPFIRE_EP_UNIT_TAIL_STAGE_GAP`; [`bridge_holes`]).
    gap_units: usize,
    /// The rank's tail and whether to print the per-forward counter line; set once.
    link: std::sync::OnceLock<(
        std::sync::Arc<crate::expert_unit_tail_rank::UnitTailRank>,
        bool,
    )>,
    state: std::cell::RefCell<UnitSlotState>,
}

impl UnitSlotStage {
    /// `stage_slots`: the stage's slot count (length of its `copies`).
    fn new(
        layers: Vec<UnitSlotLayer>,
        stage_slots: usize,
        piece_bytes: [usize; 2],
    ) -> Result<Self, String> {
        // The static tables: what `new_units` uploaded (no live slot).
        let tables = layers
            .iter()
            .map(|layer| {
                let plan = unit_stage_forward_plan(&layer.sources(), piece_bytes, &[], 0)?;
                Ok(plan.tables)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let gap_units = crate::expert_unit_tail::ep_unit_tail_stage_gap()?;
        Ok(Self {
            layers,
            piece_bytes,
            gap_units,
            link: std::sync::OnceLock::new(),
            state: std::cell::RefCell::new(UnitSlotState {
                copies: vec![None; stage_slots],
                tables,
                upload_pending: false,
                forwards: 0,
            }),
        })
    }
}

/// Bytes of zeroed tail pad a stage buffer of `size` bytes can hold behind a payload ending at
/// `used`: the kernels' over-read pad ([`ARENA_TAIL_PAD_BYTES`]) as far as the buffer reaches. A
/// dedicated buffer is sized `used + pad` (the whole pad); a lent tensor that the payload fills
/// flush has none (`0`), the case the S2 donor stage has always had.
pub(super) fn tail_pad_room(size: usize, used: usize) -> usize {
    size.saturating_sub(used).min(ARENA_TAIL_PAD_BYTES)
}

/// The stage slot of every trunk layer of `rank` under `plan`: layers with a
/// host region and the loan-mirror `donors` (whole-Resident layers of a
/// decoupled EP reserve, read from their pinned mirror while their VRAM is
/// scratch) take consecutive slots, `layers` entries long. `donors` is
/// ascending and empty without an EP reserve.
pub(super) fn unit_stage_slots(
    plan: &crate::expert_ep_units::EpUnitPlan,
    rank: usize,
    layers: usize,
    donors: &[usize],
) -> Vec<Option<usize>> {
    let mut next = 0usize;
    (0..layers)
        .map(|layer| {
            (layer < plan.geometry().layers
                && (plan.stage_host_region_bytes(rank, layer) > 0
                    || donors.binary_search(&layer).is_ok()))
            .then(|| {
                next += 1;
                next - 1
            })
        })
        .collect()
}

/// Payload capacity `C` of each unit-stage buffer: the plan's largest host
/// region, and with borrowed `donors` at least one whole layer region (a donor
/// layer's compact gate/up plus down of the largest rank), exactly the charge
/// `place_routed_experts_units_forced_staged` took.
pub(super) fn unit_stage_capacity(
    plan: &crate::expert_ep_units::EpUnitPlan,
    donors: &[usize],
) -> u64 {
    let geometry = plan.geometry();
    let donor_floor = if donors.is_empty() {
        0
    } else {
        (0..geometry.ranks)
            .map(|rank| geometry.owned_per_layer(rank) as u64)
            .max()
            .unwrap_or(0)
            .saturating_mul(geometry.unit_bytes())
    };
    plan.max_stage_host_region_bytes().max(donor_floor)
}

/// The two regions (loan mirror gate/up, down) a borrowed donor `layer` stages
/// from, and the CPU address of each region's pinned bytes: the region base is
/// the mirror's device-visible address (the address the borrowed runtime's
/// pointer-table entries carry), the bytes are the pinned host allocation the
/// copy reads.
pub(super) fn donor_stage_regions(
    sources: &ep_reserve_forward::EpStageHostSources,
    layer: usize,
) -> Result<(Vec<UnitStageRegion>, Vec<usize>), String> {
    let [(gate, gate_base), (down, down_base)] = sources
        .pair_with_device(layer)
        .ok_or_else(|| format!("layer {layer} has no loan mirror"))?;
    if gate_base == 0 || down_base == 0 {
        return Err(format!("layer {layer}: a loan mirror has no device alias"));
    }
    Ok((
        vec![
            UnitStageRegion {
                base: gate_base,
                len: gate.len(),
            },
            UnitStageRegion {
                base: down_base,
                len: down.len(),
            },
        ],
        vec![gate.as_ptr() as usize, down.as_ptr() as usize],
    ))
}

/// The next layer after `layer` of the same parity with a staged slot.
fn next_staged_layer(slots: &[Option<usize>], layer: usize) -> Option<usize> {
    (layer + 2..slots.len())
        .step_by(2)
        .find(|&next| slots[next].is_some())
}

/// Whether every one of `layers` has a stage slot.
fn slots_serve(slots: &[Option<usize>], layers: &[usize]) -> bool {
    layers
        .iter()
        .all(|&layer| slots.get(layer).copied().flatten().is_some())
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
            // already admitted at load for DECODE (`slab_decodes`); a decode-only codec (no K2:
            // W3 on gfx1100) has no Stage capability, so this keeps a measured link or an
            // explicit `1` from staging a codec the arch has no K1+K2 for.
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
                layer_copies.push(StageCopy {
                    src: cpu,
                    dst,
                    len,
                    buf: 0,
                });
            }
            copies.push(layer_copies);
        }
        let used = layouts
            .iter()
            .map(|(_, layout)| [layout.used, 0])
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
                    unit_slots: None,
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

    /// Per-expert-unit EP stage (D1), or `None` (printed) when it cannot
    /// stage. Every layer of this rank with host experts takes a slot; its
    /// compact decode-table entries that point inside its pinned host region
    /// are rebased onto the parity's stage buffer, all others (the
    /// mixed layer's VRAM arena, the zero dummy) keep their address. The dedicated
    /// stage overwrites nothing and restores nothing.
    ///
    /// A plan with a BORROWED stage ([`crate::expert_ep_units::EpUnitPlan::stage_borrow_layers`],
    /// `HIPFIRE_EP_UNIT_STAGE_BORROW=1`) has no dedicated buffers: parity `p`'s stage is the gate/up
    /// tensor (buffer 0) and the down tensor (buffer 1) of the whole-resident layer `lent[p]`, which
    /// precedes every layer it stages. A layer's host regions (a mixed layer's host arena split into
    /// its gate/up and down blocks, a host layer's or scratch donor's two aggregates) are packed
    /// first-fit into the two tensors ([`unit_stage_layout_bufs`]); the pointer tables rebase per
    /// buffer. The lent layer's own bytes are mirrored in pinned host RAM at load and copied back by
    /// the S2 borrowed-stage protocol ([`borrowed_after_moe`], `slab_join`, `slab_restore_blocking`).
    ///
    /// `host_sources` is the decoupled EP reserve's loan resolver: its layers
    /// are the whole-Resident scratch donors, whose VRAM is scratch during a
    /// borrowed epoch. `moe` is then the borrowed view (`moe_borrow[0]`), whose
    /// donor runtimes name the loan mirrors; each donor layer takes a slot of its
    /// own whose two regions are the mirror gate/up and down (device-visible base,
    /// pinned bytes), staged exactly like a host layer's aggregates.
    pub(super) fn new_units(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        moe: &[Option<Qwen4MoeLayerRuntime>],
        chunk_rows: usize,
        host_sources: Option<&ep_reserve_forward::EpStageHostSources>,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        let off = |reason: &str| {
            eprintln!("  qwen4 long-prefill expert staging: off ({reason})");
            hipfire_config::load_summary::note(
                hipfire_config::load_summary::Facet::Prefill,
                format!("long-prefill expert staging off ({reason})"),
            );
            Ok(None)
        };
        let (Some(plan), Some(placement)) =
            (bundle.weights.ep_unit_plan(), bundle.weights.ep_placement())
        else {
            return off("a unit stage needs an EP rank with a per-expert plan");
        };
        if !rdna_compute::expert_capabilities::stage_eligible(
            &gpu.arch,
            [rdna_compute::expert_capabilities::QT_MQ4_XTS],
        ) {
            return off(&format!("no MQ4 stage capability on {}", gpu.arch));
        }
        let config = &bundle.config;
        let rank = placement.rank;
        let donors: &[usize] = host_sources.map_or(&[], |sources| sources.layers());
        let borrow = plan.stage_borrow_layers();
        let geometry = plan.geometry();
        let owned = placement.local_count(config.num_experts);
        // The two tensors (compact gate/up, compact down) one lent layer holds: the two stage
        // buffers of its parity.
        let lent_bytes = [
            owned * geometry.gate_up_bytes as usize,
            owned * geometry.down_bytes as usize,
        ];
        if let Some(lent) = borrow {
            plan.check_stage_borrow()
                .map_err(|reason| invalid(format!("unit stage: {reason}")))?;
            if plan.stage_borrow_buffer_bytes(rank) != lent_bytes.map(|bytes| bytes as u64) {
                return Err(invalid(format!(
                    "unit stage: the lent tensors of rank {rank} hold {lent_bytes:?} B, the plan's geometry gives {:?} B",
                    plan.stage_borrow_buffer_bytes(rank)
                )));
            }
            if let Some(layer) = lent.iter().find(|layer| donors.contains(layer)) {
                return Err(invalid(format!(
                    "unit stage: lent stage layer {layer} is also a scratch donor"
                )));
            }
            for layer in lent {
                if !moe
                    .get(layer)
                    .and_then(Option::as_ref)
                    .is_some_and(|runtime| !runtime.host_mapped())
                {
                    return Err(invalid(format!(
                        "unit stage: lent stage layer {layer} has no device-resident MoE runtime"
                    )));
                }
            }
        }
        let slots = unit_stage_slots(plan, rank, moe.len(), donors);
        // The layers whose host region is a mixed unit layer's host arena (the units the D2 tail
        // may hold in a slot), as opposed to a host layer's aggregates or a scratch donor.
        let mut mixed_layers: Vec<usize> = Vec::new();
        // (layer, regions, CPU addresses of the regions, layout, expected rebased entries)
        let mut layouts: Vec<(
            usize,
            Vec<UnitStageRegion>,
            Vec<usize>,
            UnitStageLayoutBufs,
            usize,
        )> = Vec::new();
        for (layer, runtime) in moe.iter().enumerate() {
            let planned = slots[layer].is_some();
            let Some(runtime) = runtime.as_ref() else {
                if planned {
                    return Err(invalid(format!(
                        "unit stage: layer {layer} holds host experts but has no MoE runtime"
                    )));
                }
                continue;
            };
            // The forward's stage selection and `before_moe` wait read the
            // runtime's layer bit; it must be exactly the plan's host layers
            // (and, on the borrowed view, the donor layers).
            if runtime.host_mapped() != planned {
                return Err(invalid(format!(
                    "unit stage: layer {layer} runtime host_mapped {} disagrees with the plan ({planned})",
                    runtime.host_mapped()
                )));
            }
            if !planned {
                continue;
            }
            if borrow.is_some_and(|lent| layer <= lent[layer % 2]) {
                return Err(invalid(format!(
                    "unit stage: staged layer {layer} does not follow the lent layer of its parity"
                )));
            }
            let mut regions = Vec::with_capacity(2);
            let mut cpu = Vec::with_capacity(2);
            let donor = donors.binary_search(&layer).is_ok();
            let expected = match (donor, bundle.weights.ep_unit_layer(layer)?) {
                (true, Some(_)) => {
                    return Err(invalid(format!(
                        "unit stage: scratch donor layer {layer} is a mixed unit layer"
                    )));
                }
                (true, None) => {
                    let sources = host_sources.expect("donor layers come from the host sources");
                    if plan.layer_state(rank, layer)
                        != crate::expert_ep_units::EpLayerState::Resident
                    {
                        return Err(invalid(format!(
                            "unit stage: scratch donor layer {layer} is not whole-resident on rank {rank}"
                        )));
                    }
                    (regions, cpu) = donor_stage_regions(sources, layer)
                        .map_err(|reason| invalid(format!("unit stage: {reason}")))?;
                    owned
                }
                (false, Some(unit)) => {
                    mixed_layers.push(layer);
                    let bytes = unit.map.arena_entry_bytes(UnitTier::Host);
                    let Some(host) = gpu.host_bytes(unit.host) else {
                        return off("a mixed layer's host arena has no pinned host bytes");
                    };
                    if host.len() < bytes
                        || bytes as u64 != plan.stage_host_region_bytes(rank, layer)
                    {
                        return Err(invalid(format!(
                            "unit stage: layer {layer} host arena holds {} B, the plan stages {} B",
                            host.len(),
                            plan.stage_host_region_bytes(rank, layer)
                        )));
                    }
                    let base = unit.host.buf.as_ptr() as usize;
                    let count = unit.map.count(UnitTier::Host);
                    if borrow.is_some() {
                        // Two regions (gate/up block, down block): each lands in its own kind
                        // of lent tensor whatever the host count is.
                        let split = split_mixed_host_arena(
                            base,
                            count,
                            geometry.gate_up_bytes as usize,
                            geometry.down_bytes as usize,
                        );
                        if split[0].len + split[1].len != bytes {
                            return Err(invalid(format!(
                                "unit stage: layer {layer} host arena of {bytes} B is not {count} experts of gate/up plus down"
                            )));
                        }
                        cpu.push(host.as_ptr() as usize);
                        cpu.push(host.as_ptr() as usize + split[0].len);
                        regions.extend(split);
                    } else {
                        regions.push(UnitStageRegion { base, len: bytes });
                        cpu.push(host.as_ptr() as usize);
                    }
                    count
                }
                (false, None) => {
                    let weights = bundle
                        .weights
                        .layer(layer)
                        .ok_or_else(|| invalid("unit stage layer is not owned"))?
                        .ffn
                        .moe()
                        .ok_or_else(|| invalid("unit stage layer has no MoE FFN"))?;
                    for source in [&weights.experts_gate_up, &weights.experts_down] {
                        let owner = bundle.weights.resident(source)?;
                        let Some(host) = gpu.host_bytes(owner) else {
                            return off("a host expert owner has no pinned host bytes");
                        };
                        regions.push(UnitStageRegion {
                            base: owner.buf.as_ptr() as usize,
                            len: host.len(),
                        });
                        cpu.push(host.as_ptr() as usize);
                    }
                    let total = regions.iter().map(|region| region.len as u64).sum::<u64>();
                    if total != plan.stage_host_region_bytes(rank, layer) {
                        return Err(invalid(format!(
                            "unit stage: layer {layer} host aggregates hold {total} B, the plan stages {} B",
                            plan.stage_host_region_bytes(rank, layer)
                        )));
                    }
                    owned
                }
            };
            let layout = match borrow {
                None => unit_stage_layout(&regions).map(|single| UnitStageLayoutBufs {
                    buf: vec![0; regions.len()],
                    dst: single.dst,
                    used: [single.used, 0],
                }),
                Some(_) => unit_stage_layout_bufs(&regions, lent_bytes),
            }
            .map_err(|reason| invalid(format!("unit stage: layer {layer}: {reason}")))?;
            // Every owned host expert of both tables must lie inside the region
            // (validated here with a null stage base, before anything is allocated).
            for (name, entries) in [
                ("gate/up", &runtime.expert_gate_up_entries),
                ("down", &runtime.expert_down_entries),
            ] {
                if entries.len() != config.num_experts {
                    return Err(invalid(format!(
                        "unit stage: layer {layer} {name} table length differs from num_experts"
                    )));
                }
                let (_, rebased) =
                    unit_stage_table_bufs(entries, &regions, &layout.dst, &layout.buf, [0, 0])
                        .map_err(|reason| {
                            invalid(format!("unit stage: layer {layer}: {reason}"))
                        })?;
                if rebased != expected {
                    return Err(invalid(format!(
                        "unit stage: layer {layer} {name} table has {rebased} entries in its host region, the plan has {expected} host experts"
                    )));
                }
            }
            layouts.push((layer, regions, cpu, layout, expected));
        }
        let Some(first_host) = layouts.first().map(|(layer, ..)| *layer) else {
            return off("this rank has no host expert layer or borrowed donor layer");
        };
        // Dedicated stage: one buffer per layer parity, the largest staged payload of that
        // parity plus the zeroed tail pad; the loader charged the largest region of any layer
        // for each. A borrowed stage has no buffer of its own (the lent tensors), and
        // `unit_stage_layout_bufs` already fitted every layer into them.
        let mut sizes = [ARENA_TAIL_PAD_BYTES; 2];
        if borrow.is_none() {
            let capacity = unit_stage_capacity(plan, donors) as usize;
            for (layer, _, _, layout, _) in &layouts {
                if layout.used[0] > capacity {
                    return Err(invalid(format!(
                        "unit stage: layer {layer} stages {} B, above the {capacity} B the placement charged",
                        layout.used[0]
                    )));
                }
                let need = layout.used[0]
                    .checked_add(ARENA_TAIL_PAD_BYTES)
                    .ok_or_else(|| invalid("unit stage buffer size overflows"))?;
                sizes[layer % 2] = sizes[layer % 2].max(need);
            }
        }
        let mut copies = Vec::with_capacity(layouts.len());
        let mut used = Vec::with_capacity(layouts.len());
        for (_, regions, cpu, layout, _) in &layouts {
            copies.push(
                regions
                    .iter()
                    .zip(cpu)
                    .zip(layout.dst.iter().zip(&layout.buf))
                    .map(|((region, &src), (&dst, &buf))| StageCopy {
                        src,
                        dst,
                        len: region.len,
                        buf,
                    })
                    .collect::<Vec<_>>(),
            );
            used.push(layout.used);
        }
        // The two tensors (gate/up, down) of a lent layer.
        let lent_sources = |layer: usize| -> Result<[&TensorRef; 2], Qwen4GpuForwardError> {
            let weights = bundle
                .weights
                .layer(layer)
                .ok_or_else(|| invalid("lent stage layer is not owned"))?
                .ffn
                .moe()
                .ok_or_else(|| invalid("lent stage layer has no MoE FFN"))?;
            Ok([&weights.experts_gate_up, &weights.experts_down])
        };
        // A borrowed stage's pinned restore copies of the two lent layers sit on top of the
        // placement's pinned experts: refused here as the S2 stage refuses its own (the loader's
        // plan-time aggregate check already counted them).
        if let Some(lent) = borrow {
            let layer_bytes = (0..placement.ranks)
                .map(|rank| plan.stage_borrow_buffer_bytes(rank).iter().sum::<u64>())
                .max()
                .unwrap_or(0) as usize;
            let mirror_bytes =
                crate::expert_residency::expert_stage_mirror_bytes(layer_bytes, placement.ranks)
                    .map_err(invalid)?;
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
                return off(&format!(
                    "restore copies of the lent stage layers: {reason}"
                ));
            }
            for layer in lent {
                for (kind, source) in lent_sources(layer)?.into_iter().enumerate() {
                    let owner = bundle.weights.resident(source)?;
                    if owner.buf.is_host_mapped() || owner.buf.size() < lent_bytes[kind] {
                        return off("a lent stage layer's expert owner is not device memory of the stage size");
                    }
                }
            }
        }
        gpu.ensure_capture_stream()?;
        // Staging is an optimisation: buffers that do not fit VRAM leave every
        // host expert read in place.
        let mut buffers = Vec::with_capacity(2);
        if borrow.is_none() {
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
        }
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
            donors: borrow.unwrap_or([usize::MAX; 2]),
            donor_views: Vec::new(),
            mirrors: buffers,
            tables: Vec::with_capacity(2 * layouts.len()),
            compact_tables: Vec::new(),
            compact_keys: Vec::new(),
            full_keys: Vec::new(),
            compact_bound: Vec::new(),
            first_host,
            // EP marker only (compact-route gate): a unit stage rebases its own
            // slab tables and never redirects through sealed donor slots.
            ep_slots: Some(Vec::new()),
            bytes: if borrow.is_some() { lent_bytes } else { [0, 0] },
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
                    unit_slots: None,
                    _keepalive: std::sync::Arc::new(()),
                }),
                borrowed: borrow.is_some(),
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
            // The lent layers' own bytes, read back once into pinned restore copies (the S2
            // stage's mirrors); the views are never freed here, the bundle's weights own them.
            // Parity-major: `[even gate/up, even down, odd gate/up, odd down]`.
            for layer in borrow.into_iter().flatten() {
                for (kind, source) in lent_sources(layer)?.into_iter().enumerate() {
                    let owner = bundle.weights.resident(source)?;
                    let view = GpuTensor {
                        buf: owner.buf.byte_view(0, lent_bytes[kind]),
                        shape: vec![lent_bytes[kind]],
                        dtype: DType::Raw,
                    };
                    let mut host = vec![0u8; lent_bytes[kind]];
                    gpu.memcpy_dtoh_auto(&mut host, &view.buf)?;
                    stage
                        .mirrors
                        .push(gpu.upload_raw_host_mapped(&host, &[lent_bytes[kind]])?);
                    drop(host);
                    stage.donor_views.push(view);
                }
            }
            for (layer, regions, _, layout, _) in &layouts {
                let runtime = moe[*layer].as_ref().expect("checked above");
                let base = |buf: usize| {
                    if borrow.is_some() {
                        stage.donor_views[2 * (layer % 2) + buf].buf.as_ptr() as usize
                    } else {
                        stage.mirrors[layer % 2].buf.as_ptr() as usize
                    }
                };
                let bases = [base(0), if borrow.is_some() { base(1) } else { 0 }];
                for entries in [
                    &runtime.expert_gate_up_entries,
                    &runtime.expert_down_entries,
                ] {
                    let (table, _) =
                        unit_stage_table_bufs(entries, regions, &layout.dst, &layout.buf, bases)
                            .map_err(|reason| {
                                invalid(format!("unit stage: layer {layer} table: {reason}"))
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
        // The D2 tail's prefill visibility (its prefill flag on): keep each mixed layer's static
        // sources so a staged forward can name live tail slots ([`UnitSlotStage`]). Built only
        // then; every other load keeps exactly the static stage.
        if crate::expert_unit_tail::ep_unit_tail_prefill_requested() {
            let mut slot_layers = Vec::new();
            for (layer, regions, cpu, layout, _) in &layouts {
                if !mixed_layers.contains(layer) {
                    continue;
                }
                let runtime = moe[*layer].as_ref().expect("checked above");
                let base = |buf: usize| {
                    if borrow.is_some() {
                        stage.donor_views[2 * (layer % 2) + buf].buf.as_ptr() as usize
                    } else {
                        stage.mirrors[layer % 2].buf.as_ptr() as usize
                    }
                };
                let slot = stage
                    .slab
                    .as_ref()
                    .and_then(|meta| meta.tier.as_ref())
                    .and_then(|tier| tier.slots[*layer])
                    .expect("a mixed unit layer has a stage slot");
                slot_layers.push(UnitSlotLayer {
                    slot,
                    layer: *layer,
                    entries: [
                        runtime.expert_gate_up_entries.clone(),
                        runtime.expert_down_entries.clone(),
                    ],
                    regions: regions.clone(),
                    cpu: cpu.clone(),
                    dst: layout.dst.clone(),
                    buf: layout.buf.clone(),
                    bases: [base(0), if borrow.is_some() { base(1) } else { 0 }],
                });
            }
            if !slot_layers.is_empty() {
                match UnitSlotStage::new(
                    slot_layers,
                    layouts.len(),
                    [
                        geometry.gate_up_bytes as usize,
                        geometry.down_bytes as usize,
                    ],
                ) {
                    Ok(unit_slots) => {
                        stage
                            .slab
                            .as_mut()
                            .and_then(|meta| meta.tier.as_mut())
                            .expect("a unit stage has tier meta")
                            .unit_slots = Some(unit_slots);
                    }
                    Err(reason) => {
                        let _ = stage.free_gpu(gpu);
                        return Err(invalid(format!("unit stage: tail slot plan: {reason}")));
                    }
                }
            }
        }
        let host_experts = layouts
            .iter()
            .filter(|(layer, ..)| donors.binary_search(layer).is_err())
            .map(|(.., expected)| *expected)
            .sum::<usize>();
        let donor_note = if donors.is_empty() {
            String::new()
        } else {
            format!(
                "; {} borrowed scratch-donor layers {donors:?} read from their pinned loan mirrors",
                donors.len()
            )
        };
        if let Some(lent) = borrow {
            // The lent tensors end exactly at their payload: a region that fills one has no zeroed
            // tail pad behind it (the S2 donor stage's own flush case). The allocations' real
            // slack is only visible here.
            let slack = lent
                .iter()
                .flat_map(|&layer| {
                    lent_sources(layer)
                        .ok()
                        .into_iter()
                        .flatten()
                        .enumerate()
                        .filter_map(|(kind, source)| {
                            bundle
                                .weights
                                .resident(source)
                                .ok()
                                .map(|owner| owner.buf.size().saturating_sub(lent_bytes[kind]))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            eprintln!(
                "  qwen4 long-prefill expert staging: forwards of >= {} rows stage the {host_experts} host experts of {} EP unit layers through the tensors of layers {} and {} lent as the two parity stage buffers (gate/up {} MiB + down {} MiB each; restored from {} MiB pinned copies after the last staged layer of each parity; lent allocation slack past the payload {slack:?} B; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows}){donor_note}",
                stage.min_rows,
                layouts.len(),
                lent[0],
                lent[1],
                lent_bytes[0] >> 20,
                lent_bytes[1] >> 20,
                (2 * (lent_bytes[0] + lent_bytes[1])) >> 20,
                if stage.capture { "stage" } else { "mapped" },
                stage.min_rows_source.label()
            );
        } else {
            eprintln!(
                "  qwen4 long-prefill expert staging: forwards of >= {} rows stage the {host_experts} host experts of {} EP unit layers through two VRAM stage buffers of {} and {} MiB (resident experts keep their VRAM arena address; nothing overwritten, nothing to restore; wide-hidden capture forwards {}; min_rows from {}, prefill chunk {chunk_rows}){donor_note}",
                stage.min_rows,
                layouts.len(),
                sizes[0] >> 20,
                sizes[1] >> 20,
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

    /// The unit stage of a decoupled EP reserve whose scratch donors are explicit
    /// ([`crate::ep_reserve::EpReservePlan::explicit_donors`]): the ordinary stage
    /// admission, then [`Self::new_units`] over the borrowed view `view`
    /// (`moe_borrow[0]`) with the loan's host sources. `Ok(None)` (printed) when the
    /// stage is not admitted on this rank; the reserve then refuses the load.
    pub(super) fn new_reserve_units(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        view: &[Option<Qwen4MoeLayerRuntime>],
        chunk_rows: usize,
        host_sources: &ep_reserve_forward::EpStageHostSources,
    ) -> Result<Option<Self>, Qwen4GpuForwardError> {
        if !Self::admitted(gpu, bundle)? || !crate::expert_ep_units::ep_unit_stage_on(&gpu.arch) {
            return Ok(None);
        }
        Self::new_units(gpu, bundle, view, chunk_rows, Some(host_sources))
    }

    /// Whether this unit stage reads `layers` from its own slots (every layer a
    /// decoupled reserve borrows must be one). Always `false` for a non-unit stage.
    pub(super) fn unit_stage_serves(&self, layers: &[usize]) -> bool {
        self.slab
            .as_ref()
            .and_then(|meta| meta.tier.as_ref())
            .is_some_and(|tier| slots_serve(&tier.slots, layers))
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

    /// Stage buffer `buf` (0: gate/up side, 1: down side) of `parity` of a unit stage: a
    /// dedicated stage has the one buffer `mirrors[parity]` (`buf` is 0); a borrowed unit stage
    /// lends the lent layer's gate/up and down tensors (`donor_views[2 * parity + buf]`).
    fn unit_buffer(&self, parity: usize, buf: usize) -> &GpuTensor {
        if self.slab.as_ref().is_some_and(|meta| meta.borrowed) {
            &self.donor_views[2 * parity + buf]
        } else {
            debug_assert_eq!(buf, 0, "a dedicated stage has one buffer per parity");
            &self.mirrors[parity]
        }
    }

    /// Lent tensors per parity: two (gate/up, down) for a borrowed unit stage, one arena for a
    /// borrowed slab stage. Parity `p`'s tensors are `donor_views[per * p..per * (p + 1)]` and
    /// their restore copies `mirrors[per * p..per * (p + 1)]`.
    fn lent_per_parity(&self) -> usize {
        if self.slab.as_ref().is_some_and(|meta| meta.tier.is_some()) {
            2
        } else {
            1
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

    /// Start of a staged forward under the D2 tail ([`UnitSlotStage`]): ask the rank's tail
    /// which owned units of each mixed layer a slot serves as of the last retirement, upload the
    /// layers' slot-naming pointer tables when they changed and plan the per-layer H2D runs that
    /// skip those units ([`unit_stage_forward_plan`]). A no-op until the tail is linked, so with
    /// the tail or its prefill flag off the stage keeps its static tables and copies.
    ///
    /// The table uploads are host-asynchronous on the compute stream, ahead of every kernel of
    /// this forward and behind every kernel of the previous one, so no running kernel sees a
    /// table change. Their host bytes live in the stage state and are rewritten only after the
    /// compute stream was drained, which an earlier upload makes necessary at most once per
    /// forward that changed a table (a prefill's later chunks find the tables unchanged).
    fn unit_slots_refresh(
        &self,
        gpu: &Gpu,
        tier: &TierStageMeta,
        layers: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let Some(slots) = tier.unit_slots.as_ref() else {
            return Ok(());
        };
        let Some((tail, log)) = slots.link.get() else {
            return Ok(());
        };
        let mut state = slots.state.borrow_mut();
        let state = &mut *state;
        let mut plans = Vec::with_capacity(slots.layers.len());
        let (mut served, mut rejected, mut served_layers) = (0usize, 0usize, 0usize);
        let (mut bridged_bytes, mut bridged_pieces) = (0usize, 0usize);
        for (index, layer) in slots.layers.iter().enumerate() {
            if layer.layer >= layers {
                continue;
            }
            let live = tail.live_slots(layer.layer);
            let plan = unit_stage_forward_plan(
                &layer.sources(),
                slots.piece_bytes,
                &live,
                slots.gap_units,
            )
            .map_err(|reason| {
                invalid(format!(
                    "unit stage: layer {} slot-aware plan: {reason}",
                    layer.layer
                ))
            })?;
            served += plan.served;
            rejected += plan.rejected;
            bridged_bytes += plan.bridged_bytes;
            bridged_pieces += plan.bridged_pieces;
            served_layers += usize::from(plan.served > 0);
            plans.push((index, plan));
        }
        let changed = plans
            .iter()
            .any(|(index, plan)| state.tables[*index] != plan.tables);
        let compute = if changed {
            let compute = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("G2 compute stream missing"))?;
            if state.upload_pending {
                gpu.hip.stream_synchronize(compute)?;
                state.upload_pending = false;
            }
            Some(compute)
        } else {
            None
        };
        for (index, plan) in plans {
            let layer = &slots.layers[index];
            let UnitForwardPlan {
                tables,
                copies,
                served: layer_served,
                ..
            } = plan;
            if state.tables[index] != tables {
                let compute = compute.expect("a changed table was planned with the compute stream");
                state.tables[index] = tables;
                for kind in 0..2 {
                    let bytes = &state.tables[index][kind];
                    let target = &self.tables[2 * layer.slot + kind];
                    if let Err(error) = gpu.hip.memcpy_htod_async(
                        &target.buf.byte_view(0, bytes.len()),
                        bytes,
                        compute,
                    ) {
                        // The device table is unknown from here: drain what was enqueued, then
                        // force the next forward to upload this layer again.
                        let _ = gpu.hip.stream_synchronize(compute);
                        state.upload_pending = false;
                        state.tables[index] = [Vec::new(), Vec::new()];
                        return Err(error.into());
                    }
                }
                state.upload_pending = true;
            }
            state.copies[layer.slot] = (layer_served > 0).then_some(copies);
        }
        state.forwards += 1;
        if *log {
            let (mut tail_off, mut staged, mut tail_off_runs, mut runs) = (0usize, 0usize, 0, 0);
            for slot in tier.slots.iter().take(layers).flatten() {
                let base = &tier.copies[*slot];
                let used = state.copies[*slot].as_deref().unwrap_or(base);
                tail_off += base.iter().map(|copy| copy.len).sum::<usize>();
                staged += used.iter().map(|copy| copy.len).sum::<usize>();
                tail_off_runs += base.len();
                runs += used.len();
            }
            let mib = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
            eprintln!(
                "[unit-tail] rank {} stage forward {}: staged {:.1} MiB H2D in {runs} copy runs \
                 (tail-off stage: {:.1} MiB in {tail_off_runs}), {served} units served from tail \
                 slots over {served_layers} layers ({:.1} MiB not copied, {rejected} live slots \
                 refused); merge gap {} units: {bridged_pieces} slot-live pieces ({:.1} MiB) \
                 copied anyway inside the staged bytes",
                self.placement.map_or(0, |placement| placement.rank),
                state.forwards,
                mib(staged),
                mib(tail_off),
                mib(tail_off - staged),
                slots.gap_units,
                mib(bridged_bytes),
            );
        }
        Ok(())
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
        // Slot-aware tables and copy runs first (a borrowed stage returns below, but its tables
        // still name this forward's slots).
        if let Some(tier) = meta.tier.as_ref() {
            self.unit_slots_refresh(gpu, tier, layers)?;
        }
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
            if meta.borrowed {
                // From the first byte issued, the lent layer no longer holds its experts.
                self.set_state(parity, DonorState::Overwritten);
            }
            // The static copies, or this forward's runs minus the units the D2 tail serves from
            // a live slot ([`UnitSlotStage`]; planned by `unit_slots_refresh` in `slab_prime`).
            let slot_state = tier.unit_slots.as_ref().map(|slots| slots.state.borrow());
            let copies: &[StageCopy] = slot_state
                .as_ref()
                .and_then(|state| state.copies[slot].as_deref())
                .unwrap_or(&tier.copies[slot]);
            for copy in copies {
                let buffer = &self.unit_buffer(parity, copy.buf).buf;
                // SAFETY: `copy.src` lies inside a pinned host owner (the lease's
                // for a tier stage, the bundle's host arena or aggregate for a
                // unit stage; checked against its range at construction), which
                // outlives the stage and nothing writes after materialization.
                let source = unsafe { std::slice::from_raw_parts(copy.src as *const u8, copy.len) };
                gpu.hip.memcpy_htod_async(
                    &buffer.byte_view(copy.dst, copy.len),
                    source,
                    &self.copy_stream,
                )?;
            }
            // The previous occupant may have been larger: re-zero the pad the
            // kernels' tail over-read can reach past the last staged byte of each
            // buffer the slot uses, as far as the buffer reaches (a lent tensor
            // that the payload fills flush has no pad behind it: the S2 donor
            // stage's own case).
            for (buf, &used) in tier.used[slot].iter().enumerate() {
                if used == 0 {
                    continue;
                }
                let buffer = &self.unit_buffer(parity, buf).buf;
                let room = tail_pad_room(buffer.size(), used);
                if room > 0 {
                    gpu.hip.memset_async(
                        &buffer.byte_view(used, room),
                        0,
                        room,
                        &self.copy_stream,
                    )?;
                }
            }
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
            if self.slab.as_ref().is_some_and(|meta| meta.borrowed) {
                // A borrowed unit stage: the S2 protocol over the lent layers `donors`; a
                // staged layer (a slot) refills its parity, a lent layer also frees it.
                return self.borrowed_after_moe_step(
                    gpu,
                    layer,
                    borrowed_after_moe(
                        self.donors,
                        moe.len(),
                        |candidate| tier.slots.get(candidate).copied().flatten().is_some(),
                        layer,
                        self.state.get()[layer % 2] == DonorState::Overwritten,
                    ),
                );
            }
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
        gpu.hip
            .stream_wait_event(&self.copy_stream, &self.free[parity])?;
        let per = self.lent_per_parity();
        for at in per * parity..per * (parity + 1) {
            let host = gpu
                .host_bytes(&self.mirrors[at])
                .ok_or_else(|| invalid("G2 slab donor restore copy missing"))?;
            gpu.hip
                .memcpy_htod_async(&self.donor_views[at].buf, host, &self.copy_stream)?;
        }
        gpu.hip
            .event_record(&self.ready[parity], Some(&self.copy_stream))?;
        self.set_state(parity, DonorState::Restoring);
        Ok(())
    }

    /// What a borrowed stage does after layer `layer`'s MoE (`step`, from
    /// [`borrowed_after_moe`]): a lent layer's own MoE frees its tensors, then the next staged
    /// layer of its parity copies over them, or past the last one the lent layer is restored.
    fn borrowed_after_moe_step(
        &self,
        gpu: &Gpu,
        layer: usize,
        step: BorrowedAfterMoe,
    ) -> Result<(), Qwen4GpuForwardError> {
        let parity = layer % 2;
        if step.release {
            // The lent layer's own MoE read its tensors on the compute stream.
            let compute = gpu
                .active_stream
                .as_ref()
                .ok_or_else(|| invalid("G2 compute stream missing"))?;
            gpu.hip.event_record(&self.free[parity], Some(compute))?;
        }
        match step.next {
            BorrowedNext::Nothing => Ok(()),
            BorrowedNext::Enqueue(next) => self.slab_enqueue(gpu, next),
            BorrowedNext::Restore => self.slab_enqueue_restore(gpu, parity),
        }
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
            let per = self.lent_per_parity();
            for at in per * parity..per * (parity + 1) {
                let host = gpu
                    .host_bytes(&self.mirrors[at])
                    .ok_or_else(|| invalid("G2 slab donor restore copy missing"))?;
                gpu.hip.memcpy_htod(&self.donor_views[at].buf, host)?;
            }
            gpu.hip.device_synchronize()?;
            self.set_state(parity, DonorState::Resident);
        }
        Ok(())
    }
}

impl Qwen4GpuForward {
    /// Make the rank's D2 `tail` visible to this forward's unit stage ([`UnitSlotStage`]): from
    /// the next staged forward on, a mixed layer's pointer tables name the slot of each owned
    /// unit the tail serves, and the per-layer H2D copy skips that unit's bytes. `log` prints a
    /// counter line per staged forward (bytes staged H2D against units served from slots).
    /// `Ok(false)`: this forward has no slot-capable unit stage (no stage admitted, or the
    /// tail's prefill flag was off when it was built), so there is nothing to link.
    pub(crate) fn link_unit_tail_stage(
        &self,
        tail: std::sync::Arc<crate::expert_unit_tail_rank::UnitTailRank>,
        log: bool,
    ) -> Result<bool, String> {
        let Some(stage) = self.expert_stage.as_ref() else {
            return Ok(false);
        };
        let Some(slots) = stage
            .slab
            .as_ref()
            .and_then(|meta| meta.tier.as_ref())
            .and_then(|tier| tier.unit_slots.as_ref())
        else {
            return Ok(false);
        };
        slots
            .link
            .set((tail, log))
            .map_err(|_| "the unit stage is already linked to a D2 tail".to_string())?;
        eprintln!(
            "qwen4 EP rank {}: D2 tail prefill stage on: {} mixed staged layers name live tail \
             slots in their stage tables and skip those units' H2D bytes (copy runs merge across \
             slot-live gaps shorter than {} units)",
            stage.placement.map_or(0, |placement| placement.rank),
            slots.layers.len(),
            slots.gap_units
        );
        Ok(true)
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

    // ---- D1 unit stage (`ep_unit_stage_*`) ----

    /// Flash-Next-sized geometry: gate/up 1,740,800 B and down 870,400 B per
    /// expert, small expert count for the pure tests.
    const UNIT_GU: usize = 1_740_800;
    const UNIT_DN: usize = 870_400;

    #[test]
    fn ep_unit_stage_mixed_layer_rebases_only_host_entries_by_one_delta() {
        // Rank 0 of 2 owns even experts of a 12-expert layer: experts 0,2,4 in
        // the VRAM arena, 6,8,10 in the host arena, odd experts are the dummy.
        let (vram, host, dummy) = (0x7000_0000usize, 0x9000_0000usize, 0x5000_0000usize);
        let n_host = 3;
        let host_len = n_host * (UNIT_GU + UNIT_DN);
        let gate_up: Vec<usize> = (0..12)
            .map(|e| match e {
                e if e % 2 == 1 => dummy,
                0 | 2 | 4 => vram + (e / 2) * UNIT_GU,
                e => host + (e / 2 - 3) * UNIT_GU,
            })
            .collect();
        let down: Vec<usize> = (0..12)
            .map(|e| match e {
                e if e % 2 == 1 => dummy,
                0 | 2 | 4 => vram + 3 * UNIT_GU + (e / 2) * UNIT_DN,
                e => host + n_host * UNIT_GU + (e / 2 - 3) * UNIT_DN,
            })
            .collect();
        let regions = [UnitStageRegion {
            base: host,
            len: host_len,
        }];
        let layout = unit_stage_layout(&regions).unwrap();
        assert_eq!(layout.dst, vec![0]);
        assert_eq!(layout.used, host_len);
        let stage = 0x2_0000_0000usize;
        for entries in [&gate_up, &down] {
            let (table, rebased) = unit_stage_table(entries, &regions, &layout.dst, stage).unwrap();
            assert_eq!(rebased, n_host);
            let out = decode(&table);
            for (e, (&before, &after)) in entries.iter().zip(&out).enumerate() {
                if (6..12).contains(&e) && e % 2 == 0 {
                    // Host experts moved by one delta: stage_base - host base.
                    assert_eq!(after, before - host + stage, "expert {e}");
                } else {
                    // Resident (VRAM arena) and dummy entries are untouched.
                    assert_eq!(after, before, "expert {e}");
                }
            }
        }
    }

    #[test]
    fn ep_unit_stage_host_layer_rebases_two_aggregate_sources() {
        // A wholly host layer: two pinned aggregates, one region each, packed
        // back to back at 256-byte alignment (gate/up block, then down).
        let (gate, down_base, dummy) = (0x9000_0000usize, 0xA000_0000usize, 0x5000_0000usize);
        let owned = 4;
        let regions = [
            UnitStageRegion {
                base: gate,
                len: owned * UNIT_GU,
            },
            UnitStageRegion {
                base: down_base,
                len: owned * UNIT_DN,
            },
        ];
        let layout = unit_stage_layout(&regions).unwrap();
        assert_eq!(layout.dst, vec![0, owned * UNIT_GU]);
        assert_eq!(layout.used, owned * (UNIT_GU + UNIT_DN));
        // Odd regions pad up to the next 256 boundary.
        let odd = unit_stage_layout(&[
            UnitStageRegion {
                base: 0x1000,
                len: 100,
            },
            UnitStageRegion {
                base: 0x2000,
                len: 100,
            },
        ])
        .unwrap();
        assert_eq!(odd.dst, vec![0, 256]);
        assert_eq!(odd.used, 356);
        let gate_up: Vec<usize> = (0..8)
            .map(|e| {
                if e % 2 == 0 {
                    gate + (e / 2) * UNIT_GU
                } else {
                    dummy
                }
            })
            .collect();
        let down: Vec<usize> = (0..8)
            .map(|e| {
                if e % 2 == 0 {
                    down_base + (e / 2) * UNIT_DN
                } else {
                    dummy
                }
            })
            .collect();
        let stage = 0x2_0000_0000usize;
        let (table, rebased) = unit_stage_table(&gate_up, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, owned);
        let out = decode(&table);
        for e in 0..8 {
            if e % 2 == 0 {
                assert_eq!(out[e], stage + (e / 2) * UNIT_GU);
            } else {
                assert_eq!(out[e], dummy);
            }
        }
        let (table, rebased) = unit_stage_table(&down, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, owned);
        let out = decode(&table);
        for e in (0..8).step_by(2) {
            // The down aggregate sits after the gate/up block in the stage.
            assert_eq!(out[e], stage + owned * UNIT_GU + (e / 2) * UNIT_DN);
        }
    }

    // ---- D2 tail slots visible to the unit stage (`unit_stage_forward_plan`) ----

    use crate::expert_unit_tail_rank::LiveSlot;

    const SLOT_VRAM: usize = 0x7000_0000;
    const SLOT_HOST: usize = 0x9000_0000;
    const SLOT_DUMMY: usize = 0x5000_0000;
    const SLOT_CPU: usize = 0x7f00_0000_0000;
    const SLOT_POOL_GU: usize = 0x4_0000_0000;
    const SLOT_POOL_DN: usize = 0x5_0000_0000;

    /// A mixed layer of rank 0 of 2 (owns the even experts): experts 0, 2, 4 in the VRAM arena,
    /// the next `n_host` even experts in the host arena (gate/up block, then down block), odd
    /// experts the dummy. `split`: the host arena is two regions in two stage buffers (a
    /// borrowed stage), else one region in one buffer.
    struct MixedFixture {
        gate_up: Vec<usize>,
        down: Vec<usize>,
        regions: Vec<UnitStageRegion>,
        cpu: Vec<usize>,
        dst: Vec<usize>,
        buf: Vec<usize>,
        bases: [usize; 2],
        n_host: usize,
    }

    impl MixedFixture {
        fn new(n_host: usize, split: bool) -> Self {
            let experts = 2 * (3 + n_host);
            let gate_up = (0..experts)
                .map(|e| match e {
                    e if e % 2 == 1 => SLOT_DUMMY,
                    e if e / 2 < 3 => SLOT_VRAM + (e / 2) * UNIT_GU,
                    e => SLOT_HOST + (e / 2 - 3) * UNIT_GU,
                })
                .collect();
            let down = (0..experts)
                .map(|e| match e {
                    e if e % 2 == 1 => SLOT_DUMMY,
                    e if e / 2 < 3 => SLOT_VRAM + 3 * UNIT_GU + (e / 2) * UNIT_DN,
                    e => SLOT_HOST + n_host * UNIT_GU + (e / 2 - 3) * UNIT_DN,
                })
                .collect();
            let (regions, cpu, dst, buf) = if split {
                let regions = split_mixed_host_arena(SLOT_HOST, n_host, UNIT_GU, UNIT_DN);
                let layout =
                    unit_stage_layout_bufs(&regions, [n_host * UNIT_GU, n_host * UNIT_DN]).unwrap();
                (
                    regions.to_vec(),
                    vec![SLOT_CPU, SLOT_CPU + n_host * UNIT_GU],
                    layout.dst,
                    layout.buf,
                )
            } else {
                let regions = vec![UnitStageRegion {
                    base: SLOT_HOST,
                    len: n_host * (UNIT_GU + UNIT_DN),
                }];
                let layout = unit_stage_layout(&regions).unwrap();
                (regions, vec![SLOT_CPU], layout.dst, vec![0])
            };
            Self {
                gate_up,
                down,
                regions,
                cpu,
                dst,
                buf,
                bases: [0x2_0000_0000, 0x3_0000_0000],
                n_host,
            }
        }

        fn sources(&self) -> UnitLayerSources<'_> {
            UnitLayerSources {
                entries: [&self.gate_up, &self.down],
                regions: &self.regions,
                cpu: &self.cpu,
                dst: &self.dst,
                buf: &self.buf,
                bases: self.bases,
            }
        }

        /// The `index`th host expert (`e = 2 * (3 + index)`) live in pool slot `slot`.
        fn live(&self, index: usize, slot: usize) -> LiveSlot {
            let expert = 2 * (3 + index);
            LiveSlot {
                expert,
                canon: [self.gate_up[expert], self.down[expert]],
                slot: [SLOT_POOL_GU + slot * UNIT_GU, SLOT_POOL_DN + slot * UNIT_DN],
            }
        }

        /// The static tables `new_units` uploads (no slot): the flag-off result.
        fn static_tables(&self) -> [Vec<u8>; 2] {
            [&self.gate_up, &self.down].map(|entries| {
                unit_stage_table_bufs(entries, &self.regions, &self.dst, &self.buf, self.bases)
                    .unwrap()
                    .0
            })
        }

        /// The static copies `new_units` builds: one whole copy per region.
        fn static_copies(&self) -> Vec<StageCopy> {
            self.regions
                .iter()
                .enumerate()
                .map(|(i, region)| StageCopy {
                    src: self.cpu[i],
                    dst: self.dst[i],
                    len: region.len,
                    buf: self.buf[i],
                })
                .collect()
        }
    }

    const SLOT_PIECES: [usize; 2] = [UNIT_GU, UNIT_DN];

    #[test]
    fn slot_plan_without_live_slots_is_the_static_stage() {
        for split in [false, true] {
            let f = MixedFixture::new(4, split);
            let plan = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &[], 0).unwrap();
            assert_eq!(plan.tables, f.static_tables(), "split {split}");
            assert_eq!(plan.copies, f.static_copies(), "split {split}");
            assert_eq!((plan.served, plan.rejected), (0, 0));
        }
    }

    #[test]
    fn slot_plan_names_the_slot_and_cuts_exactly_the_live_units_out_of_the_copy() {
        let f = MixedFixture::new(4, false);
        // Host experts 1 and 3 (e = 8, 12) are live in pool slots 5 and 0.
        let live = [f.live(1, 5), f.live(3, 0)];
        let plan = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 0).unwrap();
        assert_eq!((plan.served, plan.rejected), (2, 0));
        let (static_gu, static_dn) = {
            let [gu, dn] = f.static_tables();
            (decode(&gu), decode(&dn))
        };
        let (gu, dn) = (decode(&plan.tables[0]), decode(&plan.tables[1]));
        for e in 0..f.gate_up.len() {
            match e {
                8 => assert_eq!(
                    (gu[e], dn[e]),
                    (SLOT_POOL_GU + 5 * UNIT_GU, SLOT_POOL_DN + 5 * UNIT_DN)
                ),
                12 => assert_eq!((gu[e], dn[e]), (SLOT_POOL_GU, SLOT_POOL_DN)),
                // Every other entry (VRAM arena, dummy, staged host units) is the static one.
                e => assert_eq!((gu[e], dn[e]), (static_gu[e], static_dn[e]), "expert {e}"),
            }
        }
        // Region = [4 gate/up | 4 down]; the live pieces are gate/up 1 and 3, down 1 and 3.
        let down_block = 4 * UNIT_GU;
        let run = |from: usize, to: usize| StageCopy {
            src: SLOT_CPU + from,
            dst: from,
            len: to - from,
            buf: 0,
        };
        assert_eq!(
            plan.copies,
            [
                run(0, UNIT_GU),
                run(2 * UNIT_GU, 3 * UNIT_GU),
                run(down_block, down_block + UNIT_DN),
                run(down_block + 2 * UNIT_DN, down_block + 3 * UNIT_DN),
            ]
        );
        let bytes: usize = plan.copies.iter().map(|copy| copy.len).sum();
        assert_eq!(bytes, 4 * (UNIT_GU + UNIT_DN) - 2 * (UNIT_GU + UNIT_DN));
        // Never split an expert: each staged (non-live) host unit's two pieces sit whole inside
        // one run, and each live unit's pieces are in none.
        for index in 0..f.n_host {
            let pieces = [
                (index * UNIT_GU, UNIT_GU),
                (down_block + index * UNIT_DN, UNIT_DN),
            ];
            for (from, len) in pieces {
                let covered: usize = plan
                    .copies
                    .iter()
                    .map(|copy| {
                        let (lo, hi) = (copy.dst.max(from), (copy.dst + copy.len).min(from + len));
                        hi.saturating_sub(lo)
                    })
                    .sum();
                let live = index == 1 || index == 3;
                assert_eq!(
                    covered,
                    if live { 0 } else { len },
                    "host unit {index} at {from}"
                );
                if !live {
                    assert!(
                        plan.copies
                            .iter()
                            .any(|copy| copy.dst <= from && from + len <= copy.dst + copy.len),
                        "host unit {index} piece at {from} is split across runs"
                    );
                }
            }
        }
    }

    #[test]
    fn slot_plan_merges_adjacent_live_units_and_drops_an_empty_region() {
        let f = MixedFixture::new(3, true);
        // Borrowed stage, two regions: live host experts 0 and 1 are adjacent in both blocks.
        let live = [f.live(0, 2), f.live(1, 3)];
        let plan = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 0).unwrap();
        let run = |region: usize, from: usize, to: usize| StageCopy {
            src: f.cpu[region] + from,
            dst: f.dst[region] + from,
            len: to - from,
            buf: f.buf[region],
        };
        assert_eq!(
            plan.copies,
            [
                run(0, 2 * UNIT_GU, 3 * UNIT_GU),
                run(1, 2 * UNIT_DN, 3 * UNIT_DN)
            ]
        );
        // Every unit live: nothing left to copy.
        let all: Vec<LiveSlot> = (0..3).map(|i| f.live(i, i)).collect();
        let plan = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &all, 0).unwrap();
        assert_eq!((plan.served, plan.copies.len()), (3, 0));
        // The live units' table entries name their slots, in the gate/up and the down table.
        let (gu, dn) = (decode(&plan.tables[0]), decode(&plan.tables[1]));
        for index in 0..3 {
            let e = 2 * (3 + index);
            assert_eq!(gu[e], SLOT_POOL_GU + index * UNIT_GU);
            assert_eq!(dn[e], SLOT_POOL_DN + index * UNIT_DN);
        }
    }

    #[test]
    fn slot_plan_refuses_slots_it_cannot_honour_and_stages_those_units() {
        let f = MixedFixture::new(3, false);
        let mut moved = f.live(0, 0);
        // The tail's canonical entry is not this stage's.
        moved.canon[0] += 256;
        let mut null_slot = f.live(1, 1);
        null_slot.slot[1] = 0;
        // A "live" unit whose canonical entries are VRAM-arena addresses (not in a host region).
        let core = LiveSlot {
            expert: 2,
            canon: [f.gate_up[2], f.down[2]],
            slot: [SLOT_POOL_GU, SLOT_POOL_DN],
        };
        let plan = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &[moved, null_slot, core], 0)
            .unwrap();
        assert_eq!((plan.served, plan.rejected), (0, 3));
        assert_eq!(plan.tables, f.static_tables());
        assert_eq!(plan.copies, f.static_copies());
    }

    #[test]
    fn slot_plan_refuses_malformed_live_lists() {
        let f = MixedFixture::new(3, false);
        let twice = [f.live(0, 0), f.live(0, 1)];
        assert!(unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &twice, 0).is_err());
        let mut outside = f.live(0, 0);
        outside.expert = f.gate_up.len();
        assert!(unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &[outside], 0).is_err());
        // Two units whose pieces would overlap (a piece longer than the unit stride).
        let overlapping = [f.live(0, 0), f.live(1, 1)];
        assert!(
            unit_stage_forward_plan(&f.sources(), [UNIT_GU + 4096, UNIT_DN], &overlapping, 0)
                .is_err()
        );
    }

    /// Per piece kind, every byte of host unit `index` (`0..n_host`) as `(offset, len)` in a
    /// fixture region: the unsplit fixture is one region `[gate/up block | down block]`.
    fn unit_pieces(f: &MixedFixture, index: usize) -> Vec<(usize, usize, usize)> {
        if f.regions.len() == 1 {
            vec![
                (0, index * UNIT_GU, UNIT_GU),
                (0, f.n_host * UNIT_GU + index * UNIT_DN, UNIT_DN),
            ]
        } else {
            vec![(0, index * UNIT_GU, UNIT_GU), (1, index * UNIT_DN, UNIT_DN)]
        }
    }

    /// Bytes of region `region` in `[from, from + len)` that `copies` cover (stage offsets
    /// are region-relative plus `dst`).
    fn covered(
        f: &MixedFixture,
        copies: &[StageCopy],
        region: usize,
        from: usize,
        len: usize,
    ) -> usize {
        copies
            .iter()
            .filter(|copy| {
                copy.src >= f.cpu[region] && copy.src < f.cpu[region] + f.regions[region].len
            })
            .map(|copy| {
                let start = copy.src - f.cpu[region];
                let (lo, hi) = (start.max(from), (start + copy.len).min(from + len));
                hi.saturating_sub(lo)
            })
            .sum()
    }

    #[test]
    fn bridge_holes_merges_only_short_interior_clusters() {
        let u = 100usize;
        let len = 20 * u;
        // Pieces of one kind at units 0, 3..=4, 8, 12 and 19 (the last touches the region end).
        let mut holes: Vec<(usize, usize)> = [0, 3, 4, 8, 12, 19]
            .iter()
            .map(|&i| (i * u, (i + 1) * u))
            .collect();
        // G = 0: every cluster stays (touching units 3, 4 coalesce).
        let (kept, bridged) = bridge_holes(&mut holes.clone(), len, 0).unwrap();
        assert_eq!(
            kept,
            [
                (0, u),
                (3 * u, 5 * u),
                (8 * u, 9 * u),
                (12 * u, 13 * u),
                (19 * u, 20 * u)
            ]
        );
        assert_eq!(bridged, (0, 0));
        // G = 2: the one-unit interior clusters (8, 12) are bridged; the two-unit cluster and
        // both edge clusters (no run on one side) stay.
        let (kept, bridged) = bridge_holes(&mut holes.clone(), len, 2).unwrap();
        assert_eq!(kept, [(0, u), (3 * u, 5 * u), (19 * u, 20 * u)]);
        assert_eq!(bridged, (2 * u, 2));
        // G = 3: the two-unit cluster is short too.
        let (kept, bridged) = bridge_holes(&mut holes, len, 3).unwrap();
        assert_eq!(kept, [(0, u), (19 * u, 20 * u)]);
        assert_eq!(bridged, (4 * u, 4));
        // A different-size piece touching a cluster is a different kind: never coalesced.
        let mut mixed = vec![(3 * u, 4 * u), (4 * u, 4 * u + 50), (9 * u, 10 * u)];
        let (kept, _) = bridge_holes(&mut mixed, len, 0).unwrap();
        assert_eq!(kept, [(3 * u, 4 * u), (4 * u, 4 * u + 50), (9 * u, 10 * u)]);
        // Overlapping pieces are an error whatever the gap.
        assert!(bridge_holes(&mut [(0, 150), (100, 250)], len, 4).is_err());
    }

    #[test]
    fn stage_gap_zero_is_todays_complement_and_a_gap_only_adds_whole_unreferenced_units() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for split in [false, true] {
            let f = MixedFixture::new(24, split);
            for trial in 0..64 {
                // A random live subset of the 24 host units.
                let density = 1 + trial % 6;
                let live_units: Vec<usize> = (0..f.n_host)
                    .filter(|_| (next() % 8) < density as u64)
                    .collect();
                let live: Vec<LiveSlot> = live_units
                    .iter()
                    .enumerate()
                    .map(|(slot, &i)| f.live(i, slot))
                    .collect();
                let base = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 0).unwrap();
                // G = 0: exactly the complement of the live pieces (today's plan).
                let mut expect = 0usize;
                for index in 0..f.n_host {
                    let is_live = live_units.contains(&index);
                    for (region, from, len) in unit_pieces(&f, index) {
                        assert_eq!(
                            covered(&f, &base.copies, region, from, len),
                            if is_live { 0 } else { len },
                            "split {split} trial {trial} unit {index}"
                        );
                        expect += if is_live { 0 } else { len };
                    }
                }
                let base_bytes: usize = base.copies.iter().map(|copy| copy.len).sum();
                assert_eq!(base_bytes, expect);
                assert_eq!((base.bridged_bytes, base.bridged_pieces), (0, 0));
                let mut previous_runs = base.copies.len();
                for gap in [1usize, 2, 3, 8, 64] {
                    let plan =
                        unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, gap).unwrap();
                    // Slot entries (both tables) and the served count never change.
                    assert_eq!(plan.tables, base.tables, "split {split} gap {gap}");
                    assert_eq!((plan.served, plan.rejected), (base.served, base.rejected));
                    // Every non-slot unit is wholly inside a run; a live unit is copied whole or
                    // not at all (never split), and only the bridged ones are copied.
                    let mut bridged_pieces = 0usize;
                    let mut bridged_bytes = 0usize;
                    for index in 0..f.n_host {
                        let is_live = live_units.contains(&index);
                        for (region, from, len) in unit_pieces(&f, index) {
                            let got = covered(&f, &plan.copies, region, from, len);
                            if is_live {
                                assert!(got == 0 || got == len, "split unit {index} {got}/{len}");
                                if got == len {
                                    bridged_pieces += 1;
                                    bridged_bytes += len;
                                }
                            } else {
                                assert_eq!(got, len, "non-slot unit {index} gap {gap}");
                            }
                        }
                    }
                    assert_eq!(plan.bridged_pieces, bridged_pieces, "gap {gap}");
                    assert_eq!(plan.bridged_bytes, bridged_bytes, "gap {gap}");
                    let bytes: usize = plan.copies.iter().map(|copy| copy.len).sum();
                    assert_eq!(bytes, base_bytes + plan.bridged_bytes);
                    // Bridging only merges runs: the run count never grows with the gap, runs
                    // stay ascending and disjoint inside a region.
                    assert!(plan.copies.len() <= previous_runs, "gap {gap} added runs");
                    previous_runs = plan.copies.len();
                    for pair in plan.copies.windows(2) {
                        if pair[0].buf == pair[1].buf && pair[0].dst < pair[1].dst {
                            assert!(pair[0].dst + pair[0].len <= pair[1].dst);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage_gap_merges_the_scattered_live_units_into_few_runs() {
        // Live units 1, 4, 5, 9 of 12 in one gate/up block and one down block (unsplit region).
        let f = MixedFixture::new(12, false);
        let live: Vec<LiveSlot> = [1usize, 4, 5, 9]
            .iter()
            .enumerate()
            .map(|(slot, &i)| f.live(i, slot))
            .collect();
        let base = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 0).unwrap();
        // gate/up: runs [0], [2,3], [6..=8], [10..=11]; down: [0], [2,3], [6..=8], [10,11]; the
        // last gate/up run touches the first down run (units 10, 11 | down 0): one region run
        // each side of every cut, 4 + 4 minus the join of gate/up's tail and down's head.
        assert_eq!(base.copies.len(), 7);
        // G = 2 bridges the one-unit live holes (1, 9): two fewer runs per kind, one lost join.
        let two = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 2).unwrap();
        assert_eq!(two.bridged_pieces, 4);
        assert_eq!(two.bridged_bytes, 2 * (UNIT_GU + UNIT_DN));
        assert!(two.copies.len() < base.copies.len());
        // G = 3 bridges the two-unit hole (4, 5) as well.
        let three = unit_stage_forward_plan(&f.sources(), SLOT_PIECES, &live, 3).unwrap();
        assert_eq!(three.bridged_pieces, 8);
        let bytes: usize = three.copies.iter().map(|copy| copy.len).sum();
        assert_eq!(bytes, 12 * (UNIT_GU + UNIT_DN));
        assert_eq!(three.copies.len(), 1);
        assert_eq!(three.tables, base.tables);
    }

    #[test]
    fn ep_unit_stage_layout_and_table_refuse_malformed_input() {
        let region = |base, len| UnitStageRegion { base, len };
        assert!(unit_stage_layout(&[region(0x1000, 0)]).is_err());
        assert!(unit_stage_layout(&[region(0x1000, 0x100), region(0x1080, 0x100)]).is_err());
        assert!(unit_stage_layout(&[region(usize::MAX - 4, 16)]).is_err());
        assert!(unit_stage_layout(&[region(0x1000, 0x100), region(0x1100, 0x100)]).is_ok());
        let regions = [region(0x1000, 0x100)];
        assert!(unit_stage_table(&[0x1000], &regions, &[], 0).is_err());
        assert!(unit_stage_table(&[0x1010], &regions, &[0], usize::MAX).is_err());
        // An entry one past the region end is not in it (stays unchanged).
        let (table, rebased) = unit_stage_table(&[0x1100, 0xfff], &regions, &[0], 0x9000).unwrap();
        assert_eq!(rebased, 0);
        assert_eq!(decode(&table), vec![0x1100, 0xfff]);
    }

    #[test]
    fn ep_unit_stage_next_staged_layer_scans_mixed_host_and_resident_layers() {
        use crate::expert_ep_units::{
            place_routed_experts_units, EpLayerState, EpRanking, EpUnitGeometry, EpUnitPlan,
            EpUnitSource,
        };
        // 8 layers x 8 experts, 2 ranks: rank 0 owns 4 per layer. A hand-made
        // ranking keeps layer 1 fully resident (its 4 rank-0 units), layer 3
        // and 6 partly (mixed), every other layer stays wholly host.
        let geometry = EpUnitGeometry {
            layers: 8,
            experts_per_layer: 8,
            ranks: 2,
            gate_up_bytes: UNIT_GU as u64,
            down_bytes: UNIT_DN as u64,
        };
        let id = |layer: usize, expert: usize| (layer * 8 + expert) as u32;
        let mut order: Vec<u32> = Vec::new();
        order.extend((0..8).filter(|e| e % 2 == 0).map(|e| id(1, e)));
        order.extend([id(3, 0), id(3, 2), id(6, 4)]);
        let hot: std::collections::BTreeSet<u32> = order.iter().copied().collect();
        order.extend((0..64u32).filter(|id| !hot.contains(id)));
        let units = 4 + 2 + 1;
        let charge = |layers: u64| {
            units as u64 * geometry.unit_bytes()
                + layers * 2 * crate::expert_ep_units::VRAM_ARENA_TAIL_PAD_BYTES
        };
        let plan: EpUnitPlan = place_routed_experts_units(
            &geometry,
            &EpUnitSource::Ranking {
                ranking: EpRanking {
                    rank: &order,
                    counts: None,
                    sha256: None,
                },
                rank_budget_bytes: vec![charge(3), 0],
            },
        )
        .unwrap();
        let states: Vec<_> = (0..8).map(|l| plan.layer_state(0, l)).collect();
        assert_eq!(states[1], EpLayerState::Resident);
        assert!(matches!(states[3], EpLayerState::Mixed { .. }));
        assert!(matches!(states[6], EpLayerState::Mixed { .. }));
        for l in [0, 2, 4, 5, 7] {
            assert_eq!(states[l], EpLayerState::Host, "layer {l}");
        }
        // Slots follow the host-bearing layers (everything but layer 1).
        let slots = unit_stage_slots(&plan, 0, 8, &[]);
        assert_eq!(
            slots,
            vec![
                Some(0),
                None,
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(5),
                Some(6)
            ]
        );
        // Even parity: 0 -> 2 -> 4 -> 6 (mixed, host and mixed alike).
        assert_eq!(next_staged_layer(&slots, 0), Some(2));
        assert_eq!(next_staged_layer(&slots, 2), Some(4));
        assert_eq!(next_staged_layer(&slots, 4), Some(6));
        assert_eq!(next_staged_layer(&slots, 6), None);
        // Odd parity skips the resident layer 1 entirely: 3 -> 5 -> 7.
        assert_eq!(slots[1], None);
        assert_eq!(next_staged_layer(&slots, 1), Some(3));
        assert_eq!(next_staged_layer(&slots, 3), Some(5));
        assert_eq!(next_staged_layer(&slots, 5), Some(7));
        assert_eq!(next_staged_layer(&slots, 7), None);
        // Rank 1 holds nothing (zero budget): all host, every layer a slot.
        assert!(unit_stage_slots(&plan, 1, 8, &[])
            .iter()
            .all(Option::is_some));
        // A layer past the plan (an MTP head) is never staged.
        assert_eq!(unit_stage_slots(&plan, 0, 9, &[])[8], None);
    }

    // ---- D1 unit stage over a decoupled EP reserve's donor layers ----

    use crate::expert_ep_units::{
        place_routed_experts_units_forced, place_routed_experts_units_forced_staged, EpLayerState,
        EpRanking, EpUnitGeometry, EpUnitPlan, VRAM_ARENA_TAIL_PAD_BYTES,
    };

    fn donor_geometry(layers: usize) -> EpUnitGeometry {
        EpUnitGeometry {
            layers,
            experts_per_layer: 8,
            ranks: 2,
            gate_up_bytes: UNIT_GU as u64,
            down_bytes: UNIT_DN as u64,
        }
    }

    /// `forced` layers whole on both ranks; `hot` ids of the ranking first; `units` extra
    /// resident units per rank beyond the forced layers, in `mixed` touched layers.
    fn forced_plan(
        geometry: &EpUnitGeometry,
        forced: &[usize],
        hot: &[(usize, usize)],
        extra_units: u64,
        extra_layers: u64,
        slack: u64,
    ) -> EpUnitPlan {
        let unit = geometry.unit_bytes();
        let owned = geometry.owned_per_layer(0) as u64;
        let budget = forced.len() as u64 * (owned * unit + 2 * VRAM_ARENA_TAIL_PAD_BYTES)
            + extra_units * unit
            + extra_layers * 2 * VRAM_ARENA_TAIL_PAD_BYTES
            + slack;
        let mut order: Vec<u32> = hot
            .iter()
            .map(|&(layer, expert)| (layer * geometry.experts_per_layer + expert) as u32)
            .collect();
        let taken: std::collections::BTreeSet<u32> = order.iter().copied().collect();
        order.extend((0..geometry.total_ids() as u32).filter(|id| !taken.contains(id)));
        place_routed_experts_units_forced(
            geometry,
            EpRanking {
                rank: &order,
                counts: None,
                sha256: None,
            },
            &vec![budget; geometry.ranks],
            forced,
        )
        .unwrap()
    }

    /// 8 layers: 1 and 5 forced donors, 6 forced whole-resident but NOT a donor, 3 mixed
    /// (2 of 4 units per rank resident), every other layer host.
    fn donor_plan() -> EpUnitPlan {
        let g = donor_geometry(8);
        // Rank 0 owns even experts, rank 1 odd ones: layer 3 keeps two units of each rank.
        forced_plan(&g, &[1, 5, 6], &[(3, 0), (3, 2), (3, 1), (3, 3)], 2, 1, 0)
    }

    #[test]
    fn ep_unit_stage_donor_layers_take_slots_beside_host_and_mixed_layers() {
        let plan = donor_plan();
        assert_eq!(plan.whole_resident_layers_all_ranks(), vec![1, 5, 6]);
        assert!(matches!(plan.layer_state(0, 3), EpLayerState::Mixed { .. }));
        for rank in 0..2 {
            // A whole-Resident layer has no host region: only the donor list gives it a slot.
            assert_eq!(plan.stage_host_region_bytes(rank, 1), 0);
            let plain = unit_stage_slots(&plan, rank, 8, &[]);
            assert_eq!(plain[1], None);
            assert_eq!(plain[5], None);
            assert_eq!(plain[6], None);
            let donors = unit_stage_slots(&plan, rank, 8, &[1, 5]);
            // Donors 1 and 5 join the host-bearing layers 0, 2, 3, 4, 7 in layer order; layer 6
            // (whole-resident, not a donor) stays unstaged.
            assert_eq!(
                donors,
                vec![
                    Some(0),
                    Some(1),
                    Some(2),
                    Some(3),
                    Some(4),
                    Some(5),
                    None,
                    Some(6)
                ],
                "rank {rank}"
            );
            // Without donors the plan stages exactly the host-bearing layers.
            assert_eq!(
                plain.iter().flatten().count() + 2,
                donors.iter().flatten().count()
            );
            // The parity chains run through the donors: odd 1 -> 3 -> 5 -> 7.
            assert_eq!(next_staged_layer(&donors, 1), Some(3));
            assert_eq!(next_staged_layer(&donors, 3), Some(5));
            assert_eq!(next_staged_layer(&donors, 5), Some(7));
            // Even: 0 -> 2 -> 4 -> (6 is not staged) none.
            assert_eq!(next_staged_layer(&donors, 4), None);
            assert!(slots_serve(&donors, &[1, 5]));
            assert!(!slots_serve(&plain, &[1, 5]));
            assert!(!slots_serve(&donors, &[1, 6]));
            assert!(slots_serve(&donors, &[]));
        }
        // A donor past the plan (an MTP head) never gets a slot.
        assert_eq!(unit_stage_slots(&plan, 0, 9, &[1, 5, 8])[8], None);
    }

    /// A fake loan: per donor layer two pinned buffers and two device aliases.
    struct FakeLoan {
        buffers: Vec<[Vec<u8>; 2]>,
    }

    impl FakeLoan {
        const DEVICE: [usize; 2] = [0x40_0000_0000, 0x50_0000_0000];

        fn new(layers: &[usize], owned: usize) -> Self {
            Self {
                buffers: layers
                    .iter()
                    .map(|_| [vec![0u8; owned * UNIT_GU], vec![0u8; owned * UNIT_DN]])
                    .collect(),
            }
        }

        fn device(slot: usize, index: usize) -> usize {
            Self::DEVICE[slot] + index * 0x1_0000_0000
        }

        fn sources(&self, layers: &[usize]) -> ep_reserve_forward::EpStageHostSources {
            let parts: Vec<_> = layers
                .iter()
                .zip(&self.buffers)
                .enumerate()
                .map(|(index, (&layer, pair))| {
                    (
                        layer,
                        [
                            (pair[0].as_ptr() as usize, pair[0].len()),
                            (pair[1].as_ptr() as usize, pair[1].len()),
                        ],
                        [Self::device(0, index), Self::device(1, index)],
                    )
                })
                .collect();
            ep_reserve_forward::EpStageHostSources::from_parts(&parts)
        }
    }

    #[test]
    fn ep_unit_stage_donor_regions_are_the_two_loan_mirrors() {
        let plan = donor_plan();
        let owned = plan.geometry().owned_per_layer(0);
        let donors = [1usize, 5];
        let loan = FakeLoan::new(&donors, owned);
        let sources = loan.sources(&donors);
        assert_eq!(sources.layers(), &donors);
        for (index, &layer) in donors.iter().enumerate() {
            let (regions, cpu) = donor_stage_regions(&sources, layer).unwrap();
            assert_eq!(
                regions,
                vec![
                    UnitStageRegion {
                        base: FakeLoan::device(0, index),
                        len: owned * UNIT_GU
                    },
                    UnitStageRegion {
                        base: FakeLoan::device(1, index),
                        len: owned * UNIT_DN
                    },
                ],
                "layer {layer}"
            );
            // The copy source is the pinned mirror, never the device alias.
            assert_eq!(cpu[0], loan.buffers[index][0].as_ptr() as usize);
            assert_eq!(cpu[1], loan.buffers[index][1].as_ptr() as usize);
            // Gate/up first, then down at its 256-aligned offset: a whole layer region.
            let layout = unit_stage_layout(&regions).unwrap();
            assert_eq!(layout.dst, vec![0, owned * UNIT_GU]);
            assert_eq!(
                layout.used as u64,
                plan.geometry().owned_per_layer(0) as u64 * plan.geometry().unit_bytes()
            );
        }
        // Layers with no mirror (a mixed or host layer, a non-donor resident layer) are refused.
        for layer in [0usize, 2, 3, 6] {
            assert!(
                donor_stage_regions(&sources, layer).is_err(),
                "layer {layer}"
            );
        }
        // A mirror without a device alias cannot be rebased onto.
        let broken = ep_reserve_forward::EpStageHostSources::from_parts(&[(
            1,
            [
                (
                    loan.buffers[0][0].as_ptr() as usize,
                    loan.buffers[0][0].len(),
                ),
                (
                    loan.buffers[0][1].as_ptr() as usize,
                    loan.buffers[0][1].len(),
                ),
            ],
            [0, FakeLoan::device(1, 0)],
        )]);
        assert!(donor_stage_regions(&broken, 1).is_err());
    }

    /// The compact decode table of a layer whose owned experts (every `ranks`-th) sit in a
    /// tensor of `stride`-byte experts at `base`, every other expert on the zero dummy.
    fn compact_entries(base: usize, stride: usize, rank: usize, dummy: usize) -> Vec<usize> {
        (0..8)
            .map(|expert| {
                if expert % 2 == rank {
                    base + (expert / 2) * stride
                } else {
                    dummy
                }
            })
            .collect()
    }

    #[test]
    fn ep_unit_stage_donor_table_rebases_only_its_mirror_entries() {
        let plan = donor_plan();
        let owned = plan.geometry().owned_per_layer(0);
        let donors = [1usize, 5];
        let loan = FakeLoan::new(&donors, owned);
        let sources = loan.sources(&donors);
        let (dummy, stage) = (0x5000_0000usize, 0x2_0000_0000usize);
        let (regions, _) = donor_stage_regions(&sources, 1).unwrap();
        let layout = unit_stage_layout(&regions).unwrap();
        // The borrowed runtime's tables: owned experts at the mirror's device base.
        let gate = compact_entries(FakeLoan::device(0, 0), UNIT_GU, 0, dummy);
        let down = compact_entries(FakeLoan::device(1, 0), UNIT_DN, 0, dummy);
        let (table, rebased) = unit_stage_table(&gate, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, owned);
        let out: Vec<usize> = table
            .chunks_exact(8)
            .map(|bytes| usize::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        for (expert, (&before, &after)) in gate.iter().zip(&out).enumerate() {
            if expert % 2 == 0 {
                assert_eq!(
                    after,
                    stage + (expert / 2) * UNIT_GU,
                    "gate expert {expert}"
                );
            } else {
                assert_eq!(after, before, "expert {expert} is the dummy");
            }
        }
        let (table, rebased) = unit_stage_table(&down, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, owned);
        let out: Vec<usize> = table
            .chunks_exact(8)
            .map(|bytes| usize::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        for expert in (0..8).step_by(2) {
            // The down mirror sits behind the gate/up mirror in the stage buffer.
            assert_eq!(
                out[expert],
                stage + owned * UNIT_GU + (expert / 2) * UNIT_DN
            );
        }
        // Another donor's entries (a different mirror) are not this layer's: nothing rebases, so
        // a table that names the wrong mirror fails the per-table rebase count of `new_units`.
        let other = compact_entries(FakeLoan::device(0, 1), UNIT_GU, 0, dummy);
        let (_, rebased) = unit_stage_table(&other, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, 0);
        // A mixed layer's VRAM-arena entries sit outside the donor regions as well.
        let vram = compact_entries(0x7000_0000, UNIT_GU, 0, dummy);
        let (_, rebased) = unit_stage_table(&vram, &regions, &layout.dst, stage).unwrap();
        assert_eq!(rebased, 0);
    }

    #[test]
    fn ep_unit_stage_capacity_holds_one_whole_donor_layer_region() {
        // 4 layers: donors 0 and 1 forced whole; layers 2 and 3 mixed with ONE host unit per
        // rank (3 of 4 resident). No whole-host layer, so the plan's own largest host region is
        // a single unit; only a donor layer raises the stage capacity to a whole layer.
        let g = donor_geometry(4);
        let hot: Vec<(usize, usize)> = [2usize, 3]
            .iter()
            .flat_map(|&layer| (0..6).map(move |expert| (layer, expert)))
            .collect();
        let plan = forced_plan(&g, &[0, 1], &hot, 6, 2, 0);
        assert_eq!(plan.whole_resident_layers_all_ranks(), vec![0, 1]);
        assert!(matches!(
            plan.layer_state(0, 2),
            EpLayerState::Mixed { host: 1, .. }
        ));
        let unit = g.unit_bytes();
        let whole = g.owned_per_layer(0) as u64 * unit;
        assert_eq!(plan.max_stage_host_region_bytes(), unit);
        assert_eq!(unit_stage_capacity(&plan, &[]), unit);
        assert_eq!(unit_stage_capacity(&plan, &[0, 1]), whole);
    }

    #[test]
    fn ep_unit_stage_charge_covers_what_the_forward_stages() {
        // The planner's staged charge (`place_routed_experts_units_forced_staged`) is at least the
        // capacity `new_units` checks every staged layer against, with and without donors.
        let g = donor_geometry(8);
        let hot: Vec<u32> = [(3usize, 0usize), (3, 2), (3, 1), (3, 3)]
            .iter()
            .map(|&(layer, expert)| (layer * g.experts_per_layer + expert) as u32)
            .collect();
        let mut order = hot.clone();
        order.extend((0..g.total_ids() as u32).filter(|id| !hot.contains(id)));
        let unit = g.unit_bytes();
        let owned = g.owned_per_layer(0) as u64;
        let budget = 3 * (owned * unit + 2 * VRAM_ARENA_TAIL_PAD_BYTES)
            + 4 * unit
            + 2 * 2 * VRAM_ARENA_TAIL_PAD_BYTES
            + 2 * (owned * unit + crate::expert_ep_units::EP_UNIT_STAGE_PAD_BYTES)
            + crate::expert_ep_units::ep_unit_stage_table_bytes(&g);
        for forced in [&[1usize, 5, 6][..], &[][..]] {
            let budgets = vec![budget; g.ranks];
            let staged = place_routed_experts_units_forced_staged(
                &g,
                EpRanking {
                    rank: &order,
                    counts: None,
                    sha256: None,
                },
                &budgets,
                forced,
                true,
            )
            .unwrap();
            let donors: Vec<usize> = forced.iter().copied().filter(|&layer| layer != 6).collect();
            // A plan that is a whole-layer split takes the donor stage, which stages no host
            // region of its own: only the donor floor remains of its capacity.
            let units_stage = crate::expert_ep_units::ep_unit_stage_kind(Some(&staged.plan))
                == crate::expert_ep_units::EpUnitStageKind::Units;
            let need = if units_stage || !donors.is_empty() {
                unit_stage_capacity(&staged.plan, &donors)
            } else {
                0
            };
            assert!(
                need <= staged.stage.capacity,
                "forced {forced:?}: the forward stages {need} B, the plan charged {} B",
                staged.stage.capacity
            );
            if !forced.is_empty() {
                assert!(staged.stage.capacity >= owned * unit);
            }
        }
    }

    #[test]
    fn ep_unit_stage_without_donors_is_unchanged() {
        // The no-donor path is the pre-reserve one: slots are the host-bearing layers and the
        // capacity is the plan's largest host region.
        let plan = donor_plan();
        for rank in 0..2 {
            let expected: Vec<Option<usize>> = {
                let mut next = 0;
                (0..8)
                    .map(|layer| {
                        (plan.stage_host_region_bytes(rank, layer) > 0).then(|| {
                            next += 1;
                            next - 1
                        })
                    })
                    .collect()
            };
            assert_eq!(unit_stage_slots(&plan, rank, 8, &[]), expected);
        }
        assert_eq!(
            unit_stage_capacity(&plan, &[]),
            plan.max_stage_host_region_bytes()
        );
    }

    // ---- D1 borrowed unit stage (`HIPFIRE_EP_UNIT_STAGE_BORROW`) ----

    /// Bytes of the two tensors (compact gate/up, compact down) of a whole layer of one rank of
    /// 512 experts over 2 ranks: the two stage buffers of one lent parity.
    const LENT: [usize; 2] = [256 * UNIT_GU, 256 * UNIT_DN];

    fn region(base: usize, len: usize) -> UnitStageRegion {
        UnitStageRegion { base, len }
    }

    #[test]
    fn borrowed_layout_is_first_fit_into_the_two_lent_tensors() {
        // A host layer or scratch donor: its two aggregates fill one tensor each.
        let whole = [
            region(0x1_0000_0000, LENT[0]),
            region(0x2_0000_0000, LENT[1]),
        ];
        let layout = unit_stage_layout_bufs(&whole, LENT).unwrap();
        assert_eq!(layout.buf, vec![0, 1]);
        assert_eq!(layout.dst, vec![0, 0]);
        assert_eq!(layout.used, LENT);
        // A mixed layer with `n` host experts: gate/up block then down block. Both blocks share
        // the lent gate/up tensor (adjacent, as in the host arena) while they fit, else the
        // down block takes the lent down tensor.
        for n in [1usize, 125, 170, 171, 255, 256] {
            let split = split_mixed_host_arena(0x3_0000_0000, n, UNIT_GU, UNIT_DN);
            assert_eq!(split[0], region(0x3_0000_0000, n * UNIT_GU));
            assert_eq!(split[1].base, split[0].base + split[0].len, "n {n}");
            assert_eq!(split[1].len, n * UNIT_DN);
            let layout = unit_stage_layout_bufs(&split, LENT).unwrap();
            if n * (UNIT_GU + UNIT_DN) <= LENT[0] {
                assert_eq!(layout.buf, vec![0, 0], "n {n}");
                assert_eq!(layout.dst, vec![0, n * UNIT_GU], "n {n}");
                assert_eq!(layout.used, [n * (UNIT_GU + UNIT_DN), 0], "n {n}");
            } else {
                assert_eq!(layout.buf, vec![0, 1], "n {n}");
                assert_eq!(layout.dst, vec![0, 0], "n {n}");
                assert_eq!(layout.used, [n * UNIT_GU, n * UNIT_DN], "n {n}");
            }
        }
        // Alignment: each region starts at a 256-byte boundary of its buffer.
        let odd = [region(0x100, 100), region(0x1000, 100), region(0x2000, 100)];
        let layout = unit_stage_layout_bufs(&odd, [1000, 1000]).unwrap();
        assert_eq!(layout.buf, vec![0, 0, 0]);
        assert_eq!(layout.dst, vec![0, 256, 512]);
        assert_eq!(layout.used, [612, 0]);
        // A region that fits neither tensor, an empty and an overlapping region are refused.
        assert!(unit_stage_layout_bufs(&[region(0, LENT[0] + 1)], LENT).is_err());
        assert!(
            unit_stage_layout_bufs(&[region(0, LENT[1]), region(0x10, LENT[0])], LENT).is_err()
        );
        let text = unit_stage_layout_bufs(&[region(0, 0)], LENT).unwrap_err();
        assert!(text.contains("empty"), "{text}");
        let text = unit_stage_layout_bufs(&[region(0, 64), region(32, 64)], LENT).unwrap_err();
        assert!(text.contains("overlaps"), "{text}");
        // The single-buffer layout is untouched.
        let single = unit_stage_layout(&whole).unwrap();
        assert_eq!(single.dst, vec![0, LENT[0]]);
        assert_eq!(single.used, LENT[0] + LENT[1]);
    }

    #[test]
    fn borrowed_table_rebases_only_host_entries_each_onto_its_own_buffer() {
        let regions = [region(0x1000, 0x4000), region(0x5000, 0x2000)];
        let layout = unit_stage_layout_bufs(&regions, [0x4000, 0x2000]).unwrap();
        assert_eq!(
            (layout.buf.clone(), layout.dst.clone()),
            (vec![0, 1], vec![0, 0])
        );
        let (dummy, vram) = (0xdead_0000usize, 0x9_0000usize);
        let entries = [0x1000, 0x2000, 0x5000, 0x6000, dummy, vram, 0x6fff, 0x7000];
        let bases = [0x70_0000_0000usize, 0x90_0000_0000];
        let (table, rebased) =
            unit_stage_table_bufs(&entries, &regions, &layout.dst, &layout.buf, bases).unwrap();
        // 0x7000 is one past the second region: not a host entry.
        assert_eq!(rebased, 5);
        let out: Vec<usize> = table
            .chunks_exact(8)
            .map(|bytes| usize::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            out,
            vec![
                bases[0],
                bases[0] + 0x1000,
                bases[1],
                bases[1] + 0x1000,
                dummy,
                vram,
                bases[1] + 0x1fff,
                0x7000
            ]
        );
        // The single-buffer table is the two-buffer table with every region in buffer 0.
        let single = unit_stage_layout(&regions).unwrap();
        let (a, count_a) = unit_stage_table(&entries, &regions, &single.dst, bases[0]).unwrap();
        let (b, count_b) = unit_stage_table_bufs(
            &entries,
            &regions,
            &single.dst,
            &[0, 0],
            [bases[0], bases[1]],
        )
        .unwrap();
        assert_eq!((a, count_a), (b, count_b));
        // Malformed inputs are refused.
        assert!(unit_stage_table_bufs(&entries, &regions, &[0], &[0, 1], bases).is_err());
        assert!(unit_stage_table_bufs(&entries, &regions, &[0, 0], &[0, 2], bases).is_err());
    }

    #[test]
    fn borrowed_tail_pad_is_zeroed_only_as_far_as_the_lent_tensor_reaches() {
        // A dedicated buffer is sized `used + pad`: the whole pad.
        assert_eq!(
            tail_pad_room(1000 + ARENA_TAIL_PAD_BYTES, 1000),
            ARENA_TAIL_PAD_BYTES
        );
        // Slack inside a lent tensor behind a smaller payload: the whole pad again.
        assert_eq!(tail_pad_room(LENT[0], 1 << 20), ARENA_TAIL_PAD_BYTES);
        // A payload that fills the tensor flush (a whole host layer or scratch donor): no pad,
        // exactly the S2 donor stage's case.
        assert_eq!(tail_pad_room(LENT[0], LENT[0]), 0);
        assert_eq!(tail_pad_room(LENT[1], LENT[1]), 0);
        // Fewer than a pad of slack: what is there.
        assert_eq!(tail_pad_room(LENT[1], LENT[1] - 96), 96);
        assert_eq!(tail_pad_room(10, 20), 0);
    }

    /// One staged forward of the borrowed protocol over `layers` of a stage with `slots`, stopped
    /// after the layer `upto - 1` (an aborted forward when `upto < layers`) and joined: every
    /// event as `(layer whose MoE just ran, event)`, the join's at `upto`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ev {
        Free(usize),
        Enqueue(usize),
        Restore(usize),
        JoinRestore(usize),
    }

    fn simulate_borrowed_forward(
        slots: &[Option<usize>],
        lent: [usize; 2],
        upto: usize,
    ) -> Vec<(usize, Ev)> {
        let mut overwritten = [false; 2];
        let mut events = Vec::new();
        for layer in 0..upto {
            let parity = layer % 2;
            let step = borrowed_after_moe(
                lent,
                slots.len(),
                |candidate| slots[candidate].is_some(),
                layer,
                overwritten[parity],
            );
            if step.release {
                events.push((layer, Ev::Free(parity)));
            }
            match step.next {
                BorrowedNext::Nothing => {}
                BorrowedNext::Enqueue(next) => {
                    overwritten[parity] = true;
                    events.push((layer, Ev::Enqueue(next)));
                }
                BorrowedNext::Restore => {
                    overwritten[parity] = false;
                    events.push((layer, Ev::Restore(parity)));
                }
            }
        }
        // `slab_join`: a parity still overwritten at the end of the forward is restored.
        for parity in 0..2 {
            if overwritten[parity] {
                events.push((upto, Ev::JoinRestore(parity)));
            }
        }
        events
    }

    /// 8 layers over 2 ranks, layers 0 and 1 lent, layer 5 a scratch donor, layer 3 mixed (2 of 4
    /// units resident per rank), layers 2, 4, 6, 7 host: placed by the borrowed planner.
    fn borrowed_plan() -> EpUnitPlan {
        let g = donor_geometry(8);
        let hot: Vec<u32> = [(3usize, 0usize), (3, 2), (3, 1), (3, 3)]
            .iter()
            .map(|&(layer, expert)| (layer * g.experts_per_layer + expert) as u32)
            .collect();
        let mut order = hot.clone();
        order.extend((0..g.total_ids() as u32).filter(|id| !hot.contains(id)));
        let unit = g.unit_bytes();
        let owned = g.owned_per_layer(0) as u64;
        let budget = crate::expert_ep_units::ep_unit_stage_table_bytes(&g)
            + 3 * (owned * unit + 2 * VRAM_ARENA_TAIL_PAD_BYTES)
            + 2 * unit
            + 2 * VRAM_ARENA_TAIL_PAD_BYTES;
        crate::expert_ep_units::place_routed_experts_units_borrowed_staged(
            &g,
            EpRanking {
                rank: &order,
                counts: None,
                sha256: None,
            },
            &vec![budget; g.ranks],
            &[5],
            true,
        )
        .unwrap()
        .plan
    }

    #[test]
    fn borrowed_schedule_frees_a_lent_layer_before_overwriting_it_and_restores_each_parity_last() {
        let plan = borrowed_plan();
        let lent = plan.stage_borrow_layers().unwrap();
        assert_eq!(lent, [0, 1]);
        assert_eq!(plan.whole_resident_layers_all_ranks(), vec![0, 1, 5]);
        assert!(matches!(plan.layer_state(0, 3), EpLayerState::Mixed { .. }));
        for rank in 0..2 {
            // The lent layers are whole-Resident: no host region, so no slot of their own.
            let slots = unit_stage_slots(&plan, rank, 8, &[5]);
            assert_eq!(slots[0], None);
            assert_eq!(slots[1], None);
            assert_eq!(slots.iter().flatten().count(), 6);
            let events = simulate_borrowed_forward(&slots, lent, 8);
            assert_eq!(
                events,
                vec![
                    (0, Ev::Free(0)),
                    (0, Ev::Enqueue(2)),
                    (1, Ev::Free(1)),
                    (1, Ev::Enqueue(3)),
                    (2, Ev::Enqueue(4)),
                    (3, Ev::Enqueue(5)),
                    (4, Ev::Enqueue(6)),
                    (5, Ev::Enqueue(7)),
                    (6, Ev::Restore(0)),
                    (7, Ev::Restore(1)),
                ],
                "rank {rank}"
            );
        }
    }

    #[test]
    fn borrowed_schedule_never_skips_a_restore_or_overwrites_before_the_lent_moe() {
        let plan = borrowed_plan();
        let lent = plan.stage_borrow_layers().unwrap();
        let full = unit_stage_slots(&plan, 0, 8, &[5]);
        // Parities with no staged layer (odd layers all resident) and a plan whose even chain is
        // cut short, besides the full chain, at every abort point of the forward.
        let even_only: Vec<Option<usize>> = full
            .iter()
            .enumerate()
            .map(|(layer, slot)| slot.filter(|_| layer % 2 == 0))
            .collect();
        let none: Vec<Option<usize>> = vec![None; 8];
        for slots in [&full, &even_only, &none] {
            for upto in 0..=8 {
                let events = simulate_borrowed_forward(slots, lent, upto);
                for parity in 0..2 {
                    let enqueues: Vec<usize> = events
                        .iter()
                        .filter_map(|&(_, ev)| match ev {
                            Ev::Enqueue(next) if next % 2 == parity => Some(next),
                            _ => None,
                        })
                        .collect();
                    let restores = events
                        .iter()
                        .filter(|(_, ev)| {
                            matches!(ev, Ev::Restore(p) | Ev::JoinRestore(p) if *p == parity)
                        })
                        .count();
                    // Anything that overwrote a lent layer is restored exactly once, by the
                    // last staged layer's step or by the join of an aborted forward.
                    assert_eq!(
                        restores,
                        usize::from(!enqueues.is_empty()),
                        "parity {parity} upto {upto}"
                    );
                    // Every staged layer is copied at most once, in layer order, and only
                    // layers with a slot are ever copied.
                    assert!(enqueues.windows(2).all(|pair| pair[0] + 2 == pair[1]));
                    assert!(enqueues.iter().all(|&layer| slots[layer].is_some()));
                    // The first overwrite of a parity comes after the lent layer's own MoE has
                    // issued and freed its tensors.
                    let first = events
                        .iter()
                        .position(|&(_, ev)| matches!(ev, Ev::Enqueue(n) if n % 2 == parity));
                    if let Some(first) = first {
                        let freed = events
                            .iter()
                            .position(|&(_, ev)| ev == Ev::Free(parity))
                            .expect("a parity that overwrites its lent layer frees it first");
                        assert!(freed < first, "parity {parity} upto {upto}");
                        assert!(events[first].0 >= lent[parity], "parity {parity}");
                        // A staged layer is never copied before the one two layers earlier ran.
                        assert!(events[first].0 + 2 <= enqueues[0]);
                    }
                }
            }
        }
    }

    #[test]
    fn borrowed_plan_stages_every_layer_of_every_rank_through_the_two_lent_tensors() {
        // The real geometry (48 layers x 512 experts x 2 ranks) under a pseudo-random ranking
        // and 7 scratch donors: every host-bearing layer and donor must pack into the lent
        // tensors of its rank, a mixed layer whatever its host count.
        let g = EpUnitGeometry {
            layers: 48,
            experts_per_layer: 512,
            ranks: 2,
            gate_up_bytes: UNIT_GU as u64,
            down_bytes: UNIT_DN as u64,
        };
        let mut state = 0x5EED_B0A2_0D1u64;
        let mut order: Vec<u32> = (0..g.total_ids() as u32).collect();
        for i in (1..order.len()).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            order.swap(i, (state >> 33) as usize % (i + 1));
        }
        let unit = g.unit_bytes();
        let donors = [9usize, 14, 20, 27, 31, 38, 44];
        for resident_units in [3_000u64, 8_700, 11_000] {
            let budget = resident_units * unit
                + 2 * 48 * VRAM_ARENA_TAIL_PAD_BYTES
                + crate::expert_ep_units::ep_unit_stage_table_bytes(&g);
            let Ok(staged) = crate::expert_ep_units::place_routed_experts_units_borrowed_staged(
                &g,
                EpRanking {
                    rank: &order,
                    counts: None,
                    sha256: None,
                },
                &vec![budget; 2],
                &donors,
                true,
            ) else {
                continue;
            };
            let plan = staged.plan;
            assert_eq!(plan.stage_borrow_layers(), Some([0, 1]));
            for rank in 0..2 {
                let caps = plan
                    .stage_borrow_buffer_bytes(rank)
                    .map(|bytes| bytes as usize);
                assert_eq!(caps, LENT);
                let slots = unit_stage_slots(&plan, rank, 48, &donors);
                // The lent layers (and nothing before them) are never staged.
                assert!(slots[0].is_none() && slots[1].is_none());
                for (layer, slot) in slots.iter().enumerate() {
                    if slot.is_none() {
                        continue;
                    }
                    let regions = match plan.layer_state(rank, layer) {
                        EpLayerState::Mixed { host, .. } => split_mixed_host_arena(
                            0x10_0000_0000 * (layer + 1),
                            host,
                            UNIT_GU,
                            UNIT_DN,
                        )
                        .to_vec(),
                        // A host layer's aggregates and a donor's mirrors are two regions of
                        // the whole layer's tensors.
                        _ => vec![
                            region(0x10_0000_0000 * (layer + 1), LENT[0]),
                            region(0x10_0000_0000 * (layer + 1) + LENT[0], LENT[1]),
                        ],
                    };
                    let layout = unit_stage_layout_bufs(&regions, caps).unwrap_or_else(|reason| {
                        panic!("units {resident_units} rank {rank} layer {layer}: {reason}")
                    });
                    assert!(layout.used[0] <= caps[0] && layout.used[1] <= caps[1]);
                    // A layer needing the schedule's parity buffer has a lent layer before it.
                    assert!(layer > [0usize, 1][layer % 2]);
                }
            }
        }
    }
}
