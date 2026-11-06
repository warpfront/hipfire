use crate::qwen35::{
    DeltaNetState, LayerType, Qwen35Config, Qwen35DecodeBatchState, Qwen35Scratch,
    Qwen35ScratchSet, Qwen35Weights, StateQuant,
};
use crate::Qwen35;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_adaptive::{KvAdaptive, Preset};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::kv_mode;
use hipfire_runtime::llama::KvCacheExt;
use hipfire_runtime::llama::{self, KvCache, KvDims, KvLayers, KvTarget};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::offload;

pub struct Qwen35Bundle {
    pub config: Qwen35Config,
    pub weights: Qwen35Weights,
    pub scratch: Qwen35Scratch,
    pub kv_cache: KvCache,
    pub dn_state: DeltaNetState,
    /// Adaptive KV controller when engaged at load. Moved into
    /// `LoadedModel.kv_adaptive` by `finish_qwen35_load`.
    pub kv_adaptive: Option<KvAdaptive>,
    /// Pipeline-parallel per-device scratch set. `Some` only when
    /// `LoadedModel.pp > 1` — single-GPU loads leave this `None`.
    /// Freed via `Qwen35ScratchSet::free_gpu_multi(&mut Gpus)` in the pp>1
    /// unload arm, NOT via `ArchModel::free_gpu` (which takes a single
    /// `&mut Gpu` and cannot free a per-device set).
    pub pp_scratch_set: Option<Qwen35ScratchSet>,
    /// Optional Qwen3.5-VL vision tower — `Some` when the HFQ contained
    /// `model.visual.patch_embed.proj.weight`. Remains `None` for pure
    /// text checkpoints; the bundle's text path is unaffected.
    pub vision_config: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionConfig>,
    pub vision_weights: Option<hipfire_arch_qwen35_vl::qwen35_vl::VisionWeights>,
    /// Continuous-batch decode state for Qwen3.5 (single-GPU). `Some` when
    /// `HIPFIRE_CONTINUOUS_BATCH` staged a batch (arch 5/6, pp=1, non-EP).
    /// Freed via `Qwen35DecodeBatchState::free_gpu` in `ArchModel::free_gpu`
    /// or eagerly via `LoadedModel::qwen35_mut()` in `unload_model` before
    /// `ArchModel::free_gpu`. Previously lived on `LoadedModel`.
    pub qwen35_decode_batch: Option<Qwen35DecodeBatchState>,
    /// Per-request VMM continuous-batching executor (request owners plus
    /// row-budget scratch). `None` unless the VMM batch route is staged;
    /// freed before KV/weights in every bundle free path.
    pub vmm_store: Option<crate::forward_slots::vmm::Qwen35VmmStore>,
    /// Live weight-host lease (`HIPFIRE_WEIGHT_HOST=1`): `Some` exactly when
    /// `weights` were imported from a host generation. Released only by
    /// [`free_qwen35_bundle`] / the loader's `unload_model` after the imported
    /// owners are freed and VMM teardown is clean; dropping it closes the
    /// lease without a release acknowledgement.
    pub weight_lease: Option<SharedWeightLease>,
}
/// Device bytes one draft file will take, estimated the way the loader's own
/// admission estimates a weight file: file bytes + 1/8 for the expanded/AWQ
/// tensors + 512 MiB of scratch. An over-estimate for a head that loads only its
/// own layers, which is the safe direction — it can only make the trunk spill
/// slightly more, never less. 0 for a path that does not exist.
pub(crate) fn draft_file_reserve_bytes(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    reserve_for_bytes(meta.len())
}

/// The `draft_file_reserve_bytes` formula for a known byte count: file bytes +
/// 1/8 for the expanded/AWQ tensors + 512 MiB of scratch. Split out so the
/// bundled-trailer arm (which knows its section size without a file of its
/// own) charges the same estimate as a sidecar file of the same size.
fn reserve_for_bytes(file_bytes: u64) -> u64 {
    file_bytes + file_bytes / 8 + (512 << 20)
}

/// Where the MTP head that will load beside a trunk lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MtpHeadLocation {
    /// A standalone `.mtp` sidecar file.
    Sidecar(std::path::PathBuf),
    /// A head bundled inside the trunk file itself: only the appended MTP
    /// section counts — the trunk's own bytes are already in the arch census
    /// (`qwen35_layer_bytes`), so charging the whole file would count them
    /// twice and force a maximal spill.
    Bundled { section_bytes: u64 },
}

/// Device bytes the drafts take when they load beside a spilled trunk.
///
/// Drafts stay on the GPU — they are not part of the trunk spill — so their
/// bytes come out of the capacity the placement may spend. Pure (paths and a
/// byte count in, bytes out; the only filesystem reads are the files' own
/// lengths), so unit tests pin the arithmetic without a GPU. 0 when no draft
/// is in play.
///
/// `mtp_head` is the exact head that will load (`None` = none), and
/// `dflash_draft` the DFlash draft file that will load on the slots route.
/// The single caller that both resolvers share — this is what fixes the
/// slots under-reserve, where the old inline probe charged only the MTP
/// sidecar and silently dropped a DFlash draft.
pub(crate) fn draft_device_reserve_bytes(
    mtp_head: Option<&MtpHeadLocation>,
    dflash_draft: Option<&std::path::Path>,
) -> u64 {
    let mut bytes = 0u64;
    match mtp_head {
        Some(MtpHeadLocation::Sidecar(path)) => {
            bytes = bytes.saturating_add(draft_file_reserve_bytes(path));
        }
        Some(MtpHeadLocation::Bundled { section_bytes }) => {
            if *section_bytes > 0 {
                bytes = bytes.saturating_add(reserve_for_bytes(*section_bytes));
            }
        }
        None => {}
    }
    if let Some(path) = dflash_draft {
        bytes = bytes.saturating_add(draft_file_reserve_bytes(path));
    }
    bytes
}

/// The MTP head that will load beside this trunk: the bundled trailer first,
/// then the first existing `sidecar_candidates(trunk, "mtp")` spelling
/// (last-extension replacement, then the stem spelling `model.mtp`). Both
/// loaders probe in this order (bundled, then sidecar), so the reserve always
/// covers what actually loads. A single `path + ".mtp"` concatenation misses
/// the stem spelling and charges 0 for a head that then loads,
/// under-reserving the placement.
///
/// A bundled trailer (the MTP head rides inside the trunk file itself, found
/// by `detect_bundled_mtp_offset`) reserves only the appended MTP section
/// (`file_len − offset − trailer`). The section is a second HFQM container
/// after the trunk's tensors: `HfqFile::open` parses exactly the `n_tensors`
/// index entries of the header at offset 0 (`open_at_offset`), and the census
/// (`qwen35_layer_bytes`) walks only that `tensor_infos()` — no MTP tensor
/// is ever counted there, while the head loader opens the same file
/// separately at the trailer offset (`load_mtp_head_at_offset`). The two
/// containers are disjoint, so the section size is the head's footprint with
/// no double count; the file+1/8+512MiB formula on top is the same
/// over-estimate the sidecar arm already uses for a head that loads only its
/// own layers (safe direction: it can only spill slightly more, never less).
pub(crate) fn resolve_mtp_head_file(trunk: &std::path::Path) -> Option<MtpHeadLocation> {
    if let Some(offset) = crate::mtp_head::detect_bundled_mtp_offset(trunk)
        .ok()
        .flatten()
    {
        let section_bytes = std::fs::metadata(trunk)
            .map(|meta| {
                meta.len()
                    .saturating_sub(offset)
                    .saturating_sub(crate::mtp_head::BUNDLE_TRAILER_LEN)
            })
            .unwrap_or(0);
        return Some(MtpHeadLocation::Bundled { section_bytes });
    }
    crate::mtp_head::mtp_sidecar_candidates(trunk)
        .into_iter()
        .find(|p| p.is_file())
        .map(MtpHeadLocation::Sidecar)
}

/// Resolve partial GPU offload before the first allocation.
///
/// Returns `None` when both budgets are `Full` — the unset default on every
/// model, including MoE. It stays byte-identical to the pre-offload loader:
/// no measured capacity, no admission, no refusal.
///
/// With a knob set, the capacity is `free VRAM − KV reservation − prefill floor −
/// 128 MiB`, the same KV footprint the post-weight path computes, so the
/// placement and the KV admission cannot drift.
///
/// Eligibility is `hipfire_config::memory::spill_support` over this load's own
/// facts (arch 5|6, `ctx.pp`, EP never on this route, the trunk tower-tensor
/// probe, `config.paged_experts`, `config.reap_keep`) — the same call the
/// slots twin makes, so the two cannot disagree. A configured spill on an
/// unspillable path was already refused at admission (`partial_offload_refusal`).
/// The defensive unsupported-path arm below permits only a fully resident
/// placement that fits, otherwise refusing before allocation.
fn resolve_partial_offload(
    hfq: &HfqFile,
    config: &Qwen35Config,
    ctx: &LoadCtx,
    plan: &Qwen35KvPlan,
) -> Result<Option<offload::Placement>, String> {
    use hipfire_config::memory::{
        effective_offload_budgets, spill_support, OffloadBudget, SpillSupport,
    };
    let spill_facts = spill_facts(hfq.arch_id, ctx, config);
    let support = spill_support(
        spill_facts.arch_id,
        spill_facts.tp,
        spill_facts.pp,
        spill_facts.ep_experts,
        spill_facts.paged_experts,
        spill_facts.reap_keep,
        spill_facts.is_vl,
    );
    let budgets = effective_offload_budgets();
    if budgets == (OffloadBudget::Full, OffloadBudget::Full) {
        return Ok(None);
    }
    // A configured spill on an unsupported path should already have been
    // refused at admission. Defensively allow only a pinned Full/Full plan
    // that fits; otherwise refuse before allocation. This reuses offload::plan
    // against the same capacity the placement below would spend.
    if let SpillSupport::Refused(refusal) = support {
        return fail_closed_or_resident(hfq, config, ctx, plan, refusal);
    }
    // `memory.offload_exec=cpu` with an `auto` budget is allowed: the capture
    // gate now reads the model-level host-weights flag the loader sets from the
    // resolved placement (`Gpu::set_host_cpu_weights`), so `auto` can no longer
    // leave hipGraph capture enabled over CPU-executed steps.

    let placement_inputs = placement_capacity_inputs(hfq, config, ctx, plan)?;
    // Every pinned host byte, not just the routed experts: a whole-layer spill
    // hosts each layer's own weights too, and those are the largest host consumer
    // a dense model has.
    let placement = offload::resolve_placement(
        &placement_inputs.layer_bytes,
        &placement_inputs.inputs,
        budgets,
        offload::Orientation::SuffixResident,
    )?;
    Ok(Some(placement))
}

/// The single fact bundle behind both placement resolvers: every dimension
/// `spill_support` takes, built identically by the sequential carrier and the
/// slots engine, so a future arch copies one call, not a condition list.
struct SpillFacts {
    arch_id: u32,
    tp: usize,
    pp: usize,
    ep_experts: Option<usize>,
    paged_experts: bool,
    reap_keep: bool,
    is_vl: bool,
}

/// Build the fact bundle for the sequential single-GPU route: tp is 1 by
/// construction (the tp>1 routes never call this), EP never applies, pp and
/// the parsed model flags come from the load, and VL-ness is the trunk
/// tower-tensor probe — the same index the loader's `probe_vision` asks, so
/// the two cannot classify a tower differently (`config.is_vl_text` is the
/// nested-name text wrapper, which a plain text checkpoint carries too).
fn spill_facts(arch_id: u32, ctx: &LoadCtx, config: &Qwen35Config) -> SpillFacts {
    SpillFacts {
        arch_id,
        tp: 1,
        pp: ctx.pp,
        ep_experts: None,
        paged_experts: config.paged_experts,
        reap_keep: config.reap_keep.is_some(),
        is_vl: has_vision_tower(ctx, arch_id),
    }
}

/// Whether this trunk carries the Qwen3.5-VL vision tower: the inline tower
/// tensor in the trunk index, or a sidecar tower the carrier will load — the
/// explicit `vision_path`, or the `<stem>.vl` sibling discovery unless
/// `vision_mode=off` suppresses it (the same ladder the carrier's VL block
/// applies when it resolves `vl_path`). Reads the index and the filesystem
/// only; touches no GPU state. Named for what it answers (tower presence
/// from any source), not just the inline probe that answers most of it.
fn has_vision_tower(ctx: &LoadCtx, arch_id: u32) -> bool {
    if !matches!(arch_id, 5 | 6) {
        return false;
    }
    // `LoadCtx` borrows the trunk path, not the open HFQ, so re-open
    // read-only for the single index probe (the loader gate probes the
    // already-open source instead — same tensor, same answer).
    let inline = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(ctx.path))
        .ok()
        .is_some_and(|hfq| {
            hfq.tensor_data("model.visual.patch_embed.proj.weight")
                .is_some()
        });
    if inline {
        return true;
    }
    if ctx.vision_path.is_some() {
        return true;
    }
    if ctx.vision_mode == "off" {
        return false;
    }
    hipfire_runtime::sidecar::find_vl_sidecar(std::path::Path::new(ctx.path)).is_some()
}

/// Measured capacity inputs both the fail-closed check and the placement
/// search spend: free VRAM minus the KV reservation, the prefill floor, any
/// device-resident draft, and the fixed slack. One function so the check and
/// the search cannot be charged different rooms.
struct PlacementCapacity {
    layer_bytes: offload::LayerBytes,
    inputs: offload::CapacityInputs,
}

/// Whether the placement reserves device room for the MTP head: exactly when
/// the load will attach it, i.e. the producer-capability gate admits the typed
/// mode (`on`; `auto` only with a declared measured AR floor, which this
/// producer lacks). Mirrors the sequential loader's head gate so a reservation
/// is never charged for a head that does not load.
pub(crate) fn mtp_head_reserved(spec_mtp: Option<bool>) -> bool {
    hipfire_config::mtp_mode_admits(
        hipfire_config::MtpMode::from_tri_state(spec_mtp),
        crate::MTP_MEASURED_AR_FLOOR,
    )
}

/// Device bytes the drafts this load attaches will own: the MTP head (only
/// when `mtp_head_reserved` admits it: bundled trailer, the CLI sidecar, or the
/// `.mtp` sibling) plus the DFlash draft when named. One resolution shared by
/// the placement capacity and the post-weight context cap so the two cannot
/// drift. MTP not admitted (explicit off, or `auto` without a measured AR
/// floor) means the head will not load; a named DFlash draft is still charged
/// because the slots route runs its DFlash2 chain independently of the MTP gate.
fn load_draft_reserve_bytes(trunk: &std::path::Path, ctx: &LoadCtx) -> u64 {
    let dflash_draft: Option<&std::path::Path> = ctx.draft_path.map(std::path::Path::new);
    let mtp_head: Option<MtpHeadLocation> = if mtp_head_reserved(ctx.spec.mtp) {
        match ctx.mtp_path.clone() {
            // An explicit CLI path is always a standalone sidecar file.
            Some(path) => Some(MtpHeadLocation::Sidecar(path)),
            None => resolve_mtp_head_file(trunk),
        }
    } else {
        None
    };
    draft_device_reserve_bytes(mtp_head.as_ref(), dflash_draft)
}

fn placement_capacity_inputs(
    hfq: &HfqFile,
    config: &Qwen35Config,
    ctx: &LoadCtx,
    plan: &Qwen35KvPlan,
) -> Result<PlacementCapacity, String> {
    // Planned capacity (total - own - reserve; UMA/emu keep their capped free
    // figure): a transient foreign allocation must not change placement.
    let planned = ctx
        .gpu
        .planned_capacity()
        .map_err(|e| format!("Qwen pre-weight VRAM query: {e}"))?;
    eprintln!("qwen35: admission decision: weight/KV placement capacity from {planned}");
    let free_vram = planned.bytes as u64;
    let rows = plan.dims.physical_cap.unwrap_or(plan.dims.max_seq);
    let stride =
        crate::qwen35::prefill::vmm_kv_token_bytes(config, plan.pair, plan.adaptive.is_some())
            .ok_or("Qwen KV stride overflow")?;
    let kv = offload::kv_reserve(stride as u64, rows).ok_or("Qwen KV reservation overflow")?;
    // `None` only on sizing overflow (MoE is charged through
    // `PrefillBatchScratch::projected_allocation_bytes`). Reserve 0 and say so:
    // the placement is then bounded by the KV reservation alone, and the
    // shortfall is visible rather than hidden.
    let pbs_reservation =
        crate::qwen35::prefill::minimum_prefill_reservation_bytes(config, &ctx.gpu.arch);
    let pbs = pbs_reservation.unwrap_or(0) as u64;
    // A draft stays on the GPU: MTP (bundled trailer or `.mtp` sidecar) and
    // the DFlash draft are both device-resident and are *not* part of the
    // trunk spill, so they must be reserved out of the capacity the placement
    // spends — otherwise `auto` would over-place the trunk and the head's own
    // load would OOM after the weights landed. The MTP head is reserved only
    // when the load will actually attach it (`mtp_head_reserved`): under
    // `auto` this producer declares no measured AR floor, so no head loads
    // and nothing is reserved.
    // (DSpark is a DeepSeek4 module and never attaches here.)
    let draft_reservation = load_draft_reserve_bytes(hfq.path(), ctx);
    let layer_bytes = crate::qwen35::load::qwen35_layer_bytes(hfq, config);
    let inputs = offload::CapacityInputs {
        free_vram,
        kv_bytes: kv.bytes,
        kv_rows: kv.rows,
        kv_stride: stride as u64,
        prefill_floor: pbs,
        prefill_floor_absent: pbs_reservation.is_none(),
        draft_bytes: draft_reservation,
        slack: 128u64 << 20,
    };
    Ok(PlacementCapacity {
        layer_bytes,
        inputs,
    })
}

/// Defensive arm for a configured spill on an unspillable path. Admission
/// normally refuses it before reaching this loader. Runs the pinned `Full`/`Full`
/// plan against the same capacity the placement would spend: if it fits, stay
/// resident; otherwise refuse, naming the unsupported path and the knob.
fn fail_closed_or_resident(
    hfq: &HfqFile,
    config: &Qwen35Config,
    ctx: &LoadCtx,
    plan: &Qwen35KvPlan,
    refusal: hipfire_config::memory::SpillRefusal,
) -> Result<Option<offload::Placement>, String> {
    use hipfire_config::memory::OffloadBudget;
    let capacity = placement_capacity_inputs(hfq, config, ctx, plan)?;
    let weight_room = offload::weight_capacity(
        capacity.inputs.free_vram,
        capacity.inputs.kv_bytes,
        capacity.inputs.prefill_floor,
        capacity.inputs.draft_bytes,
        capacity.inputs.slack,
    );
    // The pinned plan is the residency proof: it errs exactly when even a
    // fully resident load overruns the room — the shape that used to OOM
    // mid-load. `plan`'s tier order and its own refusal text are untouched;
    // only the `Ok` arm is read here.
    if offload::plan(
        &capacity.layer_bytes,
        (OffloadBudget::Full, OffloadBudget::Full),
        &weight_room,
        offload::Orientation::SuffixResident,
    )
    .is_ok()
    {
        return Ok(None);
    }
    let device = capacity.layer_bytes.weight_total();
    Err(unspillable_no_fit_refusal(
        refusal,
        device,
        weight_room.weight_bytes,
    ))
}

/// The shared fail-closed sentence for an unspillable path whose model does
/// not fit: names the path that cannot spill, the reason, and the knob. Both
/// placement resolvers refuse through this — the sequential twin after its
/// pinned-residency proof, the slots twin after its own — so the wording
/// cannot drift between routes. `pub(crate)`: the slots engine is the only
/// other caller.
pub(crate) fn unspillable_no_fit_refusal(
    refusal: hipfire_config::memory::SpillRefusal,
    device_bytes: u64,
    room_bytes: u64,
) -> String {
    use hipfire_config::memory::SpillRefusal;
    let reason = match refusal {
        SpillRefusal::Arch(arch_id) => format!("arch id {arch_id} has no host-placement path"),
        SpillRefusal::Tp(tp) => format!("tp={tp} loads every layer on every rank"),
        SpillRefusal::Pp(pp) => format!("pp={pp} bands own per-device layouts"),
        SpillRefusal::ExpertParallel(experts) => {
            format!("expert parallelism pins each of {experts} experts to one rank")
        }
        SpillRefusal::VisionTower => {
            "the Qwen3.5-VL vision tower loads through its own lowered loader".to_string()
        }
        SpillRefusal::PagedExperts => "the weight pager owns the expert buffers".to_string(),
        SpillRefusal::ReapKeepMap => "the REAP keep-map has no host-placement path".to_string(),
    };
    format!(
        "load refused: this Qwen3.5-MoE needs {} MiB of device weights but {reason}, so the \
         auto-fit default cannot spill it — it would OOM mid-load instead. Lower memory.max_seq, \
         use a smaller quantization, or load on a spillable path (single-GPU text trunk without \
         paged experts or a REAP keep-map; memory.gpu_layer_budget / memory.moe_expert_budget) — \
         only {} MiB is available after the KV reservation and the prefill floor",
        device_bytes / (1 << 20),
        room_bytes / (1 << 20),
    )
}

/// Build the Qwen35 GPU bundle from an HFQ source.
///
/// CPU-only config/compat validation runs **before** weight upload. Every
/// fallible GPU stage after weights is transactional: on failure, owners
/// constructed so far are explicitly `free_gpu`'d (KvCache via
/// [`KvCache::free_gpu`]) and the original error is preserved with any
/// cleanup/VMM-pending context appended.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen35Bundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("qwen35: directory source unsupported".into());
    };

    let config_placement_started = std::time::Instant::now();
    let config = <Qwen35 as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;

    // ── CPU-only parse + compatibility (zero GPU allocation) ─────────
    let mut plan = plan_qwen35_gpu_stages(&config, ctx)?;
    // Placement is resolved once, here, against the measured capacity a
    // configured spill must fit — before any allocation. `None` (no knob) keeps
    // the load byte-identical to the pre-offload loader.
    let placement = resolve_partial_offload(&hfq, &config, ctx, &plan)?;
    let dn_quant = parse_state_quant(ctx.state_quant_override)?;
    eprintln!("  DeltaNet state: {}", state_quant_label(dn_quant));
    warn_tiny_model_state(&hfq, dn_quant);
    rdna_compute::load_phases::add(
        "qwen35_config_placement",
        rdna_compute::load_phases::ns_since(config_placement_started),
        0,
    );

    // ── Weights (first GPU ownership): private upload, or import from the
    //    weight host when HIPFIRE_WEIGHT_HOST=1. No private-load fallback
    //    once opted in. ─────────────────────────────────────────────────
    let weights_started = std::time::Instant::now();
    let socket = weight_host_socket()?;
    let route = if socket.is_some() {
        "import"
    } else {
        "private"
    };
    let (weights, mut weight_lease) = match socket {
        None => (
            <Qwen35 as Architecture>::load_weights_with_placement(
                &mut hfq,
                &config,
                ctx.gpu,
                placement.as_ref(),
            )?,
            None,
        ),
        Some(socket) => {
            let (weights, lease) = import_from_weight_host(&socket, &hfq, &config, ctx)?;
            (weights, Some(lease))
        }
    };
    hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
    let weights_ms = weights_started.elapsed().as_secs_f64() * 1e3;
    // The real card bound is determined only now: weights (including any
    // expanded tensors) already own VRAM, while KV and lazy PBS do not.
    if let Some(mut sequence) = ctx.sequence {
        let reserve = sequence.reserve_bytes;
        let capacity = (|| -> Result<(usize, usize, usize, usize, u64), String> {
            let planned = ctx
                .gpu
                .planned_capacity()
                .map_err(|e| format!("Qwen post-weight VRAM query: {e}"))?;
            eprintln!("qwen35: admission decision: max_seq/KV budget capacity from {planned}");
            let free = planned.bytes;
            let minimum =
                crate::qwen35::prefill::minimum_prefill_reservation_bytes(&config, &ctx.gpu.arch)
                    .ok_or("Qwen minimum PBS sizing overflow")?;
            let stride = crate::qwen35::prefill::vmm_kv_token_bytes(
                &config,
                plan.pair,
                plan.adaptive.is_some(),
            )
            .ok_or("Qwen KV stride overflow")?;
            let draft_reserve = load_draft_reserve_bytes(hfq.path(), ctx);
            let card_cap = free
                .saturating_sub(minimum)
                .saturating_sub(reserve)
                .saturating_sub(128 << 20)
                .saturating_sub(usize::try_from(draft_reserve).unwrap_or(usize::MAX))
                / stride;
            Ok((free, minimum, stride, card_cap, draft_reserve))
        })();
        let (free, minimum, stride, card_cap, draft_reserve) = match capacity {
            Ok(v) => v,
            Err(e) => {
                weights.free_gpu(ctx.gpu);
                return Err(release_after_failed_load(e, weight_lease.take(), ctx.gpu));
            }
        };
        eprintln!(
            "max_seq budget: free_after_weights={free} min_prefill={minimum} vision_reserve={reserve} draft_reserve={draft_reserve} page_rounding={} kv_bytes_per_token={stride}",
            128 << 20,
        );
        if sequence.automatic {
            if card_cap < 512 {
                weights.free_gpu(ctx.gpu);
                return Err(release_after_failed_load(
                    "Qwen VMM load refused: measured post-weight VRAM cannot hold minimum PBS and KV".into(),
                    weight_lease.take(),
                    ctx.gpu,
                ));
            }
            ctx.max_seq = sequence.model_ctx.min(card_cap);
        }
        sequence.card_cap = card_cap;
        ctx.sequence = Some(sequence);
        plan = match plan_qwen35_gpu_stages(&config, ctx) {
            Ok(v) => v,
            Err(e) => {
                weights.free_gpu(ctx.gpu);
                return Err(release_after_failed_load(e, weight_lease.take(), ctx.gpu));
            }
        };
    }

    // ── KV (transactional: free weights on fail) ─────────────────────
    let kv_started = std::time::Instant::now();
    let (kv, kv_adaptive) = match construct_kv_cache(&config, ctx, plan) {
        Ok(v) => v,
        Err(e) => {
            weights.free_gpu(ctx.gpu);
            let e = append_cleanup_context(e, note_vmm_after_free(ctx.gpu));
            return Err(release_after_failed_load(e, weight_lease.take(), ctx.gpu));
        }
    };
    let kv_ms = kv_started.elapsed().as_secs_f64() * 1e3;

    // ── DeltaNet state (free kv + weights on fail) ───────────────────
    let dn_started = std::time::Instant::now();
    let dn = match DeltaNetState::new_with_quant(ctx.gpu, &config, dn_quant) {
        Ok(v) => v,
        Err(e) => {
            let cleanup = free_kv_and_weights(kv, weights, ctx.gpu);
            let e = append_cleanup_context(format!("{e}"), cleanup);
            return Err(release_after_failed_load(e, weight_lease.take(), ctx.gpu));
        }
    };
    let dn_ms = dn_started.elapsed().as_secs_f64() * 1e3;

    // ── Scratch (free dn + kv + weights on fail) ─────────────────────
    // Flash-attention partials address physical KV slots, not the logical
    // context range.  An eviction-bounded cache can make this cap much smaller
    // than max_seq, so sizing from max_seq retains memory for unreachable tiles.
    let scratch_started = std::time::Instant::now();
    let scratch = match Qwen35Scratch::new_with_kv_max(ctx.gpu, &config, 2048, kv.physical_cap) {
        Ok(v) => v,
        Err(e) => {
            let cleanup = free_dn_kv_and_weights(dn, kv, weights, ctx.gpu);
            let e = append_cleanup_context(format!("{e}"), cleanup);
            return Err(release_after_failed_load(e, weight_lease.take(), ctx.gpu));
        }
    };
    let scratch_ms = scratch_started.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "QWEN35_LOAD_PHASES route={route} weights_ms={weights_ms:.3} kv_ms={kv_ms:.3} dn_ms={dn_ms:.3} scratch_ms={scratch_ms:.3}"
    );
    for (name, ms) in [
        ("qwen35_weights", weights_ms),
        ("qwen35_kv", kv_ms),
        ("qwen35_deltanet", dn_ms),
        ("qwen35_scratch", scratch_ms),
    ] {
        rdna_compute::load_phases::add(name, (ms * 1e6) as u64, 0);
    }

    Ok(Qwen35Bundle {
        config,
        weights,
        scratch,
        kv_cache: kv,
        dn_state: dn,
        kv_adaptive,
        pp_scratch_set: None,
        vision_config: None,
        vision_weights: None,
        qwen35_decode_batch: None,
        vmm_store: None,
        weight_lease,
    })
}

/// CPU-resolved KV construction inputs. Built before any weight upload so
/// malformed adaptive / V / compat configs never touch device memory.
struct Qwen35KvPlan {
    pair: kv_mode::KvPair,
    is_kv_layer: Vec<bool>,
    dims: KvDims,
    /// `None` = static path. `Some` carries a fully-built controller (floors
    /// authoritative) ready for adaptive KV construction.
    adaptive: Option<KvAdaptive>,
}

/// All CPU-only validation that must precede `load_weights`.
fn plan_qwen35_gpu_stages(config: &Qwen35Config, ctx: &LoadCtx) -> Result<Qwen35KvPlan, String> {
    let mode_raw = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_mode.clone());

    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::FullAttention)
        .collect();

    // `kv_v_override` is admission's resolved V (typed > HIPFIRE_KV_V); never
    // re-read the env here or VMM sizing and the loaded pair can diverge.
    let k_raw = ctx.kv_k_override.filter(|s| !s.is_empty());
    let v_raw = ctx.kv_v_override.filter(|s| !s.is_empty());
    let kv_adaptive_spec = ctx
        .kv_adaptive_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| hipfire_runtime::config::get().kv_adaptive.clone());
    let adaptive_req = parse_kv_adaptive(&kv_adaptive_spec)?;
    let auto_native = kv_mode::qwen35_native_auto_eligible(
        ctx.gpu.arch.as_str(),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        ctx.pp,
        adaptive_req.is_some(),
        ctx.cask.sidecar.is_some(),
    );
    let policy =
        kv_mode::qwen35_policy_for_native(&kv_mode::QWEN35_HFQ_POLICY, &mode_raw, auto_native);

    let pair = kv_mode::resolve_kv_pair(
        &mode_raw,
        k_raw,
        v_raw,
        &policy,
        ctx.gpu.arch.as_str(),
        ctx.qwen_default_q8,
    )
    .map_err(|e| e.to_string())?;
    let mode = pair.k();
    let effective = match pair {
        kv_mode::KvPair::Native(_) => kv_mode::qwen_k_display_name(mode).to_string(),
        kv_mode::KvPair::Split(_, v) => format!(
            "K={} V={}",
            kv_mode::qwen_k_display_name(mode),
            kv_mode::qwen_v_display_name(v),
        ),
    };
    match pair {
        kv_mode::KvPair::Native(_) => {
            eprintln!("  KV cache: requested mode={mode_raw}, effective KV={effective}")
        }
        kv_mode::KvPair::Split(_, v) => eprintln!(
            "  KV cache: requested mode={mode_raw}, effective K={}, effective V={}",
            kv_mode::qwen_k_display_name(mode),
            kv_mode::qwen_v_display_name(v),
        ),
    }
    hipfire_config::load_summary::note(hipfire_config::load_summary::Facet::Kv, effective);

    validate_native_kv_admission(mode, config, ctx, adaptive_req.is_some(), v_raw.is_some())?;
    validate_adaptive_cask_handoff(
        adaptive_req.is_some(),
        ctx.cask.sidecar.is_some(),
        ctx.cask.handoff_tokens,
        ctx.kv_backend,
        ctx.cask.cask_m_folding,
        ctx.max_seq,
    )?;
    // DDTree + CASK/TriAttention eviction (#272): refuse before any allocation.
    // Mirrors `load_dflash_state`'s budget resolution (env wins, else the spec param).
    validate_ddtree_cask(
        ctx.gpu
            .flags
            .ddtree_budget
            .or(ctx.spec.ddtree_budget)
            .unwrap_or(0)
            > 0,
        ctx.draft_path.is_some_and(|d| !d.is_empty()),
        ctx.cask.sidecar.is_some(),
    )?;
    validate_gfx1100_split_verify_kv(
        crate::qwen35::prefill::gfx1100_split_verify_min_ctx(&*ctx.gpu, config).is_some(),
        adaptive_req.is_none()
            && matches!(
                pair,
                kv_mode::KvPair::Native(hipfire_runtime::kv_mode::KvMode::Q8)
                    | kv_mode::KvPair::Split(
                        hipfire_runtime::kv_mode::KvMode::Q8,
                        llama::VMode::Q8
                    )
            ),
        ctx.kv_backend,
    )?;

    // Adaptive DFlash is out of scope: refuse explicit adaptive + DFlash draft.
    if adaptive_req.is_some() {
        if let Some(draft) = ctx.draft_path {
            if !draft.is_empty() {
                return Err(
                    "kv_adaptive is incompatible with DFlash (adaptive×DFlash is out of scope); \
                     disable one of HIPFIRE_KV_ADAPTIVE / draft path"
                        .into(),
                );
            }
        }
        if ctx.pp > 1 {
            return Err("kv_adaptive requires single-GPU (pp=1)".into());
        }
        if v_raw.is_some() {
            return Err(
                "kv_adaptive cannot combine with an explicit V override \
                 (typed kv_v / HIPFIRE_KV_V); adaptive always starts V=q8 and downshifts via the controller"
                    .into(),
            );
        }
        if config.head_dim != 256 {
            return Err(format!(
                "kv_adaptive requires head_dim=256 (got {})",
                config.head_dim
            ));
        }
    }

    let physical_cap = ctx.cask.physical_cap(ctx.max_seq)?;
    let dims = KvDims {
        layers: KvLayers::Mask(is_kv_layer.clone()),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(physical_cap),
    };

    if let Some((preset, k_floor, v_floor)) = adaptive_req {
        // Build controller first — floors and thresholds are authoritative.
        let mut ad = match preset {
            Some(p) => KvAdaptive::from_preset(p, ctx.max_seq, config.n_kv_heads, config.head_dim),
            None => KvAdaptive::new(
                ctx.max_seq,
                config.n_kv_heads,
                config.head_dim,
                k_floor,
                v_floor,
            ),
        };
        if ad.current_cap() < hipfire_runtime::llama::PREFILL_MAX_BATCH {
            return Err(format!(
                "kv_adaptive={kv_adaptive_spec}: max_seq={} too small: start-tier capacity {} < prefill chunk {}",
                ctx.max_seq,
                ad.current_cap(),
                hipfire_runtime::llama::PREFILL_MAX_BATCH,
            ));
        }
        if ctx.cask.sidecar.is_some() {
            ad.configure_eviction_handoff(ctx.cask.handoff_tokens);
        }
        Ok(Qwen35KvPlan {
            pair,
            is_kv_layer,
            dims,
            adaptive: Some(ad),
        })
    } else {
        // Static path: final K mode + optional Lloyd-V known before allocation.
        validate_cask_static_layout(ctx.cask.sidecar.is_some(), pair)?;
        if let kv_mode::KvPair::Split(_, static_v) = pair {
            if !matches!(static_v, llama::VMode::Q8)
                && !matches!(
                    mode,
                    kv_mode::KvMode::Fwht2 | kv_mode::KvMode::Fwht3 | kv_mode::KvMode::Fwht4
                )
            {
                return Err(format!(
                    "V={} requires an FWHT K mode (fwht2/3/4); resolved mode is {:?} (K={})",
                    kv_mode::qwen_v_display_name(static_v),
                    mode,
                    kv_mode::qwen_k_display_name(mode)
                ));
            }
        }
        Ok(Qwen35KvPlan {
            pair,
            is_kv_layer,
            dims,
            adaptive: None,
        })
    }
}

/// Admission for native fp8-E4M3 / flat-bf16 KV on the Qwen route. Authored
/// native presets need `kv_mode::qwen35_native_supported` (exact gfx1201,
/// single GPU, non-adaptive, no CASK eviction, and a mode-specific exact
/// geometry: fp8 H24/Hkv4/D256 or H16/Hkv2/D256, bf16 H24/Hkv4/D256) plus
/// neutral V. Implicit auto is gated separately by
/// `kv_mode::qwen35_native_auto_eligible` and instead selects Q8.
pub fn validate_native_kv_admission(
    mode: hipfire_runtime::kv_mode::KvMode,
    config: &Qwen35Config,
    ctx: &LoadCtx,
    adaptive: bool,
    authored_v: bool,
) -> Result<(), String> {
    use hipfire_runtime::kv_mode::KvMode;
    if !matches!(mode, KvMode::Fp8 | KvMode::Bf16) {
        return Ok(());
    }
    if let Some(why) = kv_mode::qwen35_native_refusal(
        mode,
        ctx.gpu.arch.as_str(),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        ctx.pp,
        adaptive,
        ctx.cask.sidecar.is_some(),
    ) {
        return Err(why);
    }
    if authored_v {
        return Err(format!(
            "kv_mode {mode:?} requires neutral V (no typed kv_v / HIPFIRE_KV_V); native Fp8/Bf16 pairs are indivisible"
        ));
    }
    Ok(())
}

/// #272: DDTree keeps a CPU `target_hidden_host` shadow that eviction does not
/// compact (`position` shrinks, the shadow keeps the pre-eviction rows), which
/// panics at `target_hidden_host size mismatches position`. CASK/TriAttention
/// eviction is deprecated since 0.4.0, so the combination is refused instead
/// of growing a host-mirror contract. `ddtree` is "budget > 0"; it only matters
/// with a DFlash draft attached.
pub fn validate_ddtree_cask(ddtree: bool, has_draft: bool, cask: bool) -> Result<(), String> {
    if ddtree && has_draft && cask {
        return Err(
            "DDTree (speculation.ddtree_budget / HIPFIRE_DDTREE_BUDGET > 0) cannot be combined with \
             CASK/TriAttention eviction (memory.cask.sidecar): eviction does not compact DDTree's \
             hidden-state shadow and the run would panic (#272). CASK is deprecated since 0.4.0; \
             drop the CASK sidecar or set ddtree_budget = 0 (chain DFlash)."
                .into(),
        );
    }
    Ok(())
}

fn validate_adaptive_cask_handoff(
    adaptive: bool,
    sidecar: bool,
    handoff_tokens: usize,
    backend: KvBackend,
    cask_m_folding: bool,
    max_seq: usize,
) -> Result<(), String> {
    if !adaptive {
        return if handoff_tokens == 0 {
            Ok(())
        } else {
            Err("memory.cask.handoff_tokens requires memory.kv_adaptive != off".into())
        };
    }
    if !sidecar {
        return if handoff_tokens == 0 {
            Ok(())
        } else {
            Err(
                "memory.cask.handoff_tokens requires memory.cask.sidecar (or auto-attached sidecar)"
                    .into(),
            )
        };
    }
    if handoff_tokens == 0 {
        return Err(
            "kv_adaptive and CASK require memory.cask.handoff_tokens > 0 for an explicit one-way handoff"
                .into(),
        );
    }
    if backend != KvBackend::Vmm {
        return Err("kv_adaptive -> CASK handoff currently requires memory.kv_backend=vmm".into());
    }
    if cask_m_folding {
        return Err(
            "kv_adaptive -> CASK handoff supports plain TriAttention eviction only; CASK m-folding needs FWHT/Lloyd fold kernels"
                .into(),
        );
    }
    if handoff_tokens > max_seq {
        return Err(format!(
            "memory.cask.handoff_tokens={handoff_tokens} exceeds max_seq={max_seq} and can never activate"
        ));
    }
    Ok(())
}

/// `kernel.gfx1100_fa2_split_verify` admits VMM KV only. When the opt-in
/// applies to this model (`split_envelope`: exact gfx1100, dense
/// H24/KV4/D256, parent `kernel.verify_attn` on) and the cache is plain Q8,
/// a legacy backend is refused before any upload rather than serving the
/// split route from a legacy cache or silently dropping the opt-in.
#[doc(hidden)]
pub fn validate_gfx1100_split_verify_kv(
    split_envelope: bool,
    plain_q8: bool,
    backend: KvBackend,
) -> Result<(), String> {
    if split_envelope && plain_q8 && backend != KvBackend::Vmm {
        return Err(
            "kernel.gfx1100_fa2_split_verify admits VMM KV only, but this load resolved \
             memory.kv_backend=legacy; use memory.kv_backend=vmm or set \
             kernel.gfx1100_fa2_split_verify=false"
                .into(),
        );
    }
    Ok(())
}

/// The static CASK path remains limited to Q8 V and Givens legacy-asym K.
/// Adaptive FWHT/Lloyd layouts use the separately validated plain-TriAttention
/// handoff; m-folding still lacks the corresponding fold/requant kernels.
fn validate_cask_static_layout(cask_enabled: bool, pair: kv_mode::KvPair) -> Result<(), String> {
    if !cask_enabled {
        return Ok(());
    }
    if !matches!(pair, kv_mode::KvPair::Split(_, llama::VMode::Q8)) {
        return Err(format!(
            "CASK currently requires split V=q8 (resolved KV is {pair:?})"
        ));
    }
    if !matches!(
        pair.k(),
        hipfire_runtime::kv_mode::KvMode::Q8
            | hipfire_runtime::kv_mode::KvMode::Asym2
            | hipfire_runtime::kv_mode::KvMode::Asym3
            | hipfire_runtime::kv_mode::KvMode::Asym4
    ) {
        return Err(format!(
            "CASK currently supports Q8/legacy-asym2/legacy-asym3/legacy-asym4 K only \
             (resolved mode is {:?}); FWHT/Lloyd combinations are rejected — use legacy-asym3",
            pair.k()
        ));
    }
    Ok(())
}

fn construct_kv_cache(
    config: &Qwen35Config,
    ctx: &mut LoadCtx,
    plan: Qwen35KvPlan,
) -> Result<(KvCache, Option<KvAdaptive>), String> {
    if let Some(ad) = plan.adaptive {
        // Floors come from the controller, not separately parsed hints.
        let k_floor = ad.k_floor;
        let v_floor = ad.v_floor;
        // Adaptive always starts exactly FWHT4/Q8 regardless of resolved static mode.
        let start_mode = hipfire_runtime::kv_mode::KvMode::Fwht4;
        let k_floor_bph = k_floor.bytes_per_head(config.head_dim);

        let kv = match ctx.kv_backend {
            KvBackend::Vmm => {
                // Floor-reserved VMM arenas; current encoding FWHT4/Q8.
                KvCache::new_gpu_vmm_adaptive_filtered(
                    ctx.gpu,
                    &plan.is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    ctx.max_seq,
                    k_floor_bph,
                    v_floor,
                )
                .map_err(|e| format!("{e}"))?
            }
            KvBackend::Legacy => {
                // Legacy: allocate start-tier FWHT4/Q8 then floor-resize in place.
                // If floor-resize fails, free the start-tier cache explicitly.
                let mut kv = <KvCache as KvCacheExt>::from_mode_with_backend(
                    start_mode,
                    KvBackend::Legacy,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?;
                if let Err(e) = kv.set_adaptive_floor_alloc(ctx.gpu, v_floor, k_floor_bph) {
                    let cleanup = kv.free_gpu(ctx.gpu).map_err(|fe| fe.to_string());
                    return Err(append_cleanup_context(format!("{e}"), cleanup));
                }
                kv
            }
        };

        eprintln!(
            "[adaptive-kv] engaged: backend={:?} pattern={:?} k_floor={:?} v_floor={:?} thresholds={:?} start_cap={} (max_seq={}, reserve at floor, start FWHT4/Q8)",
            ctx.kv_backend,
            ad.steps,
            ad.k_floor,
            ad.v_floor,
            ad.thresholds,
            ad.current_cap(),
            ctx.max_seq,
        );
        Ok((kv, Some(ad)))
    } else {
        let mode = plan.pair.k();
        let physical_cap = plan
            .dims
            .physical_cap
            .expect("Qwen3.5 KV plan always resolves physical_cap");
        let kv = match (ctx.kv_backend, plan.pair) {
            (KvBackend::Vmm, kv_mode::KvPair::Native(_)) => {
                <KvCache as KvCacheExt>::from_mode_with_backend(
                    mode,
                    KvBackend::Vmm,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?
            }
            (KvBackend::Vmm, kv_mode::KvPair::Split(_, vm)) => {
                KvCache::new_gpu_vmm_capped_filtered(
                    ctx.gpu,
                    &plan.is_kv_layer,
                    config.n_kv_heads,
                    config.head_dim,
                    ctx.max_seq,
                    physical_cap,
                    mode,
                    vm,
                )
                .map_err(|e| format!("{e}"))?
            }
            (KvBackend::Legacy, kv_mode::KvPair::Native(_))
            | (KvBackend::Legacy, kv_mode::KvPair::Split(_, llama::VMode::Q8)) => {
                <KvCache as KvCacheExt>::from_mode_with_backend(
                    mode,
                    KvBackend::Legacy,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?
            }
            (KvBackend::Legacy, kv_mode::KvPair::Split(_, vm)) => {
                let mut kv = <KvCache as KvCacheExt>::from_mode_with_backend(
                    mode,
                    KvBackend::Legacy,
                    KvTarget::Single(ctx.gpu),
                    &plan.dims,
                )
                .map_err(|e| format!("{e}"))?;
                if let Err(e) = kv.set_v_mode_realloc(ctx.gpu, vm) {
                    let cleanup = kv.free_gpu(ctx.gpu).map_err(|fe| fe.to_string());
                    return Err(append_cleanup_context(format!("{e}"), cleanup));
                }
                kv
            }
        };
        if let kv_mode::KvPair::Split(_, vm) = plan.pair {
            if !matches!(vm, llama::VMode::Q8) {
                eprintln!(
                    "[hipfire-arch-qwen35] V-cache mode override → {} (256-wide lloyd-V on fwht K={})",
                    kv_mode::qwen_v_display_name(vm),
                    kv_mode::qwen_k_display_name(mode)
                );
            }
        }
        Ok((kv, None))
    }
}

pub fn free_qwen35_bundle(bundle: Qwen35Bundle, gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    let Qwen35Bundle {
        config: _,
        weights,
        scratch,
        kv_cache,
        dn_state,
        kv_adaptive: _,
        pp_scratch_set,
        vision_config: _,
        vision_weights,
        qwen35_decode_batch,
        vmm_store,
        weight_lease,
    } = bundle;
    debug_assert!(
        pp_scratch_set.is_none(),
        "free_qwen35_bundle: pp_scratch_set must be None on single-GPU free"
    );
    let _ = pp_scratch_set;
    if let Some(store) = vmm_store {
        if let Err(e) = store.free_gpu(gpu) {
            eprintln!("free_qwen35_bundle: VMM batch store free: {e}");
        }
    }
    if let Some(batch) = qwen35_decode_batch {
        let _ = batch.free_gpu(gpu);
    }
    // Match unload_model Qwen35 order: kv → scratch → weights → dn → vision.
    let mut first: Option<String> = None;
    if let Err(e) = kv_cache.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    scratch.free_gpu(gpu);
    weights.free_gpu(gpu);
    dn_state.free_gpu(gpu);
    if let Some(vw) = vision_weights {
        vw.free_gpu(gpu);
    }
    let vmm = note_vmm_after_free(gpu);
    let result = match (first, vmm) {
        (None, Ok(())) => Ok(()),
        (Some(e), Ok(())) => Err(e),
        (None, Err(c)) => Err(c),
        (Some(e), Err(c)) => Err(format!("{e}; cleanup also failed: {c}")),
    };
    match (weight_lease, result) {
        (None, result) => result,
        (Some(lease), Ok(())) => {
            // Captured graphs embed imported addresses; drop them before
            // telling the host this tenant holds no references.
            gpu.invalidate_graph_state();
            lease.release()
        }
        (Some(lease), Err(e)) => {
            drop(lease);
            Err(format!(
                "{e}; weight-host lease closed without release (teardown not proven clean)"
            ))
        }
    }
}

/// Live lease on one weight-host generation, held by [`Qwen35Bundle`] for as
/// long as imported weights may be referenced. Dropping it closes the lease
/// without a release acknowledgement: the host then keeps the lease until this
/// process exits.
pub struct SharedWeightLease {
    #[cfg(target_os = "linux")]
    inner: hipfire_runtime::weight_host::WeightHostLease,
    #[cfg(not(target_os = "linux"))]
    _never: std::convert::Infallible,
}

impl SharedWeightLease {
    /// `Ok(false)` once the host closed the lease (drain or exit).
    pub fn host_alive(&self) -> Result<bool, String> {
        #[cfg(target_os = "linux")]
        return self.inner.host_alive().map_err(|e| e.to_string());
        #[cfg(not(target_os = "linux"))]
        match self._never {}
    }

    /// Send `Release` (closing every allocation FD first). Call only after
    /// every imported owner is freed and VMM teardown reported clean.
    pub fn release(self) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            let mut inner = self.inner;
            inner.close_allocation_fds();
            inner
                .release()
                .map_err(|e| format!("weight-host release: {e}"))
        }
        #[cfg(not(target_os = "linux"))]
        match self._never {}
    }
}

/// `HIPFIRE_WEIGHT_HOST` opt-in, read at bootstrap (it selects the GPU claim
/// before the process config exists). `Ok(None)`: unset or the `0` kill
/// switch, incumbent private loading. `Ok(Some(socket))`: import from the
/// weight host at `HIPFIRE_WEIGHT_HOST_SOCKET`. Anything else fails closed.
pub fn weight_host_socket() -> Result<Option<std::path::PathBuf>, String> {
    #[cfg(target_os = "linux")]
    {
        hipfire_runtime::weight_host::weight_host_socket_from_env().map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    match std::env::var("HIPFIRE_WEIGHT_HOST").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => Ok(None),
        Ok("1") => Err("HIPFIRE_WEIGHT_HOST=1 is supported on Linux only".into()),
        _ => Err("HIPFIRE_WEIGHT_HOST must be unset, 0 or 1".into()),
    }
}

/// Failed load after an import: release the lease only when the imported
/// owners are gone and VMM teardown is clean, else close it unacknowledged.
fn release_after_failed_load(
    error: String,
    lease: Option<SharedWeightLease>,
    gpu: &mut rdna_compute::Gpu,
) -> String {
    let Some(lease) = lease else {
        return error;
    };
    match note_vmm_after_free(gpu) {
        Ok(()) => match lease.release() {
            Ok(()) => error,
            Err(release) => format!("{error}; {release}"),
        },
        Err(cleanup) => format!("{error}; {cleanup}; weight-host lease closed without release"),
    }
}

/// Weight-host v1 envelope for an import: single device, plain dense decode.
/// Checked before connecting, so an unsupported request never takes a lease.
fn refuse_outside_weight_host_envelope(ctx: &LoadCtx) -> Result<(), String> {
    let refuse = |what: &str| Err(format!("HIPFIRE_WEIGHT_HOST=1 does not support {what}"));
    if ctx.pp != 1 {
        return refuse("pp > 1");
    }
    if ctx.draft_path.is_some() || ctx.gemma4_drafter_path.is_some() {
        return refuse("a draft model");
    }
    if ctx.mtp_path.is_some() || ctx.spec.mtp == Some(true) {
        return refuse("MTP");
    }
    if ctx.spec.dflash == Some(true) || ctx.spec.dspark == Some(true) {
        return refuse("DFlash/DSpark");
    }
    if ctx.vision_path.is_some() {
        return refuse("a vision tower");
    }
    if ctx.gpu.arch != "gfx1201" {
        return refuse(&format!("{} (v1 is gfx1201 only)", ctx.gpu.arch));
    }
    Ok(())
}

/// Card identity both sides of the handshake compute the same way: HIP UUID,
/// PCI bus ID and architecture of the bound device.
#[cfg(target_os = "linux")]
pub fn weight_host_card_identity(
    gpu: &rdna_compute::Gpu,
) -> Result<hipfire_runtime::weight_host::CardIdentity, String> {
    let uuid = gpu
        .hip
        .device_uuid(gpu.device_id)
        .map_err(|e| format!("card UUID: {e}"))?;
    let pci_bus_id = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .map_err(|e| format!("card PCI bus ID: {e}"))?;
    Ok(hipfire_runtime::weight_host::CardIdentity {
        uuid,
        pci_bus_id,
        device_arch: gpu.arch.clone(),
    })
}

/// Source certificate the host publishes beside its socket (same private
/// directory): `<socket>.source-cert`.
pub fn weight_host_source_certificate_path(socket: &std::path::Path) -> std::path::PathBuf {
    let mut name = socket
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".source-cert");
    socket.with_file_name(name)
}

#[cfg(target_os = "linux")]
mod loader_abi_generated {
    include!(concat!(env!("OUT_DIR"), "/loader_abi.rs"));
}

/// `LoaderAbiSha256` inputs of this build (build.rs: loader closure source
/// digests, quant representation constants, CPU toolchain and features).
/// Forward kernels are not inputs, so compute-only builds share one ABI.
#[cfg(target_os = "linux")]
pub fn loader_abi() -> hipfire_runtime::weight_host::LoaderAbi {
    use hipfire_runtime::weight_host::{FieldValue, LoaderAbi, Sha256Digest};
    let mut abi = LoaderAbi::new("hipfire.qwen35.dense.loader-abi/v1").with(
        "representation_tag",
        FieldValue::str(crate::shared_weights::DENSE_REPRESENTATION_TAG),
    );
    for (name, digest) in loader_abi_generated::LOADER_ABI_COMPONENTS {
        abi.components
            .insert((*name).to_owned(), Sha256Digest(*digest));
    }
    abi
}

/// `ModelContentSha256` of the base HFQ, the only v1 source.
#[cfg(target_os = "linux")]
pub fn weight_host_source_sha256(
    cert: &hipfire_runtime::weight_host::SourceCertificate,
) -> Result<hipfire_runtime::weight_host::Sha256Digest, String> {
    hipfire_runtime::weight_host::model_content_sha256(&[cert.to_source_digest()])
        .map_err(|e| e.to_string())
}

/// Diagnostics-only provenance: never compared or hashed.
#[cfg(target_os = "linux")]
pub fn weight_host_provenance(role: &str) -> hipfire_runtime::weight_host::BuildProvenance {
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    hipfire_runtime::weight_host::BuildProvenance {
        label: format!("{role} {exe} pid {}", std::process::id()),
        ..Default::default()
    }
}

#[cfg(target_os = "linux")]
fn import_from_weight_host(
    socket: &std::path::Path,
    hfq: &HfqFile,
    config: &Qwen35Config,
    ctx: &mut LoadCtx,
) -> Result<(Qwen35Weights, SharedWeightLease), String> {
    use hipfire_runtime::weight_host as wh;
    refuse_outside_weight_host_envelope(ctx)?;
    let connect_started = std::time::Instant::now();
    let admission_token = std::env::var("HIPFIRE_FABRIC_ADMISSION").map_err(|_| {
        "HIPFIRE_WEIGHT_HOST=1 needs HIPFIRE_FABRIC_ADMISSION (dev-fabric correct-class job)"
            .to_string()
    })?;
    let card = weight_host_card_identity(ctx.gpu)?;
    let cert_path = weight_host_source_certificate_path(socket);
    // Identity check only (fstat), no content scan: the host hashed once.
    let (_source, cert) =
        wh::SourceCertificate::verify_source_path(&cert_path, std::path::Path::new(ctx.path))
            .map_err(|e| {
                format!(
                    "weight host source certificate {}: {e}",
                    cert_path.display()
                )
            })?;
    let source_sha256 = weight_host_source_sha256(&cert)?;
    crate::shared_weights::apply_source_dispatch_flags(ctx.gpu, hfq);
    let recipe =
        crate::shared_weights::resolve_dense_recipe(ctx.gpu, config).map_err(|e| e.to_string())?;
    let hello = wh::HostHello {
        card,
        source_sha256,
        loader_abi_sha256: loader_abi().sha256().map_err(|e| e.to_string())?,
        recipe,
        admission_token,
        provenance: weight_host_provenance("tenant"),
        options: wh::ConnectOptions {
            handshake_timeout: std::time::Duration::from_secs(30),
            expected_host_pid: None,
        },
    };
    let mut lease = wh::WeightHostClient::connect(socket, hello)
        .map_err(|e| format!("weight host {}: {e}", socket.display()))?;
    let connect_ms = connect_started.elapsed().as_secs_f64() * 1e3;
    let import_started = std::time::Instant::now();
    // Failures before ImportAck drop the lease unacknowledged (Release is only
    // valid after an ack); the host keeps it charged until this process exits.
    let weights = crate::shared_weights::import_dense_weights(ctx.gpu, config, &lease)
        .map_err(|e| format!("weight host import (rolled back): {e}"))?;
    let imported_bytes = crate::shared_weights::dense_imported_physical_bytes(lease.manifest());
    if let Err(e) = lease.ack_import(imported_bytes) {
        weights.free_gpu(ctx.gpu);
        return Err(append_cleanup_context(
            format!("weight host import ack: {e}"),
            note_vmm_after_free(ctx.gpu),
        ));
    }
    let import_map_ms = import_started.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "WEIGHT_HOST_IMPORT epoch={} layout_sha256={} imported_bytes={imported_bytes} fds={} connect_ms={connect_ms:.3} import_map_ms={import_map_ms:.3} ack=ok",
        lease.epoch(),
        lease.layout_sha256(),
        lease.allocation_fds().len(),
    );
    Ok((weights, SharedWeightLease { inner: lease }))
}

#[cfg(not(target_os = "linux"))]
fn import_from_weight_host(
    _socket: &std::path::Path,
    _hfq: &HfqFile,
    _config: &Qwen35Config,
    _ctx: &mut LoadCtx,
) -> Result<(Qwen35Weights, SharedWeightLease), String> {
    Err("HIPFIRE_WEIGHT_HOST=1 is supported on Linux only".into())
}

fn free_kv_and_weights(
    kv: KvCache,
    weights: Qwen35Weights,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut first: Option<String> = None;
    if let Err(e) = kv.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    weights.free_gpu(gpu);
    match first {
        Some(e) => Err(append_cleanup_context(e, note_vmm_after_free(gpu))),
        None => note_vmm_after_free(gpu),
    }
}

fn free_dn_kv_and_weights(
    dn: DeltaNetState,
    kv: KvCache,
    weights: Qwen35Weights,
    gpu: &mut rdna_compute::Gpu,
) -> Result<(), String> {
    let mut first: Option<String> = None;
    if let Err(e) = kv.free_gpu(gpu) {
        first = Some(e.to_string());
    }
    dn.free_gpu(gpu);
    weights.free_gpu(gpu);
    match first {
        Some(e) => Err(append_cleanup_context(e, note_vmm_after_free(gpu))),
        None => note_vmm_after_free(gpu),
    }
}

fn note_vmm_after_free(gpu: &mut rdna_compute::Gpu) -> Result<(), String> {
    gpu.ensure_vmm_cleaned().map_err(|e| {
        format!("pending VMM teardown after free ({e}); retry unload or restart the process")
    })
}

/// Preserve the primary operation error; append cleanup failure context when present.
fn append_cleanup_context(op_err: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => op_err,
        Err(c) => format!("{op_err}; cleanup also failed: {c}"),
    }
}

// ─── Helper: StateQuant parsing ─────────────────────────────────────

fn parse_state_quant(mode: Option<&str>) -> Result<StateQuant, String> {
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

fn state_quant_label(q: StateQuant) -> &'static str {
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

// ─── Helper: parameter count + tiny-model warning ─────────────────

fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

fn warn_tiny_model_state(hfq: &HfqFile, q: StateQuant) {
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS && q != StateQuant::FP32 {
        eprintln!(
            "  warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {})",
            params as f64 / 1.0e9,
            state_quant_label(q)
        );
    }
}

// ─── Helper: KV adaptive parsing ──────────────────────────────────

/// Parse adaptive policy. `Ok(None)` = off. Malformed/unsupported explicit
/// requests are hard errors (no silent ignore).
fn parse_kv_adaptive(
    s: &str,
) -> Result<
    Option<(
        Option<Preset>,
        hipfire_runtime::kv_adaptive::KMode,
        hipfire_runtime::llama::VMode,
    )>,
    String,
> {
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;
    match s {
        "" | "off" => Ok(None),
        // Floors agree with KvAdaptive::from_preset: C=F4/L4, B=F3/L3, A=F2/L2.
        "conservative" => Ok(Some((
            Some(Preset::Conservative),
            KMode::Fwht4,
            VMode::Lloyd4,
        ))),
        "balanced" => Ok(Some((Some(Preset::Balanced), KMode::Fwht3, VMode::Lloyd3))),
        "aggressive" => Ok(Some((
            Some(Preset::Aggressive),
            KMode::Fwht2,
            VMode::Lloyd2,
        ))),
        other if other.starts_with("advanced:") => {
            let spec = &other["advanced:".len()..];
            let mut k = None;
            let mut v = None;
            for kvp in spec.split(',') {
                let mut it = kvp.splitn(2, '=');
                match (it.next(), it.next()) {
                    (Some("k"), Some("fwht4")) => k = Some(KMode::Fwht4),
                    (Some("k"), Some("fwht3")) => k = Some(KMode::Fwht3),
                    (Some("k"), Some("fwht2")) => k = Some(KMode::Fwht2),
                    (Some("v"), Some("lloyd4")) => v = Some(VMode::Lloyd4),
                    (Some("v"), Some("lloyd3")) => v = Some(VMode::Lloyd3),
                    (Some("v"), Some("lloyd2")) => v = Some(VMode::Lloyd2),
                    (Some(key), Some(val)) => {
                        return Err(format!(
                            "kv_adaptive='{other}' unknown key/value {key}={val} \
                             (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "kv_adaptive='{other}' malformed \
                             (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                        ));
                    }
                }
            }
            match (k, v) {
                (Some(k), Some(v)) => Ok(Some((None, k, v))),
                _ => Err(format!(
                    "kv_adaptive='{other}' incomplete \
                     (expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>)"
                )),
            }
        }
        other => Err(format!(
            "kv_adaptive='{other}' unknown \
             (expected off|conservative|balanced|aggressive|advanced:k=..,v=..)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use hipfire_runtime::llama::VMode;

    #[test]
    fn ddtree_with_cask_is_refused_up_front() {
        let e = validate_ddtree_cask(true, true, true).unwrap_err();
        assert!(
            e.contains("DDTree") && e.contains("CASK") && e.contains("#272"),
            "{e}"
        );
        // Each leg alone, or no draft (ddtree inert), is accepted.
        assert!(validate_ddtree_cask(true, true, false).is_ok());
        assert!(validate_ddtree_cask(false, true, true).is_ok());
        assert!(validate_ddtree_cask(true, false, true).is_ok());
        assert!(validate_ddtree_cask(false, false, false).is_ok());
    }
    #[test]
    fn parse_presets_agree_with_controller_floors() {
        let cases = [
            (
                "conservative",
                Preset::Conservative,
                KMode::Fwht4,
                VMode::Lloyd4,
            ),
            ("balanced", Preset::Balanced, KMode::Fwht3, VMode::Lloyd3),
            (
                "aggressive",
                Preset::Aggressive,
                KMode::Fwht2,
                VMode::Lloyd2,
            ),
        ];
        for (spec, preset, k, v) in cases {
            let parsed = parse_kv_adaptive(spec).unwrap().unwrap();
            assert_eq!(parsed.0, Some(preset));
            assert_eq!(parsed.1, k);
            assert_eq!(parsed.2, v);
            let ad = KvAdaptive::from_preset(preset, 10_000, 4, 256);
            assert_eq!(
                (ad.k_floor, ad.v_floor),
                (k, v),
                "parse floors must match KvAdaptive::from_preset for {spec}"
            );
        }
    }

    #[test]
    fn parse_off_and_errors() {
        assert!(parse_kv_adaptive("off").unwrap().is_none());
        assert!(parse_kv_adaptive("").unwrap().is_none());
        assert!(parse_kv_adaptive("nope").is_err());
        assert!(parse_kv_adaptive("advanced:k=fwht2").is_err());
        assert!(parse_kv_adaptive("advanced:k=fwht9,v=lloyd2").is_err());
    }

    /// Malformed adaptive must fail in the CPU parse/plan seam — before any
    /// weight upload would run. Observable contract: pure helpers error with
    /// no GPU required.
    #[test]
    fn malformed_adaptive_fails_in_cpu_plan() {
        // Direct plan path: parse_kv_adaptive is the CPU seam for malformed specs.
        let err = parse_kv_adaptive("not-a-preset").unwrap_err();
        assert!(err.contains("unknown"), "{err}");
        let err = parse_kv_adaptive("advanced:k=fwht2").unwrap_err();
        assert!(
            err.contains("incomplete") || err.contains("malformed"),
            "{err}"
        );
    }

    #[test]
    fn adaptive_cask_handoff_compatibility_matrix_is_fail_closed() {
        let check = |adaptive, sidecar, handoff, backend, folding| {
            validate_adaptive_cask_handoff(adaptive, sidecar, handoff, backend, folding, 2048)
        };

        assert!(check(true, true, 128, KvBackend::Vmm, false).is_ok());
        assert!(check(true, true, 0, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("handoff_tokens > 0"));
        assert!(check(true, true, 128, KvBackend::Legacy, false)
            .unwrap_err()
            .contains("requires memory.kv_backend=vmm"));
        assert!(check(true, true, 128, KvBackend::Vmm, true)
            .unwrap_err()
            .contains("m-folding"));
        assert!(check(true, false, 128, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("requires memory.cask.sidecar"));
        assert!(check(false, false, 128, KvBackend::Vmm, false)
            .unwrap_err()
            .contains("requires memory.kv_adaptive"));
        assert!(
            validate_adaptive_cask_handoff(true, true, 2049, KvBackend::Vmm, false, 2048)
                .unwrap_err()
                .contains("exceeds max_seq")
        );
    }

    #[test]
    fn gfx1100_split_verify_refuses_legacy_kv() {
        assert!(validate_gfx1100_split_verify_kv(true, true, KvBackend::Vmm).is_ok());
        assert!(
            validate_gfx1100_split_verify_kv(true, true, KvBackend::Legacy)
                .unwrap_err()
                .contains("admits VMM KV only")
        );
        // Outside the opt-in envelope or off plain Q8 the route never runs,
        // so the backend is not this check's concern.
        assert!(validate_gfx1100_split_verify_kv(false, true, KvBackend::Legacy).is_ok());
        assert!(validate_gfx1100_split_verify_kv(true, false, KvBackend::Legacy).is_ok());
    }

    #[test]
    fn append_cleanup_preserves_primary_error() {
        let s = append_cleanup_context("kv failed".into(), Ok(()));
        assert_eq!(s, "kv failed");
        let s = append_cleanup_context("kv failed".into(), Err("pending VMM teardown".into()));
        assert!(s.starts_with("kv failed; cleanup also failed:"), "{s}");
        assert!(s.contains("pending VMM"), "{s}");
    }

    #[test]
    fn cask_static_layout_accepts_only_current_compaction_formats() {
        use hipfire_runtime::kv_mode::{KvMode, KvPair};

        for mode in [KvMode::Q8, KvMode::Asym2, KvMode::Asym3, KvMode::Asym4] {
            validate_cask_static_layout(true, KvPair::Split(mode, VMode::Q8)).unwrap();
        }
        for mode in [KvMode::Fwht2, KvMode::Fwht3, KvMode::Fwht4] {
            let err =
                validate_cask_static_layout(true, KvPair::Split(mode, VMode::Q8)).unwrap_err();
            assert!(err.contains("legacy-asym"), "{err}");
        }
        let err = validate_cask_static_layout(true, KvPair::Split(KvMode::Fwht3, VMode::Lloyd3))
            .unwrap_err();
        assert!(err.contains("V=q8"), "{err}");
        assert!(
            validate_cask_static_layout(true, KvPair::Native(KvMode::Fp8))
                .unwrap_err()
                .contains("V=q8")
        );
    }

    #[test]
    fn cask_layout_guard_is_inert_when_cask_is_disabled() {
        use hipfire_runtime::kv_mode::{KvMode, KvPair};

        validate_cask_static_layout(false, KvPair::Split(KvMode::Fwht3, VMode::Lloyd3)).unwrap();
    }

    /// The draft-reserve arithmetic, pinned without a GPU: one file reserves
    /// file + file/8 + 512 MiB, and MTP + DFlash charge independently and add.
    #[test]
    fn draft_reserve_charges_mtp_and_dflash() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-draft-reserve-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let write_sized = |name: &str, len: u64| {
            let path = dir.join(name);
            std::fs::write(&path, vec![0u8; len as usize]).expect("temp file");
            path
        };
        let file_len = 8 << 20;
        let mtp = write_sized("model.mtp", file_len);
        let dflash = write_sized("draft.hfq", file_len);
        let expect = file_len + file_len / 8 + (512 << 20);
        assert_eq!(draft_file_reserve_bytes(&mtp), expect);
        assert_eq!(draft_file_reserve_bytes(&dflash), expect);
        assert_eq!(draft_file_reserve_bytes(&dir.join("missing.mtp")), 0);
        // Each side alone, then both together: the single helper both
        // resolvers call, so a slots load with a DFlash sidecar charges it.
        let mtp_sidecar = MtpHeadLocation::Sidecar(mtp);
        assert_eq!(draft_device_reserve_bytes(Some(&mtp_sidecar), None), expect);
        assert_eq!(draft_device_reserve_bytes(None, Some(&dflash)), expect);
        assert_eq!(
            draft_device_reserve_bytes(Some(&mtp_sidecar), Some(&dflash)),
            expect.saturating_mul(2)
        );
        // A bundled head charges only its appended section, never the trunk.
        let bundled = MtpHeadLocation::Bundled {
            section_bytes: file_len,
        };
        assert_eq!(draft_device_reserve_bytes(Some(&bundled), None), expect);
        assert_eq!(
            draft_device_reserve_bytes(Some(&bundled), Some(&dflash)),
            expect.saturating_mul(2)
        );
        assert_eq!(draft_device_reserve_bytes(None, None), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The placement reserves the MTP head only where the load attaches it:
    /// qwen35 declares no measured AR floor, so `auto` reserves nothing, an
    /// explicit `on` reserves, `off` never. A DFlash draft is charged
    /// independently of this verdict (`draft_device_reserve_bytes`).
    #[test]
    fn placement_reserves_the_mtp_head_only_when_the_load_attaches_it() {
        assert!(!crate::MTP_MEASURED_AR_FLOOR);
        assert!(
            !mtp_head_reserved(None),
            "auto without a floor: no head, no reservation"
        );
        assert!(
            mtp_head_reserved(Some(true)),
            "explicit on reserves the head"
        );
        assert!(!mtp_head_reserved(Some(false)), "off never reserves");
    }

    /// The MTP head probe both resolvers share: the stem sidecar spelling
    /// resolves, the last-extension spelling shadows it, and a bundled
    /// trailer resolves to the appended section size (its MTP container is
    /// disjoint from the trunk index the census walks — see
    /// `resolve_mtp_head_file`).
    #[test]
    fn mtp_head_probe_prefers_sidecar_spelling_then_bundle() {
        let dir =
            std::env::temp_dir().join(format!("hipfire-mtp-head-file-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let trunk = dir.join("model.mq4v2.hfq");
        std::fs::write(&trunk, [0u8; 64]).expect("temp file");
        // No sidecar, no trailer: no head.
        assert!(resolve_mtp_head_file(&trunk).is_none());
        // Stem spelling (`model.mtp`) resolves.
        let stem = dir.join("model.mtp");
        std::fs::write(&stem, [0u8; 64]).expect("temp file");
        assert!(matches!(
            resolve_mtp_head_file(&trunk),
            Some(MtpHeadLocation::Sidecar(p)) if p == stem
        ));
        // The last-extension spelling shadows the stem when both exist.
        let specific = dir.join("model.mq4v2.mtp");
        std::fs::write(&specific, [0u8; 64]).expect("temp file");
        assert!(matches!(
            resolve_mtp_head_file(&trunk),
            Some(MtpHeadLocation::Sidecar(p)) if p == specific
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
