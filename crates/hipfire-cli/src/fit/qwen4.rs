// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free Flash-Next placement bounds, using the carrier's real byte census
//! and placement/host gates. A partial reserve can disprove a load, but cannot
//! promise that the GPU's full reserve will leave the same expert split.

use hipfire_arch_qwen4::expert_hotset as hotsets;
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::weights::Qwen4TierRecipe;
use hipfire_runtime::hfq::HfqFile;
use serde_json::Value;

use super::types::{gib, ArchFit, Finding, FitDevice, FitHost, PlacementPlan};

pub(crate) struct Qwen4Request<'a> {
    pub hfq: &'a HfqFile,
    pub device: &'a FitDevice,
    pub host: &'a FitHost,
    pub max_seq: usize,
    pub params: &'a Value,
}

/// Only exact inventory facts and policy inputs: `decide` reads no environment,
/// filesystem, payload or GPU. The reserve argument is the *base* lower bound;
/// the real placement helpers add their own margin and dedicated stage charge.
struct Inventory {
    non_expert: u64,
    experts: u64,
    layer_experts: u64,
    trunk_layers: usize,
    moe_layers: usize,
    slab_stage: bool,
    force_auto: bool,
    all_host: u64,
    pinned_host: Option<u64>,
    /// An eligible sidecar may replace whole-layer placement. Its live reserve
    /// and tier plan cannot be inferred from a whole-layer bound.
    ranked_hotset: bool,
}

pub(crate) fn analyze(req: &Qwen4Request<'_>) -> ArchFit {
    match analyze_inventory(req) {
        Ok(report) => report,
        // Source admission owns hard artifact/config refusals. This stage must
        // not turn an unavailable byte inventory into a capacity refusal.
        Err(error) => unknown(error),
    }
}

fn analyze_inventory(req: &Qwen4Request<'_>) -> Result<ArchFit, String> {
    let mut receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(req.hfq, Some(&req.device.arch))
        .map_err(|error| error.to_string())?;
    let recipe = &receipt.manifest.identity.recipe;
    let config = &receipt.config;
    // The loader-private qwen4_entry_bytes (qwen4_slab.rs:28-42) is a
    // recipe-arena lookup, not a sizing formula. Read the same public plan.
    // Ordinary data_size equals tensor_data().len(), without faulting payload
    // pages (and remains usable for a sparse, header-only source replica).
    let bytes_of = |entry: &hipfire_runtime::weight_manifest::WeightEntry| {
        if let Qwen4TierRecipe::FlashNextSlab { arenas, .. } = &recipe {
            if entry
                .name
                .ends_with(hipfire_arch_qwen4::expert_arena::ARENA_SUFFIX)
            {
                return entry
                    .layer
                    .and_then(|layer| arenas.layer(layer))
                    .map(|layer| layer.bytes as u64);
            }
        }
        req.hfq
            .find_tensor_info(&entry.name)
            .map(|info| info.data_size as u64)
    };
    let (non_expert, layer_experts) =
        residency::resident_split_ranks(&receipt.manifest.weights, bytes_of, 1)?;
    let experts = residency::routed_expert_bytes_ranks(&receipt.manifest.weights, bytes_of, 1)?;
    let explicit = residency::expert_vram_layers_from_env()?;
    // This invokes the same pin-wins policy without pretending to know whether
    // a lower-bound resident fit is the final resident fit.
    let policy = residency::resolve_expert_vram_layers(explicit, || Ok(false))?;
    let pinned = match policy {
        Some(residency::ExpertVramLayers::Layers(layers)) => Some(layers),
        _ => None,
    };
    let slab_stage = residency::dedicated_slab_stage(
        residency::expert_stage_placement_enabled(&req.device.arch),
        matches!(recipe, Qwen4TierRecipe::FlashNextSlab { .. }),
        false,
    );
    let hotset_policy = hotsets::HotsetPolicy::from_env()?;
    let placement_policy = hotsets::PlacementPolicy::from_env()?;
    let hotset_budget = hotsets::HotsetBudget::from_env()?;
    let hotset_env = hotsets::HotsetEnv {
        gpu_arch: &req.device.arch,
        unified_memory: req.device.unified_memory,
        expert_parallel: false,
        explicit_vram_layers: explicit.is_some(),
        source: match &recipe {
            Qwen4TierRecipe::FlashNextPacked { .. } => hotsets::HotsetSource::Aggregate,
            Qwen4TierRecipe::FlashNextSlab { .. } => hotsets::HotsetSource::SlabArena,
            Qwen4TierRecipe::StockFp8 => hotsets::HotsetSource::StockFp8,
            _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
        },
    };
    let whole_first = hotsets::decide_whole_first(placement_policy, &hotset_env, false);
    let ranked_hotset = whole_first != hotsets::WholeFirstDecision::Active
        && hotsets::preflight(&hotset_policy, req.hfq.path(), &hotset_env).is_ok();
    let mut details = reserve_notes(req.device);
    let mut notes = Vec::new();
    if hotset_budget.is_fixed() {
        notes.push("A pinned hot-expert budget is never widened or shrunk; whether it applies is decided at load.".into());
        details.push("Pinned hot-expert budget: never widened or shrunk; applicability, sidecar binding and the live tier-plan budget are decided at load (expert_hotset.rs:113-138; carriers.rs:2106-2262).".into());
    }
    if whole_first == hotsets::WholeFirstDecision::Active {
        notes.push("Whole-first placement policy: whole-layer placement is kept; the donor-stage plan is decided at load.".into());
        details.push("whole-first policy: whole-layer placement retained; donor-stage/remainder plan decided at load (carriers.rs:2491-2505).".into());
    }
    if let Some(layers) = pinned {
        notes.push(format!("{}={layers} pins the first {layers} expert layers in VRAM; the loader never widens a pin.", residency::EXPERT_VRAM_LAYERS_ENV));
        details.push(format!("{}={layers} pins the trunk-layer prefix; the loader never widens it. MTP experts remain host-mapped on a split (expert_residency.rs:2041-2057).", residency::EXPERT_VRAM_LAYERS_ENV));
    } else if explicit.is_some() {
        notes.push(format!("{}=auto explicitly requests split placement even when a fully resident lower bound fits.", residency::EXPERT_VRAM_LAYERS_ENV));
    }
    if ranked_hotset {
        notes.push(
            "A ranked expert hot set applies: the exact hot-expert split is decided at load."
                .into(),
        );
        details.push("Eligible ranked hotset: whole-layer counts are not a tier-plan prediction. Sidecar binding, optional fallback and exact hot-expert budget are decided at load; the host lower bound below allows arbitrary expert residency and omits nonnegative tier-table/stage charges (carriers.rs:2133-2269).".into());
    }
    let trunk_layers = config.num_hidden_layers;
    let moe_layers = residency::moe_layer_count(config);
    notes.push(format!("Model inventory: {} non-expert weights, {} routed experts ({} per expert layer), {moe_layers} expert layers.", gib(non_expert), gib(experts), gib(layer_experts)));
    details.push(format!("Exact HFQ inventory: {} non-expert, {} routed experts, {} largest expert layer; {trunk_layers} trunk / {moe_layers} total MoE layers. The admitted manifest counts MTP weights even when the native speculator is off (expert_residency.rs:1096-1099,2217-2219).", gib(non_expert), gib(experts), gib(layer_experts)));
    // Mutate only the local admitted manifest, not the source. Using the real
    // host census also charges per-allocation HOST_MAPPED_PAD_BYTES. Raw
    // expert payload alone is not a sound upper bound for a host-side FITS.
    let pinned_host = if let Some(layers) = pinned {
        residency::place_routed_experts(&mut receipt.manifest.weights, layers);
        Some(residency::host_mapped_bytes_ranks(&receipt.manifest.weights, bytes_of, 1)?[0])
    } else {
        None
    };
    residency::place_routed_experts(&mut receipt.manifest.weights, 0);
    let all_host = residency::host_mapped_bytes_ranks(&receipt.manifest.weights, bytes_of, 1)?[0];
    details.push("Host upper bound and pinned charges use place_routed_experts + host_mapped_bytes_ranks, including the engine's per-entry allocation padding; automatic host lower bounds omit padding and donor restore mirrors (expert_residency.rs:2044-2057,2125-2143,2146-2154).".into());
    let inv = Inventory {
        non_expert,
        experts,
        layer_experts,
        trunk_layers,
        moe_layers,
        slab_stage,
        force_auto: explicit == Some(residency::ExpertVramLayers::Auto),
        all_host,
        pinned_host,
        ranked_hotset,
    };
    let base_lb = base_reserve_lb(req.device, config)?;
    let mut report = decide(&inv, req.device, req.host, base_lb, pinned);
    report.notes.extend(notes);
    report.details.extend(details);
    let explicit_context = req
        .params
        .get("max_seq")
        .is_some_and(|value| !value.is_null());
    if req.device.unified_memory {
        report.notes.push(format!("Context is decided at load: automatic = min(model context, card capacity measured after the load); an explicit max_seq is re-checked after the load. Admitted: {} tokens.", req.max_seq));
        report.details.push("qwen4_memory_admission needs bundle.context_growth_bytes(gpu, t), including live gather scratch and resident PLE (carriers.rs:2934-2976; admission.rs:755-797).".into());
        if explicit_context {
            report.findings.push(Finding::undetermined("context", format!("decided at load: explicit max_seq={} is re-checked against post-load context growth and free memory", req.max_seq), None));
        }
    } else {
        report.notes.push(format!("On a discrete GPU an omitted max_seq is {} tokens and an explicit one is not checked at load (docs/CONFIG.md:817-818); admitted max_seq {}.", hipfire_arch_qwen4::config::QWEN4_DEFAULT_CONTEXT, req.max_seq));
        if report
            .placement
            .host_mapped_bytes
            .is_none_or(|bytes| bytes > 0)
            || report.placement.decided_at_load
        {
            if req
                .params
                .pointer("/memory/offload_exec")
                .is_none_or(|value| value.is_null())
                && req
                    .params
                    .get("offload_exec")
                    .is_none_or(|value| value.is_null())
                && hipfire_config::process_value("HIPFIRE_OFFLOAD_EXEC").is_none()
            {
                report.notes.push("D2 CPU split: decided at model load from this model's CPU arm and live link samples; the calibration record supplies no D2 figure (docs/CONFIG.md:1205-1206).".into());
            }
        }
    }
    Ok(report)
}

/// These pool terms are unconditional in the real placement census:
/// `include_short_tails=true`, and mq_fixed is independent of rows/dimensions.
/// Use the owners' byte constants, never duplicate their sizing formulas.
fn base_reserve_lb(
    device: &FitDevice,
    config: &hipfire_arch_qwen4::Qwen4Config,
) -> Result<u64, String> {
    // The carrier's routed_i4 workspace eligibility, not a byte estimator
    // (expert_residency.rs:2666-2685). This term is independent of chunk rows.
    let routed_i4 = !config.ffn.is_dense()
        && config.num_experts > 0
        && config.num_experts_per_tok > 0
        && config.moe_intermediate_size > 0
        && config.hidden_size.is_multiple_of(256)
        && config.moe_intermediate_size.is_multiple_of(128);
    let slab_union = if routed_i4 {
        rdna_compute::scratch::qwen4_slab_union_bytes(rdna_compute::slab_moe::SLAB_MAX_LANES)
            .ok_or("slab lane-union bytes overflow")? as u64
    } else {
        0
    };
    residency::runtime_overhead_bytes(&device.arch, device.unified_memory, 1)
        .unwrap_or(0)
        .checked_add(rdna_compute::scratch::MQ_FIXED_WORKSPACE_BYTES as u64)
        .and_then(|bytes| {
            bytes.checked_add(rdna_compute::tensor_ops::GDN_PAIR_COUNTERS_BYTES as u64)
        })
        .and_then(|bytes| bytes.checked_add(slab_union))
        .ok_or_else(|| "partial Qwen4 reserve census overflows".into())
}

fn reserve_notes(device: &FitDevice) -> Vec<String> {
    vec![
        format!("Reserve lower bound includes HIP runtime={} via runtime_overhead_bytes(arch, unified, 1), zero where no measured term exists (expert_residency.rs:62-69). AutoReserve::bytes=max(historical,census) >= census >= this partial census (expert_residency.rs:1375-1390).", gib(residency::runtime_overhead_bytes(&device.arch, device.unified_memory, 1).unwrap_or(0))),
        format!("Reserve lower bound also includes unconditional mq_fixed={} and gdn_pair_counters={} using the owners' public byte constants; placement always includes short tails (rdna-compute/src/scratch.rs:1350,1440-1457; tensor_ops.rs:496; expert_residency.rs:2685).", gib(rdna_compute::scratch::MQ_FIXED_WORKSPACE_BYTES as u64), gib(rdna_compute::tensor_ops::GDN_PAIR_COUNTERS_BYTES as u64)),
        "Reserve lower bound includes slab_union via qwen4_slab_union_bytes(SLAB_MAX_LANES) exactly when the carrier's routed_i4 geometry admits it; this descriptor is independent of chunk rows (expert_residency.rs:2666-2685; rdna-compute/src/scratch.rs:1329,1425-1429; slab_moe.rs:486).".into(),
        "Placement includes the real PlacementCharges margin and, only for automatic slab split placement, dedicated long-prefill stage buffers; a resident load has no stage, and UMA's resident gate uses the base reserve without discrete placement charges (expert_residency.rs:1495-1546,1795-1814; carriers.rs:2304-2328).".into(),
        "Omitted QSA context and recurrent/fixed state census: state_census requires resolve_state_format's GPU flags, QSA layout and VMM allocation granularity; its arch-only QSA/GDN resolvers are private. No substitute state format or context estimator (state.rs:817,979-1002; expert_residency.rs:476-501,2562).".into(),
        "Omitted forward scratch and kernel pool workspace terms fp16_x, ksplit_det, int4_xq, moe_group, down_i4q, slab_tiles, gdn_dense_a and gdn_dense_ef: source_forward_scratch_mode needs GPU/payload route facts; qwen4_prefill_chunk_requested needs a live gfx1100 host-link sample, and its injected/explicit-row parsers are crate-private. The stored calibration link is not the live host_link(gpu) input; no copied chunk resolver (carriers.rs:1937-1943,2005-2013; gpu_forward.rs:145-174; expert_residency.rs:2650-2690; rdna-compute/src/scratch.rs:1402-1457).".into(),
        "Omitted F16 weight shadows: enabled by GPU route flags and chunk threshold, not the cold manifest alone (expert_residency.rs:2635-2648). Omitted QSA selector scratch: GPU route/layout dependent (expert_residency.rs:2697-2702).".into(),
        "Omitted native MTP reserve: capture/fill scratch depend on the GPU; host-spill attachment uses the real mtp_host_spill_policy with gfx1201's admitted list, not a second MTP sizing formula (carriers.rs:1427-1436; admission.rs:347-354; expert_residency.rs:2563-2582).".into(),
        "Omitted QSA gather growth: route and already-committed GPU scratch are unknown (expert_residency.rs:2588-2602). Omitted historical reserve floor: auto_vram_reserve needs the unresolved context/state/scratch/chunk/MTP/gather inputs (expert_residency.rs:2609-2624). All omitted terms are nonnegative; the reported expert residency is an upper bound, never a load promise.".into(),
    ]
}

fn unknown(detail: impl Into<String>) -> ArchFit {
    ArchFit {
        placement: PlacementPlan {
            summary: "decided at load".into(),
            expert_layers_resident: None,
            host_mapped_bytes: None,
            decided_at_load: true,
        },
        findings: vec![Finding::undetermined("experts", detail, None)],
        notes: Vec::new(),
        details: Vec::new(),
    }
}

fn decide(
    inv: &Inventory,
    device: &FitDevice,
    host: &FitHost,
    reserve_lb: u64,
    pinned: Option<usize>,
) -> ArchFit {
    let Some(free) = device.free else {
        let mut report =
            unknown("free device memory is unavailable; expert placement is decided at load");
        if !device.unified_memory {
            host_findings(
                &mut report,
                inv.pinned_host.unwrap_or(0),
                inv.pinned_host.unwrap_or(inv.all_host),
                host,
            );
        }
        return report;
    };
    if device.unified_memory {
        let mut report = unknown("resident weights pass only a reserve lower-bound check; the complete reserve is decided at load");
        report.placement.summary = "fully resident".into();
        report.placement.expert_layers_resident = Some((inv.moe_layers, inv.moe_layers));
        report.placement.host_mapped_bytes = Some(0);
        if !residency::fits_fully_resident(free, inv.non_expert, inv.experts, reserve_lb) {
            // Same resident gate and refusal as carriers.rs:2312-2328; only
            // the reserve is a labelled lower bound instead of a GPU census.
            report.findings = vec![Finding::fails("vram", format!("qwen4: unified memory cannot hold the resident load ({} MiB routed experts + {} MiB non-expert weights + {} MiB reserve (lower bound) exceed {} MiB free device memory); free host memory, or set memory.oom_guard = off to load anyway", inv.experts >> 20, inv.non_expert >> 20, reserve_lb >> 20, free >> 20), "free host memory, a GPU with more free memory, or a smaller tier")];
        }
        if pinned.is_some() || inv.force_auto {
            report.findings.push(Finding::undetermined("experts", format!("qwen4: {} puts experts in host RAM, which only a discrete GPU supports; unset it on this unified-memory device", residency::EXPERT_VRAM_LAYERS_ENV), Some(format!("unset {}", residency::EXPERT_VRAM_LAYERS_ENV))));
        }
        return report;
    }
    let layers = (|| {
        if inv.ranked_hotset {
            // Do not apply slab/whole-layer stage charges to a tier plan.
            // Zero-stage PlacementCharges is also a lower bound for ranked
            // placement, which charges tables and its own stage on top.
            let charges = residency::PlacementCharges::new(inv.layer_experts, false)?;
            residency::auto_placement_layers(
                free,
                inv.non_expert,
                inv.layer_experts,
                inv.trunk_layers,
                reserve_lb,
                &charges,
            )?;
            return Ok(None);
        }
        if pinned.is_some() || inv.force_auto {
            let charges = residency::PlacementCharges::new(
                inv.layer_experts,
                pinned.is_none() && inv.slab_stage,
            )?;
            let bound = residency::auto_placement_layers(
                free,
                inv.non_expert,
                inv.layer_experts,
                inv.trunk_layers,
                reserve_lb,
                &charges,
            )?;
            return Ok(Some(pinned.unwrap_or(bound).min(inv.trunk_layers)));
        }
        residency::incumbent_placement_layers(
            free,
            inv.non_expert,
            inv.experts,
            inv.layer_experts,
            inv.trunk_layers,
            reserve_lb,
            inv.slab_stage,
        )
    })();
    let layers = match layers {
        Ok(layers) => layers,
        Err(error) => {
            let mut report = unknown("expert placement cannot fit its lower-bound reserve");
            report.placement.summary =
                "no placement: the non-expert weights and the load reserve exceed free VRAM".into();
            report.placement.decided_at_load = false;
            report.findings = vec![Finding::fails(
                "vram",
                error,
                "a GPU with more free memory, or a smaller tier",
            )];
            return report;
        }
    };
    let (lower, upper, resident) = if inv.ranked_hotset {
        // Even arbitrary expert residency cannot consume more than the free
        // bytes left after the non-expert payload. Omitting every reserve and
        // hotset table here only makes this host lower bound more permissive.
        (
            inv.experts
                .saturating_sub(free.saturating_sub(inv.non_expert)),
            inv.all_host,
            None,
        )
    } else if let Some(exact) = inv.pinned_host {
        (
            exact,
            exact,
            Some((layers.unwrap_or(inv.moe_layers), inv.moe_layers)),
        )
    } else {
        let resident = layers.unwrap_or(inv.moe_layers);
        let lower = if layers.is_none() {
            0
        } else {
            inv.experts
                .saturating_sub(inv.layer_experts.saturating_mul(resident as u64))
        };
        (lower, inv.all_host, Some((resident, inv.moe_layers)))
    };
    let exact = pinned.is_some() && !inv.ranked_hotset;
    let summary = if inv.ranked_hotset {
        format!(
            "ranked hotset decided at load; at least {} host-mapped",
            gib(lower)
        )
    } else if layers.is_none() {
        "fully resident".into()
    } else if exact {
        format!(
            "{} of {} expert layers pinned in VRAM; {} host-mapped",
            resident.unwrap().0,
            inv.moe_layers,
            gib(lower)
        )
    } else {
        format!(
            "at most {} of {} expert layers in VRAM; at least {} host-mapped",
            resident.unwrap().0,
            inv.moe_layers,
            gib(lower)
        )
    };
    let mut report = ArchFit {
        placement: PlacementPlan {
            summary,
            expert_layers_resident: resident,
            host_mapped_bytes: Some(lower),
            decided_at_load: !exact,
        },
        findings: vec![Finding::undetermined(
            "experts",
            if exact {
                "expert placement is pinned, not widened; its full GPU reserve and pinned device payload are checked at load".to_owned()
            } else {
                format!(
                "decided at load: {} non-expert weights plus a load reserve of at least {} leave at most {} of the {} free for routed experts; the GPU's complete reserve (context, scratch, MTP) may keep fewer expert layers resident, and the load fails only if it leaves no room for the non-expert weights",
                gib(inv.non_expert),
                gib(reserve_lb),
                gib(free.saturating_sub(inv.non_expert).saturating_sub(reserve_lb)),
                gib(free),
            )
            },
            None,
        )],
        notes: Vec::new(),
        details: Vec::new(),
    };
    host_findings(&mut report, lower, upper, host);
    report
}

fn host_findings(report: &mut ArchFit, lower: u64, upper: u64, host: &FitHost) {
    let need = lower.saturating_add(residency::HOST_RAM_HEADROOM_BYTES);
    let ram_fix = format!("need {} host RAM including the gate's headroom, versus {} MemAvailable - CMA plus {} TTM pool; free host memory, keep more expert layers in VRAM (a GPU with more memory) or a smaller tier", gib(need), host.mem_available.map(gib).unwrap_or_else(|| "unreadable".into()), gib(host.ttm_pool));
    let range = || {
        format!(
            "decided at load: needs between {} and {} of host memory depending on the load reserve",
            gib(lower),
            gib(upper)
        )
    };
    report.findings.push(
        match residency::check_host_ram(lower, host.mem_available, host.ttm_pool) {
            Err(error) => Finding::fails("host_ram", error, ram_fix.as_str()),
            Ok(()) => match residency::check_host_ram(upper, host.mem_available, host.ttm_pool) {
                Ok(()) => Finding::fits(
                    "host_ram",
                    format!(
                        "the real host RAM gate passes even at {} host-mapped",
                        gib(upper)
                    ),
                ),
                Err(_) => Finding::undetermined("host_ram", range(), None),
            },
        },
    );
    let gtt = host.gtt_limit.map(|limit_bytes| residency::GttBudget {
        limit_bytes,
        used_bytes: host.gtt_used.unwrap_or(0),
    });
    // If a cap is known but usage is not, zero usage is a safe *lower-bound*
    // refusal check, never proof that the upper host bound passes the gate.
    report.findings.push(match residency::check_gtt_cap(lower, gtt) {
        Err(error) => Finding::fails("gtt", error, format!("{ram_fix}; raise ttm.pages_limit (docs/CONFIG.md:827; the gate names the live cap), or use HSA_USERPTR_FOR_PAGED_MEM=1 for pageable userptr memory, which host pressure can stall")),
        Ok(()) if host.gtt_limit.is_some() && host.gtt_used.is_none() && upper > 0 => Finding::undetermined("gtt", "decided at load: TTM cap is known but live amdgpu GTT usage is unreadable", None),
        Ok(()) => match residency::check_gtt_cap(upper, gtt) {
            Ok(()) => Finding::fits("gtt", if gtt.is_some() { format!("the real GTT cap gate passes even at {} host-mapped", gib(upper)) } else { "no GTT cap applies (the real gate skips None)".into() }),
            Err(_) => Finding::undetermined("gtt", range(), None),
        },
    });
}

#[cfg(test)]
mod tests {
    use super::super::types::Verdict;
    use super::*;

    const GIB: u64 = 1 << 30;

    fn device(free: Option<u64>) -> FitDevice {
        FitDevice {
            index: 0,
            arch: "gfx1201".into(),
            unified_memory: false,
            vram_total: free,
            free,
            link: None,
            capacity: None,
            capacity_shown: None,
            env_selector: "0".into(),
            config_selector: "GPU-0".into(),
        }
    }

    fn host(ram: u64, cap: Option<u64>) -> FitHost {
        FitHost {
            ram_total: Some(ram),
            mem_available: Some(ram),
            ttm_pool: 0,
            gtt_limit: cap,
            gtt_used: Some(0),
            devices: Vec::new(),
        }
    }

    fn flash_sized() -> Inventory {
        // Synthetic 60 GiB routed payload, not a measured artifact byte count:
        // reproduces docs/EMU.md:224-231's 38–61 GB host-mapped range at 16 GB.
        Inventory {
            non_expert: 4 * GIB,
            experts: 60 * GIB,
            layer_experts: 5 * GIB / 4,
            trunk_layers: 48,
            moe_layers: 48,
            slab_stage: false,
            force_auto: false,
            all_host: 60 * GIB + 96 * residency::HOST_MAPPED_PAD_BYTES,
            pinned_host: None,
            ranked_hotset: false,
        }
    }

    fn finding(report: &ArchFit, id: &str) -> Finding {
        report
            .findings
            .iter()
            .find(|finding| finding.id == id)
            .unwrap()
            .clone()
    }

    #[test]
    fn emu_16_gib_64_gb_host_31_gib_ttm_refuses_with_real_message() {
        let inv = flash_sized();
        let report = decide(
            &inv,
            &device(Some(16 * GIB)),
            &host(64_000_000_000, Some(31 * GIB)),
            0,
            None,
        );
        let lower = report.placement.host_mapped_bytes.unwrap();
        let error = residency::check_gtt_cap(
            lower,
            Some(residency::GttBudget {
                limit_bytes: 31 * GIB,
                used_bytes: 0,
            }),
        )
        .unwrap_err();
        assert_eq!(finding(&report, "gtt").verdict, Verdict::DoesNotFit);
        assert_eq!(finding(&report, "gtt").detail, error);
        assert!(finding(&report, "gtt")
            .fix
            .unwrap()
            .contains("ttm.pages_limit"));
    }

    #[test]
    fn same_inventory_128_gb_host_without_cap_passes_host_gates() {
        let report = decide(
            &flash_sized(),
            &device(Some(16 * GIB)),
            &host(128_000_000_000, None),
            0,
            None,
        );
        assert_eq!(finding(&report, "host_ram").verdict, Verdict::Fits);
        assert_eq!(finding(&report, "gtt").verdict, Verdict::Fits);
        assert!(report.placement.decided_at_load);
    }

    #[test]
    fn resident_lower_bound_on_32_gib_is_not_a_load_promise() {
        let mut inv = flash_sized();
        inv.experts = 12 * GIB;
        inv.layer_experts = GIB / 4;
        inv.all_host = inv.experts + 96 * residency::HOST_MAPPED_PAD_BYTES;
        let report = decide(
            &inv,
            &device(Some(32 * GIB)),
            &host(128 * GIB, None),
            0,
            None,
        );
        assert_eq!(report.placement.summary, "fully resident");
        assert_eq!(report.placement.expert_layers_resident, Some((48, 48)));
        assert_eq!(report.placement.host_mapped_bytes, Some(0));
        assert!(report.placement.decided_at_load);
        assert_eq!(finding(&report, "experts").verdict, Verdict::Undetermined);
    }

    #[test]
    fn explicit_pin_is_never_widened_and_host_charge_is_exact() {
        let mut inv = flash_sized();
        inv.pinned_host = Some(55 * GIB);
        let report = decide(
            &inv,
            &device(Some(128 * GIB)),
            &host(128 * GIB, None),
            0,
            Some(4),
        );
        assert_eq!(report.placement.expert_layers_resident, Some((4, 48)));
        assert_eq!(report.placement.host_mapped_bytes, Some(55 * GIB));
        assert!(!report.placement.decided_at_load);
    }

    #[test]
    fn non_expert_and_reserve_refusal_preserves_gate_message() {
        let inv = flash_sized();
        let charges = residency::PlacementCharges::new(inv.layer_experts, inv.slab_stage).unwrap();
        let error = residency::auto_placement_layers(
            4 * GIB,
            inv.non_expert,
            inv.layer_experts,
            inv.trunk_layers,
            GIB,
            &charges,
        )
        .unwrap_err();
        let report = decide(
            &inv,
            &device(Some(4 * GIB)),
            &host(128 * GIB, None),
            GIB,
            None,
        );
        assert_eq!(finding(&report, "vram").verdict, Verdict::DoesNotFit);
        assert_eq!(finding(&report, "vram").detail, error);
    }

    #[test]
    fn host_ram_refusal_preserves_gate_message_and_headroom_fix() {
        let report = decide(
            &flash_sized(),
            &device(Some(16 * GIB)),
            &host(32 * GIB, None),
            0,
            None,
        );
        let lower = report.placement.host_mapped_bytes.unwrap();
        assert_eq!(
            finding(&report, "host_ram").detail,
            residency::check_host_ram(lower, Some(32 * GIB), 0).unwrap_err()
        );
        assert_eq!(finding(&report, "host_ram").verdict, Verdict::DoesNotFit);
        assert!(finding(&report, "host_ram")
            .fix
            .unwrap()
            .contains("headroom"));
    }

    #[test]
    fn partial_host_gate_range_is_undetermined() {
        let report = decide(
            &flash_sized(),
            &device(Some(32 * GIB)),
            &host(128 * GIB, Some(40 * GIB)),
            0,
            None,
        );
        assert_eq!(finding(&report, "gtt").verdict, Verdict::Undetermined);
        assert!(finding(&report, "gtt").fix.is_none());
    }

    #[test]
    fn missing_free_memory_is_undetermined() {
        let report = decide(
            &flash_sized(),
            &device(None),
            &host(128 * GIB, None),
            0,
            None,
        );
        assert_eq!(finding(&report, "experts").verdict, Verdict::Undetermined);
        assert!(report.placement.expert_layers_resident.is_none());
    }

    #[test]
    fn known_cap_with_unknown_usage_cannot_prove_host_fit() {
        let mut machine = host(128 * GIB, Some(100 * GIB));
        machine.gtt_used = None;
        let report = decide(&flash_sized(), &device(Some(16 * GIB)), &machine, 0, None);
        assert_eq!(finding(&report, "gtt").verdict, Verdict::Undetermined);
    }

    #[test]
    fn unified_resident_gate_uses_entire_weight_inventory() {
        let mut gpu = device(Some(16 * GIB));
        gpu.unified_memory = true;
        let report = decide(&flash_sized(), &gpu, &host(128 * GIB, None), 0, None);
        assert_eq!(finding(&report, "vram").verdict, Verdict::DoesNotFit);
        assert_eq!(report.placement.host_mapped_bytes, Some(0));
    }

    #[test]
    fn ranked_hotset_does_not_apply_whole_layer_host_lower_bound() {
        let mut inv = flash_sized();
        inv.ranked_hotset = true;
        let report = decide(
            &inv,
            &device(Some(32 * GIB)),
            &host(128 * GIB, None),
            0,
            None,
        );
        assert!(report.placement.expert_layers_resident.is_none());
        assert_eq!(report.placement.host_mapped_bytes, Some(32 * GIB));
        assert!(report.placement.decided_at_load);
    }

    /// Optional source census only; never run in the CPU unit-test suite.
    #[test]
    #[ignore = "requires the operator's local Flash-Next artifact; reads headers only"]
    fn print_local_flash_next_inventory() {
        let hfq = HfqFile::open(std::path::Path::new(
            "/home/kaden/.hipfire/models/qwen3.8-flash-next-gptq3.mq4",
        ))
        .unwrap();
        let gpu = device(Some(16 * GIB));
        let machine = host(64_000_000_000, Some(31 * GIB));
        let params = serde_json::json!({});
        let report = analyze(&Qwen4Request {
            hfq: &hfq,
            device: &gpu,
            host: &machine,
            max_seq: hipfire_arch_qwen4::config::QWEN4_DEFAULT_CONTEXT,
            params: &params,
        });
        for note in report.notes {
            eprintln!("{note}");
        }
    }
}
