// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The single-device, no-MTP Qwen4 load the S1 validation tools share
//! (`qwen4_slowlink_gate`, `qwen4_greedy_ids`, `qwen4_kld`, the parity CLI
//! and the hardware tests).
//!
//! It is the product loader's admission-first order, without the product's
//! MTP, PLE hot store and hot set: the S1 profile is selected FIRST
//! ([`crate::slowlink_admission::select_load_profile`], from the exact
//! placement census [`crate::expert_residency::placement_inputs`]), then the
//! routed-expert placement it implies, then the bundle, with the profile
//! attached before the forward is built. Candidate and baseline arms differ
//! only in `HIPFIRE_QWEN4_SLOWLINK_PREFILL`, so they are separate processes
//! (the config snapshot is read once).

use crate::kv_backend::Qwen4KvBackend;
use rdna_compute::Gpu;
use std::path::Path;

use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};

/// What the shared loader built.
pub struct SlowlinkLoaded {
    pub bundle: crate::bundle::Qwen4Bundle,
    pub vocab: usize,
    pub eos_token: u32,
    pub num_layers: usize,
    pub arch_id: u32,
    /// Routed-expert tensors host-mapped by the placement (0 = all resident).
    pub host_mapped_tensors: usize,
    /// Trunk layers whose routed experts are resident in VRAM.
    pub resident_layers: usize,
    /// How the placement was decided: `slowlink-profile` (the S1 admission's
    /// joint placement), `incumbent-auto` (the product's automatic placement
    /// at the incumbent chunk), `explicit-env` (a fixed
    /// `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS`, which declines S1) or
    /// `unified-memory` (everything resident).
    pub placement: &'static str,
    /// The S1 profile was admitted and attached to the bundle.
    pub slowlink_profile: bool,
    pub ple_format: String,
    pub tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    /// The artifact's chat template (rendering is the caller's).
    pub chat_template: Option<String>,
}

/// The fulfillment payload of one manifest entry: a slab layer arena (one
/// container span), a checked source range (unified memory or an external
/// entry) or the mapped bytes borrowed (a resident or host-mapped discrete
/// entry), as the product loader reads them.
fn payload<'a>(
    source: &'a hipfire_runtime::hfq::HfqModelSource,
    recipe: &crate::weights::Qwen4TierRecipe,
    entry: &WeightEntry,
    use_ranges: bool,
) -> Result<hipfire_runtime::model_source::SourcePayload<'a>, String> {
    use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
    if let Some(payload) = crate::expert_arena::recipe_arena_payload(source, recipe, entry) {
        return payload;
    }
    if use_ranges || entry.residency.is_external() {
        return source
            .tensor_range(&entry.name)
            .map_err(|error| error.to_string())?
            .map(SourcePayload::Range)
            .ok_or_else(|| {
                if entry.residency.is_external() {
                    format!("missing external tensor '{}'", entry.name)
                } else {
                    format!("missing resident tensor '{}'", entry.name)
                }
            });
    }
    let (info, bytes) = source
        .tensor_data(&entry.name)
        .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
    Ok(SourcePayload::Borrowed { info, bytes })
}

/// Load one real Qwen4 HFQM artifact the way the product loader does for a
/// single-device, no-MTP load: arena payloads, the S1 admission first, then
/// the placement it implies (the profile's resident layers when S1 is
/// admitted, else the incumbent automatic placement at the placement chunk),
/// the borrowed host-mapped payload fallback, no PLE hot store.
/// `session_cache_budget = Some(bytes)` attaches a session cache of that
/// budget. An explicit `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS` pins the placement
/// and declines S1, exactly as in the product. For the S1 gate and the
/// hardware tests, whose incumbent arm intentionally names the product's
/// automatic placement; examples that must keep their own baseline placement
/// use [`slowlink_load_if_selected`]. `purpose` is the explicit admission
/// intent: `Product` for the product runtime, `PrimitiveSelftest` only for the
/// diagnostic primitives gate, which sizes its own larger transient reserve.
pub fn slowlink_load_bundle(
    gpu: &mut Gpu,
    path: &Path,
    max_seq: usize,
    session_cache_budget: Option<u64>,
    purpose: crate::slowlink_admission::AdmissionPurpose,
) -> Result<SlowlinkLoaded, String> {
    load_inner(gpu, path, max_seq, session_cache_budget, true, purpose)?
        .ok_or_else(|| "slowlink load produced no bundle".to_string())
}

/// The S1 load only when the selector and admission choose S1 for this load
/// (forced `1` or an eligible automatic selection); `Ok(None)` BEFORE any GPU
/// allocation otherwise (selector off or unset-and-declined, fast/unified/
/// unmeasured link, an explicit expert layer count, any other decline), so the
/// caller's own loader and placement run byte for byte as before. A forced
/// selector that is eligible but over capacity is an `Err` with its exact
/// deficit, as in the product. This is the product selection (greedy/KLD/
/// parity examples), so it always admits as
/// [`crate::slowlink_admission::AdmissionPurpose::Product`] and never pays the
/// diagnostic reserve.
pub fn slowlink_load_if_selected(
    gpu: &mut Gpu,
    path: &Path,
    max_seq: usize,
    session_cache_budget: Option<u64>,
) -> Result<Option<SlowlinkLoaded>, String> {
    load_inner(
        gpu,
        path,
        max_seq,
        session_cache_budget,
        false,
        crate::slowlink_admission::AdmissionPurpose::Product,
    )
}

fn load_inner(
    gpu: &mut Gpu,
    path: &Path,
    max_seq: usize,
    session_cache_budget: Option<u64>,
    product_fallback: bool,
    purpose: crate::slowlink_admission::AdmissionPurpose,
) -> Result<Option<SlowlinkLoaded>, String> {
    use crate::expert_residency as residency;
    use crate::slowlink_admission as s1;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(path)
        .map_err(|error| format!("open HFQM {}: {error}", path.display()))?;
    let choice = hipfire_runtime::ple_sidecar::attach_validation_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
    )
    .map_err(|error| format!("qwen4 PLE rows: {error}"))?;
    let receipt = crate::admit_hfqm_artifact_on(&hfq, Some(&gpu.arch))
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let trunk_overlay = crate::resolve_trunk_overlay(&hfq, path, &gpu.arch, &receipt.config)
        .map_err(|error| format!("qwen4 trunk overlay: {error}"))?;
    let scratch_mode =
        crate::artifact::source_forward_scratch_mode(gpu, &hfq, &receipt.config, &receipt.manifest)
            .map_err(|error| format!("qwen4 forward scratch plan failed: {error}"))?;
    // The one artifact scan the symmetric IU4 window route
    // (`HIPFIRE_QWEN4_MOE_SYM_IU4=1`) needs: the source-byte symmetry of every routed trunk
    // layer, read from the HFQ before the model source takes ownership of it, and only when the
    // route's other terms hold (`SymIu4RouteFacts::scan_needed`). A unified-memory load never
    // selects S1 and may have dropped its mapping.
    let routed_symmetric: Option<Result<bool, String>> = (!gpu.is_uma()
        && crate::slowlink_admission::SymIu4RouteFacts::of_gpu(gpu).scan_needed(
            crate::slowlink_admission::window_route_of(&receipt.manifest.weights),
        ))
    .then(|| {
        crate::artifact::routed_headers_symmetric_verdict(&hfq, &receipt.manifest)
            .map_err(|error| error.to_string())
    });
    let unified = gpu.is_uma();
    if unified {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, gpu);
    let npu_identity = crate::npu_trunk::model_index_identity(&hfq, &gpu.arch);
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| format!("tokenizer: {error}"))?;
    let chat_template = hfq.chat_template();
    let arch_id = hfq.arch_id;
    // The session-cache domain binds the artifact, tokenizer and device; it
    // is read from the HFQ file before the source takes ownership of it.
    let cache_domain = session_cache_budget.map(|_| {
        hipfire_runtime::serve_contract::CacheDomain::for_model(
            &hfq,
            &tokenizer,
            None,
            "qwen4",
            gpu.device_id,
        )
    });
    let recipe = receipt.manifest.identity.recipe.clone();
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    // The cached process snapshot, as the product reads developer knobs.
    let state_quant = hipfire_config::developer_var("HIPFIRE_STATE_QUANT").unwrap_or_default();
    let state_format = crate::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &state_quant,
        gpu,
        &receipt.config,
    )?;
    let backend = Qwen4KvBackend::automatic(gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|error| format!("qwen4: VMM granularity: {error}"))?,
        Qwen4KvBackend::Legacy => 1,
    };
    // Entry sizes as the product loader sizes them: a slab layer arena from
    // the recipe's plan, anything else from the mapped tensor.
    let bytes_of = |entry: &WeightEntry| -> Option<u64> {
        if let crate::weights::Qwen4TierRecipe::FlashNextSlab { arenas, .. } = &recipe {
            if entry.name.ends_with(crate::expert_arena::ARENA_SUFFIX) {
                return entry
                    .layer
                    .and_then(|layer| arenas.layer(layer))
                    .map(|layer| layer.bytes as u64);
            }
        }
        use hipfire_runtime::model_source::ModelSource as _;
        source
            .tensor_data(&entry.name)
            .map(|(_, bytes)| bytes.len() as u64)
    };
    let routed_experts = !receipt.config.ffn.is_dense();
    let mut weights = receipt.manifest.weights.clone();
    let explicit = residency::expert_vram_layers_from_env()?;
    let mut weight_arenas = None;
    let mut resident_layer_set = None;
    let (resident_layers, placement, profile) = if !product_fallback
        && (unified || matches!(explicit, Some(residency::ExpertVramLayers::Layers(_))))
    {
        // A unified-memory or explicitly pinned load never selects S1: the
        // caller's own loader decides its placement.
        return Ok(None);
    } else if unified {
        (receipt.config.num_hidden_layers, "unified-memory", None)
    } else if let Some(residency::ExpertVramLayers::Layers(layers)) = explicit {
        residency::place_routed_experts(&mut weights, layers);
        (layers, "explicit-env", None)
    } else {
        // Automatic placement (an explicit `auto` declines S1 through the
        // shape's `explicit_vram_layers`, as in the product).
        let chunk_env = hipfire_config::developer_var("HIPFIRE_PREFILL_CHUNK_ROWS").ok();
        let shape = s1::LoadShape::from_process(
            &gpu.arch,
            false,
            s1::StructuralShape {
                // Sealed layout/panel/codec/kernel verdicts admit (never the
                // recipe name): unsupported modes refuse before materialization.
                expert_parallel: false,
                pipeline_band: false,
                routed_experts,
                expert_keep: receipt.config.expert_keep.is_some(),
                hotset_sidecar: false,
                whole_first: false,
            },
            max_seq,
            chunk_env.as_deref(),
        )?;
        let load = s1::SingleDeviceLoad {
            gpu: &*gpu,
            config: &receipt.config,
            weights: &weights,
            bytes_of: &bytes_of,
            backend,
            max_seq,
            vmm_granularity,
            native_mtp: false,
            mtp_kept: false,
            max_k: 0,
            state_format,
            scratch_mode,
            purpose,
            routed_source_symmetric: &|| routed_symmetric.clone().unwrap_or(Ok(false)),
        };
        // Device free memory just before the host-link initialization. The
        // selector runs the link closure only after its probe-free
        // preconditions pass (default-off, UMA and other ineligible loads never
        // reach it), so those query nothing; an eligible load pays ONE
        // `hipMemGetInfo` here even when the link then measures fast or fails
        // to measure (the class is unknown until the probe runs). The
        // admission's own free sample is taken after that initialization.
        let pre_init_free = std::cell::Cell::new(None::<u64>);
        let profile = s1::select_load_profile(
            s1::SlowlinkMode::from_env(),
            &shape,
            || {
                pre_init_free.set(load.gpu.device_mem_info().ok().map(|(free, _)| free as u64));
                crate::host_link::host_link(load.gpu)
            },
            || {
                let arenas = s1::single_device_arena_plan(&load)?;
                let facts =
                    s1::single_device_facts_with_arenas(&load, &arenas, pre_init_free.get())?;
                weight_arenas = Some(arenas);
                Ok(facts)
            },
        )?;
        if profile.is_none() && !product_fallback {
            // Not selected: the caller keeps its own (baseline) loader and
            // placement exactly; nothing was allocated.
            return Ok(None);
        }
        let layers = match &profile {
            Some(p) => p.resident_layers,
            None => {
                // The product's automatic placement at its own chunk (the
                // shared incumbent resolver; `None`: fully resident).
                let dedicated_stage = residency::dedicated_slab_stage(
                    residency::expert_stage_placement_enabled(&gpu.arch),
                    matches!(
                        recipe,
                        crate::weights::Qwen4TierRecipe::FlashNextSlab { .. }
                    ),
                    false,
                );
                residency::incumbent_auto_resident_layers(
                    load.gpu,
                    load.config,
                    load.weights,
                    load.bytes_of,
                    backend,
                    max_seq,
                    vmm_granularity,
                    state_format,
                    scratch_mode,
                    dedicated_stage,
                    0,
                )?
                .unwrap_or(load.config.num_hidden_layers)
            }
        };
        if profile.is_some()
            && s1::resident_layer_switch_is_override(
                hipfire_config::developer_var(s1::S1_RESIDENT_LAYERS_ENV)
                    .ok()
                    .as_deref(),
            )
        {
            let resident = s1::s1_resident_layer_set(receipt.config.num_hidden_layers, layers)?;
            residency::place_routed_experts_set(&mut weights, &resident);
            resident_layer_set = Some(resident);
        } else {
            residency::place_routed_experts(&mut weights, layers);
        }
        let placement = if profile.is_some() {
            "slowlink-profile"
        } else {
            "incumbent-auto"
        };
        (layers, placement, profile)
    };
    if profile.is_none() {
        weight_arenas = None;
    }
    let host_mapped_tensors = weights
        .iter()
        .filter(|entry| entry.residency == WeightResidency::HostMapped)
        .count();
    let payload_source = |entry: &WeightEntry| payload(&source, &recipe, entry, unified);
    let transaction = if profile.is_some() {
        let arenas = weight_arenas.take().ok_or_else(|| {
            "qwen4: admitted S1 profile has no charged weight arena plan".to_string()
        })?;
        hipfire_runtime::weight_store::fulfill_manifest_from_payloads_with_arenas(
            &weights,
            &mesh,
            receipt.config.num_hidden_layers,
            gpu,
            expected,
            arenas,
            payload_source,
        )
    } else {
        hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
            &weights,
            &mesh,
            receipt.config.num_hidden_layers,
            gpu,
            expected,
            payload_source,
        )
    }
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let vocab = receipt.config.vocab_size;
    let eos_token = receipt.config.eos_token_id;
    let num_layers = receipt.config.num_hidden_layers;
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_ple_rows(
        receipt.config,
        &recipe,
        transaction,
        &receipt.placements,
        gpu,
        max_seq,
        receipt.ple,
        receipt.ple_rows,
        state_format,
        backend,
        None,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    bundle.attach_npu_trunk_identity(npu_identity);
    if let Some(overlay) = trunk_overlay {
        if let Err(error) = bundle.attach_trunk_overlay(overlay) {
            let teardown = match bundle.free_gpu(gpu) {
                Ok(()) => "bundle released to the device pool".to_string(),
                Err(free) => format!("bundle teardown reported {free}"),
            };
            return Err(format!("qwen4 trunk overlay: {error}; {teardown}"));
        }
    }
    let slowlink_profile = profile.is_some();
    // The admitted profile is attached before the forward is built: the
    // forward sizes its tile scratch and arenas from it.
    if let Some(profile) = profile {
        bundle.attach_slowlink_profile(profile);
    }
    // A failed attach (including a named S1 owner-binding failure) leaves no
    // half-built load behind: the bundle is torn down and the teardown's own
    // disposition is part of the error, never a silent success.
    if let Err(error) = bundle.attach_forward(gpu, max_seq, scratch_mode) {
        let teardown = match bundle.free_gpu(gpu) {
            Ok(()) => "bundle released to the device pool".to_string(),
            Err(free) => format!("bundle teardown reported {free}"),
        };
        return Err(format!("qwen4 forward setup failed: {error}; {teardown}"));
    }
    if let Some(domain) = cache_domain {
        let cache = hipfire_runtime::session_cache::SessionCache::new(
            domain,
            session_cache_budget.unwrap_or(0),
        );
        bundle.attach_session_cache(cache);
    }
    if let Some(resident) = &resident_layer_set {
        eprintln!(
            "qwen4 slowlink load: placement={placement} resident layers={resident_layers} \
             resident_layer_set={resident:?} host-mapped tensors={host_mapped_tensors} \
             slowlink profile={slowlink_profile}"
        );
    } else {
        eprintln!(
            "qwen4 slowlink load: placement={placement} resident layers={resident_layers} \
             host-mapped tensors={host_mapped_tensors} slowlink profile={slowlink_profile}"
        );
    }
    Ok(Some(SlowlinkLoaded {
        bundle,
        vocab,
        eos_token,
        num_layers,
        arch_id,
        host_mapped_tensors,
        resident_layers,
        placement,
        slowlink_profile,
        ple_format: choice.format.name().to_string(),
        tokenizer,
        chat_template,
    }))
}
