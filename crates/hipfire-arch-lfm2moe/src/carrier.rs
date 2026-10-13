// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{lfm2_layer_bytes, Lfm2MoeState, Lfm2MoeWeights};
use crate::spec_impl::Lfm2MoeBundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;
use hipfire_runtime::offload::{self, Placement};

fn resolve_eos_tok(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, candidates: &[&str]) -> u32 {
    for s in candidates {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 {
            return ids[0];
        }
    }
    1
}

fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))
    } else {
        Err("no tokenizer.json found in model directory".into())
    }
}

/// Build the LFM2.5-MoE GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the `Lfm2MoeCarrier::load` model work. Preserves every
/// early-return and error string byte-for-byte. The per-source pp refusals,
/// config/weights seam (`Lfm2MoeConfig::from_hfq` / `config_from_source` and
/// `Lfm2MoeWeights::load` / `load_weights_from_source`), `maybe_screen_mmq`,
/// `Lfm2MoeState::new_with_max_seq`, and eos candidate list
/// `["<|im_end|>", "</s>", "<|endoftext|>"]` (fallback `1`) are arch knowledge
/// and live here. `dir_diag`, `resolve_source_meta`, `build_speculator`, and
/// `LoadedModel::skeleton` stay in the loader (loader-private).
pub fn load_lfm2moe_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Lfm2MoeBundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "lfm2moe: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "lfm2moe: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            let config = Lfm2MoeConfig::from_hfq(&hfq)?;
            let spill = resolve_lfm2_partial_offload(&hfq, &config, ctx)?;
            let resident = Placement::all_device(config.num_hidden_layers);
            let weights = Lfm2MoeWeights::load_with_placement(
                &mut hfq,
                &config,
                ctx.gpu,
                spill.as_ref().unwrap_or(&resident),
            )?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let eos_tok = resolve_eos_tok(&tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
                vision_config: None,
                vision_weights: None,
            })
        }
        ModelSource::Dir(source) => {
            if hipfire_config::memory::spill_requested() {
                return Err("lfm2moe: partial GPU offload (memory.gpu_layer_budget / \
                     memory.moe_expert_budget) needs the HFQ loader; this is a safetensors \
                     directory. Quantize to HFQ or unset both keys"
                    .into());
            }
            let config = crate::config::config_from_source(&source)
                .ok_or_else(|| "lfm2moe: failed to parse config from safetensors".to_string())?;
            let weights = crate::lfm2moe::load_weights_from_source(&source, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer = tokenizer_from_dir(&source)?;
            let eos_tok = resolve_eos_tok(&tokenizer, &["<|im_end|>", "</s>", "<|endoftext|>"]);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
                // Safetensors Dir loads stay text-only — the qwen35 carrier
                // makes the same choice; VL artifacts are HFQ-only here.
                vision_config: None,
                vision_weights: None,
            })
        }
    }
}

// ─────────────────────── Partial GPU offload (arch 11) ───────────────────────
//
// The shared placement machinery lives in `hipfire_runtime::offload`; this
// arch supplies only its byte census (`lfm2_layer_bytes`) and the measured
// capacity. Tier order, admission checks and the operator-facing lines are
// the runtime's, so the LFM2.5 policy cannot drift from Qwen3.5's.

/// Resolve partial GPU offload before the first allocation.
///
/// Returns `None` when both budgets are `Full` — the unset default, and the
/// zero-diff path: no measured capacity, no admission, no refusal, byte
/// output identical to the pre-offload loader.
///
/// Eligibility is `hipfire_config::memory::spill_support` over this load's
/// own facts (arch 11, tp 1 — the tp>1 routes never reach this carrier —
/// pp 1 — refused at the top of `load_lfm2moe_bundle` — no EP on this
/// route, no paged experts, the config's REAP keep-map, and the VL tower
/// probe). A configured spill on an unspillable path was already refused at
/// admission (`partial_offload_refusal`); the defensive arm below permits
/// only a fully resident placement that fits, otherwise refusing before
/// allocation.
fn resolve_lfm2_partial_offload(
    hfq: &hipfire_runtime::hfq::HfqFile,
    config: &Lfm2MoeConfig,
    ctx: &LoadCtx,
) -> Result<Option<Placement>, String> {
    use hipfire_config::memory::{
        effective_offload_budgets, offload_exec, spill_support, OffloadBudget, OffloadExec,
        SpillSupport,
    };
    let budgets = effective_offload_budgets();
    if budgets == (OffloadBudget::Full, OffloadBudget::Full) {
        return Ok(None);
    }
    let support = spill_support(
        crate::ARCH_ID,
        1,
        1,
        None,
        false,
        config.reap_keep.is_some(),
        lfm2_has_vision_tower(hfq),
    );
    if let SpillSupport::Refused(refusal) = support {
        return fail_closed_or_resident(hfq, config, ctx, refusal);
    }
    // `memory.offload_exec` decides WHO multiplies a spilled layer's weights.
    // Only `pcie` is implemented for arch 11: this crate's dense projections
    // launch through the llama-crate `weight_gemv`/`weight_gemv_residual`
    // helpers, not `hipfire-dispatch`'s `execute_steps`, so the CPU-exec seam
    // (`cpu_exec::plan_step`) never sees them and a `cpu`/`passback` request
    // would silently keep every spilled GEMV on the GPU over PCIe. Refuse
    // rather than half-honour the request.
    if offload_exec() != OffloadExec::Pcie {
        return Err(format!(
            "lfm2moe: memory.offload_exec={} is not implemented for arch 11 (LFM2.5 launches \
             its projections outside the CPU-exec seam); use pcie or unset the key",
            offload_exec()
        ));
    }
    let capacity = lfm2_placement_capacity(hfq, config, ctx)?;
    let placement = offload::resolve_placement(
        &capacity.layer_bytes,
        &capacity.inputs,
        budgets,
        offload::Orientation::SuffixResident,
    )?;
    Ok(Some(placement))
}

/// Whether this trunk carries the LFM2-VL vision tower: the same two facts
/// the loader's admission probe reads (tower tensor + `vision_config`
/// metadata) — a tower loads through its own lowered VL loader and cannot be
/// spilled.
fn lfm2_has_vision_tower(hfq: &hipfire_runtime::hfq::HfqFile) -> bool {
    let has_tensor = hfq
        .tensor_data("model.vision_tower.vision_model.embeddings.patch_embedding.weight")
        .is_some();
    has_tensor || hipfire_arch_lfm2_vl::vision_config_from_hfq(hfq).is_some()
}

struct Lfm2PlacementCapacity {
    layer_bytes: offload::LayerBytes,
    inputs: offload::CapacityInputs,
}

/// The census + every measured claim on device memory, mirroring the Qwen3.5
/// carrier's `placement_capacity_inputs`:
///
/// * free VRAM, read now;
/// * the KV reservation this load will allocate right after the weights
///   (`Lfm2MoeState::new_with_max_seq` builds a flat Q8 cache over
///   `num_attention_layers()` layers of `ctx.max_seq` rows — the stride comes
///   from the same `SlotKvTierPlan` the packed layouts describe, so census and
///   allocation cannot drift);
/// * prefill floor 0 (`prefill_floor_absent` — this arch has no prefill
///   sizing model; the shortfall stays visible in the capacity line rather
///   than hidden);
/// * a device-resident DFlash draft (`ctx.draft_path`), charged by file size
///   the way Qwen3.5 charges its sidecars;
/// * 128 MiB slack for the state scratch, conv-state rings, expert pointer
///   tables and logits row nobody sizes.
fn lfm2_placement_capacity(
    hfq: &hipfire_runtime::hfq::HfqFile,
    config: &Lfm2MoeConfig,
    ctx: &LoadCtx,
) -> Result<Lfm2PlacementCapacity, String> {
    let free_vram = ctx
        .gpu
        .hip
        .get_vram_info()
        .map_err(|e| format!("lfm2moe: pre-weight VRAM query: {e}"))?
        .0 as u64;
    // The state resolves KV through `HFQ_Q8_ONLY_POLICY` → Q8; resolve the
    // same way here instead of hard-coding one tier's stride.
    let mode =
        hipfire_runtime::kv_mode::resolve("", &hipfire_runtime::kv_mode::HFQ_Q8_ONLY_POLICY).mode;
    let tier = hipfire_runtime::kv_mode::SlotKvTierPlan::resolve(
        mode,
        config.num_key_value_heads,
        config.head_dim,
    )
    .map_err(|e| format!("lfm2moe: KV tier plan: {}", e.message))?;
    let n_attn = config.num_attention_layers().max(1);
    let stride = (tier.k_bytes_per_pos + tier.v_bytes_per_pos) as u64 * n_attn as u64;
    let kv = offload::kv_reserve(stride, ctx.max_seq).ok_or("lfm2moe: KV reservation overflow")?;
    let draft_bytes = ctx
        .draft_path
        .map(std::path::Path::new)
        .map(draft_file_reserve_bytes)
        .unwrap_or(0);
    let layer_bytes = lfm2_layer_bytes(hfq, config);
    let inputs = offload::CapacityInputs {
        free_vram,
        kv_bytes: kv.bytes,
        kv_rows: kv.rows,
        kv_stride: stride,
        prefill_floor: 0,
        prefill_floor_absent: true,
        draft_bytes,
        slack: 128u64 << 20,
    };
    Ok(Lfm2PlacementCapacity {
        layer_bytes,
        inputs,
    })
}

/// File bytes + 1/8 for the expanded/AWQ tensors + 512 MiB of scratch — the
/// Qwen3.5 carrier's `draft_file_reserve_bytes` formula, kept identical so a
/// draft costs the same estimate on both arches.
fn draft_file_reserve_bytes(path: &std::path::Path) -> u64 {
    let Ok(meta) = std::fs::metadata(path) else {
        return 0;
    };
    let file_bytes = meta.len();
    file_bytes + file_bytes / 8 + (512 << 20)
}

/// Defensive arm for a configured spill on an unspillable path (admission
/// normally refuses it first). Runs the pinned `Full`/`Full` plan against the
/// same capacity the placement would spend: if the fully resident model fits,
/// stay resident; otherwise refuse, naming the unspillable path.
fn fail_closed_or_resident(
    hfq: &hipfire_runtime::hfq::HfqFile,
    config: &Lfm2MoeConfig,
    ctx: &LoadCtx,
    refusal: hipfire_config::memory::SpillRefusal,
) -> Result<Option<Placement>, String> {
    use hipfire_config::memory::{OffloadBudget, SpillRefusal};
    let capacity = lfm2_placement_capacity(hfq, config, ctx)?;
    let weight_room = offload::weight_capacity(
        capacity.inputs.free_vram,
        capacity.inputs.kv_bytes,
        capacity.inputs.prefill_floor,
        capacity.inputs.draft_bytes,
        capacity.inputs.slack,
    );
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
    let reason = match refusal {
        SpillRefusal::Arch(arch_id) => format!("arch id {arch_id} has no host-placement path"),
        SpillRefusal::Tp(tp) => format!("tp={tp} loads every layer on every rank"),
        SpillRefusal::Pp(pp) => format!("pp={pp} bands own per-device layouts"),
        SpillRefusal::ExpertParallel(experts) => {
            format!("expert parallelism pins each of {experts} experts to one rank")
        }
        SpillRefusal::VisionTower => {
            "the LFM2-VL vision tower loads through its own lowered loader".to_string()
        }
        SpillRefusal::PagedExperts => "the weight pager owns the expert buffers".to_string(),
        SpillRefusal::ReapKeepMap => {
            "the pruned (REAP keep-map) expert load has no host-placement path".to_string()
        }
    };
    let device = capacity.layer_bytes.weight_total();
    Err(format!(
        "load refused: partial GPU offload was requested but cannot run here ({reason}), and \
         the fully resident model needs {} MiB of device weights against {} MiB available; \
         adjust memory.gpu_layer_budget / memory.moe_expert_budget, lower memory.max_seq, or \
         use a smaller quantization",
        device / (1 << 20),
        weight_room.weight_bytes / (1 << 20),
    ))
}
