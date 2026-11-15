// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The oracle load of a stock-FP8 Flash-Next artifact together with its
//! `<model>.mtp` head sidecar (`#[path = "common/stock_mtp_load.rs"]`; the
//! `qwen4_mtp_g942_cost` example includes it from `../tests/common/`).
//!
//! The product loader admits native MTP on the stock-FP8 recipe on gfx942 when
//! the head sidecar is attached (`speculation.mtp = on` binds it). This is the
//! standalone oracle load, the recipe `qwen4_mtp_head_dump` uses: the sidecar bound by
//! `bind_stock_mtp_sidecar` and merged at admission
//! (`admit_hfqm_artifact_with(.., Some(&bound))`), every `mtp.*` payload served
//! `Borrowed` from the sidecar's own mapping, everything else from the base, a
//! single fully resident device (a fixed `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` is
//! honored, `auto` refused), then `attach_forward`, `attach_mtp` and the loader's
//! int8 routed-down head settle. Nothing about the bundle differs from the
//! served one except how it was assembled.

#![allow(dead_code)]

use hipfire_arch_qwen4::artifact::{admit_hfqm_artifact_with, source_forward_scratch_mode};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::stock_mtp_sidecar::{
    bind_stock_mtp_sidecar, qwen4_stock_mtp_sidecar_path, stock_mtp_attached_line,
    stock_mtp_refusal_line,
};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::{ModelSource as _, SourcePayload};
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::Gpu;
use std::path::Path;

/// Whether `model` has the stock-FP8 `<model>.mtp` head sidecar next to it (the
/// artifacts this oracle load serves).
pub fn has_stock_mtp_sidecar(model: &Path) -> bool {
    qwen4_stock_mtp_sidecar_path(model).is_file()
}

/// Load `model` plus its `<model>.mtp` sidecar into a bundle with forward
/// resources and the native MTP head attached, for a context of `max_seq`.
/// `state_request` is the GDN state request handed to `resolve_state_format`
/// (`q8` / `fp32`; empty = the product auto); the caller checks the resolved
/// format. Prints the sidecar attach line.
pub fn load_stock_mtp_bundle(
    gpu: &mut Gpu,
    model: &Path,
    max_seq: usize,
    state_request: &str,
) -> Result<Qwen4Bundle, String> {
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let sidecar_path = qwen4_stock_mtp_sidecar_path(model);
    let bound = bind_stock_mtp_sidecar(&hfq, &sidecar_path)
        .map_err(|reason| stock_mtp_refusal_line(&sidecar_path, &reason))?;
    println!("{}", stock_mtp_attached_line(&bound));
    let receipt = admit_hfqm_artifact_with(&hfq, Some(&gpu.arch), false, Some(&bound))
        .map_err(|e| e.to_string())?;
    if receipt.config.mtp.is_none() || receipt.mtp_sidecar.is_none() {
        return Err("admission did not attach the MTP head sidecar".into());
    }
    let scratch_mode = source_forward_scratch_mode(gpu, &hfq, &receipt.config, &receipt.manifest)
        .map_err(|e| e.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, gpu);
    let source = HfqModelSource::from_hfq(hfq);
    // The sidecar is served Borrowed from its own mapping (never a range: a
    // second backing file would fail the direct reader's identity check).
    let mut side = HfqModelSource::from_hfq(bound.hfq);
    side.set_prefer_ranges(false);
    let weights = {
        let mut weights = receipt.manifest.weights.clone();
        match hipfire_arch_qwen4::expert_residency::expert_vram_layers_from_env()? {
            None => {}
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Layers(layers)) => {
                if gpu.is_uma() {
                    return Err("host-mapped experts are only for discrete GPUs".into());
                }
                let moved = hipfire_arch_qwen4::expert_residency::place_routed_experts(
                    &mut weights,
                    layers,
                );
                eprintln!(
                    "stock_mtp_load: routed experts: layers 0..{layers} in VRAM, {moved} tensors host-mapped"
                );
            }
            Some(hipfire_arch_qwen4::expert_residency::ExpertVramLayers::Auto) => {
                return Err(
                    "the stock-FP8 oracle load takes a fixed expert layer count, not auto".into(),
                );
            }
        }
        weights
    };
    let transaction = fulfill_manifest_from_payloads(
        &weights,
        &mesh,
        receipt.config.num_hidden_layers,
        gpu,
        expected,
        |entry| {
            // The one total routing rule: `mtp.*` is the sidecar's, everything
            // else the base's.
            if entry.name.starts_with("mtp.") {
                let (info, bytes) = side
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing sidecar tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            if let Some(payload) = hipfire_arch_qwen4::expert_arena::recipe_arena_payload(
                &source,
                &receipt.manifest.identity.recipe,
                entry,
            ) {
                return payload;
            }
            if entry.residency == hipfire_runtime::weight_manifest::WeightResidency::HostMapped {
                let (info, bytes) = source
                    .tensor_data(&entry.name)
                    .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
                return Ok(SourcePayload::Borrowed { info, bytes });
            }
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| e.to_string())?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        state_request,
        gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        &receipt.manifest.identity.recipe,
        transaction,
        &receipt.placements,
        gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(gpu, max_seq, scratch_mode)
        .map_err(|e| e.to_string())?;
    bundle.attach_mtp(gpu, max_seq).map_err(|e| e.to_string())?;
    // The loader settles the int8 routed-down verdict right after its head-attach step.
    bundle
        .settle_moe_down_a8_head(gpu, true)
        .map_err(|e| e.to_string())?;
    Ok(bundle)
}
