// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! qwen4_legacy_mq4_rowtile_oracle — real ORIGINAL G1 consumer for the
//! frozen PLAN59 legacy aggregate MQ4 tile family.
//!
//! Ordinary production load through the actual carriers
//! (`hipfire_loader::{Qwen4RankShared, open_qwen4_rank_source, load_qwen4_rank}`
//! yielding the ordinary production `.bundle`): source admission with the
//! arch-preferred PLE rows sidecar, trunk overlay, expert hot set / automatic
//! placement, weight fulfillment, assembly and forward program. No private
//! duplicate model loader, no synthetic expert-parallel plan, no session
//! seed, no forced expert-layer override. Speculation is explicitly off via
//! the actual `hipfire_runtime::loader_api::SpecLoadCfg` API
//! (`ngram_draft/mtp/dflash/dspark` all `Some(false)`, `native_mtp: false`,
//! `max_k: 0`); pins, tokenizer, canonical PLE selection and normal env
//! semantics are preserved. Runs the canonical prompt prefill, then performs
//! `hi` sequential real singleton `forward_token` greedy continuation
//! steps with a live lab G1 borrowed observer installed at the ACTUAL legacy
//! launch sites (GU ROWS8 branch in `gemv.rs`, down in `moe.rs`).
//!
//! For EACH requested logical R lo..=hi (default 1..=64, including tails):
//! - collects hi real X rows / selected top-10 ids / router probabilities
//!   and the stable borrowed weight identity (window seal bound once per
//!   window from the dispatch stage's `RoutedExpertWeights::immutable_identity`
//!   — the genuinely immutable ORIGINAL aggregate owner, not merely the same
//!   pointer; NO weight payload copies, ever — only small activation/id/prob
//!   values are copied as the forward collector is allowed to do),
//! - launches the producer's EXISTING fixed-8 candidate tile ABI against the
//!   same original per-row AR reference: GU 56B
//!   (`expert_ptrs@0,topk_indices@8,x@16,y_gate@24,y_up@32,M@40=output
//!   rows,K_TOP@44=10,n_exp@48,R@52`, grid `[ceil(M/8),K_TOP,ceil(R/8)]`,
//!   block 256, LDS 10240 re-used per token-8) and down ORIGINAL 52B
//!   (`expert_ptrs@0,topk_indices@8,hidden@16,outputs@24,M@32,K@36,
//!   N@40=total R,n_exp@44,k_top@48`, grid `[M,k_top,ceil(N/8)]`, block 32),
//! - compares every GU word, the actual existing activation output, every
//!   unweighted down word and the canonical rank combine as the existing
//!   ABIs dictate — never slot counts or head argmax only.
//!
//! The two deterministic pathological seeds derive ONLY mutable activations,
//! routes and router probabilities (never weight bytes): an all-invalid-slot
//! seed and a tail/poison seed with device guards. Router probabilities are
//! the dispatch stage's own `topk_weights` buffer (NOT a kernel launch
//! operand of the down kernel — intentionally absent from the launch-site
//! borrow). Refusal, a missing required stage or shape, or any nonzero
//! mismatch exits nonzero — never a synthetic-only qual PASS. No G2/G3/G4
//! content substitutes G1.
//!
//! C1 code comes directly from the existing loaded-owner raw-device getter.
//! Optional `--reference-gu-co` / `--reference-down-co` files must match that
//! actual raw SHA. GU C1 is K2560 ROWS8 ABI52 (not generic top10 ABI56);
//! candidate GU is ABI56 and both down stages are ABI52.
//! `--candidate-co` is the real emitted `legacy_mq4_rowtiles_pm_<arch>` image.
//! `--candidate-sha256`, when supplied, pins its decoded raw device bytes.
//! Stage-specific builders bind and validate actual module lookup names.
//!
//! CLI: `--pins pins.json --out OUT [--n-ctx 8192] [--rows A-B]
//!   [--reference-gu-co REF_GU.co] [--reference-down-co REF_DN.co]
//!   --candidate-co CAND.co [--candidate-sha256 HEX]`.
//! `--preload-plan-only` emits JSON to stdout (or `--out FILE`).
//! It reads metadata/prompt/code, allocates the transparent 15-slot native
//! workspace, and prewarms code modules; it performs no model/PLE tensor
//! payload fulfillment and no fixtures. Quotes are source-planned,
//! unmeasured, nonqualification budgets, not peaks or fit certificates.
//! Resource planning uses the existing public planner
//! `hipfire_loader::qwen4_preload_resource_plan` (metadata-only). The real
//! native workspace is allocated BEFORE the quote and the normal load; the
//! SAME physical owners are retained for every comparison cell via
//! guards/canaries/poison and logical views (no second allocation set).
//! Real free VRAM is queried with `Gpu::device_mem_info` AFTER the native
//! allocation; the normal producer load uses the remaining memory.
//! Pins use `qwen4-rowtile-pins-v1`. Model and weight sidecar pins require
//! prior stat-bound full SHA verification in the read-only runtime cache;
//! payload SHA is not recomputed here. Prompt and code bytes are hashed here.
//! Model remains alive through every comparison. Only small pointer tables,
//! activations, routes, probabilities, outputs and guards are oracle-owned.
//! A success receipt requires all 3*(hi-lo+1) requested cells to match.
//! A partial logical range contributes to aggregate Original G1; it is not
//! a whole-range certificate. The full production model is loaded per run;
//! small comparison buffers do not imply four full models fit concurrently.
//! Missing GPU/HIP, loaded owner, stage, shape, or any mismatch fails nonzero.

use hip_bridge::{HipError, HipResult};
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_loader::{
    load_qwen4_rank, open_qwen4_rank_source, qwen4_preload_resource_plan, Qwen4RankShared,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::argmax;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::verified_cache::{StatKey, VerifiedCache};
use rdna_compute::{lab_extract_device_bytes, DType, Gpu, GpuTensor, LabG1Witness};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const ROWS: usize = 64;
const K_TOP: usize = 10;
const USAGE: &str = "usage: qwen4_legacy_mq4_rowtile_oracle --pins pins.json [--preload-plan-only] [--out OUT] [--n-ctx 8192] [--rows A-B] [--reference-gu-co REF_GU.co] [--reference-down-co REF_DN.co] --candidate-co CAND.co [--candidate-sha256 RAW_HEX]";
const ORACLE_MODE: &str = "legacy-mq4-g1-rowtile-consumer-v1";

/// Frozen LP module/image family for the G1 tile image.
const LEGACY_MODULE: &str = "legacy_mq4_rowtiles";
const LEGACY_IMAGE_PREFIX: &str = "legacy_mq4_rowtiles_pm_";

fn es<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

fn hip_err(context: &str) -> HipError {
    HipError::new(0, context)
}

fn sha_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha(sha: &str) -> bool {
    sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())
}

/// One contiguous inclusive range within the observer's physical storage.
fn parse_rows(spec: &str) -> Result<(usize, usize), String> {
    let bad = || format!("bad --rows {spec:?}: expected A-B or R within 1..={ROWS}");
    let spec = spec.trim();
    let (a, b) = spec.split_once('-').unwrap_or((spec, spec));
    let lo: usize = a.trim().parse().map_err(|_| bad())?;
    let hi: usize = b.trim().parse().map_err(|_| bad())?;
    if lo == 0 || hi < lo || hi > ROWS {
        return Err(bad());
    }
    Ok((lo, hi))
}

// ---------------------------------------------------------------------------
// Pins (same schema as qwen4_rowtile_capture; no second convention)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Pin {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Pins {
    schema: String,
    arch: String,
    model: Pin,
    prompt: Pin,
    sidecars: Vec<Pin>,
}

#[derive(Clone)]
struct ResolvedPin {
    canonical: PathBuf,
    sha256: String,
}

fn resolve_pin(parent: &Path, pin: &Pin, label: &str) -> Result<(PathBuf, ResolvedPin), String> {
    if pin.path.is_empty() || Path::new(&pin.path).is_absolute() || pin.path.contains("..") {
        return Err(format!("{label}: path must be nonempty and pins-relative"));
    }
    if !valid_sha(&pin.sha256) {
        return Err(format!("{label}: sha256 must be 64 lowercase hex"));
    }
    let joined = parent.join(&pin.path);
    let canonical = joined
        .canonicalize()
        .map_err(|e| format!("{label} {}: {e}", joined.display()))?;
    Ok((
        joined,
        ResolvedPin {
            canonical,
            sha256: pin.sha256.to_lowercase(),
        },
    ))
}

/// Reuse a prior full SHA only for this exact canonical path and live stat.
/// No payload read, cache write, alias fallback, or streaming fallback.
fn verify_cached(
    cache: &VerifiedCache,
    joined: &Path,
    pin: &mut ResolvedPin,
    label: &str,
) -> Result<(), String> {
    if joined.canonicalize().map_err(es)? != pin.canonical {
        return Err(format!("{label}: canonical path drift"));
    }
    let metadata = std::fs::metadata(&pin.canonical)
        .map_err(|e| format!("{label} {}: {e}", pin.canonical.display()))?;
    let actual = cache
        .verified_digest(&pin.canonical, &StatKey::of(&metadata))
        .ok_or_else(|| format!(
            "{label} {}: missing prior stat-bound full SHA in runtime verified cache; refusing without payload rehash",
            pin.canonical.display()
        ))?;
    if actual != pin.sha256 {
        return Err(format!(
            "{label} {}: SHA256 mismatch actual={actual} pinned={}",
            pin.canonical.display(),
            pin.sha256
        ));
    }
    pin.sha256 = actual;
    Ok(())
}

fn verify_small(
    joined: &Path,
    canonical: &Path,
    want: &str,
    label: &str,
) -> Result<Vec<u8>, String> {
    let bytes =
        std::fs::read(joined).map_err(|e| format!("read {} {}: {e}", label, joined.display()))?;
    if sha_hex(&bytes) != want {
        return Err(format!("{label}: sha mismatch"));
    }
    if joined.canonicalize().map_err(es)?.as_path() != canonical {
        return Err(format!("{label}: canonical path drift"));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Stock producer load + transparent native workspace.
//
// Ownership: `Qwen4RankShared` is the ordinary request (spec explicitly off
// via the actual `SpecLoadCfg` API); `open_qwen4_rank_source` opens the real
// source with PLE rows resolved as admission does for rank 0; the public
// planner `qwen4_preload_resource_plan` quotes metadata-only; then
// `load_qwen4_rank` fulfills the ordinary production `.bundle`. No synthetic
// EP/session/layer override, no private duplicate loader.
// ---------------------------------------------------------------------------

struct NativeWorkspace {
    slots: Vec<GpuTensor>,
    // Points into the Box<Gpu> owned by run/Model; never moved or outlived.
    gpu: std::ptr::NonNull<Gpu>,
    device_bytes: u64,
    free_before: u64,
    free_after: u64,
    closed: bool,
}

impl NativeWorkspace {
    /// Allocate the actual 15-slot native workspace on the live device BEFORE
    /// any quote or model load. Every slot is a real device owner with
    /// poison guards; logical per-cell views alias these same owners (no
    /// second allocation set). Slots cover the widest comparison extents:
    /// 2 pointer tables, X/ids/probs, 4 GU, 2 activated/rotated, 2 down,
    /// 2 combines. Two retained logits owners (prefix + row logits) are real
    /// natives too, allocated here and reused through capture.
    fn allocate(
        gpu: &mut Box<Gpu>,
        rows_hi: usize,
        vocab: u64,
        n_exp: u64,
    ) -> Result<Self, String> {
        native_extra_bytes(rows_hi, vocab, n_exp, 1)?;
        let free_before = gpu.device_mem_info().map_err(es)?.0 as u64;
        let table = quote_mul(n_exp, 8, "pointer VALUES")?;
        let x = 2560 * 4;
        let route = K_TOP as u64 * 4;
        let gu = K_TOP as u64 * 640 * 4;
        let down = K_TOP as u64 * 2560 * 4;
        let combine = 2560 * 4;
        let rows = rows_hi as u64;
        let padded = quote_mul(quote_add(rows, 7, "padded rows")? / 8, 8, "padded rows")?;
        let capacities = [
            table,
            table,
            quote_mul(rows, x, "comparison X")?,
            quote_mul(rows, route, "comparison ids")?,
            quote_mul(rows, route, "comparison probabilities")?,
            quote_mul(padded, gu, "candidate gate")?,
            quote_mul(padded, gu, "candidate up")?,
            quote_mul(padded, gu, "reference gate")?,
            quote_mul(padded, gu, "reference up")?,
            quote_mul(padded, gu, "candidate activation")?,
            quote_mul(padded, gu, "reference activation")?,
            quote_mul(padded, down, "candidate down")?,
            quote_mul(padded, down, "reference down")?,
            quote_mul(rows, combine, "candidate combine")?,
            quote_mul(rows, combine, "reference combine")?,
        ];
        let mut workspace = Self {
            slots: Vec::with_capacity(17),
            gpu: std::ptr::NonNull::from(gpu.as_mut()),
            device_bytes: 0,
            free_before,
            free_after: free_before,
            closed: false,
        };
        let logits = quote_mul(vocab, 4, "vocabulary logits")?;
        for capacity in [logits, logits].into_iter().chain(capacities) {
            let guarded = quote_add(capacity, 2 * GUARD_BYTES as u64, "buffer guards")?;
            let len = usize::try_from(guarded).map_err(|_| "native slot extent exceeds usize")?;
            let poison = vec![POISON; len];
            let owner = gpu.upload_raw(&poison, &[len]).map_err(es)?;
            workspace.slots.push(owner);
            workspace.device_bytes =
                quote_add(workspace.device_bytes, guarded, "actual native requests")?;
        }
        gpu.hip.device_synchronize().map_err(es)?;
        workspace.free_after = gpu.device_mem_info().map_err(es)?.0 as u64;
        Ok(workspace)
    }

    fn check_logits_guards(&self, gpu: &Gpu) -> Result<(), String> {
        gpu.hip.device_synchronize().map_err(es)?;
        for (slot, owner) in self.slots[..2].iter().enumerate() {
            let bytes = gpu.download_raw_bytes(owner).map_err(es)?;
            if bytes[..GUARD_BYTES]
                .iter()
                .chain(&bytes[bytes.len() - GUARD_BYTES..])
                .any(|&b| b != POISON)
            {
                return Err(format!(
                    "logits slot {slot}: native allocation guard changed"
                ));
            }
        }
        Ok(())
    }

    fn close_checked(&mut self) -> Result<(), String> {
        if self.closed {
            return Ok(());
        }
        // Mark before cleanup: consumed owners are never retried by Drop.
        self.closed = true;
        // SAFETY: the private owner points into the still-live boxed Gpu.
        let gpu = unsafe { self.gpu.as_ref() };
        let mut first_error = gpu
            .hip
            .device_synchronize()
            .err()
            .map(|error| format!("native workspace cleanup synchronization: {error}"));
        for (slot, owner) in self.slots.drain(..).enumerate() {
            if let Err(error) = gpu.hip.free(owner.buf) {
                first_error.get_or_insert_with(|| {
                    format!("native workspace cleanup slot {slot}: {error}")
                });
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for NativeWorkspace {
    fn drop(&mut self) {
        if let Err(error) = self.close_checked() {
            eprintln!("{error}");
        }
    }
}

/// Ordinary `Qwen4RankShared` request: explicit spec-off via the actual
/// `SpecLoadCfg` API. `None` would inherit the ambient ladder; `Some(false)`
/// is the explicit off verdict the normal load path consumes.
fn spec_off() -> SpecLoadCfg {
    SpecLoadCfg {
        ngram_draft: Some(false),
        mtp: Some(false),
        dflash: Some(false),
        dspark: Some(false),
        ..SpecLoadCfg::default()
    }
}

fn shared_request(gpu: &Gpu, path: &Path, n_ctx: usize) -> Result<Qwen4RankShared, String> {
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(gpu);
    let vmm_granularity = match backend {
        hipfire_arch_qwen4::Qwen4KvBackend::Vmm => gpu.vmm_recommended_granularity().map_err(es)?,
        hipfire_arch_qwen4::Qwen4KvBackend::Legacy => 1,
    };
    Ok(Qwen4RankShared {
        continuous_batch: false,
        path: path.to_str().ok_or("model path is not UTF-8")?.to_owned(),
        max_seq: n_ctx,
        sequence: None,
        vision_reserve_bytes: 0,
        backend,
        vmm_granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: hipfire_config::process_value("HIPFIRE_STATE_QUANT").filter(|v| !v.is_empty()),
        spec: spec_off(),
        native_mtp: false,
        max_k: 0,
    })
}

struct Model {
    workspace: NativeWorkspace,
    gpu: Box<Gpu>,
    bundle: std::mem::ManuallyDrop<Qwen4Bundle>,
    vocab: usize,
    max_experts: usize,
    tokenizer: Tokenizer,
    arch: String,
    recipe: String,
    ple_rows_source: String,
    trunk_overlay_candidate: Option<PathBuf>,
    model_pin: ResolvedPin,
    prompt_pin: ResolvedPin,
    prompt_text: String,
    measured_snapshot: Option<serde_json::Value>,
    measured_identity: Option<serde_json::Value>,
    trunk_marker_present: bool,
    hotset_applied: bool,
    bundle_live: bool,
}

fn recipe_name(recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe) -> String {
    use hipfire_arch_qwen4::weights::Qwen4TierRecipe as R;
    match recipe {
        R::FlashNextPacked {
            trunk_tier,
            mtp_source,
        } => {
            format!("FlashNextPacked(trunk_tier={trunk_tier:?},mtp_source={mtp_source})")
        }
        R::FlashNextSlab {
            trunk_tier,
            mtp_source,
            ..
        } => {
            format!("FlashNextSlab(trunk_tier={trunk_tier:?},mtp_source={mtp_source})")
        }
        R::StockFp8 => "StockFp8".to_string(),
        R::SyntheticSourceBf16 => "SyntheticSourceBf16".to_string(),
        R::SyntheticPacked => "SyntheticPacked".to_string(),
    }
}

fn is_flash_next(recipe: &hipfire_arch_qwen4::weights::Qwen4TierRecipe) -> bool {
    use hipfire_arch_qwen4::weights::Qwen4TierRecipe as R;
    matches!(recipe, R::FlashNextPacked { .. } | R::FlashNextSlab { .. })
}

struct ModelLoadPlan {
    source: ModelSource,
    optics: (usize, usize),
    shared: Qwen4RankShared,
    quote: serde_json::Value,
}

impl Model {
    /// Plan the ordinary stock load metadata-only BEFORE any model payload
    /// fulfillment. The native workspace already exists (transparent
    /// allocation) and code modules are prewarmed, so fresh capacity
    /// already excludes their real device owners.
    fn preload_resource_plan(
        gpu: &Gpu,
        shared: Qwen4RankShared,
        source: ModelSource,
        prompt: &[u8],
        sidecars: &[(PathBuf, ResolvedPin)],
        native: &NativeWorkspace,
        code: &PrewarmedCode,
        n_ctx: usize,
        rows_lo: usize,
        rows_hi: usize,
    ) -> Result<(ModelLoadPlan, Tokenizer, Vec<u32>), String> {
        let hfq = match &source {
            ModelSource::Hfq(hfq) => hfq,
            ModelSource::Dir(_) => return Err("G1 requires the pinned HFQM source".into()),
        };
        let path = Path::new(&shared.path);
        let host_code_buffers = code.host_code_buffers;
        if let Some(ple) = hfq.ple_rows() {
            let canonical = ple.path().canonicalize().map_err(es)?;
            if !sidecars.iter().any(|(_, pin)| pin.canonical == canonical) {
                return Err(format!(
                    "resolved PLE sidecar {} is not pinned",
                    canonical.display()
                ));
            }
        }
        let source_seal = hfq.source_identity().map_err(es)?;
        let model_path = path.canonicalize().map_err(es)?;
        for file in &source_seal.files {
            if file.canonical_path != model_path
                && !sidecars
                    .iter()
                    .any(|(_, pin)| pin.canonical == file.canonical_path)
            {
                return Err(format!(
                    "resolved source file {} is not pinned",
                    file.canonical_path.display()
                ));
            }
        }
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(es)?;
        let text = std::str::from_utf8(prompt)
            .map_err(|_| "prompt file is not valid UTF-8".to_string())?;
        let prompt_ids = tokenizer.encode(text);
        if prompt_ids.is_empty() {
            return Err("prompt encodes to zero tokens".into());
        }
        if prompt_ids
            .len()
            .checked_add(rows_hi)
            .is_none_or(|n| n > n_ctx)
        {
            return Err(format!(
                "prompt len {} + {rows_hi} exceeds n_ctx {n_ctx}",
                prompt_ids.len()
            ));
        }
        let optics = Self::optics(&hfq)?;
        // Existing public metadata-only planner: admits the HFQ artifact and
        // joins the existing placement/census/host/PLE/staging/runtime
        // authorities. No fulfillment, no allocation, no fit declaration.
        let base = qwen4_preload_resource_plan(gpu, &shared, &hfq)?;
        let granule = quote_number(&base, "/context/granule_bytes")?;
        let extra = native_extra_bytes(rows_hi, optics.0 as u64, optics.1 as u64, granule)?;
        let native_device = extra.device_allocations;
        let prompt_arrays = quote_add(
            prompt.len() as u64,
            quote_mul(prompt_ids.len() as u64, 4, "prefix token allocation")?,
            "prefix host arrays",
        )?;
        let native_host_arrays = quote_sum(
            &[extra.host_arrays, prompt_arrays, host_code_buffers],
            "all native host arrays",
        )?;
        let uma_device_host = if gpu.is_uma() { native_device } else { 0 };
        let host_extra = quote_add(
            native_host_arrays,
            uma_device_host,
            "native host backing union",
        )?;
        let host = quote_add(
            quote_number(&base, "/whole_load_host_union_bytes")?,
            host_extra,
            "whole G1 host union",
        )?;
        let device = quote_add(
            quote_number(&base, "/component_union/device_union_upper_bound_bytes")?,
            native_device,
            "whole G1 device union",
        )?;
        let pinned = quote_add(
            quote_number(&base, "/whole_load_join/pinned_gtt_demand_bytes")?,
            uma_device_host,
            "whole G1 pinned/GTT union",
        )?;
        // Fresh free VRAM AFTER the real native allocation: queried here, so
        // the normal producer load below uses the remaining memory.
        let (free, total) = gpu.device_mem_info().map_err(es)?;
        let environment: std::collections::BTreeMap<_, _> = std::env::vars()
            .filter(|(key, _)| key.starts_with("HIPFIRE_"))
            .collect();
        let quote = json!({
            "schema": "qwen4_legacy_mq4_rowtile_oracle.preload_plan.v1",
            "producer": ORACLE_MODE,
            "source": file!(),
            "loader": "hipfire_loader::{Qwen4RankShared, open_qwen4_rank_source, load_qwen4_rank} (ordinary production serve load)",
            "planner": "hipfire_loader::qwen4_preload_resource_plan (existing public metadata-only planner)",
            "spec": "explicit off via SpecLoadCfg { ngram_draft/mtp/dflash/dspark: Some(false) }, native_mtp false, max_k 0",
            "planning_only": true, "source_planned": true, "unmeasured": true,
            "evidence_kind": "source_planned_unmeasured_nonqualification",
            "complete": false, "qualification": "nonqualification",
            "nonqualification": true,
            "model": path.display().to_string(),
            "context": base["context"],
            "parameters": {
                "logical_r": [rows_lo, rows_hi], "capture_extent": rows_hi,
                "physical_observer_rows": ROWS, "seed_families": 3,
                "cells": 3 * (rows_hi - rows_lo + 1),
                "prompt_tokens": prompt_ids.len(), "vocab": optics.0,
                "experts": optics.1, "top_k": K_TOP,
                "gu_m": 1280, "hidden": 2560, "intermediate": 640,
                "native_mtp_enabled": false,
                "native_workspace_arena_slots": 15,
                "native_workspace_logits_owners": 2,
                "native_workspace_device_bytes": native.device_bytes,
                "native_workspace_host_bytes": 0,
                "ple_format": format!("{:?}", hipfire_config::memory::ple_format()),
                "environment": environment,
            },
            "native_extras": {
                "capture_host_values_bytes": extra.capture,
                "capture_replacement_transient_bytes": extra.capture_transient,
                "guarded_arena_bytes": extra.arena,
                "guarded_arena_rounded_bytes": extra.arena_rounded,
                "guarded_allocation_count": 17, "arena_allocation_count": 15, "guard_bytes_each_end": GUARD_BYTES,
                "comparison_host_array_union_bytes": extra.compare_transient,
                "prefix_host_arrays_bytes": prompt_arrays,
                "workspace_retained_device_bytes": native.device_bytes,
                "workspace_retained_host_bytes": 0,
                "host_array_upper_bound_bytes": native_host_arrays,
                "lifetime_union": "SUM captured rows + retained seed/ids/positions + MAX(capture replacement+row logits, one comparison cell); SUM all 15 guarded arena owners plus two guarded logits roots once; MAX across cells/seeds",
                "device_allocation_upper_bound_bytes": native_device,
                "uma_device_backing_added_to_host_bytes": uma_device_host,
                "activation_fwht_workspace": "base reserve.workspace.mq_fixed (once)",
            },
            "code_modules": {
                "prewarmed_before_weights": true, "retained_through_comparison": true,
                "candidate_sha256": code.candidate.container_sha,
                "reference_gu_sha256": code.reference_gu.container_sha,
                "reference_down_sha256": code.reference_down.container_sha,
                "accounting": "actual loaded module/driver allocations excluded from fresh free capacity; never file length as VRAM",
            },
            "budget": {
                "host_union_upper_bound_bytes": host,
                "host_headroom_bytes": residency::HOST_RAM_HEADROOM_BYTES,
                "host_reservation_budget_bytes": quote_add(host, residency::HOST_RAM_HEADROOM_BYTES, "host budget with headroom")?,
                "pinned_gtt_demand_bytes": pinned,
                "device_allocation_upper_bound_bytes": device,
                "model_only_remaining_device_bytes": quote_number(&base, "/component_union/device_union_upper_bound_bytes")?,
                "runtime_allowance_already_in_base_bytes": quote_number(&base, "/component_union/runtime_allowance_bytes")?,
                "note": "planned reservation budget, not measured peak/RSS or qualification; existing runtime reserve and host headroom cover runtime/tokenizer/driver/bookkeeping once, not additional allocations charged twice",
            },
            "fresh_after_native": {"free_before_native_bytes": native.free_before, "free_immediately_after_native_bytes": native.free_after, "free_after_native_and_code_bytes": free, "free_device_bytes": free, "total_device_bytes": total,
                "capacity_scope": "this prewarmed process/card only after the real native workspace; not another card or cross-process sharing"},
            "union": base,
        });
        Ok((
            ModelLoadPlan {
                source,
                optics,
                shared,
                quote,
            },
            tokenizer,
            prompt_ids,
        ))
    }

    /// Read (vocab, experts) optics from metadata only; geometry refusals are
    /// named, never zero-filled.
    fn optics(hfq: &HfqFile) -> Result<(usize, usize), String> {
        let receipt = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, None)
            .map_err(|e| format!("qwen4 admission: {e}"))?;
        if !is_flash_next(&receipt.manifest.identity.recipe) {
            return Err(format!(
                "refusing non-FlashNext recipe {}: no G1 schema",
                recipe_name(&receipt.manifest.identity.recipe)
            ));
        }
        if receipt.config.vocab_size == 0 || receipt.config.num_experts == 0 {
            return Err("G1 resource quote: missing capture/vocabulary/expert geometry".into());
        }
        Ok((receipt.config.vocab_size, receipt.config.num_experts))
    }

    /// Logical view into the retained native workspace (same physical
    /// owners; guards/canaries/poison preserved outside the view extent).
    fn workspace_view(&self, slot: usize, offset: usize, len: usize) -> Result<GpuTensor, String> {
        let ws = self.workspace_slots();
        let base = ws.get(slot).ok_or("native workspace slot out of range")?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > base.buf.size().saturating_sub(2 * GUARD_BYTES))
        {
            return Err("native workspace view exceeds guarded extent".into());
        }
        Ok(GpuTensor {
            buf: base.buf.byte_view(GUARD_BYTES + offset, len),
            shape: vec![len / 4],
            dtype: DType::F32,
        })
    }

    /// Workspace slots live beside the model on the same `Gpu`; borrow them.
    fn workspace_slots(&self) -> &[GpuTensor] {
        &self.workspace.slots
    }

    /// Ordinary production load: single rank, spec explicitly off, no
    /// synthetic EP/session/layer override. Consumes the same `Gpu` that
    /// owns the native workspace and code modules.

    fn load(
        mut gpu: Box<Gpu>,
        tokenizer: Tokenizer,
        plan: ModelLoadPlan,
        workspace: NativeWorkspace,
        expected_arch: &str,
        model: (PathBuf, ResolvedPin),
        prompt_bytes: Vec<u8>,
        prompt: (PathBuf, ResolvedPin),
        n_ctx: usize,
    ) -> Result<Self, String> {
        if gpu.arch != expected_arch {
            return Err("G1 resource plan GPU arch changed before fulfillment".into());
        }
        let ModelLoadPlan {
            source,
            optics: (vocab, max_experts),
            shared,
            ..
        } = plan;
        let model_pin = &model.1;
        let path: &Path = &model_pin.canonical;
        let model_bytes = std::fs::metadata(path)
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        let arch = gpu.arch.clone();
        let prompt_text = String::from_utf8(prompt_bytes)
            .map_err(|_| "prompt file is not valid UTF-8".to_string())?;
        let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
        if loaded.bundle.config.vocab_size != vocab
            || loaded.bundle.config.num_experts != max_experts
        {
            let _ = loaded.bundle.free_gpu(&mut gpu);
            return Err("G1 stock load: bundle optics drifted from planned metadata".into());
        }
        if loaded.native_mtp {
            let _ = loaded.bundle.free_gpu(&mut gpu);
            return Err("G1 stock load: native MTP attached despite explicit spec-off".into());
        }
        if loaded.max_k != 0 {
            let _ = loaded.bundle.free_gpu(&mut gpu);
            return Err("G1 stock load: max_k nonzero despite explicit spec-off".into());
        }
        let ple_rows_source = match loaded.bundle.ple_source() {
            Some(_) => "ple-source: attached (ordinary carrier)",
            None => "ple-source: none (ordinary carrier resolved PLE-0)",
        }
        .to_string();
        let recipe = recipe_name(&loaded.bundle.weights.manifest.identity.recipe);
        let trunk_overlay_candidate = hipfire_config::memory::trunk_overlay_sidecar_path(path);
        let trunk_overlay_candidate = trunk_overlay_candidate
            .is_file()
            .then_some(trunk_overlay_candidate);
        // Activation presence is a named absence, never presence=>applied:
        // the settled marker cannot distinguish pre-existing native IU4 from
        // an applied trunk overlay (`carriers.rs` overlay-ambiguity rule).
        let trunk_applied = loaded.bundle.config.trunk_iu4_default.is_some();
        let measured_snapshot = match loaded
            .measured_resource_snapshot
            .as_ref()
            .map(|snapshot| {
                serde_json::from_str::<serde_json::Value>(snapshot.canonical_json())
                    .map_err(|error| format!("G1 measured resource snapshot JSON: {error}"))
            })
            .transpose()
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let _ = loaded.bundle.free_gpu(&mut gpu);
                return Err(error);
            }
        };
        let measured_identity = match loaded
            .measured_load_identity
            .as_ref()
            .map(|identity| {
                serde_json::from_str::<serde_json::Value>(identity.canonical_json())
                    .map_err(|error| format!("G1 measured load identity JSON: {error}"))
            })
            .transpose()
        {
            Ok(identity) => identity,
            Err(error) => {
                let _ = loaded.bundle.free_gpu(&mut gpu);
                return Err(error);
            }
        };
        let hotset_applied = loaded.bundle.weights.expert_tiers().is_some();
        eprintln!(
            "qwen4_legacy_mq4_rowtile_oracle: stock-loaded {} ({:.2} GB) on {} (vocab {vocab}), trunk marker {}",
            path.display(),
            model_bytes as f64 / 1e9,
            arch,
            if trunk_applied { "present (ambiguous: pre-existing or applied)" } else { "absent" }
        );
        Ok(Self {
            arch: gpu.arch.clone(),
            gpu,
            workspace,
            bundle: std::mem::ManuallyDrop::new(loaded.bundle),
            bundle_live: true,
            vocab,
            max_experts,
            tokenizer,
            recipe,
            ple_rows_source,
            trunk_overlay_candidate,
            model_pin: model.1.clone(),
            prompt_pin: prompt.1.clone(),
            prompt_text,
            measured_snapshot,
            measured_identity,
            trunk_marker_present: trunk_applied,
            hotset_applied,
        })
    }

    fn close_checked(&mut self) -> Result<(), String> {
        // Always attempt both independent owners, retaining the first error.
        let mut first_error = self.workspace.close_checked().err();
        if self.bundle_live {
            self.bundle_live = false;
            // SAFETY: the live flag consumes this owner exactly once.
            let bundle = unsafe { std::mem::ManuallyDrop::take(&mut self.bundle) };
            if let Err(error) = bundle.free_gpu(&mut self.gpu) {
                first_error
                    .get_or_insert_with(|| format!("ordinary Qwen4 bundle cleanup: {error}"));
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if let Err(error) = self.close_checked() {
            eprintln!("{error}");
        }
    }
}

// ---------------------------------------------------------------------------
// Borrowed-reference capture: one arbiter row of real operands.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RowObs {
    /// Fired at least once with a real hot legacy route.
    hot: bool,
    /// Borrowed per-row blobs copied synchronously (NOT weight payload):
    /// X row (F32), ids row, router probs row, gate/up reference words,
    /// down reference expanded words. Weights stay borrowed in place.
    x_row: Vec<u8>,
    ids_row: Vec<u8>,
    /// Device-pointer VALUES of the resident expert table (n_exp u64s, NOT
    /// weight bytes): staged into an owned pointer table aliasing the SAME
    /// resident owners for the per-R tile launches.
    table_row: Vec<u8>,
    down_table: Vec<u8>,
    rotated: Vec<u8>,
    probs_row: Vec<u8>,
    gate_rows: Vec<u8>,
    up_rows: Vec<u8>,
    down_rows: Vec<u8>,
    expr_set: Option<u64>,
    m: usize,
    mi: usize,
    down_m: usize,
    down_k: usize,
    bf16_round_trip: bool,
    n_exp: usize,
    module: String,
    object_sha: String,
    device_sha: String,
}

struct ObsState {
    rows: [RowObs; ROWS],
    row: usize,
    max_experts: usize,
    error: Option<String>,
}

fn fail(state: &mut ObsState, message: String) -> HipError {
    if state.error.is_none() {
        state.error = Some(message.clone());
    }
    hip_err(&message)
}

fn download_prefix(
    gpu: &Gpu,
    tensor: &GpuTensor,
    len: usize,
    label: &str,
) -> Result<Vec<u8>, HipError> {
    if tensor.buf.size() < len {
        return Err(hip_err(&format!(
            "{label}: device buffer {} bytes < meaningful extent {len}",
            tensor.buf.size()
        )));
    }
    gpu.hip.device_synchronize()?;
    let mut out = vec![0u8; len];
    gpu.hip.memcpy_dtoh(&mut out, &tensor.buf)?;
    Ok(out)
}

fn check_finite_f32(bytes: &[u8], label: &str) -> Result<(), String> {
    if bytes.len() % 4 != 0 {
        return Err(format!("{label}: byte len not F32-aligned"));
    }
    for (i, w) in bytes.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        if !v.is_finite() {
            return Err(format!("{label}: non-finite F32 word {i}"));
        }
    }
    Ok(())
}

/// Live G1 borrowed observer: fires at the ACTUAL legacy launch boundary,
/// downloads only small activation/id/prob/reference words synchronously,
/// never weight payload, retains nothing, never alters selection/recording.
fn observe(state: &Arc<Mutex<ObsState>>, gpu: &Gpu, witness: &LabG1Witness<'_>) -> HipResult<()> {
    let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(message) = st.error.clone() {
        return Err(hip_err(&message));
    }
    if gpu.graphs.capture_mode || gpu.replay.is_recording() {
        return Err(fail(
            &mut st,
            "capture during graph capture or replay recording".to_string(),
        ));
    }
    let row = st.row;
    if row >= ROWS {
        return Ok(());
    }
    let seal = gpu.lab_g1_window().ok_or_else(|| {
        fail(
            &mut st,
            "g1 window: no pipeline-bound seal at launch boundary".to_string(),
        )
    })?;
    if st.rows[row]
        .expr_set
        .is_some_and(|identity| identity != seal.expert_set_identity)
    {
        return Ok(());
    }
    if seal.router_probability_words.len() != K_TOP {
        return Err(fail(
            &mut st,
            "g1 window: probability extent is not top-10".into(),
        ));
    }
    let mut slot = std::mem::take(&mut st.rows[row]);
    match witness {
        LabG1Witness::GateUp(b) => {
            if b.k_top != K_TOP || b.rows != 1 {
                return Err(fail(
                    &mut st,
                    format!("gate_up: k_top={} rows={} (need 10/1)", b.k_top, b.rows),
                ));
            }
            if b.m != 1280 {
                return Err(fail(&mut st, format!("gate_up: noncanonical M={}", b.m)));
            }
            if b.n_exp != seal.n_exp
                || b.n_exp == 0
                || b.n_exp > st.max_experts
                || seal.expert_set_identity == 0
            {
                return Err(fail(&mut st, "gate_up: window seal mismatch".to_string()));
            }
            let mi = b.m / 2;
            let x = download_prefix(gpu, b.x, 2560 * 4, "gate_up X")
                .map_err(|e| fail(&mut st, format!("gate_up X: {e}")))?;
            check_finite_f32(&x, "gate_up X").map_err(|e| fail(&mut st, e))?;
            let ids = download_prefix(gpu, b.topk_indices, K_TOP * 4, "gate_up ids")
                .map_err(|e| fail(&mut st, format!("gate_up ids: {e}")))?;
            // Small pointer-table snapshot: device-pointer VALUES only
            // (n_exp u64s, NOT weight bytes), so the per-R engine can stage
            // an owned pointer table aliasing the SAME resident owners.
            let table = download_prefix(gpu, b.expert_ptrs, b.n_exp * 8, "gate_up table")
                .map_err(|e| fail(&mut st, format!("gate_up table: {e}")))?;
            let gate = download_prefix(gpu, b.y_gate, K_TOP * mi * 4, "gate_up gate")
                .map_err(|e| fail(&mut st, format!("gate_up gate: {e}")))?;
            check_finite_f32(&gate, "gate_up gate").map_err(|e| fail(&mut st, e))?;
            let up = download_prefix(gpu, b.y_up, K_TOP * mi * 4, "gate_up up")
                .map_err(|e| fail(&mut st, format!("gate_up up: {e}")))?;
            check_finite_f32(&up, "gate_up up").map_err(|e| fail(&mut st, e))?;
            // Actual loaded ROWS8 reference symbol + device digest: the C1
            // reference for GU is the actual K2560 ROWS8 ABI52 entry — NOT
            // the generic top10 entry (which carries the K word, ABI56).
            if slot.module.is_empty() {
                let actual = "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched";
                let (module, object) = gpu
                    .lab_kernel_identity(actual)
                    .map_err(|e| fail(&mut st, format!("gate_up identity: {e}")))?;
                let (_, _, device) = gpu
                    .lab_kernel_device_identity(actual)
                    .map_err(|e| fail(&mut st, format!("gate_up device identity: {e}")))?;
                slot.module = module;
                slot.object_sha = object;
                slot.device_sha = device;
            }
            slot.x_row = x;
            slot.ids_row = ids;
            slot.table_row = table;
            slot.gate_rows = gate;
            slot.up_rows = up;
            slot.expr_set = Some(seal.expert_set_identity);
            slot.bf16_round_trip = seal.bf16_round_trip;
            slot.m = b.m;
            slot.mi = mi;
            slot.n_exp = b.n_exp;
            slot.hot = true;
        }
        LabG1Witness::Down(b) => {
            if b.k_top != K_TOP || b.rows != 1 {
                return Err(fail(
                    &mut st,
                    format!("down: k_top={} rows={} (need 10/1)", b.k_top, b.rows),
                ));
            }
            if (b.m, b.k) != (2560, 640) {
                return Err(fail(
                    &mut st,
                    format!("down: noncanonical M/K={}/{}", b.m, b.k),
                ));
            }
            if b.n_exp != seal.n_exp
                || b.n_exp == 0
                || b.n_exp > st.max_experts
                || seal.expert_set_identity == 0
            {
                return Err(fail(&mut st, "down: window seal mismatch".to_string()));
            }
            let ids = download_prefix(gpu, b.topk_indices, K_TOP * 4, "down ids")
                .map_err(|e| fail(&mut st, format!("down ids: {e}")))?;
            let mut probs = Vec::with_capacity(K_TOP * 4);
            for word in &seal.router_probability_words {
                probs.extend_from_slice(&word.to_le_bytes());
            }
            check_finite_f32(&probs, "down probs").map_err(|e| fail(&mut st, e))?;
            let down = download_prefix(gpu, b.expert_outputs, K_TOP * b.m * 4, "down expanded")
                .map_err(|e| fail(&mut st, format!("down expanded: {e}")))?;
            check_finite_f32(&down, "down expanded").map_err(|e| fail(&mut st, e))?;
            let x = download_prefix(gpu, b.hidden, K_TOP * b.k * 4, "down hidden")
                .map_err(|e| fail(&mut st, format!("down hidden: {e}")))?;
            check_finite_f32(&x, "down hidden").map_err(|e| fail(&mut st, e))?;
            let table = download_prefix(gpu, b.expert_ptrs, b.n_exp * 8, "down table")
                .map_err(|e| fail(&mut st, format!("down table: {e}")))?;
            slot.down_table = table;
            slot.rotated = x;
            if slot.ids_row != ids && !slot.ids_row.is_empty() {
                return Err(fail(
                    &mut st,
                    "down ids differ from gate_up ids".to_string(),
                ));
            }
            slot.ids_row = ids;
            slot.probs_row = probs;
            slot.down_rows = down;
            slot.down_m = b.m;
            slot.down_k = b.k;
            slot.hot = true;
        }
    }
    st.rows[row] = slot;
    Ok(())
}

// ---------------------------------------------------------------------------
// Candidate/reference CO loading: real typed symbols, stage-classified.
// ---------------------------------------------------------------------------

/// Actual loaded candidate/reference helper: loads an explicit external CO
/// file and binds REAL typed symbols via `module_get_function`. The stored
/// actual lookup name classifies the ABI (GU56 vs C1-GU52 vs down52);
/// stage-specific launch builders validate it — no generic mixed-ABI raw
/// launch, no nameless whole-image waiver (the combined legacy image holds
/// GU56 + down52, which differ, so no single-ABI whole-image claim applies).
struct LoadedImage {
    /// Keeps every bound function's actual module owner alive.
    module: hip_bridge::Module,
    /// Actual raw device SHAs by bound symbol (pinned per symbol).
    device_sha: HashMap<String, String>,
    container_sha: String,
    image_path: String,
    host_input_capacity: usize,
}

fn bound_symbol_kind(symbol: &str) -> Result<&'static str, String> {
    if symbol.contains("gate_up") && symbol.contains("_t8_pm_") {
        return Ok("gu56");
    }
    if symbol.contains("down") && symbol.contains("_t8_pm_") {
        return Ok("down52");
    }
    if symbol == "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched" {
        return Ok("c1gu52");
    }
    if symbol == "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded" {
        return Ok("c1down52");
    }
    Err(format!("unclassifiable symbol {symbol:?}"))
}

/// Candidate GU symbol for this arch; gfx1100 has NO gate_up twin (frozen).
fn candidate_gu_symbol(arch: &str) -> Result<String, String> {
    match arch {
        "gfx1201" | "gfx1151" => Ok(format!(
            "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_{arch}"
        )),
        other => Err(format!("no gate_up _t8 twin for arch {other:?}")),
    }
}

/// Candidate down symbol for this arch.
fn candidate_down_symbol(arch: &str) -> Result<String, String> {
    match arch {
        "gfx1201" | "gfx1151" | "gfx1100" => Ok(format!(
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_{arch}"
        )),
        other => Err(format!("no down _t8 twin for arch {other:?}")),
    }
}

fn load_image(
    gpu: &Gpu,
    path: &Path,
    symbols: &[String],
    label: &str,
) -> Result<(LoadedImage, Vec<u8>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {label} {}: {e}", path.display()))?;
    if bytes.is_empty() {
        return Err(format!("{label}: empty code object"));
    }
    let container_sha = sha_hex(&bytes);
    let device = lab_extract_device_bytes(&bytes, &gpu.arch).ok_or_else(|| {
        format!(
            "{label}: no decodable {} device image (refusing container fallback)",
            gpu.arch
        )
    })?;
    // Decode once to bytes for the actual SHA pin; no source/module labels.
    let device_sha_all = sha_hex(device);
    let module = gpu.hip.module_load_data(&bytes).map_err(es)?;
    let mut device_sha = HashMap::new();
    for symbol in symbols {
        bound_symbol_kind(symbol)?;
        gpu.hip
            .module_get_function(&module, symbol)
            .map_err(|e| format!("{label}: bind {symbol:?}: {e}"))?;
        device_sha.insert(symbol.clone(), device_sha_all.clone());
    }
    Ok((
        LoadedImage {
            module,
            device_sha,
            container_sha,
            image_path: path.display().to_string(),
            host_input_capacity: bytes.capacity(),
        },
        bytes,
    ))
}

/// Borrow the exact resident original code owner; optional external files must
/// decode to those same raw bytes, rather than merely sharing a symbol name.
fn load_original(gpu: &Gpu, path: Option<&Path>, symbol: &str) -> Result<LoadedImage, String> {
    let raw = gpu.lab_kernel_device_bytes(symbol).map_err(es)?;
    let raw_sha = sha_hex(raw);
    if let Some(path) = path {
        let (image, _) = load_image(gpu, path, &[symbol.to_owned()], "original")?;
        if image.device_sha.get(symbol) != Some(&raw_sha) {
            return Err(format!(
                "original {symbol}: external raw SHA differs from loaded owner"
            ));
        }
        return Ok(image);
    }
    bound_symbol_kind(symbol)?;
    let module = gpu.hip.module_load_data(raw).map_err(es)?;
    gpu.hip.module_get_function(&module, symbol).map_err(es)?;
    Ok(LoadedImage {
        module,
        device_sha: HashMap::from([(symbol.to_owned(), raw_sha.clone())]),
        container_sha: raw_sha,
        image_path: format!("loaded-owner:{symbol}"),
        host_input_capacity: 0, // raw image is borrowed from the loaded owner
    })
}

/// Actual K2560 ROWS8 C1 symbol: 52 bytes, no generic K or appended R.
fn launch_gu_original(
    gpu: &Gpu,
    image: &LoadedImage,
    table: u64,
    ids: u64,
    x: u64,
    gate: u64,
    up: u64,
    m: i32,
    n_exp: i32,
) -> Result<(), String> {
    let function = gpu
        .hip
        .module_get_function(
            &image.module,
            "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
        )
        .map_err(es)?;
    let mut args = hip_bridge::KernargBlob::new();
    args.push_ptr(table as *const std::ffi::c_void);
    args.push_ptr(ids as *const std::ffi::c_void);
    args.push_ptr(x as *const std::ffi::c_void);
    args.push_ptr(gate as *const std::ffi::c_void);
    args.push_ptr(up as *const std::ffi::c_void);
    args.push_i32(m);
    args.push_i32(10);
    args.push_i32(n_exp);
    let mut blob = args.into_vec();
    // SAFETY: resident immutable weights remain Model-owned; oracle buffers
    // outlive this synchronous native C1 submission.
    unsafe {
        gpu.hip.launch_kernel_blob(
            &function,
            [m as u32 / 8, 10, 1],
            [256, 1, 1],
            0,
            None,
            &mut blob,
        )
    }
    .map_err(es)?;
    gpu.hip.device_synchronize().map_err(es)
}

/// GU56 launch, bound by actual literal module-function lookups.
/// Inputs are oracle-owned small buffers aliasing Model-owned immutable
/// expert payload addresses; outputs are guarded oracle-owned scratch.
fn launch_gu_tile(
    gpu: &Gpu,
    img: &LoadedImage,
    symbol: &str,
    expert_ptrs: u64,
    topk: u64,
    x: u64,
    y_gate: u64,
    y_up: u64,
    m: i32,
    k_top: i32,
    n_exp: i32,
    r: i32,
) -> Result<(), String> {
    let f = match symbol {
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201" => {
            gpu.hip.module_get_function(
                &img.module,
                "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1201",
            )
        }
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151" => {
            gpu.hip.module_get_function(
                &img.module,
                "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched_t8_pm_gfx1151",
            )
        }
        _ => return Err(format!("GU56: unsupported actual symbol {symbol:?}")),
    }
    .map_err(es)?;
    let grid = [(m as u32).div_ceil(8), k_top as u32, (r as u32).div_ceil(8)];
    let mut ka = hip_bridge::KernargBlob::new();
    ka.push_ptr(expert_ptrs as *const std::ffi::c_void);
    ka.push_ptr(topk as *const std::ffi::c_void);
    ka.push_ptr(x as *const std::ffi::c_void);
    ka.push_ptr(y_gate as *const std::ffi::c_void);
    ka.push_ptr(y_up as *const std::ffi::c_void);
    ka.push_i32(m);
    ka.push_i32(k_top);
    ka.push_i32(n_exp);
    ka.push_i32(r);
    let mut blob = ka.into_vec();
    // SAFETY: all pointers address live owner/oracle buffers; 56B layout is
    // the frozen producer ABI; the launch is synchronous with a sync below.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&f, grid, [256, 1, 1], 10_240, None, &mut blob)
    }
    .map_err(es)?;
    gpu.hip.device_synchronize().map_err(es)?;
    Ok(())
}

/// Down tile launch: REAL typed symbol + ORIGINAL frozen 52B ABI (N = R).
/// Validates the stored actual lookup name is a `down52`/`c1down52` symbol.
fn launch_down_tile(
    gpu: &Gpu,
    img: &LoadedImage,
    symbol: &str,
    expert_ptrs: u64,
    topk: u64,
    hidden: u64,
    outputs: u64,
    m: i32,
    k: i32,
    n: i32,
    n_exp: i32,
    k_top: i32,
) -> Result<(), String> {
    let f = match symbol {
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201" => {
            gpu.hip.module_get_function(
                &img.module,
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1201",
            )
        }
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151" => {
            gpu.hip.module_get_function(
                &img.module,
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1151",
            )
        }
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1100" => {
            gpu.hip.module_get_function(
                &img.module,
                "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_t8_pm_gfx1100",
            )
        }
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded" => gpu.hip.module_get_function(
            &img.module,
            "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
        ),
        _ => return Err(format!("Down52: unsupported actual symbol {symbol:?}")),
    }
    .map_err(es)?;
    let rows = if bound_symbol_kind(symbol)? == "c1down52" {
        n as u32
    } else {
        (n as u32).div_ceil(8)
    };
    let grid = [m as u32, k_top as u32, rows];
    let mut ka = hip_bridge::KernargBlob::new();
    ka.push_ptr(expert_ptrs as *const std::ffi::c_void);
    ka.push_ptr(topk as *const std::ffi::c_void);
    ka.push_ptr(hidden as *const std::ffi::c_void);
    ka.push_ptr(outputs as *const std::ffi::c_void);
    ka.push_i32(m);
    ka.push_i32(k);
    ka.push_i32(n);
    ka.push_i32(n_exp);
    ka.push_i32(k_top);
    let mut blob = ka.into_vec();
    // SAFETY: same contract as the GU tile; 52B is the original ABI.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&f, grid, [32, 1, 1], 0, None, &mut blob)
    }
    .map_err(es)?;
    gpu.hip.device_synchronize().map_err(es)?;
    Ok(())
}

struct PrewarmedCode {
    candidate: LoadedImage,
    reference_gu: LoadedImage,
    reference_down: LoadedImage,
    gu_symbol: String,
    down_symbol: String,
    candidate_bytes: Vec<u8>,
    host_code_buffers: u64,
}

/// Prewarm code modules through the SAME `Gpu` that already owns the native
/// workspace (`gpu` is borrowed, never re-initialized): stage the candidate
/// image, C1 originals and kernel owners there; the workspace stays alive.
fn prewarm_code(
    gpu: &mut Gpu,
    expected_arch: &str,
    candidate: &Path,
    reference_gu: Option<&Path>,
    reference_down: Option<&Path>,
    candidate_sha: Option<&str>,
) -> Result<PrewarmedCode, String> {
    if gpu.arch != expected_arch {
        return Err(format!(
            "pins arch {expected_arch:?} != actual GPU arch {:?}",
            gpu.arch
        ));
    }
    let gu_symbol = candidate_gu_symbol(&gpu.arch)?;
    let down_symbol = candidate_down_symbol(&gpu.arch)?;
    let (candidate, candidate_bytes) = load_image(
        &gpu,
        candidate,
        &[gu_symbol.clone(), down_symbol.clone()],
        "candidate",
    )?;
    if let Some(want) = candidate_sha {
        if candidate.device_sha.get(&gu_symbol).map(String::as_str) != Some(want) {
            return Err("candidate raw device SHA mismatch".into());
        }
    }
    // The loaded-owner getter deliberately refuses an unloaded C1 symbol.
    // Preload through the very same public kernel owner as native dispatch;
    // do not substitute an embedded/source digest for that loaded identity.
    gpu.ensure_kernel_public(
        "gemv_mq4g256v2_moe_gate_up_top10_indexed_batched",
        rdna_compute::kernels::GEMV_MQ4G256V2_MOE_GATE_UP_TOP10_INDEXED_BATCHED_SRC,
        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
    )
    .map_err(es)?;
    gpu.ensure_kernel_public(
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
        rdna_compute::kernels::GEMV_MQ4G128V2_MOE_DOWN_TOP10_INDEXED_BATCHED_EXPANDED_SRC,
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
    )
    .map_err(es)?;
    let reference_gu = load_original(
        &gpu,
        reference_gu,
        "gemv_mq4g256v2_moe_gate_up_k2560_rows8_indexed_batched",
    )?;
    let reference_down = load_original(
        &gpu,
        reference_down,
        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
    )?;
    let host_code_buffers = quote_add(
        candidate_bytes.capacity() as u64,
        reference_gu
            .host_input_capacity
            .max(reference_down.host_input_capacity) as u64,
        "code input buffer union",
    )?;
    // These exact module owners stay alive through the SAME Model load and
    // comparisons. Fresh capacity queried after this point already excludes
    // their actual code/driver allocation; file length is NOT VRAM.
    // The native workspace borrowed `gpu` stays owned by the caller.
    gpu.hip.device_synchronize().map_err(es)?;
    Ok(PrewarmedCode {
        candidate,
        reference_gu,
        reference_down,
        gu_symbol,
        down_symbol,
        candidate_bytes,
        host_code_buffers,
    })
}

fn collect_observed(rows: &[RowObs], select: fn(&RowObs) -> &[u8]) -> Vec<u8> {
    // Geometry was checked before comparison. Avoid collect/FlatMap's
    // geometric spare capacity so the host quote matches these allocations.
    let mut bytes = Vec::with_capacity(rows.iter().map(|row| select(row).len()).sum());
    for row in rows {
        bytes.extend_from_slice(select(row));
    }
    bytes
}

// Metadata quotes describe allocation requests, not a measured peak or a fit.
// Keep this accounting next to the constructors it describes: the physical
// capture is rows_hi, while each of the three seed families reuses one arena.
fn quote_number(value: &serde_json::Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("G1 resource quote: missing numeric {pointer}"))
}

fn enforce_preload_capacity(quote: &serde_json::Value) -> Result<(), String> {
    use hipfire_arch_qwen4::expert_residency as residency;
    // The report-only supported-scratch-mode union is not a device fit
    // verdict. The normal producer owns device admission against fresh
    // capacity on this same GPU, after the real native owners and code
    // prewarm. Keep the disjoint whole-host and pinned/GTT guards here.
    // Reuse the actual host/GTT guards, not an independent admission policy.
    // Host headroom is added inside check_host_ram exactly once; pageable
    // oracle arrays never become an invented pinned/GTT charge.
    residency::check_host_ram(
        quote_number(quote, "/budget/host_union_upper_bound_bytes")?,
        rdna_compute::kv_slots::mem_available_bytes(),
        residency::ttm_pool_estimate(),
    )?;
    residency::check_gtt_cap(
        quote_number(quote, "/budget/pinned_gtt_demand_bytes")?,
        residency::gtt_budget(),
    )
}

fn quote_add(a: u64, b: u64, component: &str) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("G1 resource quote: {component} overflows"))
}

fn quote_mul(a: u64, b: u64, component: &str) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("G1 resource quote: {component} overflows"))
}

fn quote_sum(values: &[u64], component: &str) -> Result<u64, String> {
    values
        .iter()
        .try_fold(0, |sum, &n| quote_add(sum, n, component))
}

fn quote_round(bytes: u64, granule: u64) -> Result<u64, String> {
    if granule == 0 {
        return Err("G1 resource quote: device granule is zero".into());
    }
    quote_mul(
        quote_add(bytes, granule - 1, "allocation rounding")? / granule,
        granule,
        "allocation rounding",
    )
}

#[derive(Debug)]
struct NativeExtraBytes {
    capture: u64,
    capture_transient: u64,
    arena: u64,
    arena_rounded: u64,
    compare_transient: u64,
    host_arrays: u64,
    device_allocations: u64,
}

fn native_extra_bytes(
    rows_hi: usize,
    vocab: u64,
    n_exp: u64,
    granule: u64,
) -> Result<NativeExtraBytes, String> {
    if !(1..=ROWS).contains(&rows_hi) || vocab == 0 || n_exp == 0 {
        return Err("G1 resource quote: missing capture/vocabulary/expert geometry".into());
    }
    let rows = rows_hi as u64;
    let padded = quote_mul(quote_add(rows, 7, "padded rows")? / 8, 8, "padded rows")?;
    let table = quote_mul(n_exp, 8, "pointer VALUES")?;
    let x = 2560 * 4;
    let route = K_TOP as u64 * 4;
    let gu = K_TOP as u64 * 640 * 4;
    let down = K_TOP as u64 * 2560 * 4;
    let combine = 2560 * 4;
    let row = quote_sum(
        &[x, route, route, table, table, gu, gu, gu, down],
        "capture row",
    )?;
    let capture = quote_mul(rows, row, "captured rows")?;
    // A slot can already contain a prior layer's observation while replacement
    // downloads are alive. GU and down callbacks do not run concurrently.
    let capture_transient = quote_sum(&[x, route, table, gu, gu], "GU replacement")?.max(
        quote_sum(&[route, route, table, gu, down], "down replacement")?,
    );
    // Exact 15 allocation capacities in run_compare: 2 pointer tables,
    // X/ids/probs, 4 GU, 2 activated/rotated, 2 down, 2 combines.
    let capacities = [
        table,
        table,
        quote_mul(rows, x, "comparison X")?,
        quote_mul(rows, route, "comparison ids")?,
        quote_mul(rows, route, "comparison probabilities")?,
        quote_mul(padded, gu, "candidate gate")?,
        quote_mul(padded, gu, "candidate up")?,
        quote_mul(padded, gu, "reference gate")?,
        quote_mul(padded, gu, "reference up")?,
        quote_mul(padded, gu, "candidate activation")?,
        quote_mul(padded, gu, "reference activation")?,
        quote_mul(padded, down, "candidate down")?,
        quote_mul(padded, down, "reference down")?,
        quote_mul(rows, combine, "candidate combine")?,
        quote_mul(rows, combine, "reference combine")?,
    ];
    let mut arena = 0;
    let mut arena_rounded = 0;
    let mut largest_guarded = 0;
    for capacity in capacities {
        let guarded = quote_add(capacity, 2 * GUARD_BYTES as u64, "buffer guards")?;
        arena = quote_add(arena, guarded, "guarded arena")?;
        arena_rounded = quote_add(
            arena_rounded,
            quote_round(guarded, granule)?,
            "rounded arena",
        )?;
        largest_guarded = largest_guarded.max(guarded);
    }
    // Every GPU owner retains its host `initial`. Input concatenations stay
    // alive throughout the cell. Provenance blocks and equal() downloads are
    // mutually exclusive; guard checking overlaps the final zero input.
    let inputs = quote_mul(
        rows,
        quote_sum(&[x, route, route], "cell inputs")?,
        "cell inputs",
    )?;
    let provenance_or_download = quote_mul(rows, 3 * gu, "GU provenance")?.max(quote_mul(
        rows,
        2 * down,
        "down provenance/equality",
    )?);
    let zeros_and_guards = quote_add(
        quote_mul(rows, combine, "combine zero input")?,
        largest_guarded.max(quote_mul(rows, 2 * combine, "combine equality")?),
        "combine/guards transient",
    )?;
    let compare_transient = quote_sum(
        &[arena, inputs, provenance_or_download.max(zeros_and_guards)],
        "comparison host arrays",
    )?;
    let logits = quote_mul(vocab, 4, "vocabulary logits")?;
    let continuation = quote_mul(
        rows,
        (std::mem::size_of::<u32>() + std::mem::size_of::<usize>()) as u64,
        "continuation ids/positions",
    )?;
    let host_arrays = quote_sum(
        &[
            capture,
            continuation,
            std::mem::size_of::<ObsState>() as u64,
            logits, // the prefix seed Vec remains in scope through comparison
            quote_add(logits, capture_transient, "capture phase")?.max(compare_transient),
        ],
        "native host array union",
    )?;
    // Both retained logits roots have the same guards as the 15 arena roots.
    let device_allocations = quote_add(
        quote_mul(
            2,
            quote_round(
                quote_add(logits, 2 * GUARD_BYTES as u64, "logits guards")?,
                granule,
            )?,
            "two guarded logits",
        )?,
        arena_rounded,
        "native device allocation union",
    )?;
    Ok(NativeExtraBytes {
        capture,
        capture_transient,
        arena,
        arena_rounded,
        compare_transient,
        host_arrays,
        device_allocations,
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen4_legacy_mq4_rowtile_oracle: error: {e}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run() -> Result<(), String> {
    let mut pins_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut n_ctx: usize = 8192;
    let (mut rows_lo, mut rows_hi) = (1, ROWS);
    let mut ref_gu: Option<PathBuf> = None;
    let mut ref_dn: Option<PathBuf> = None;
    let mut cand: Option<PathBuf> = None;
    let mut cand_sha: Option<String> = None;
    let mut preload_plan_only = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--preload-plan-only") => preload_plan_only = true,
            Some("--pins") => {
                pins_path = Some(PathBuf::from(args.next().ok_or("--pins requires a path")?));
            }
            Some("--out") => {
                out_path = Some(PathBuf::from(args.next().ok_or("--out requires a path")?));
            }
            Some("--n-ctx") => {
                let value = args.next().ok_or("--n-ctx requires a value")?;
                n_ctx = value
                    .to_str()
                    .and_then(|v| v.parse().ok())
                    .filter(|&v| v > 0)
                    .ok_or("--n-ctx requires a positive integer")?;
            }
            Some("--rows") => {
                let value = args.next().ok_or("--rows requires A-B or R")?;
                (rows_lo, rows_hi) = parse_rows(value.to_str().ok_or("--rows must be UTF-8")?)?;
            }
            Some("--reference-gu-co") => {
                ref_gu = Some(PathBuf::from(
                    args.next().ok_or("--reference-gu-co requires a path")?,
                ));
            }
            Some("--reference-down-co") => {
                ref_dn = Some(PathBuf::from(
                    args.next().ok_or("--reference-down-co requires a path")?,
                ));
            }
            Some("--candidate-co") => {
                cand = Some(PathBuf::from(
                    args.next().ok_or("--candidate-co requires a path")?,
                ));
            }
            Some("--candidate-sha256") => {
                cand_sha = Some(
                    args.next()
                        .ok_or("--candidate-sha256 requires a value")?
                        .to_str()
                        .ok_or("--candidate-sha256 must be UTF-8")?
                        .to_lowercase(),
                );
            }
            _ => return Err(format!("{USAGE}")),
        }
    }
    let pins_path = pins_path.ok_or(USAGE)?;
    let cand = cand.ok_or("--candidate-co is required")?;
    if let Some(sha) = &cand_sha {
        if !valid_sha(sha) {
            return Err("--candidate-sha256 must be 64 hex".into());
        }
    }

    match hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND").as_deref() {
        Some("off") => {}
        other => {
            return Err(format!(
                "refusing capture: HIPFIRE_REPLAY_BACKEND must be explicitly \"off\", got {other:?}"
            ));
        }
    }
    for var in ["HIPFIRE_GRAPH", "HIPFIRE_AR_GRAPH"] {
        if hipfire_config::process_value(var).as_deref() == Some("1") {
            return Err(format!("refusing: {var}=1 enables graph capture"));
        }
    }

    let pins_bytes =
        std::fs::read(&pins_path).map_err(|e| format!("read {}: {e}", pins_path.display()))?;
    let pins: Pins = serde_json::from_slice(&pins_bytes).map_err(|e| format!("pins: {e}"))?;
    if pins.schema != "qwen4-rowtile-pins-v1" {
        return Err(format!(
            "pins schema {:?} != qwen4-rowtile-pins-v1",
            pins.schema
        ));
    }
    if !matches!(pins.arch.as_str(), "gfx1151" | "gfx1201" | "gfx1100") {
        return Err(format!(
            "pins arch {:?} outside gfx1151|gfx1201|gfx1100",
            pins.arch
        ));
    }
    let pins_parent = pins_path.parent().unwrap_or_else(|| Path::new("."));
    let mut model = resolve_pin(pins_parent, &pins.model, "model")?;
    let prompt = resolve_pin(pins_parent, &pins.prompt, "prompt")?;
    let mut sidecars = Vec::with_capacity(pins.sidecars.len());
    for (i, pin) in pins.sidecars.iter().enumerate() {
        sidecars.push(resolve_pin(pins_parent, pin, &format!("sidecars[{i}]"))?);
    }
    if !preload_plan_only {
        let home = hipfire_config::hipfire_home()
            .map_err(|e| format!("runtime verified cache home unavailable: {e}"))?;
        let cache = VerifiedCache::in_home(&home);
        verify_cached(&cache, &model.0, &mut model.1, "model")?;
        for (i, (joined, pin)) in sidecars.iter_mut().enumerate() {
            verify_cached(&cache, joined, pin, &format!("sidecars[{i}]"))?;
        }
        let out = out_path.as_ref().ok_or(USAGE)?;
        if std::fs::symlink_metadata(out).is_ok() {
            return Err(format!("refusing: {} exists", out.display()));
        }
    }
    let prompt_bytes = verify_small(&prompt.0, &prompt.1.canonical, &prompt.1.sha256, "prompt")?;
    // Match ordinary daemon startup before its first HIP call; the existing
    // model-specific policy preserves explicit operator overrides.
    hipfire_loader::prepare_host_memory_for(&model.1.canonical);
    // Order: live Gpu first, then the real native workspace BEFORE any quote
    // or load, then code prewarm on that same Gpu, then the public planner
    // quote, then the normal producer load on the remaining memory.
    let mut gpu = Box::new(Gpu::init().map_err(es)?);
    if gpu.arch != pins.arch {
        return Err(format!(
            "pins arch {:?} != actual GPU arch {:?}",
            pins.arch, gpu.arch
        ));
    }
    let shared = shared_request(&gpu, &model.1.canonical, n_ctx)?;
    let source = open_qwen4_rank_source(&shared, &gpu.arch)?;
    let optics = match &source {
        ModelSource::Hfq(hfq) => Model::optics(hfq)?,
        ModelSource::Dir(_) => return Err("G1 requires the pinned HFQM source".into()),
    };
    let mut native =
        NativeWorkspace::allocate(&mut gpu, rows_hi, optics.0 as u64, optics.1 as u64)?;
    let code = prewarm_code(
        &mut gpu,
        &pins.arch,
        &cand,
        ref_gu.as_deref(),
        ref_dn.as_deref(),
        cand_sha.as_deref(),
    )?;
    let (mut plan, tokenizer, prompt_ids) = Model::preload_resource_plan(
        &gpu,
        shared,
        source,
        &prompt_bytes,
        &sidecars,
        &native,
        &code,
        n_ctx,
        rows_lo,
        rows_hi,
    )?;
    if preload_plan_only {
        let encoded = serde_json::to_string_pretty(&plan.quote).map_err(es)?;
        native.close_checked()?;
        match out_path {
            Some(path) => {
                use std::io::Write as _;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|e| format!("quote {}: {e}", path.display()))?;
                file.write_all(encoded.as_bytes()).map_err(es)?;
                file.write_all(b"\n").map_err(es)?;
            }
            None => println!("{encoded}"),
        }
        return Ok(());
    }
    let out_path = out_path.ok_or(USAGE)?;
    let resource_quote = std::mem::take(&mut plan.quote);
    enforce_preload_capacity(&resource_quote)?;
    let mut m = Model::load(
        gpu,
        tokenizer,
        plan,
        native,
        &pins.arch,
        model,
        prompt_bytes,
        prompt,
        n_ctx,
    )?;
    let prompt_len = prompt_ids.len();
    eprintln!(
        "qwen4_legacy_mq4_rowtile_oracle: arch={} prompt_len={prompt_len} n_ctx={n_ctx}",
        m.arch
    );

    let cand_img = code.candidate;
    let ref_gu_img = code.reference_gu;
    let ref_dn_img = code.reference_down;
    let gu_sym = code.gu_symbol;
    let dn_sym = code.down_symbol;
    // Keep the already-accounted code input allocation alive; no second read.
    let _candidate_bytes = code.candidate_bytes;

    // Reuse the transparent native workspace owners for capture logits:
    // slots 0/1 are the retained prefix/row logits (logical views, no second
    // allocation); the workspace itself stays owned by the same Gpu.
    let prompt_logits = m.workspace_view(0, 0, m.vocab * 4)?;
    let row_logits = m.workspace_view(1, 0, m.vocab * 4)?;

    m.bundle
        .prefill_final(&mut m.gpu, &prompt_ids, 0, &prompt_logits)
        .map_err(|e| format!("prefill_final: {e}"))?;
    m.workspace.check_logits_guards(&m.gpu)?;
    let seed = m.gpu.download_f32(&prompt_logits).map_err(es)?;
    if seed.iter().any(|v| !v.is_finite()) {
        return Err("non-finite prefill logits".into());
    }
    let mut token = argmax(&seed);
    eprintln!("qwen4_legacy_mq4_rowtile_oracle: prefill done, seed token={token}");

    // Sanity: candidate module name must be the frozen family module.
    let cand_image_name = format!("{LEGACY_IMAGE_PREFIX}{}", m.arch);
    eprintln!("qwen4_legacy_mq4_rowtile_oracle: module {LEGACY_MODULE} image {cand_image_name}");

    let state = Arc::new(Mutex::new(ObsState {
        rows: std::array::from_fn(|_| RowObs::default()),
        row: 0,
        max_experts: m.max_experts,
        error: None,
    }));
    let observer_state = Arc::clone(&state);
    let observer = Arc::new(
        move |gpu: &Gpu, witness: &LabG1Witness<'_>| -> HipResult<()> {
            observe(&observer_state, gpu, witness)
        },
    );
    let old_window = m.gpu.lab_g1_window();
    let old = m.gpu.set_lab_g1_observer(Some(observer));
    let capture_result = (|| -> Result<(Vec<u32>, Vec<usize>), String> {
        let mut tokens = Vec::with_capacity(rows_hi);
        let mut positions = Vec::with_capacity(rows_hi);
        for r in 0..rows_hi {
            let position = prompt_len + r;
            {
                let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(message) = st.error.clone() {
                    return Err(message);
                }
                st.row = r;
            }
            m.bundle
                .forward_token(&mut m.gpu, token, &row_logits, None)
                .map_err(|e| {
                    let suffix = state
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .error
                        .clone()
                        .map(|message| format!(" ({message})"))
                        .unwrap_or_default();
                    format!("forward_token row {r}{suffix}: {e}")
                })?;
            m.workspace.check_logits_guards(&m.gpu)?;
            let logits = m.gpu.download_f32(&row_logits).map_err(es)?;
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(format!("non-finite logits at row {r}"));
            }
            tokens.push(token);
            positions.push(position);
            token = argmax(&logits);
            if r % 16 == 15 {
                eprintln!(
                    "qwen4_legacy_mq4_rowtile_oracle: row {}/{} token={token}",
                    r + 1,
                    rows_hi
                );
            }
        }
        Ok((tokens, positions))
    })();
    m.gpu.set_lab_g1_observer(old);
    m.gpu.set_lab_g1_window(old_window);
    let (tokens, positions) = capture_result?;
    // Workspace logits views borrow slots 0/1: nothing to free here; the
    // same owners stay alive through comparison and are released with the
    // workspace. Logits canaries were checked after each captured row.
    let _ = (prompt_logits, row_logits);

    // Every row must show a REAL hot legacy stage; an all-invalid/masked
    // window is a fatal empty-observer success, never a pass.
    {
        let st = state.lock().unwrap_or_else(|e| e.into_inner());
        for r in 0..rows_hi {
            if !st.rows[r].hot {
                return Err(format!(
                    "row {r}: no real hot legacy G1 stage observed (missing actual stage fatal)"
                ));
            }
        }
    }

    // Pin receipts: actual raw device SHAs of every bound symbol.
    let mut pin_lines = Vec::new();
    for (label, img) in [
        ("candidate", &cand_img),
        ("reference-gu", &ref_gu_img),
        ("reference-down", &ref_dn_img),
    ] {
        for (sym, sha) in &img.device_sha {
            pin_lines.push(json!({
                "image": label,
                "path": img.image_path,
                "container_sha256": img.container_sha,
                "symbol": sym,
                "raw_device_sha256": sha,
            }));
        }
    }

    run_compare(
        &mut m,
        &state,
        &cand_img,
        &gu_sym,
        &dn_sym,
        &ref_gu_img,
        &ref_dn_img,
        rows_lo,
        rows_hi,
    )?;
    m.close_checked()?;
    std::fs::create_dir(&out_path).map_err(|e| format!("mkdir {}: {e}", out_path.display()))?;
    let capture_authority = {
        let st = state.lock().unwrap_or_else(|e| e.into_inner());
        let first = &st.rows[0];
        json!({
            "immutable_expert_identity": first.expr_set,
            "native_gu_module": first.module,
            "native_gu_container_sha256": first.object_sha,
            "native_gu_raw_device_sha256": first.device_sha,
            "resident_experts": first.n_exp,
            "bf16_round_trip": first.bf16_round_trip,
            "gu_pointer_table_sha256": sha_hex(&first.table_row),
            "down_pointer_table_sha256": sha_hex(&first.down_table),
            "ownership": "Full model retained through every comparison; checked cleanup completed before receipt; only pointer VALUES copied, no weight payload read/hash/copy",
        })
    };
    let cells = 3 * (rows_hi - rows_lo + 1);
    let comparison_scope = if rows_lo == 1 && rows_hi == ROWS {
        "full logical range for Original G1; not G2/G3/G4 certification"
    } else {
        "partial logical range for aggregate Original G1; not a whole-range certificate"
    };
    eprintln!(
        "legacy-mq4 ORIGINAL G1: {cells}/{cells} full-word cells matched; R={rows_lo}..={rows_hi}; {comparison_scope}; guards intact; checked cleanup complete"
    );
    let index = json!({
        "schema": "legacy-mq4-g1-rowtile-oracle.v1",
        "mode": ORACLE_MODE,
        "preload_resource_plan": resource_quote,
        "arch": m.arch,
        "model": {"path": m.model_pin.canonical.display().to_string(), "declared_sha256": m.model_pin.sha256, "sha256": m.model_pin.sha256, "authority": "runtime verified cache; prior stat-bound full SHA; not recomputed"},
        "payload_verification": {"authority": "runtime verified cache; prior stat-bound full SHA", "recomputed": false},
        "prompt": {"path": m.prompt_pin.canonical.display().to_string(), "sha256": m.prompt_pin.sha256},
        "recipe": m.recipe,
        "ple_rows_source": m.ple_rows_source,
        "trunk_overlay_candidate": m.trunk_overlay_candidate.as_ref().map(|p| p.display().to_string()),
        "trunk_overlay_note": "presence is candidacy only; settled marker cannot distinguish pre-existing native IU4 from applied (named ambiguity, never presence=>applied)",
        "measured_snapshot": m.measured_snapshot,
        "measured_load_identity": m.measured_identity,
        "producer_evidence_absence": "null identity/snapshot means ordinary producer did not expose verified evidence; no substitute fabricated",
        "hotset_applied": m.hotset_applied,
        "hotset_authority": "actual settled bundle.weights.expert_tiers(); not adjacent file presence",
        "trunk_overlay_activation": {"status": "unavailable_from_public_bundle_getter", "settled_iu4_marker_present": m.trunk_marker_present,
            "reason": "marker may pre-exist or follow Applied; ordinary returned identity retained separately when available"},
        "rows": rows_hi,
        "captured_ar_rows": rows_hi,
        "positions": positions,
        "tokens": tokens,
        "pins": pin_lines,
        "capture_authority": capture_authority,
        "comparison": {"status": "PASS", "cells": cells, "logical_r": [rows_lo, rows_hi],
            "scope": comparison_scope,
            "seeds": ["actual-ar", "all-invalid", "mixed-invalid-tail"],
            "stages": ["gate", "up", "activation-fwht", "unweighted-down", "canonical-combine"],
            "guards": "immutable inputs, allocation prefixes/suffixes, padded dead token rows",
            "weights": "borrowed immutable model-owned resident payloads"},
    });
    std::fs::write(
        out_path.join("index.json"),
        serde_json::to_string_pretty(&index).map_err(es)?,
    )
    .map_err(|e| format!("write index.json: {e}"))?;
    eprintln!(
        "qwen4_legacy_mq4_rowtile_oracle: wrote {}",
        out_path.display()
    );
    Ok(())
}

/// Per-R (requested inclusive range) full-word compare vs the original
/// per-row AR reference. `compare` runs on the SAME borrowed owner buffers
/// (in-place, no weight copies); tiles reuse retained oracle-owned scratch.
/// Any nonzero mismatch exits nonzero; invalid slots/tails/poison/device
/// guards are honest (dead-tail rows store nothing and are never compared).
#[allow(clippy::too_many_arguments)]
fn run_compare(
    m: &mut Model,
    state: &Arc<Mutex<ObsState>>,
    cand_img: &LoadedImage,
    gu_sym: &str,
    dn_sym: &str,
    ref_gu_img: &LoadedImage,
    ref_dn_img: &LoadedImage,
    rows_lo: usize,
    rows_hi: usize,
) -> Result<(), String> {
    let st = state.lock().unwrap_or_else(|e| e.into_inner());
    let first = &st.rows[0];
    if first.m != 1280
        || first.mi != 640
        || first.down_m != 2560
        || first.down_k != 640
        || first.n_exp == 0
        || first.expr_set.is_none()
    {
        return Err("missing canonical legacy GU/Down geometry or immutable owner".into());
    }
    for (row, slot) in st.rows[..rows_hi].iter().enumerate() {
        if !slot.hot
            || slot.expr_set != first.expr_set
            || slot.table_row != first.table_row
            || slot.down_table != first.down_table
            || slot.n_exp != first.n_exp
            || slot.bf16_round_trip != first.bf16_round_trip
            || (slot.m, slot.mi, slot.down_m, slot.down_k) != (1280, 640, 2560, 640)
            || slot.x_row.len() != 2560 * 4
            || slot.ids_row.len() != 10 * 4
            || slot.probs_row.len() != 10 * 4
            || slot.rotated.len() != 10 * 640 * 4
            || slot.gate_rows.len() != 10 * 640 * 4
            || slot.up_rows.len() != 10 * 640 * 4
            || slot.down_rows.len() != 10 * 2560 * 4
        {
            return Err(format!(
                "row {row}: missing stage, shape, or immutable owner drift"
            ));
        }
    }
    let gpu = &mut m.gpu;
    // Model loading/AR capture happened after prewarm. Bind the comparison
    // module to the actual singleton owner still loaded on that same Gpu,
    // not just to the prewarm symbol name or its by-value ABI.
    let original_down = "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded";
    let (_, _, actual_down_sha) = gpu.lab_kernel_device_identity(original_down).map_err(es)?;
    if ref_dn_img.device_sha.get(original_down) != Some(&actual_down_sha) {
        return Err("down reference differs from the actual native AR owner".into());
    }
    // Same physical owners for every comparison: logical per-cell views over
    // the retained 15-slot workspace (2 pointer tables, X/ids/probs, 4 GU,
    // 2 activated/rotated, 2 down, 2 combines). No second allocations.
    let workspace_views: &[GpuTensor] = &m.workspace.slots[2..];
    assert_eq!(workspace_views.len(), 15);
    for seed in 0..3 {
        for rows in rows_lo..=rows_hi {
            let mut arena = OracleBuffers::default();
            let result = (|| -> Result<(), String> {
                let mut x = Vec::with_capacity(rows * 2560 * 4);
                let mut ids = Vec::with_capacity(rows * 10 * 4);
                let mut probabilities = Vec::with_capacity(rows * 10 * 4);
                for row in 0..rows {
                    let slot = &st.rows[row];
                    x.extend_from_slice(&slot.x_row);
                    ids.extend_from_slice(&slot.ids_row);
                    probabilities.extend_from_slice(&slot.probs_row);
                }
                if seed != 0 {
                    for (i, word) in x.chunks_exact_mut(4).enumerate() {
                        let value = if seed == 1 {
                            0.0
                        } else {
                            match i % 8 {
                                0 => -0.0,
                                1 => 0.0,
                                2 => 0.000_061_035_156,
                                3 => -0.000_061_035_156,
                                4 => 8.0,
                                5 => -8.0,
                                6 => 0.5,
                                _ => -0.5,
                            }
                        };
                        word.copy_from_slice(&f32::to_le_bytes(value));
                    }
                    for (i, word) in ids.chunks_exact_mut(4).enumerate() {
                        let id = if seed == 1 {
                            u32::MAX
                        } else {
                            match i % 5 {
                                0 => u32::MAX,
                                1 => first.n_exp as u32,
                                2 => 0,
                                3 => (first.n_exp - 1) as u32,
                                _ => 0,
                            }
                        };
                        word.copy_from_slice(&id.to_le_bytes());
                    }
                    for (i, word) in probabilities.chunks_exact_mut(4).enumerate() {
                        let value: f32 = if seed == 1 {
                            0.0
                        } else {
                            match i % 4 {
                                0 => 0.0,
                                1 => 0.5,
                                2 => 0.25,
                                _ => 0.125,
                            }
                        };
                        word.copy_from_slice(&value.to_le_bytes());
                    }
                }
                let table = arena.input(gpu, workspace_views, 0, &first.table_row, DType::Raw)?;
                let down_table =
                    arena.input(gpu, workspace_views, 1, &first.down_table, DType::Raw)?;
                let xi = arena.input(gpu, workspace_views, 2, &x, DType::F32)?;
                let indices = arena.input(gpu, workspace_views, 3, &ids, DType::F32)?;
                let probs = arena.input(gpu, workspace_views, 4, &probabilities, DType::F32)?;
                let extent = rows * 10 * 640;
                let padded = rows.div_ceil(8) * 8;
                let cg = arena.output(gpu, workspace_views, 5, extent, padded * 10 * 640)?;
                let cu = arena.output(gpu, workspace_views, 6, extent, padded * 10 * 640)?;
                let rg = arena.reference(gpu, workspace_views, 7, extent, padded * 10 * 640)?;
                let ru = arena.reference(gpu, workspace_views, 8, extent, padded * 10 * 640)?;
                launch_gu_tile(
                    gpu,
                    cand_img,
                    gu_sym,
                    arena.ptr(table),
                    arena.ptr(indices),
                    arena.ptr(xi),
                    arena.ptr(cg),
                    arena.ptr(cu),
                    1280,
                    10,
                    first.n_exp as i32,
                    rows as i32,
                )?;
                for row in 0..rows {
                    launch_gu_original(
                        gpu,
                        ref_gu_img,
                        arena.ptr(table),
                        arena.ptr(indices) + (row * 10 * 4) as u64,
                        arena.ptr(xi) + (row * 2560 * 4) as u64,
                        arena.ptr(rg) + (row * 10 * 640 * 4) as u64,
                        arena.ptr(ru) + (row * 10 * 640 * 4) as u64,
                        1280,
                        first.n_exp as i32,
                    )?;
                }
                arena.equal(gpu, cg, rg, "GU gate")?;
                arena.equal(gpu, cu, ru, "GU up")?;
                if seed == 0 {
                    let gates = collect_observed(&st.rows[..rows], |r| &r.gate_rows);
                    let ups = collect_observed(&st.rows[..rows], |r| &r.up_rows);
                    exact_words(&arena.bytes(gpu, rg)?, &gates, "native GU gate provenance")?;
                    exact_words(&arena.bytes(gpu, ru)?, &ups, "native GU up provenance")?;
                }
                let ch = arena.output(gpu, workspace_views, 9, extent, padded * 10 * 640)?;
                let rh = arena.reference(gpu, workspace_views, 10, extent, padded * 10 * 640)?;
                for (gate, up, hidden) in [(cg, cu, ch), (rg, ru, rh)] {
                    if first.bf16_round_trip {
                        gpu.silu_mul_bf16_rt_rotate_x_mq_128_v2(
                            arena.view(gate),
                            arena.view(up),
                            arena.view(hidden),
                            640,
                            rows * 10,
                        )
                        .map_err(es)?;
                    } else {
                        gpu.silu_mul_f32(arena.view(gate), arena.view(up), arena.view(hidden))
                            .map_err(es)?;
                        gpu.rotate_x_mq_128_v2(
                            arena.view(hidden),
                            arena.view(hidden),
                            640,
                            rows * 10,
                        )
                        .map_err(es)?;
                    }
                }
                arena.equal(gpu, ch, rh, "activation/FWHT")?;
                if seed == 0 {
                    let actual = collect_observed(&st.rows[..rows], |r| &r.rotated);
                    exact_words(
                        &arena.bytes(gpu, rh)?,
                        &actual,
                        "native activation provenance",
                    )?;
                }
                let cd = arena.output(
                    gpu,
                    workspace_views,
                    11,
                    rows * 10 * 2560,
                    padded * 10 * 2560,
                )?;
                let rd = arena.reference(
                    gpu,
                    workspace_views,
                    12,
                    rows * 10 * 2560,
                    padded * 10 * 2560,
                )?;
                launch_down_tile(
                    gpu,
                    cand_img,
                    dn_sym,
                    arena.ptr(down_table),
                    arena.ptr(indices),
                    arena.ptr(ch),
                    arena.ptr(cd),
                    2560,
                    640,
                    rows as i32,
                    first.n_exp as i32,
                    10,
                )?;
                for row in 0..rows {
                    launch_down_tile(
                        gpu,
                        ref_dn_img,
                        "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded",
                        arena.ptr(down_table),
                        arena.ptr(indices) + (row * 10 * 4) as u64,
                        arena.ptr(rh) + (row * 10 * 640 * 4) as u64,
                        arena.ptr(rd) + (row * 10 * 2560 * 4) as u64,
                        2560,
                        640,
                        1,
                        first.n_exp as i32,
                        10,
                    )?;
                }
                if seed == 0 {
                    let actual = collect_observed(&st.rows[..rows], |r| &r.down_rows);
                    exact_words(&arena.bytes(gpu, rd)?, &actual, "native down provenance")?;
                }
                arena.equal(gpu, cd, rd, "unweighted down")?;
                let zeros = vec![0u8; rows * 2560 * 4];
                let cc =
                    arena.mutable(gpu, workspace_views, 13, &zeros, DType::F32, zeros.len())?;
                let rc =
                    arena.mutable(gpu, workspace_views, 14, &zeros, DType::F32, zeros.len())?;
                gpu.moe_down_combine_top10_batched(
                    arena.view(cd),
                    arena.view(indices),
                    arena.view(probs),
                    arena.view(cc),
                    2560,
                    rows,
                    first.n_exp,
                )
                .map_err(es)?;
                // Independent native singleton rank-order folds, not a host approximation.
                for row in 0..rows {
                    let out = arena.view(rd).sub_offset(row * 10 * 2560, 10 * 2560);
                    let ids = arena.view(indices).sub_offset(row * 10, 10);
                    let probability = arena.view(probs).sub_offset(row * 10, 10);
                    let residual = arena.view(rc).sub_offset(row * 2560, 2560);
                    gpu.moe_down_combine_top10_batched(
                        &out,
                        &ids,
                        &probability,
                        &residual,
                        2560,
                        1,
                        first.n_exp,
                    )
                    .map_err(es)?;
                }
                arena.equal(gpu, cc, rc, "canonical rank combine")?;
                Ok(())
            })();
            // Logical views only: verified and dropped; the same 15 physical
            // owners persist for the next cell/canary check.
            let cleanup = arena.release(gpu);
            result.map_err(|e| format!("seed={seed} R={rows}: {e}"))?;
            cleanup?;
        }
    }
    Ok(())
}

const GUARD_BYTES: usize = 256;
const POISON: u8 = 0xa5;

struct OracleBuffer {
    owner: GpuTensor,
    view: GpuTensor,
    initial: Vec<u8>,
    writable: usize,
}

// The exact owner count is known before a comparison cell starts.
struct OracleBuffers {
    buffers: Vec<OracleBuffer>,
}

impl Default for OracleBuffers {
    fn default() -> Self {
        Self {
            buffers: Vec::with_capacity(15),
        }
    }
}

impl OracleBuffers {
    fn input(
        &mut self,
        gpu: &Gpu,
        workspace: &[GpuTensor],
        slot: usize,
        data: &[u8],
        dtype: DType,
    ) -> Result<usize, String> {
        self.stage(gpu, workspace, slot, data, dtype, data.len(), 0)
    }

    fn mutable(
        &mut self,
        gpu: &Gpu,
        workspace: &[GpuTensor],
        slot: usize,
        data: &[u8],
        dtype: DType,
        capacity: usize,
    ) -> Result<usize, String> {
        self.stage(gpu, workspace, slot, data, dtype, capacity, data.len())
    }

    fn output(
        &mut self,
        gpu: &Gpu,
        workspace: &[GpuTensor],
        slot: usize,
        words: usize,
        capacity_words: usize,
    ) -> Result<usize, String> {
        self.mutable(
            gpu,
            workspace,
            slot,
            &vec![POISON; words * 4],
            DType::F32,
            capacity_words * 4,
        )
    }

    fn reference(
        &mut self,
        gpu: &Gpu,
        workspace: &[GpuTensor],
        slot: usize,
        words: usize,
        capacity_words: usize,
    ) -> Result<usize, String> {
        self.mutable(
            gpu,
            workspace,
            slot,
            &vec![0x5a; words * 4],
            DType::F32,
            capacity_words * 4,
        )
    }

    /// Stage one logical cell view: copy `data` into the retained native
    /// workspace owner at `slot` (guarded, poison-canary discipline) and
    /// alias the SAME physical owner via a byte view. No second allocation
    /// set: every `OracleBuffers` cell reuses the 15 preallocated slots.
    /// `slot` selects the widest-extent owner for this role:
    /// 0/1 pointer tables, 2 X, 3 ids, 4 probs, 5/6 candidate GU, 7/8
    /// reference GU, 9/10 activations, 11/12 down, 13/14 combines.
    fn stage(
        &mut self,
        gpu: &Gpu,
        workspace: &[GpuTensor],
        slot: usize,
        data: &[u8],
        dtype: DType,
        capacity: usize,
        writable: usize,
    ) -> Result<usize, String> {
        if capacity < data.len() || data.len() % dtype.size() != 0 {
            return Err("oracle buffer extent is invalid".into());
        }
        let base = workspace
            .get(slot)
            .ok_or("native workspace slot out of range")?;
        if capacity > base.buf.size().saturating_sub(2 * GUARD_BYTES) {
            return Err("native workspace slot extent is too small".into());
        }
        let mut initial = vec![POISON; base.buf.size()];
        initial[GUARD_BYTES..GUARD_BYTES + data.len()].copy_from_slice(data);
        gpu.hip.device_synchronize().map_err(es)?;
        gpu.hip
            .memcpy_htod(&base.buf.byte_view(0, initial.len()), &initial)
            .map_err(es)?;
        gpu.hip.device_synchronize().map_err(es)?;
        let view = GpuTensor {
            buf: base.buf.byte_view(GUARD_BYTES, data.len()),
            shape: vec![data.len() / dtype.size().max(1)],
            dtype,
        };
        let index = self.buffers.len();
        self.buffers.push(OracleBuffer {
            owner: GpuTensor {
                buf: base.buf.byte_view(0, initial.len()),
                shape: vec![initial.len()],
                dtype: DType::Raw,
            },
            view,
            initial,
            writable,
        });
        Ok(index)
    }

    fn view(&self, index: usize) -> &GpuTensor {
        &self.buffers[index].view
    }

    fn ptr(&self, index: usize) -> u64 {
        self.view(index).buf.as_ptr() as u64
    }

    fn bytes(&self, gpu: &Gpu, index: usize) -> Result<Vec<u8>, String> {
        gpu.hip.device_synchronize().map_err(es)?;
        gpu.download_raw_bytes(self.view(index)).map_err(es)
    }

    fn equal(&self, gpu: &Gpu, a: usize, b: usize, label: &str) -> Result<(), String> {
        exact_words(&self.bytes(gpu, a)?, &self.bytes(gpu, b)?, label)
    }

    fn check_guards(&self, gpu: &Gpu) -> Result<(), String> {
        gpu.hip.device_synchronize().map_err(es)?;
        for (index, buffer) in self.buffers.iter().enumerate() {
            let bytes = gpu.download_raw_bytes(&buffer.owner).map_err(es)?;
            for (offset, (&actual, &initial)) in bytes.iter().zip(&buffer.initial).enumerate() {
                let writable = offset >= GUARD_BYTES && offset < GUARD_BYTES + buffer.writable;
                if !writable && actual != initial {
                    return Err(format!(
                        "buffer {index}: input/guard/dead-tail byte {offset} changed"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Release logical views only: the physical owners stay with the
    /// retained native workspace (no second allocation set to free). Guards
    /// are verified before release; canary bytes live on the shared owners.
    fn release(self, gpu: &Gpu) -> Result<(), String> {
        self.check_guards(gpu)?;
        Ok(())
    }
}

fn exact_words(actual: &[u8], expected: &[u8], label: &str) -> Result<(), String> {
    if actual.len() != expected.len() || actual.len() % 4 != 0 {
        return Err(format!(
            "{label}: extent mismatch {} vs {}",
            actual.len(),
            expected.len()
        ));
    }
    for (word, (a, b)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        if a != b {
            return Err(format!("{label}: word {word} differs {a:02x?} != {b:02x?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod resource_quote_tests {
    use super::*;

    #[test]
    fn capture_extent_is_high_endpoint_not_requested_cell_count() {
        let (lo, hi) = parse_rows("49-64").unwrap();
        assert_eq!(3 * (hi - lo + 1), 48);
        let q = native_extra_bytes(hi, 1000, 512, 1).unwrap();
        let row_values = 2560 * 4 + 2 * 10 * 4 + 2 * 512 * 8 + 3 * 10 * 640 * 4 + 10 * 2560 * 4;
        assert_eq!(q.capture, 64 * row_values);
        assert_eq!(
            q.capture,
            native_extra_bytes(64, 1000, 512, 1).unwrap().capture
        );
    }

    #[test]
    fn arena_sums_overlapping_owners_but_not_seeds_or_cells() {
        let q = native_extra_bytes(64, 1000, 512, 1).unwrap();
        // Fifteen guarded owners, six hidden-width and two expanded-down
        // capacities, two combines, X/routes and two pointer tables.
        let expected = 15 * 2 * 256
            + 2 * 512 * 8
            + 64 * (2560 * 4 + 2 * 10 * 4 + 2 * 2560 * 4)
            + 64 * (6 * 10 * 640 * 4 + 2 * 10 * 2560 * 4);
        assert_eq!(q.arena, expected);
        assert_eq!(
            q.device_allocations,
            expected + 2 * (1000 * 4 + 2 * GUARD_BYTES as u64)
        );
        assert_eq!(q.arena_rounded, q.arena);
        assert!(q.host_arrays >= q.capture + q.compare_transient);
    }

    #[test]
    fn tails_and_per_allocation_rounding_are_charged() {
        let eight = native_extra_bytes(8, 1, 512, 65536).unwrap();
        let nine = native_extra_bytes(9, 1, 512, 65536).unwrap();
        assert!(nine.arena > eight.arena + 8 * (6 * 10 * 640 * 4));
        assert!(nine.arena_rounded >= nine.arena);
        assert_eq!(nine.device_allocations - nine.arena_rounded, 2 * 65536);
        assert!(nine.capture_transient < nine.capture);
    }

    #[test]
    fn host_transients_use_phase_max_not_a_sum() {
        let rows = 64;
        let q = native_extra_bytes(rows, 1, 512, 1).unwrap();
        let retained = q.capture
            + (rows * (4 + std::mem::size_of::<usize>())) as u64
            + std::mem::size_of::<ObsState>() as u64;
        assert!(q.compare_transient > 4 + q.capture_transient);
        assert_eq!(q.host_arrays, retained + 4 + q.compare_transient);

        let q = native_extra_bytes(1, 10_000_000, 512, 1).unwrap();
        let retained = q.capture
            + (4 + std::mem::size_of::<usize>()) as u64
            + std::mem::size_of::<ObsState>() as u64;
        assert!(40_000_000 + q.capture_transient > q.compare_transient);
        assert_eq!(q.host_arrays, retained + 80_000_000 + q.capture_transient);
    }

    #[test]
    fn missing_geometry_and_overflow_are_named_refusals() {
        assert!(native_extra_bytes(0, 1, 1, 1).is_err());
        assert!(native_extra_bytes(65, 1, 1, 1).is_err());
        assert!(native_extra_bytes(1, 0, 1, 1).is_err());
        assert!(native_extra_bytes(1, 1, 0, 1).is_err());
        assert!(native_extra_bytes(1, 1, 1, 0)
            .unwrap_err()
            .contains("granule"));
        assert!(native_extra_bytes(64, u64::MAX, 512, 1)
            .unwrap_err()
            .contains("overflows"));
        assert!(native_extra_bytes(64, 1, u64::MAX, 1)
            .unwrap_err()
            .contains("pointer VALUES"));
        assert!(quote_round(u64::MAX, 65536)
            .unwrap_err()
            .contains("rounding"));
    }
}
