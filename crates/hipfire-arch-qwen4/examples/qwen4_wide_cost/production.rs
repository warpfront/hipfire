// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Production load of `qwen4_wide_verify`: the exact serve load
//! (`hipfire_loader::load_qwen4_rank`: source admission, split base + PLE rows
//! sidecar, trunk overlay, expert hot set / automatic placement, weight
//! fulfillment, assembly, forward program and the native MTP head) plus the
//! identity receipts of the artifacts it applied, in the shape of the
//! production identity harness (`tests/greedy_mtp_identity_hw.rs`,
//! `tests/common/hw_identity.rs`).

use super::{es, hw_identity, Args, LoadInfo, Model};
use hipfire_arch_qwen4::expert_residency as residency;
use hipfire_arch_qwen4::weights::Qwen4TierRecipe;
use hipfire_arch_qwen4::Qwen4KvBackend;
use hipfire_loader::{load_qwen4_rank, Qwen4RankShared};
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
use hipfire_runtime::ple_sidecar::{PleRowsChoice, PleRowsOrigin};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const REPLAY_BACKEND_ENV: &str = "HIPFIRE_REPLAY_BACKEND";

/// Raw resource-plan payload schema consumed verbatim by E
/// (`fn-oracle-v2/full_load_resources.py`): exactly `schema`, `complete`,
/// `production` and `snapshot`. Never add extra `resource_plan/...` keys.
pub(super) const RESOURCE_PLAN_SCHEMA: &str = "qwen4_wide_verify.resource_plan.v1";
/// Source-planned preload payload schema: exactly `schema`, `planning_only`,
/// `evidence_kind`, `complete`, `qualification`, `planning_context`,
/// `loader_config` and `union`. `complete` is always false and
/// `qualification` always `nonqualification`: never production RAW, never a
/// fit, never a qualification. Never add `production`/`snapshot` keys.
pub(super) const PRELOAD_PLAN_SCHEMA: &str = "qwen4_wide_verify.preload_plan.v1";
pub(super) const PRELOAD_PLAN_EVIDENCE_KIND: &str = "source_planned_unmeasured_nonqualification";
/// Planning context when `--preload-plan-only` runs without an explicit
/// `--n-ctx`: the finite 544 bench (`max(512,128+1)+32`).
pub(super) const PRELOAD_PLAN_DEFAULT_N_CTX: usize = 544;

/// The launch backend this process requests, canonical (`hip`, `auto`,
/// `shadow`); unset or empty is `hip`. An unknown value is refused.
pub(super) fn replay_backend_request(raw: Option<&str>) -> Result<&'static str, String> {
    match raw.map(|value| value.to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("hip") | Some("off") => Ok("hip"),
        Some("auto") | Some("redline") => Ok("auto"),
        Some("shadow") => Ok("shadow"),
        Some(other) => Err(format!(
            "{REPLAY_BACKEND_ENV}={other:?} is not a known backend (hip, auto, shadow)"
        )),
    }
}

fn replay_backend_name(request: rdna_compute::replay::ReplayBackendRequest) -> &'static str {
    use rdna_compute::replay::ReplayBackendRequest;
    match request {
        ReplayBackendRequest::Hip => "hip",
        ReplayBackendRequest::Shadow => "shadow",
        ReplayBackendRequest::Auto => "auto",
    }
}

/// What the product load does with `<model>.expert-hotset` on this device:
/// `Ok(path)` applies, `Err(reason)` does not.
fn hotset_expectation(
    model: &Path,
    gpu: &Gpu,
    recipe: &Qwen4TierRecipe,
) -> Result<Result<PathBuf, String>, String> {
    use hipfire_arch_qwen4::expert_hotset as hotsets;
    let policy = hotsets::HotsetPolicy::from_env()?;
    let placement = hotsets::PlacementPolicy::from_env()?;
    let env = hotsets::HotsetEnv {
        gpu_arch: &gpu.arch,
        unified_memory: gpu.is_uma(),
        expert_parallel: false,
        explicit_vram_layers: residency::expert_vram_layers_from_env()?.is_some(),
        source: match recipe {
            Qwen4TierRecipe::FlashNextPacked { .. } => hotsets::HotsetSource::Aggregate,
            Qwen4TierRecipe::FlashNextSlab { .. } => hotsets::HotsetSource::SlabArena,
            Qwen4TierRecipe::StockFp8 => hotsets::HotsetSource::StockFp8,
            _ => hotsets::HotsetSource::Nonaggregate("not an aggregate MQ4 artifact".into()),
        },
    };
    if hotsets::decide_whole_first(placement, &env, false) == hotsets::WholeFirstDecision::Active {
        return Ok(Err(format!(
            "{}=whole-first keeps whole-layer placement",
            hotsets::PLACEMENT_POLICY_ENV
        )));
    }
    Ok(hotsets::preflight(&policy, model, &env).map_err(|skip| skip.to_string()))
}

/// The hot set the load actually bound against the expectation; a hot set the
/// device should have used but the load skipped (or one applied where it
/// should not be) is an error, never a quiet different placement.
fn hotset_receipt(
    expected: &Result<PathBuf, String>,
    applied: Option<&str>,
    model_id: &Value,
) -> Result<Value, String> {
    match (expected, applied) {
        (Ok(path), Some(sha)) => {
            if model_id["expert-hotset"]["sha256"].as_str() != Some(sha) {
                return Err(format!(
                    "the load applied a hot set with sha256 {sha}, not the hashed {} ({})",
                    path.display(),
                    model_id["expert-hotset"]["sha256"]
                ));
            }
            Ok(json!({"applied": true, "path": path.display().to_string(), "sha256": sha}))
        }
        (Ok(path), None) => Err(format!(
            "{} applies on this device but the load placed no hot set (whole-layer placement); refusing an unfaithful placement",
            path.display()
        )),
        (Err(reason), None) => Ok(json!({"applied": false, "not_applicable": reason})),
        (Err(reason), Some(sha)) => Err(format!(
            "the load applied hot set {sha} though it is not applicable here ({reason})"
        )),
    }
}

/// The PLE rows source the load resolved; a sidecar must be exactly the
/// `<model>.ple-<format>` file the identity hashed.
fn ple_rows_receipt(
    choice: Option<&PleRowsChoice>,
    model: &Path,
    model_id: &Value,
) -> Result<Value, String> {
    let Some(choice) = choice else {
        return Ok(json!({"format": null, "origin": "none", "path": null}));
    };
    let format = choice.format.name();
    match &choice.origin {
        PleRowsOrigin::Embedded => Ok(json!({
            "format": format,
            "origin": "embedded",
            "path": model.display().to_string(),
        })),
        PleRowsOrigin::Sidecar(path) => {
            let mut hashed = model.as_os_str().to_os_string();
            hashed.push(format!(".ple-{format}"));
            if path.as_os_str() != hashed.as_os_str() {
                return Err(format!(
                    "PLE rows resolved to {}, not the hashed {}",
                    path.display(),
                    PathBuf::from(hashed).display()
                ));
            }
            let key = format!("ple-{format}");
            let sha = model_id[key.as_str()]["sha256"]
                .as_str()
                .ok_or_else(|| format!("identity holds no sha256 for {key}"))?;
            Ok(json!({
                "format": format,
                "origin": "sidecar",
                "path": path.display().to_string(),
                "sha256": sha,
            }))
        }
    }
}

/// Load `args.model` exactly as `hipfire serve` does and read back the
/// receipts of what was applied.
pub(super) fn load(args: &Args) -> Result<Model, String> {
    let path = args.model.as_path();
    let model_str = path
        .to_str()
        .ok_or_else(|| format!("{} is not UTF-8", path.display()))?;
    let cache_dir = args
        .out
        .parent()
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let model_id = hw_identity::model_identity(path, cache_dir)?;
    let model_bytes = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let mut source = ModelSource::from_path(model_str)?;
    let tokenizer = match &source {
        ModelSource::Hfq(hfq) => Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer: {e}"))?,
        _ => return Err("model is not an HFQ artifact".into()),
    };
    hipfire_loader::prepare_host_memory_for(path);
    let mut gpu = Gpu::init().map_err(es)?;
    let replay_backend = replay_backend_name(gpu.replay.request());
    let requested_backend =
        replay_backend_request(std::env::var(REPLAY_BACKEND_ENV).ok().as_deref())?;
    if replay_backend != requested_backend {
        return Err(format!(
            "{REPLAY_BACKEND_ENV} resolved to the {replay_backend} backend, not {requested_backend}"
        ));
    }
    // Source admission as the daemon runs it before the carrier loads.
    let ple_choice =
        hipfire_loader::admission::resolve_qwen4_ple_rows(&mut source, model_str, &gpu.arch)?;
    let ple_receipt = ple_rows_receipt(ple_choice.as_ref(), path, &model_id)?;
    let (overlay_candidate, overlay_marker_was_set, hotset_expected) = {
        let ModelSource::Hfq(hfq) = &source else {
            return Err("model source is not an HFQ artifact".into());
        };
        let admitted = hipfire_arch_qwen4::admit_hfqm_artifact_on(hfq, Some(&gpu.arch))
            .map_err(|e| e.to_string())?;
        let overlay =
            hipfire_arch_qwen4::resolve_trunk_overlay(hfq, path, &gpu.arch, &admitted.config)?
                .map(|overlay| overlay.path().display().to_string());
        let hotset = hotset_expectation(path, &gpu, &admitted.manifest.identity.recipe)?;
        (overlay, admitted.config.trunk_iu4_default.is_some(), hotset)
    };
    let state_request = std::env::var("HIPFIRE_STATE_QUANT")
        .ok()
        .filter(|value| !value.is_empty());
    let backend = Qwen4KvBackend::automatic(&gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("qwen4: VMM granularity: {e}"))?,
        Qwen4KvBackend::Legacy => 1,
    };
    let shared = Qwen4RankShared {
        continuous_batch: false,
        path: model_str.to_string(),
        max_seq: args.n_ctx,
        sequence: None,
        backend,
        vmm_granularity,
        kv_mode: hipfire_runtime::config::get().kv_mode.clone(),
        state_quant: state_request.clone(),
        // Native-MTP oracle: explicitly request the existing spec.mtp head/capture
        // reservation before load so identity/snapshot sees the honest census
        // for every native comparison arm.
        spec: SpecLoadCfg {
            mtp: Some(true),
            ..SpecLoadCfg::default()
        },
        native_mtp: true,
        max_k: args.max_k,
    };
    let loaded = load_qwen4_rank(&mut gpu, None, &shared, None, None, source)?;
    if loaded.max_seq < args.n_ctx {
        return Err(format!(
            "the loader admitted max_seq {}, below --n-ctx {}",
            loaded.max_seq, args.n_ctx
        ));
    }
    let bundle = loaded.bundle;
    let state_format = bundle.state_format();
    let hotset_receipt = hotset_receipt(
        &hotset_expected,
        bundle
            .weights
            .expert_tiers()
            .and_then(|lease| lease.hotset_sidecar_sha256()),
        &model_id,
    )?;
    let vocab = bundle.config.vocab_size;
    let split_armed = !bundle.split_layers().is_empty() && bundle.split_digest().is_some();
    // Match the normal producer's settled-state test: attachment can refuse
    // a resolved candidate while continuing with base weights. A native IU4
    // marker predating attachment cannot prove this candidate was applied.
    if overlay_candidate.is_some() && overlay_marker_was_set {
        return Err(
            "pre-existing trunk overlay marker cannot prove this candidate was applied".into(),
        );
    }
    let overlay_applied = overlay_candidate.is_some() && bundle.config.trunk_iu4_default.is_some();
    let overlay_expected = overlay_candidate.filter(|_| overlay_applied);
    let overlay_sha = overlay_expected
        .as_ref()
        .map(|_| model_id["trunk-overlay"]["sha256"].clone());
    // Actual loaded descriptor for the FN chooser: the cached canonical JSON
    // parsed once here (outside any window), never hand-built. Unavailable
    // stays an honest refusal; the ordinary load is unaffected.
    let measured_load = match loaded.measured_load_identity.as_deref() {
        Some(identity) => match serde_json::from_str::<Value>(identity.canonical_json()) {
            Ok(canonical) => canonical,
            Err(error) => json!({
                "available": false,
                "reason": format!("measured load canonical parse failed: {error}"),
            }),
        },
        None => json!({
            "available": false,
            "reason": "measured load descriptor unavailable: real incumbent, no verified identity",
        }),
    };
    let production = json!({
        "loader": "hipfire_loader::load_qwen4_rank (production serve load)",
        "exe": hw_identity::binary_identity(),
        "compiled_sources": hw_identity::compiled_sources(),
        "config": hw_identity::config_identity(&hw_identity::env_snapshot(&[], &[]), &model_id),
        "receipt": hw_identity::receipt(&gpu, path),
        "model": model_id,
        "ple_rows": ple_receipt,
        "trunk_overlay": {
            "expected_path": overlay_expected,
            "sha256": overlay_sha,
            "applied_evidence": "resolved candidate with no pre-existing trunk IU4 marker and settled bundle.config.trunk_iu4_default",
        },
        "expert_hotset": hotset_receipt,
        "gdn_state": {
            "request": state_request,
            "resolved": state_format.gdn.name(),
        },
        "state_format": format!("{state_format:?}"),
        "state_storage": {
            "qsa_kv": format!("{:?}", state_format.qsa),
            "gdn": state_format.gdn.name(),
            "qsa_f16_arena": state_format.qsa_f16_arena,
            "qsa_pooled_bf16": state_format.qsa_pooled_bf16,
        },
        "kv_backend": backend.name(),
        "replay_backend": replay_backend,
        "max_seq": loaded.max_seq,
        "card_cap": loaded.card_cap,
        "native_mtp": loaded.native_mtp,
        "max_k": args.max_k,
        "measured_load": measured_load,
    });
    let ple_line = ple_receipt_summary(&production["ple_rows"]);
    eprintln!(
        "qwen4_wide_verify: production load of {} ({:.2} GB) on {}: {ple_line}, native_mtp={}, max_k={}",
        path.display(),
        model_bytes as f64 / 1e9,
        gpu.arch,
        loaded.native_mtp,
        args.max_k
    );
    let info = LoadInfo {
        arch: gpu.arch.to_string(),
        ple_line,
        ple_note: None,
        ple_rows_source: production["ple_rows"]["path"]
            .as_str()
            .unwrap_or("none")
            .to_string(),
        overlay_applied,
        model_bytes,
        uma: gpu.is_uma(),
        expert_vram_layers: None,
        offload_exec: hipfire_config::memory::offload_exec(),
        partition_digest: None,
        split_armed,
        loader: "production",
        production,
    };
    Ok(Model {
        gpu,
        bundle,
        vocab,
        tokenizer,
        info,
        measured_identity: loaded.measured_load_identity,
        measured_snapshot: loaded.measured_resource_snapshot,
    })
}

/// Load-only allocation-plan probe: run the real production serve load, then
/// read the loader's already-admitted 64-row capture allocation, its typed
/// `measured_resource_snapshot` and cached canonical `measured_load_identity`,
/// and emit the exact raw `qwen4_wide_verify.resource_plan.v1` shape E
/// consumes (`schema`, `complete`, `production`, `snapshot`). No prompts are
/// decoded, no drafts/windows/timing run, and nothing is armed or grown
/// post-load: the 64 rows must already be allocated BY THE LOADER under the
/// caller's existing flags (the 64-row reserve comes through `--max-k`);
/// anything else is a named `Err`, never an implicit widen. Never
/// decodes/drafts/forwards prompts, never invents prices/timings/identity: a
/// missing identity, snapshot or 64-row allocation, a mismatched identity
/// binding, or an unsupported (PP/EP/...) source is a named `Err` reason,
/// never a fake `complete`. No env is set internally.
pub(super) fn resource_plan(args: &Args) -> Result<Value, String> {
    let model = load(args)?;
    let ring_rows = model.bundle.native_mtp_capture_rows();
    if ring_rows != 64 {
        return Err(format!(
            "resource plan: unsupported source: loader allocated {ring_rows} capture rows, not the row-exact 64"
        ));
    }
    let identity = model.measured_identity.as_ref().ok_or_else(|| {
        "resource plan: measured identity pending: loader supplied no measured-load identity"
            .to_owned()
    })?;
    let snapshot = model.measured_snapshot.as_ref().ok_or_else(|| {
        "resource plan: measured snapshot pending: loader supplied no resource snapshot".to_owned()
    })?;
    if !Arc::ptr_eq(snapshot.identity(), identity)
        && snapshot.identity().canonical_json() != identity.canonical_json()
    {
        return Err(
            "resource plan: identity binding mismatch: snapshot identity differs from loader identity"
                .to_owned(),
        );
    }
    if snapshot.value().get("upper_bound").is_none() {
        return Err(
            "resource plan: snapshot pending: no source-proven upper_bound block".to_owned(),
        );
    }
    Ok(json!({
        "schema": RESOURCE_PLAN_SCHEMA,
        "complete": true,
        "production": model.info.production.clone(),
        "snapshot": snapshot.value().clone(),
    }))
}

/// Source-planned preload probe: metadata-only, before any production load.
/// Opens the HFQ headers/index only (`HfqFile::open`, never payload reads),
/// initialises the GPU bare for the typed predicates/quotes the existing
/// authorities need, and calls the loader's metadata-only
/// `qwen4_preload_resource_plan` with the ORDINARY loader ask: resolved
/// backend / bases, `speculation.mtp`/`mtp_k` lifted from the process runtime
/// config with the existing `parse_tri_state` API (`HIPFIRE_MTP_MODE`,
/// default auto; `HIPFIRE_MTP_K`), explicit `--n-ctx` honored, planning
/// default the finite 544 bench (never the oracle 8192 default, never
/// forced capture64). B resolves the effective MTP/max-k policy from that
/// ask (retained default, head presence, floor, spill) exactly as the
/// single-device load does. No `load()` call, no `hw_identity`
/// binary/source hashing, no sidecar verification/hash, no symmetric payload
/// scan, no weight fulfillment, no worker startup, no capture arming. Output
/// says `planning_only`/`source_planned`/`unmeasured`/`nonqualification`
/// with `complete: false`, `qualification: nonqualification` explicitly:
/// never production RAW, never a fit, never a qualification. A missing
/// required header/index field or overflow is a named `Err`, never a fake
/// plan.
pub(super) fn preload_plan(args: &Args) -> Result<Value, String> {
    use hipfire_runtime::hfq::HfqFile;

    if args.preload_plan_only && args.resource_plan_only {
        return Err(
            "preload plan: --preload-plan-only and --resource-plan-only are exclusive".into(),
        );
    }
    let planning_n_ctx = if args.n_ctx_given {
        args.n_ctx
    } else if args.n_ctx == PRELOAD_PLAN_DEFAULT_N_CTX {
        // Callers (and the arg parser) default exactly here; treat it as the
        // intended finite bench, not as an oracle override to repair.
        args.n_ctx
    } else {
        // Defensive: any other inherited default (e.g. the oracle 8192) is
        // NOT the planning bench -- plan the finite bench explicitly.
        PRELOAD_PLAN_DEFAULT_N_CTX
    };
    if planning_n_ctx == 0 {
        return Err("preload plan: planning context is zero".into());
    }
    let path = args.model.as_path();
    // Metadata-only open: header/index parse for `bytes_of` tenure. Never
    // payload tensors, never sidecar hash/verification reads.
    let mut hfq =
        HfqFile::open(path).map_err(|e| format!("preload plan: open {}: {e}", path.display()))?;
    hipfire_loader::prepare_host_memory_for(path);
    // Bare GPU init: typed predicates/quotes only, no model allocation. The
    // GPU arch drives the ordinary `memory.ple_format = auto` resolution
    // below, exactly as the loaders do before admission.
    let gpu = Gpu::init().map_err(es)?;
    // Ordinary PLE rows selection: header-only attach of the embedded rows
    // or the `<model>.ple-q4/8` sidecar the serve load would bind
    // (`derive_missing_q4: false`, the loader's no-derive policy). A missing
    // required sidecar is a named refusal for B to surface, never a silent
    // PLE-0. No `open_table_store` rank reads, no row-payload hash/verify.
    let ple_choice = hipfire_runtime::ple_sidecar::attach_validation_ple_rows(
        &mut hfq,
        path,
        hipfire_config::memory::ple_format(),
        &gpu.arch,
    )
    .map_err(|e| format!("preload plan: PLE rows: {e}"))?;
    // Ordinary serve-path config, same sources the daemon forwards: the
    // process `speculation.mtp` tri-state (`HIPFIRE_MTP_MODE`, default auto)
    // lifted with the existing `parse_tri_state` API, and `spec.mtp_k`
    // staying runtime auto (`None`) unless the caller EXPLICITLY passed
    // `--max-k` (tracked by `args.max_k_given`, mirroring `n_ctx_given`):
    // the resolved oracle default in `args.max_k` must never leak into the
    // ask. Runtime `HIPFIRE_MTP_K` stays visible to B through the same
    // config fallback the load uses. Nothing here forces MTP on or off: B
    // resolves the effective policy (`native_mtp` / `max_k` via its private
    // `qwen4_native_mtp` + `qwen4_native_mtp_k`) exactly as the
    // single-device load does.
    let runtime_cfg = hipfire_runtime::config::get();
    let spec = SpecLoadCfg {
        mtp: hipfire_config::parse_tri_state(&runtime_cfg.mtp_mode),
        mtp_k: preload_spec_mtp_k(args.max_k_given, args.max_k),
        ..SpecLoadCfg::default()
    };
    let backend = Qwen4KvBackend::automatic(&gpu);
    let vmm_granularity = match backend {
        Qwen4KvBackend::Vmm => gpu
            .vmm_recommended_granularity()
            .map_err(|e| format!("preload plan: qwen4 VMM granularity: {e}"))?,
        Qwen4KvBackend::Legacy => 1,
    };
    // `shared.max_k` carries the ordinary effective value B's census plans
    // (explicit/runtime ask clamped, else the GPU auto cap), computed with
    // the same authorities B's `qwen4_native_mtp_k` uses before placement.
    // B still reports its own effective resolution in `union.context`.
    let shared_max_k = preload_effective_max_k(spec.mtp_k, runtime_cfg.mtp_k, &gpu.arch);
    let path_str = path
        .to_str()
        .ok_or_else(|| format!("preload plan: {} is not UTF-8", path.display()))?
        .to_string();
    // The ordinary-resolved request B charges: single-device `pp = tp = ep
    // = 1`, retained-Redline fact, measured AR floor and head presence
    // resolved exactly as `Qwen4Carrier::load` does before placement — the
    // same policy B applies locally to its effective fields. No oracle
    // forcing: `None` auto stays auto unless config/flags say otherwise.
    let retained = preload_retained_default(&gpu.arch, &path_str);
    let floor = hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR;
    let floor_native_mtp = preload_native_mtp_for_floor(spec.mtp, retained, floor);
    let max_k = preload_effective_max_k(spec.mtp_k, runtime_cfg.mtp_k, &gpu.arch);
    // Flash-Next profiles always carry the trunk MTP head: admit it into
    // the request exactly as the admitted manifest would for the floor
    // verdict. Developer profiles have no bundled head to resolve
    // metadata-only: an explicit `on` is refused there (the head is
    // absent), while auto/off keep AR — the same `for_head` verdict,
    // without a manifest read.
    let flash_next = hfq.metadata_json.contains("flash-next");
    let native_mtp = if flash_next {
        floor_native_mtp
    } else if spec.mtp == Some(true) {
        return Err(
            "preload plan: native MTP (--spec mtp) was requested, but this profile has no MTP head"
                .into(),
        );
    } else {
        false
    };
    let shared = Qwen4RankShared {
        continuous_batch: false,
        path: path_str,
        max_seq: planning_n_ctx,
        sequence: None,
        backend,
        vmm_granularity,
        kv_mode: runtime_cfg.kv_mode.clone(),
        state_quant: std::env::var("HIPFIRE_STATE_QUANT")
            .ok()
            .filter(|value| !value.is_empty()),
        spec,
        native_mtp,
        max_k,
    };
    let union = hipfire_loader::qwen4_preload_resource_plan(&gpu, &shared, &hfq)?;
    // Effective fields come from B's resolved union, never the ask above.
    let effective = union.get("context").cloned().unwrap_or(Value::Null);
    Ok(json!({
        "schema": PRELOAD_PLAN_SCHEMA,
        "planning_only": true,
        "evidence_kind": PRELOAD_PLAN_EVIDENCE_KIND,
        "complete": false,
        "qualification": "nonqualification",
        "source_planned": true,
        "unmeasured": true,
        "nonqualification": true,
        "planning_context": {
            "max_seq": planning_n_ctx,
            "explicit_n_ctx": args.n_ctx_given,
            "max_k": max_k,
            "explicit_max_k": args.max_k_given,
            "kv_backend": backend.name(),
            "state_request": shared.state_quant,
        },
        "loader_config": {
            "spec_mtp": spec.mtp,
            "spec_mtp_k": spec.mtp_k,
            "spec_source": "runtime speculation.mtp/mtp_k (HIPFIRE_MTP_MODE/HIPFIRE_MTP_K, auto default; --max-k only when explicit)",
            "ple_rows": {
                "format": ple_choice.format.name(),
                "origin": match &ple_choice.origin {
                    hipfire_runtime::ple_sidecar::PleRowsOrigin::Embedded => {
                        Value::String("embedded".into())
                    }
                    hipfire_runtime::ple_sidecar::PleRowsOrigin::Sidecar(sidecar) => {
                        json!({"sidecar": sidecar.display().to_string()})
                    }
                },
                "derived": ple_choice.derived,
                "note": ple_choice.note,
            },
            "native_mtp_request": shared.native_mtp,
            "native_mtp_request_note": "ordinary-resolved request (B charges/reports its own effective fields alongside)",
            "effective": effective,
            "capture_rows": "unallocated (no capture64 arming)",
        },
        "union": union,
    }))
}

/// The `spec.mtp_k` ask for the preload plan: runtime auto (`None`) unless
/// the caller explicitly passed `--max-k` (post-load `args.max_k` carries
/// the resolved oracle default, which must never leak into the ask).
pub(super) fn preload_spec_mtp_k(max_k_given: bool, max_k: usize) -> Option<usize> {
    max_k_given.then_some(max_k)
}

/// The ordinary retained-Redline fact for a single-device Qwen4 planning
/// load (`pp = tp = ep = 1`), via the same public
/// `retained_redline_default` authority the loader's admission uses.
pub(super) fn preload_retained_default(gpu_arch: &str, model_path: &str) -> bool {
    hipfire_runtime::config::retained_redline_default(gpu_arch, "qwen4", model_path, 1, 1, 1, true)
}

/// The ordinary floor policy for a single-device Qwen4 planning load: the
/// same `MtpMode` tri-state table B's private `qwen4_native_mtp_for_floor`
/// applies — `off` stays AR, `on` attaches (refusing the retained default),
/// `auto` attaches only off-retained with the producer's measured AR floor.
pub(super) fn preload_native_mtp_for_floor(
    spec_mtp: Option<bool>,
    retained: bool,
    measured_ar_floor: bool,
) -> bool {
    use hipfire_config::MtpMode;
    match MtpMode::from_tri_state(spec_mtp) {
        MtpMode::Off => false,
        MtpMode::On => !retained,
        auto @ MtpMode::Auto => {
            !retained && hipfire_config::mtp_mode_admits(auto, measured_ar_floor)
        }
    }
}

/// The ordinary effective `shared.max_k` B's census plans: the explicit or
/// runtime `mtp_k` ask clamped to the physical executor width
/// (`spec_economics::DERIVED_MAX_DRAFTS`), else the GPU row-exact auto cap
/// (`mtp_k_default` over this arch's verify rows) — the same authorities
/// B's `qwen4_native_mtp_k` uses before placement.
pub(super) fn preload_effective_max_k(
    spec_mtp_k: Option<usize>,
    runtime_mtp_k: Option<usize>,
    gpu_arch: &str,
) -> usize {
    use hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS;
    spec_mtp_k
        .or(runtime_mtp_k)
        .map(|k| k.clamp(1, DERIVED_MAX_DRAFTS))
        .unwrap_or_else(|| {
            hipfire_config::mtp_k_default(
                hipfire_arch_qwen4::ARCH_ID,
                rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(gpu_arch),
            )
            .max(1)
        })
}

fn ple_receipt_summary(receipt: &Value) -> String {
    format!(
        "ple rows: format={} origin={}",
        receipt["format"].as_str().unwrap_or("none"),
        receipt["origin"].as_str().unwrap_or("none")
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn replay_backend_request_canonicalizes_and_refuses_unknowns() {
        assert_eq!(replay_backend_request(None), Ok("hip"));
        assert_eq!(replay_backend_request(Some("")), Ok("hip"));
        assert_eq!(replay_backend_request(Some("OFF")), Ok("hip"));
        assert_eq!(replay_backend_request(Some("Redline")), Ok("auto"));
        assert_eq!(replay_backend_request(Some("shadow")), Ok("shadow"));
        assert!(replay_backend_request(Some("graph")).is_err());
    }

    #[test]
    fn hotset_receipt_never_accepts_a_mismatched_or_unapplied_hot_set() {
        let id = json!({"expert-hotset": {"sha256": "aa"}});
        let path = PathBuf::from("m.expert-hotset");
        let applied = hotset_receipt(&Ok(path.clone()), Some("aa"), &id).unwrap();
        assert_eq!(applied["applied"], json!(true));
        assert!(hotset_receipt(&Ok(path.clone()), Some("bb"), &id).is_err());
        assert!(hotset_receipt(&Ok(path), None, &id).is_err());
        let skipped = hotset_receipt(&Err("off".into()), None, &id).unwrap();
        assert_eq!(skipped["applied"], json!(false));
        assert!(hotset_receipt(&Err("off".into()), Some("aa"), &id).is_err());
    }

    #[test]
    fn ple_receipt_requires_the_hashed_sidecar() {
        let model = Path::new("/m/model.mq4");
        let id = json!({"ple-q8": {"sha256": "cc"}});
        let none = ple_rows_receipt(None, model, &id).unwrap();
        assert_eq!(none["origin"], json!("none"));
        assert_eq!(
            ple_receipt_summary(&none),
            "ple rows: format=none origin=none"
        );
    }

    fn synthetic_identity_fields(capture_rows: &str) -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::from([
            ("schema".to_owned(), "qwen4-measured-load.v1".to_owned()),
            ("model.sha256".to_owned(), "ab".repeat(32)),
            ("model.bytes".to_owned(), "123456".to_owned()),
            ("ple.sha256".to_owned(), "cd".repeat(32)),
            ("ple.bytes".to_owned(), "7890".to_owned()),
            ("ple.encoding".to_owned(), "q8-synthetic".to_owned()),
            ("overlay".to_owned(), "none".to_owned()),
            ("hotset".to_owned(), "none".to_owned()),
            (
                "placement.digest".to_owned(),
                "synthetic-placement-digest-0".to_owned(),
            ),
            (
                "recipe".to_owned(),
                "synthetic-recipe-trunk-fp8-mtp-test-only".to_owned(),
            ),
            (
                "state".to_owned(),
                "SyntheticTestState(TestOnly)".to_owned(),
            ),
            ("kv_backend".to_owned(), "synthetic-test-backend".to_owned()),
            ("max_seq".to_owned(), "2048".to_owned()),
            ("capture_rows".to_owned(), capture_rows.to_owned()),
            ("gpu.arch".to_owned(), "gfx1201-synthetic".to_owned()),
            ("gpu.uuid".to_owned(), "SYNTHETIC-GPU-UUID-0".to_owned()),
            (
                "scope".to_owned(),
                "synthetic-scope/chunk0/state/test-only".to_owned(),
            ),
            (
                "flags".to_owned(),
                "SyntheticFlags(test_only:true)".to_owned(),
            ),
        ])
    }

    #[test]
    fn synthetic_identity_round_trips_canonical_json() {
        use hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity;
        let identity = Qwen4MeasuredLoadIdentity::from_fields(synthetic_identity_fields("64"))
            .expect("synthetic valid descriptor");
        let parsed: std::collections::BTreeMap<String, String> =
            serde_json::from_str(identity.canonical_json()).expect("canonical parses");
        assert_eq!(parsed, synthetic_identity_fields("64"));
        assert_eq!(parsed["capture_rows"], "64");
    }

    #[test]
    fn synthetic_raw_plan_has_exact_keys_and_rejects_timing_or_short_capture() {
        use hipfire_arch_qwen4::measured_identity::Qwen4MeasuredLoadIdentity;
        let identity = Qwen4MeasuredLoadIdentity::from_fields(synthetic_identity_fields("64"))
            .expect("synthetic valid descriptor");
        let measured: Value =
            serde_json::from_str(identity.canonical_json()).expect("canonical parses");
        let raw = json!({
            "schema": RESOURCE_PLAN_SCHEMA,
            "complete": true,
            "production": {"loader": "hipfire_loader::load_qwen4_rank (production serve load)"},
            "snapshot": {
                "schema": "fn-oracle.load-resource-snapshot.v1",
                "evidence_kind": "checked_allocation_plan_not_peak",
                "measured_load": measured,
                "upper_bound": {"evidence_kind": "conservative_component_union_upper_bound_not_peak"},
            },
        });
        assert_eq!(raw["schema"], json!(RESOURCE_PLAN_SCHEMA));
        assert_eq!(raw["complete"], json!(true));
        for key in ["windows", "prompts", "widths", "prices"] {
            assert!(raw.get(key).is_none(), "raw payload carries timing rows");
        }
        assert_eq!(
            raw["snapshot"]["measured_load"]["capture_rows"],
            json!("64")
        );
        let short = Qwen4MeasuredLoadIdentity::from_fields(synthetic_identity_fields("8"))
            .expect("synthetic valid descriptor");
        let short_measured: Value =
            serde_json::from_str(short.canonical_json()).expect("canonical parses");
        assert_ne!(
            short_measured["capture_rows"],
            json!("64"),
            "a non-64 identity never labels complete"
        );
    }

    #[test]
    fn preload_plan_schema_is_planning_only_nonqualification() {
        assert_eq!(PRELOAD_PLAN_SCHEMA, "qwen4_wide_verify.preload_plan.v1");
        assert_ne!(PRELOAD_PLAN_SCHEMA, RESOURCE_PLAN_SCHEMA);
        assert_eq!(
            PRELOAD_PLAN_EVIDENCE_KIND,
            "source_planned_unmeasured_nonqualification"
        );
        // The finite 544 bench is the planning default, never the 8192 oracle.
        assert_eq!(PRELOAD_PLAN_DEFAULT_N_CTX, 544);
        assert_eq!(PRELOAD_PLAN_DEFAULT_N_CTX, 512usize.max(128 + 1) + 32);
        assert_ne!(PRELOAD_PLAN_DEFAULT_N_CTX, 8192);
        // The emitted shape below must carry exactly these keys: no
        // `production` RAW, no `snapshot`, no timing rows.
        let planned = json!({
            "schema": PRELOAD_PLAN_SCHEMA,
            "planning_only": true,
            "evidence_kind": PRELOAD_PLAN_EVIDENCE_KIND,
            "complete": false,
            "qualification": "nonqualification",
            "source_planned": true,
            "unmeasured": true,
            "nonqualification": true,
            "planning_context": {"max_seq": 544usize, "explicit_n_ctx": false},
            "loader_config": {"spec_source": "runtime speculation.mtp/mtp_k"},
            "union": {"components": []},
        });
        assert_eq!(planned["planning_only"], json!(true));
        assert_eq!(planned["complete"], json!(false));
        assert_eq!(planned["qualification"], json!("nonqualification"));
        assert_eq!(planned["source_planned"], json!(true));
        assert_eq!(planned["unmeasured"], json!(true));
        assert_eq!(planned["nonqualification"], json!(true));
        assert_eq!(planned["planning_context"]["max_seq"], json!(544));
        for key in [
            "production",
            "snapshot",
            "predictions",
            "windows",
            "prompts",
            "widths",
            "prices",
        ] {
            assert!(planned.get(key).is_none(), "planned payload carries {key}");
        }
    }

    #[test]
    fn preload_spec_ask_lifts_the_runtime_tri_state_without_forcing_either_arm() {
        // Real `hipfire_config::parse_tri_state` (not a re-spelled match):
        // the ordinary serve ask passes auto/on/off through, never pinning a
        // planning verdict. B reports the effective arm in `union.context`.
        assert_eq!(hipfire_config::parse_tri_state("auto"), None);
        assert_eq!(hipfire_config::parse_tri_state("on"), Some(true));
        assert_eq!(hipfire_config::parse_tri_state("off"), Some(false));
        let spec = SpecLoadCfg {
            mtp: hipfire_config::parse_tri_state("auto"),
            mtp_k: preload_spec_mtp_k(false, 3),
            ..SpecLoadCfg::default()
        };
        assert_eq!(spec.mtp, None);
        assert_eq!(spec.mtp_k, None, "absent --max-k stays runtime auto");
        assert!(!(spec.mtp == Some(true)), "auto ask must not force MTP on");
        assert_eq!(preload_spec_mtp_k(true, 7), Some(7));
        // The real floor table the single-device load applies: `off` stays
        // AR, `on` attaches off-retained, `auto` needs the measured floor.
        // Qwen4 declares the floor, so auto arms off-retained (B's verdict).
        assert!(hipfire_arch_qwen4::MTP_MEASURED_AR_FLOOR);
        assert!(!preload_native_mtp_for_floor(Some(false), false, true));
        assert!(preload_native_mtp_for_floor(Some(true), false, true));
        assert!(!preload_native_mtp_for_floor(Some(true), true, true));
        assert!(preload_native_mtp_for_floor(None, false, true));
        assert!(!preload_native_mtp_for_floor(None, true, true));
        assert!(!preload_native_mtp_for_floor(None, false, false));
    }

    #[test]
    fn preload_max_k_ask_stays_auto_without_explicit_flag() {
        // The oracle's finite default is not a loader ask. Preserve AUTO in
        // SpecLoadCfg; only the effective capacity uses the GPU's row-exact
        // default, not the executor's physical storage ceiling.
        use hipfire_runtime::spec_economics::DERIVED_MAX_DRAFTS;
        let auto = SpecLoadCfg {
            mtp_k: preload_spec_mtp_k(false, 3),
            ..SpecLoadCfg::default()
        };
        assert_eq!(auto.mtp_k, None, "absent --max-k stays runtime auto");
        for arch in ["gfx1201", "gfx1151"] {
            assert_eq!(
                preload_effective_max_k(auto.mtp_k, None, arch),
                hipfire_config::mtp_k_default(
                    hipfire_arch_qwen4::ARCH_ID,
                    rdna_compute::prefill_invariant::qwen4_row_exact_verify_rows(arch),
                )
                .max(1)
            );
        }
        let explicit = SpecLoadCfg {
            mtp_k: preload_spec_mtp_k(true, 7),
            ..SpecLoadCfg::default()
        };
        assert_eq!(explicit.mtp_k, Some(7));
        assert_eq!(preload_effective_max_k(explicit.mtp_k, None, "gfx1201"), 7);
        assert_eq!(
            preload_effective_max_k(explicit.mtp_k, Some(5), "gfx1201"),
            7
        );
        assert_eq!(
            preload_effective_max_k(preload_spec_mtp_k(true, usize::MAX), None, "gfx1201"),
            DERIVED_MAX_DRAFTS
        );
        assert_eq!(preload_effective_max_k(auto.mtp_k, Some(5), "gfx1201"), 5);
    }

    #[test]
    fn preload_ple_choice_reports_format_origin_without_deriving() {
        // The real choice shape B must agree with: a header-only attach
        // reports `{format, origin, derived: false, note}` — never a
        // derivation, never a payload read. The probe refuses to plan a
        // silent PLE-0: `derived` is always false on this path.
        let choice = hipfire_runtime::ple_sidecar::PleRowsChoice {
            format: hipfire_config::memory::PleFormat::Q8,
            origin: hipfire_runtime::ple_sidecar::PleRowsOrigin::Embedded,
            derived: false,
            note: None,
        };
        assert_eq!(choice.format.name(), "q8");
        assert!(!choice.derived);
        let sidecar = hipfire_runtime::ple_sidecar::PleRowsChoice {
            format: hipfire_config::memory::PleFormat::Q4,
            origin: hipfire_runtime::ple_sidecar::PleRowsOrigin::Sidecar(PathBuf::from("m.ple-q4")),
            derived: false,
            note: None,
        };
        assert_eq!(sidecar.format.name(), "q4");
        assert!(!sidecar.derived);
    }
}
