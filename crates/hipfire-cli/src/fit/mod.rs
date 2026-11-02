// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Read-only load admission: the loader decides, this command explains its
//! decision using machine counters and published receipts. Never initialize HIP
//! or start a daemon here; GPU-dependent placement remains explicitly bounded.

mod capacity;
mod host;
mod qwen4;
mod remote;
pub(crate) mod types;

use anyhow::Result;
use capacity::{BestDevice, GpuCapacity};
use clap::Args;
use hipfire_loader::admission::{
    admit_source_with_options, qwen4_kv_adaptive_requested, qwen_default_q8_enabled,
    EffectiveTopology, KvBackendHints, KvBackendRequest, Qwen35CardProjection, SourceAdmission,
    SourceAdmissionOptions, QWEN35_MIN_AUTO_CONTEXT,
};
use hipfire_registry::{receipts, ModelEntry, RegistryV1};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::loader_api::{CaskConfig, ModelSource, SpecLoadCfg};
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use types::{gib, Finding, FitDevice, FitHost, PlacementPlan, Verdict};

/// `plan.basis` in the JSON report.
const PLAN_BASIS: &str = "pre-load";
/// The text plan's `basis` line: what its numbers are, and what the load adds.
const PLAN_BASIS_TEXT: &str = "pre-load estimate from file headers + sysfs; the load re-measures free VRAM after weights, drafts and scratch (max_seq fit) and may admit less";

#[derive(Args, Debug)]
pub(crate) struct FitArgs {
    /// Installed model, alias, file, path, or registry tag.
    pub model: String,
    /// Context to admit (otherwise use the same configured/automatic bound as run).
    #[arg(long, value_parser = parse_context)]
    pub ctx: Option<usize>,
    /// Do not fetch headers for models that are not installed.
    #[arg(long)]
    pub no_fetch: bool,
    /// Print one hipfire-fit-1 JSON object.
    #[arg(short = 'j', long)]
    pub json: bool,
    /// Also print the source-cited derivation lines (always in --json).
    #[arg(short = 'v', long)]
    pub verbose: bool,
    /// One-shot KV format for this load, validated as `run --kv-mode` does.
    #[arg(long)]
    pub kv_mode: Option<String>,
    /// Qwen-only K-format override (same values as `run --kv-k`).
    #[arg(long = "kv-k")]
    pub kv_k: Option<String>,
    /// Qwen-only V-format override (same values as `run --kv-v`).
    #[arg(long = "kv-v")]
    pub kv_v: Option<String>,
    /// Speculation selector for this load, as `run --spec`: off, auto, ngram,
    /// dflash, mtp, or dspark. Overrides `speculation.mode`.
    #[arg(
        long = "spec",
        alias = "speculation",
        value_parser = ["off", "auto", "ngram", "dflash", "mtp", "dspark"]
    )]
    pub spec: Option<String>,
    /// Tensor/expert-parallel degree for this load (same parser and range as
    /// `run --tp`); admission answers for it exactly as the daemon would.
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=64))]
    pub tp: Option<u64>,
}

impl FitArgs {
    /// The same one-shot overrides `hipfire run` layers over the config.
    fn load_overrides(&self) -> crate::LoadOverrides<'_> {
        crate::LoadOverrides {
            kv_mode: self.kv_mode.as_deref(),
            kv_k: self.kv_k.as_deref(),
            kv_v: self.kv_v.as_deref(),
            speculation: self.spec.as_deref(),
            ..Default::default()
        }
    }
}

/// The one-shot overrides this fit applied, echoed in the report (`null` =
/// not set, the config decides).
#[derive(Serialize)]
struct OverridesEcho {
    kv_mode: Option<String>,
    kv_k: Option<String>,
    kv_v: Option<String>,
    spec: Option<String>,
    tp: Option<u64>,
}

impl From<&FitArgs> for OverridesEcho {
    fn from(args: &FitArgs) -> Self {
        Self {
            kv_mode: args.kv_mode.clone(),
            kv_k: args.kv_k.clone(),
            kv_v: args.kv_v.clone(),
            spec: args.spec.clone(),
            tp: args.tp,
        }
    }
}

impl OverridesEcho {
    /// The flags as `hipfire run` would take them, each led by a space.
    fn flags(&self) -> String {
        [
            ("--kv-mode", &self.kv_mode),
            ("--kv-k", &self.kv_k),
            ("--kv-v", &self.kv_v),
            ("--spec", &self.spec),
        ]
        .iter()
        .filter_map(|(flag, value)| value.as_deref().map(|value| format!(" {flag} {value}")))
        .chain(self.tp.map(|tp| format!(" --tp {tp}")))
        .collect()
    }
}

fn parse_context(raw: &str) -> std::result::Result<usize, String> {
    raw.parse::<usize>()
        .ok()
        .filter(|n| (512..=1_048_576).contains(n))
        .ok_or_else(|| "--ctx must be an integer between 512 and 1048576".into())
}

#[derive(Serialize)]
struct ModelInfo {
    tag: String,
    file: Option<String>,
    installed: bool,
    replica: bool,
    size_bytes: Option<u64>,
    arch_id: Option<u32>,
    quant: Option<String>,
}

#[derive(Serialize)]
struct Route {
    name: String,
    value: String,
    why: String,
}

#[derive(Serialize)]
struct Plan {
    /// Where the plan's numbers come from: `pre-load` (file headers and
    /// sysfs; the load re-measures before it admits).
    basis: &'static str,
    device_index: usize,
    topology: String,
    kv_backend: String,
    kv_backend_reason: Option<String>,
    kv_mode: String,
    max_seq: usize,
    bound: String,
    model_ctx: Option<usize>,
    projection: Option<Value>,
    /// The visible GPU that holds the most context (additive; set when more
    /// than one GPU is visible and the plan is a checked single-GPU one).
    #[serde(skip_serializing_if = "Option::is_none")]
    best_device: Option<BestDevice>,
    /// Every visible GPU's projection, for the machine rows, the best-GPU line
    /// and the tier table.
    #[serde(skip)]
    gpus: Vec<GpuCapacity>,
    /// The context was left automatic (the loader's own minimum applies).
    #[serde(skip)]
    automatic: bool,
    /// `hardware.devices` is set, so it alone selects a GPU.
    #[serde(skip)]
    devices_pinned: bool,
    placement: PlacementPlan,
    decode_route: String,
    /// Why the load takes that route (the setting that selected it, or the
    /// one that kept AR).
    decode_reason: String,
    routes: Vec<Route>,
}

#[derive(Serialize)]
struct TierRow {
    tag: String,
    size_bytes: Option<u64>,
    verdict: Verdict,
    summary: String,
    /// The GPU the tier's capacity was projected on (its best one).
    #[serde(skip_serializing_if = "Option::is_none")]
    best_gpu: Option<usize>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    hipfire: Value,
    input: String,
    /// The one-shot load overrides this report was computed with.
    overrides: OverridesEcho,
    model: ModelInfo,
    machine: FitHost,
    plan: Option<Plan>,
    findings: Vec<Finding>,
    verdict: Verdict,
    tiers: Vec<TierRow>,
    receipts: Vec<Value>,
    notes: Vec<String>,
    /// Source-cited derivation lines (`--verbose` in text).
    details: Vec<String>,
}

struct Analysis {
    model: ModelInfo,
    plan: Option<Plan>,
    findings: Vec<Finding>,
    notes: Vec<String>,
    details: Vec<String>,
}

/// Give admission the visible-device count without opening the HIP runtime.
///
/// The loader's `visible_device_count` (crates/hipfire-loader/src/admission.rs
/// `visible_device_count`) reads `hardware.devices`, then
/// `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`, and only then dlopens HIP for
/// `hipGetDeviceCount` — which initializes the ROCm runtime and opens
/// `/dev/kfd`. When none of the first three is set, every KFD GPU is visible to
/// the daemon, so this process (which never loads HIP itself) states exactly
/// that set in ROCr order: the count admission reads is the count the daemon's
/// HIP would report, and fit stays GPU-free.
fn supply_visible_device_count(process: &hipfire_config::ProcessConfig, machine: &FitHost) {
    let configured = process
        .legacy_value("HIPFIRE_DEVICES")
        .is_some_and(|v| !v.trim().is_empty());
    let inherited = [
        hipfire_config::devices::HIP_VISIBLE_DEVICES,
        hipfire_config::devices::ROCR_VISIBLE_DEVICES,
    ]
    .iter()
    .any(|name| std::env::var(name).is_ok_and(|value| !value.trim().is_empty()));
    if configured || inherited || machine.devices.is_empty() {
        return;
    }
    let all = (0..machine.devices.len())
        .map(|index| index.to_string())
        .collect::<Vec<_>>()
        .join(",");
    std::env::set_var(hipfire_config::devices::ROCR_VISIBLE_DEVICES, all);
}

/// `hardware.devices` is set: it alone selects the GPU (it wins over
/// `HIP_VISIBLE_DEVICES`, hipfire_config::devices).
fn devices_pinned() -> bool {
    hipfire_config::active_process_config()
        .and_then(|process| process.legacy_value("HIPFIRE_DEVICES"))
        .is_some_and(|spec| !spec.trim().is_empty())
}

/// The admission projection on every visible GPU, from the same facts and
/// function the plan's own device uses.
fn visible_gpu_capacities(
    source: &ModelSource,
    inputs: &AdmissionInputs<'_>,
    machine: &FitHost,
) -> Vec<GpuCapacity> {
    machine
        .devices
        .iter()
        .map(|device| {
            GpuCapacity::new(
                device,
                hipfire_loader::admission::qwen35_card_projection(
                    source,
                    &device.arch,
                    inputs.hints(device),
                ),
            )
        })
        .collect()
}

fn fold_findings(findings: &[Finding]) -> Verdict {
    findings.iter().fold(Verdict::Fits, |v, f| v.and(f.verdict))
}

fn exit_code(verdict: Verdict) -> i32 {
    match verdict {
        Verdict::Fits => 0,
        Verdict::DoesNotFit => 1,
        Verdict::Undetermined => 2,
    }
}

pub(crate) fn fit_command(paths: &crate::Paths, args: FitArgs) -> Result<()> {
    let loaded = crate::load_registry(&paths.registry);
    let registry = &loaded.registry;
    let card = crate::model_card(paths, registry, &args.model);
    let resolved = crate::resolved_for_model(paths, &args.model, card.tag(), card.entry())?;
    let process = hipfire_config::ProcessConfig::from_resolved(&resolved)?;
    // Same immutable startup snapshot as run's spawn_configured, installed
    // before host/preflight can freeze process_value's compatibility table.
    // Family tiers use this one process policy; differing overlays are labelled.
    hipfire_config::install_process_config(process).map_err(|_| {
        anyhow::anyhow!("fit process configuration was initialized before model resolution")
    })?;
    let process = hipfire_config::active_process_config().expect("just installed");
    // Match daemon/main.rs:776's physical -> logical normalization, including
    // ACTIVE_DEVICES used by admission's mesh. Fit never reserves a card: the
    // claim callback only asks the pure selector to retain its ordered result.
    let visibility_error = hipfire_config::devices::apply_device_visibility(process, &mut |_| {
        Ok(hipfire_config::devices::Claim::Claimed)
    })
    .err();
    let (mut machine, host_notes) = host::detect_with_notes(paths);
    let mut findings = preflight_findings(paths);
    if let Some(error) = visibility_error {
        findings.push(Finding::fails("admission", error.to_string(),
            "set `hardware.devices` to a listed GPU, or unset incompatible ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES overrides"));
    }
    for note in host_notes
        .iter()
        .filter(|note| note.starts_with("load refused:"))
    {
        findings.push(Finding::fails("admission", note,
            note.split_once("fix:").map(|(_, fix)| fix.trim()).unwrap_or("correct hardware.devices or the GPU visibility override, then run `hipfire doctor`")));
    }
    if machine.devices.is_empty() {
        findings.push(Finding::undetermined(
            "vram",
            "no AMD GPU detected in the KFD topology",
            Some("run `hipfire doctor`".into()),
        ));
    } else if machine.devices[0].free.is_none() {
        findings.push(Finding::undetermined(
            "vram",
            "free device memory counter is unavailable; capacity cannot be determined",
            Some("run `hipfire doctor` and check amdgpu sysfs memory counters".into()),
        ));
    }
    supply_visible_device_count(process, &machine);
    let analysis = analyze_model(paths, registry, &args.model, &args, &machine)?;
    let mut tiers = Vec::new();
    let family = crate::pull_select::family_tiers(registry, &analysis.model.tag);
    // Every other tier runs the same admission concurrently: each one walks its
    // own artifact's source identity (seconds per tier on a large MoE), and the
    // analyses share only read-only state (the installed process config, the
    // registry, the machine facts) and write disjoint replica cache entries.
    let others: Vec<Option<Result<Analysis>>> = if family.len() > 1 {
        std::thread::scope(|scope| {
            let handles: Vec<_> = family
                .iter()
                .map(|tier| {
                    let tag = tier.tag;
                    let machine = &machine;
                    let args = &args;
                    (tag != analysis.model.tag).then(|| {
                        scope.spawn(move || analyze_model(paths, registry, tag, args, machine))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.map(|handle| {
                        handle
                            .join()
                            .unwrap_or_else(|_| Err(anyhow::anyhow!("tier analysis panicked")))
                    })
                })
                .collect()
        })
    } else {
        Vec::new()
    };
    for (tier, other) in family.iter().zip(others) {
        let owned;
        let evaluated = match other {
            None => &analysis,
            Some(Ok(analysis)) => {
                owned = analysis;
                &owned
            }
            // One tier that cannot be analysed (a failed header fetch, an
            // unreadable local file) is a row of its own, not a failed fit.
            Some(Err(error)) => {
                tiers.push(TierRow {
                    tag: tier.tag.into(),
                    size_bytes: tier.entry.size_bytes,
                    verdict: Verdict::Undetermined,
                    summary: format!("not evaluated: {error:#}"),
                    best_gpu: None,
                });
                continue;
            }
        };
        // The tier's own fit: machine-level blockers (the runtime preflight,
        // device visibility) are reported once above and apply to every tier.
        // A tier whose context was projected per GPU is judged on its best GPU
        // (its largest context, or that its weights do not fit anywhere).
        let on_best = evaluated
            .plan
            .as_ref()
            .filter(|p| !p.gpus.is_empty())
            .and_then(|p| {
                capacity::tier_on_best_gpu(
                    &p.gpus,
                    p.max_seq,
                    p.automatic,
                    machine.devices.len() > 1,
                )
            });
        let (verdict, mut summary, best_gpu) = match on_best {
            Some(best) => {
                let rest = evaluated
                    .findings
                    .iter()
                    .filter(|f| f.id != "context")
                    .fold(Verdict::Fits, |v, f| v.and(f.verdict));
                (rest.and(best.verdict), best.summary, Some(best.gpu))
            }
            None => {
                let verdict = fold_findings(&evaluated.findings);
                // A tier that does not fit is summarized by the first term that
                // fails; otherwise by its capacity and placement.
                let failing = evaluated
                    .findings
                    .iter()
                    .find(|f| f.verdict == Verdict::DoesNotFit);
                let summary = match failing {
                    Some(finding) => format!("{}: {}", finding.id, finding.detail),
                    None => evaluated
                        .plan
                        .as_ref()
                        .map(|p| {
                            p.projection
                                .as_ref()
                                .and_then(|v| v["card_cap"].as_u64())
                                .map(|cap| format!("up to {cap} tokens; {}", p.placement.summary))
                                .unwrap_or_else(|| p.placement.summary.clone())
                        })
                        .unwrap_or_else(|| {
                            evaluated
                                .findings
                                .iter()
                                .map(|f| f.detail.as_str())
                                .collect::<Vec<_>>()
                                .join("; ")
                        }),
                };
                (verdict, summary, None)
            }
        };
        let differs = crate::resolved_for_model(paths, tier.tag, Some(tier.tag), Some(tier.entry))
            .ok()
            .and_then(|r| hipfire_config::ProcessConfig::from_resolved(&r).ok())
            .is_some_and(|tier_process| &tier_process != process);
        if differs {
            summary.push_str(&format!(
                "; per-model settings differ: run `hipfire fit {}` for its exact answer",
                tier.tag
            ));
        }
        tiers.push(TierRow {
            tag: tier.tag.into(),
            size_bytes: evaluated.model.size_bytes,
            verdict,
            summary,
            best_gpu,
        });
    }
    // Each visible GPU's projected capacity for the fitted model, beside its
    // memory facts in `machine`.
    if let Some(plan) = &analysis.plan {
        for gpu in &plan.gpus {
            if let Some(device) = machine.devices.iter_mut().find(|d| d.index == gpu.index) {
                device.capacity = gpu.capacity().map(|tokens| tokens as u64);
                device.capacity_shown = gpu
                    .projection
                    .map(|p| capacity::capped(p.card_cap, Some(p.model_ctx)).0 as u64);
            }
        }
    }
    let published = matching_receipts(&analysis.model.tag, &machine, analysis.plan.as_ref());
    findings.extend(analysis.findings);
    let verdict = fold_findings(&findings);
    let mut notes = analysis.notes;
    notes.extend(host_notes);
    notes.push("Free memory is sampled without opening the GPU; HIP reports slightly less once a process opens it. Fit reserves no cards: device availability, locks and memory counters can change before the real load.".into());
    notes.push("Admission assumes HIP VMM is available on native Linux ROCm; the loader still applies its platform refusal rules.".into());
    let report = Report {
        schema: "hipfire-fit-1",
        hipfire: json!({"version": env!("HIPFIRE_BUILD_VERSION"), "commit": crate::BUILD_COMMIT}),
        input: safe_name(&args.model),
        overrides: OverridesEcho::from(&args),
        model: analysis.model,
        machine,
        plan: analysis.plan,
        findings,
        verdict,
        tiers,
        receipts: published,
        notes,
        details: analysis.details,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&json_report(&report)?)?);
    } else {
        print!("{}", render_text(&report, args.verbose));
    }
    std::io::stdout().flush()?;
    std::process::exit(exit_code(verdict));
}

#[cfg(target_os = "linux")]
fn preflight_findings(paths: &crate::Paths) -> Vec<Finding> {
    let preflight = crate::doctor::runtime::preflight(paths);
    if preflight.blocked() {
        // CheckStatus lives in doctor's private report module. Use its stable
        // serialized status rather than duplicating that module's status type.
        let failed = preflight
            .checks
            .iter()
            .filter(|check| {
                serde_json::to_value(check.status).ok().as_ref() == Some(&json!("fail"))
            })
            .map(|check| check.id)
            .collect::<Vec<_>>()
            .join(", ");
        vec![Finding::fails(
            "preflight",
            format!("runtime preflight blocked: {failed}"),
            "run `hipfire doctor` and apply its fix lines",
        )]
    } else {
        vec![Finding::fits("preflight", "runtime preflight ok")]
    }
}

#[cfg(not(target_os = "linux"))]
fn preflight_findings(_: &crate::Paths) -> Vec<Finding> {
    Vec::new()
}

fn analyze_model(
    paths: &crate::Paths,
    registry: &RegistryV1,
    input: &str,
    args: &FitArgs,
    machine: &FitHost,
) -> Result<Analysis> {
    let card = crate::model_card(paths, registry, input);
    let entry = card.entry();
    let tag = card.tag().unwrap_or(input);
    let local = crate::find_model_path(paths, registry, input);
    if local.is_none() && entry.is_none() {
        return Err(crate::unknown_model_error(registry, input));
    }
    let mut result = Analysis {
        model: ModelInfo {
            tag: safe_name(tag),
            file: local
                .as_deref()
                .and_then(Path::file_name)
                .map(|s| s.to_string_lossy().into_owned())
                .or_else(|| entry.map(|e| safe_name(&e.file))),
            installed: local.is_some(),
            replica: false,
            size_bytes: entry.and_then(|e| e.size_bytes),
            arch_id: entry.and_then(|e| e.arch_id),
            quant: entry.and_then(|e| e.quant.clone()),
        },
        plan: None,
        findings: Vec::new(),
        notes: Vec::new(),
        details: Vec::new(),
    };
    let path = match local {
        Some(path) => path,
        None => {
            let entry = entry.expect("registry entry checked above");
            if args.no_fetch {
                result.findings.push(missing_header(
                    tag,
                    "not installed; fit needs the model header",
                ));
                return Ok(result);
            }
            match remote::header_replica(paths, tag, entry) {
                Ok(replica) => {
                    result.model.replica = true;
                    result.notes.push(format!(
                        "{} header replica: {} bytes fetched; weights have not been downloaded",
                        if replica.cached { "cached" } else { "sparse" },
                        replica.fetched_bytes
                    ));
                    replica.path
                }
                Err(error) => {
                    result.findings.push(missing_header(
                        tag,
                        &format!("not installed; fit needs the model header: {error}"),
                    ));
                    return Ok(result);
                }
            }
        }
    };
    result.model.size_bytes = std::fs::metadata(&path)
        .ok()
        .map(|m| m.len())
        .or(result.model.size_bytes);
    let FitLoad {
        params,
        selector,
        draft_missing,
    } = match model_params(paths, input, Some(tag), entry, &path, args) {
        Ok(load) => load,
        Err(error) => {
            let detail = error.to_string();
            result
                .findings
                .push(Finding::fails("admission", &detail, admission_fix(&detail)));
            return Ok(result);
        }
    };
    if let Some(missing) = &draft_missing {
        report_missing_draft(&mut result, &params, tag, missing);
    }
    let Some(device) = machine.devices.first() else {
        return Ok(result);
    };
    let inputs = admission_inputs(&params);
    let admitted = match inputs.admit(&path, device) {
        Ok(admitted) => admitted,
        Err(detail) => match inputs.capacity_refusal(&path, device) {
            // An automatic context refused for lack of room is a capacity
            // answer (per GPU, per tier), not a setting to correct.
            Some(admitted) => admitted,
            None => {
                result
                    .findings
                    .push(Finding::fails("admission", &detail, admission_fix(&detail)));
                return Ok(result);
            }
        },
    };
    result.model.arch_id = Some(admitted.arch_id);
    if entry.is_none() {
        // Same filename-label convention as pull_select::file_label. The HFQ
        // container extension is not a claim about each tensor's mixed dtype.
        result.model.quant = path.extension().map(|v| v.to_string_lossy().into_owned());
    }
    let model_ctx = admitted
        .sequence
        .as_ref()
        .map(|s| s.model_ctx)
        .or_else(|| source_context(&admitted.source));
    let decode = decode_plan(
        &params,
        &DecodeFacts {
            selector: &selector,
            from_flag: args.spec.is_some(),
            // The carrier's own declaration. Unknown across ranks: the load
            // attaches MTP there only on an explicit request.
            mtp_measured_ar_floor: matches!(admitted.topology, EffectiveTopology::Single)
                .then(|| admitted.carrier.map(|c| c.mtp_measured_ar_floor()))
                .flatten(),
            draft_missing: draft_missing.as_ref(),
        },
    );
    let mut plan = Plan {
        basis: PLAN_BASIS,
        device_index: device.index,
        topology: plan_topology_key(&admitted),
        kv_backend: admitted.kv_backend.as_str().into(),
        kv_backend_reason: admitted.kv_backend_reason.clone(),
        kv_mode: match admitted.sequence.as_ref() {
            Some(sequence) => sequence.kv_mode.to_owned(),
            // Qwen4 resolves its KV mode the way admission does
            // (admission.rs `resolve_qwen4`), so the plan names the
            // effective mode instead of the requested `auto`.
            None if admitted.arch_id == 16 => hipfire_runtime::kv_mode::resolve_qwen4(
                params["kv_mode"].as_str().unwrap_or(""),
                &device.arch,
            )
            .map(|kv| hipfire_runtime::kv_mode::qwen_k_display_name(kv.mode).to_owned())
            .unwrap_or_else(|_| params["kv_mode"].as_str().unwrap_or("auto").into()),
            None => params["kv_mode"].as_str().unwrap_or("auto").into(),
        },
        max_seq: admitted.max_seq,
        bound: admitted
            .sequence
            .as_ref()
            .map(|s| s.bound)
            .unwrap_or(if inputs.max_seq == 0 {
                "automatic"
            } else {
                "user"
            })
            .into(),
        model_ctx,
        projection: None,
        best_device: None,
        gpus: Vec::new(),
        automatic: inputs.max_seq == 0,
        devices_pinned: devices_pinned(),
        placement: PlacementPlan {
            summary: "fully resident".into(),
            expert_layers_resident: None,
            host_mapped_bytes: Some(0),
            decided_at_load: false,
        },
        decode_route: decode.route,
        decode_reason: decode.reason,
        routes: link_routes(device),
    };
    match admitted.topology {
        EffectiveTopology::Pipeline(_) | EffectiveTopology::Expert(_) => {
            plan.placement.summary = "per-rank placement decided at load".into();
            plan.placement.decided_at_load = true;
            result.findings.push(Finding::undetermined(
                "context",
                "multi-GPU capacity is sized per rank at load",
                None,
            ));
            for rank in &machine.devices {
                result.notes.push(format!(
                    "GPU{} {}: {} free",
                    rank.index,
                    rank.arch,
                    rank.free.map(gib).unwrap_or_else(|| "unknown".into())
                ));
            }
        }
        EffectiveTopology::Single if matches!(admitted.arch_id, 5 | 6) => {
            match admitted.sequence.as_ref().and_then(|s| s.projection) {
                // The projection admission itself refused on: the load's own
                // context check, so its verdict is the fit verdict.
                Some(projection) => {
                    let automatic = inputs.max_seq == 0;
                    let gpus = visible_gpu_capacities(&admitted.source, &inputs, machine);
                    let best = capacity::best_gpu(&gpus);
                    let better = best.filter(|b| {
                        b.index != device.index
                            && b.capacity().is_some_and(|c| c > projection.card_cap)
                    });
                    result.findings.push(capacity::context_finding(
                        projection,
                        admitted.max_seq,
                        automatic,
                        device,
                        better,
                    ));
                    plan.model_ctx = Some(projection.model_ctx);
                    plan.projection = Some(projection_json(projection));
                    if machine.devices.len() > 1 {
                        plan.best_device = best.and_then(capacity::best_device);
                    }
                    plan.gpus = gpus;
                }
                // No stored projection (MoE on a discrete GPU, or legacy KV):
                // the daemon does not check this context before load, so fit
                // cannot decide it. The VMM projection is advisory only and
                // never replaces the plan's resolved KV mode.
                None => {
                    let projection = hipfire_loader::admission::qwen35_card_projection(
                        &admitted.source,
                        &device.arch,
                        inputs.hints(device),
                    );
                    let reason = admitted.sequence_reason.unwrap_or("legacy KV backend");
                    result.findings.push(unchecked_context_finding(
                        projection,
                        admitted.max_seq,
                        reason,
                    ));
                }
            }
        }
        EffectiveTopology::Single if admitted.arch_id == 16 => {
            if let ModelSource::Hfq(hfq) = &admitted.source {
                let analysis = qwen4::analyze(&qwen4::Qwen4Request {
                    hfq,
                    device,
                    host: machine,
                    max_seq: admitted.max_seq,
                    params: &params,
                });
                plan.placement = analysis.placement;
                result.findings.extend(analysis.findings);
                result.notes.extend(analysis.notes);
                result.details.extend(analysis.details);
            } else {
                result.findings.push(Finding::undetermined(
                    "context",
                    "Qwen4 placement requires an HFQ weight manifest",
                    Some("choose a published HFQ tier with `hipfire pull`".into()),
                ));
            }
        }
        EffectiveTopology::Single => {
            match (result.model.size_bytes, device.free) {
                (Some(bytes), Some(free)) if bytes <= free => result.findings.push(Finding::fits(
                    "weights",
                    format!("weights {} of {} free", gib(bytes), gib(free)),
                )),
                (Some(bytes), Some(free)) => result.findings.push(Finding::fails(
                    "weights",
                    format!("weights {} exceed {} free", gib(bytes), gib(free)),
                    "pick a smaller tier or free GPU memory",
                )),
                _ => result.findings.push(Finding::undetermined(
                    "weights",
                    "weight size or free GPU memory is unavailable",
                    Some("run `hipfire doctor`".into()),
                )),
            }
            result.findings.push(Finding::undetermined(
                "context",
                format!(
                    "this architecture sizes its KV at load (no pre-load projection); {} tokens",
                    admitted.max_seq
                ),
                None,
            ));
        }
    }
    result.plan = Some(plan);
    Ok(result)
}

fn missing_header(tag: &str, detail: &str) -> Finding {
    Finding::undetermined(
        "model",
        detail,
        Some(format!("`hipfire pull {tag}` or rerun without --no-fetch")),
    )
}

/// The load parameters fit admits, plus what only fit explains.
struct FitLoad {
    params: Value,
    /// The effective speculation selector: `--spec`, else `speculation.mode`.
    selector: String,
    /// A registry DFlash draft the final mode wants that is not installed.
    draft_missing: Option<MissingDraft>,
}

struct MissingDraft {
    file: String,
    /// The registry's size for the draft; fit never fetches it.
    size_bytes: Option<u64>,
}

fn model_params(
    paths: &crate::Paths,
    input: &str,
    tag: Option<&str>,
    entry: Option<&ModelEntry>,
    path: &Path,
    args: &FitArgs,
) -> Result<FitLoad> {
    let resolved = crate::resolved_for_model(paths, input, tag, entry)?;
    let overrides = args.load_overrides();
    // The one sequence `hipfire run` builds its params with: config, `--kv-mode`,
    // `--kv-k`/`--kv-v`, the effective `--spec` selector, the draft it keeps.
    let mut params = crate::load_params_with_overrides(
        &resolved,
        entry,
        &paths.models,
        path,
        0,
        tag,
        &overrides,
    )?;
    let selector = crate::effective_speculation_selector(&resolved, overrides.speculation)?;
    // run_command's resolve_dflash_sidecar for the final selector (config-off +
    // `--spec dflash` is only visible here). It refuses a missing draft; fit
    // keeps the plan and reports the missing draft itself.
    let mut draft_missing = match crate::locate_dflash_sidecar(&params, entry, &paths.models, path)?
    {
        crate::DflashSidecar::Unresolved => None,
        crate::DflashSidecar::Found(draft) => {
            params["draft"] = json!(draft.display().to_string());
            None
        }
        crate::DflashSidecar::NotPulled(file) => Some(MissingDraft {
            size_bytes: entry
                .and_then(|entry| entry.dflash.as_ref())
                .and_then(|sidecar| sidecar.size_bytes),
            file,
        }),
    };
    // main.rs run_command forwards --max-seq after load_params, without any
    // max_tokens inflation; --ctx has precisely that same precedence.
    if let Some(ctx) = args.ctx {
        params["max_seq"] = json!(ctx);
    }
    // run_command forwards `--tp` after load_params the same way; a degree of
    // 1 is the default and is not sent.
    if let Some(tp) = args.tp.filter(|&tp| tp > 1) {
        params["tp"] = json!(tp);
    }
    // daemon/main.rs:1675-1693 and vision_ladder.rs:49-59: sample the
    // developer override ladders outside the pure params mapper, including
    // explicit empty values (opt out). The mapper applies each mode gate.
    for (variable, key) in [
        ("HIPFIRE_DFLASH_DRAFT", "draft"),
        ("HIPFIRE_VISION_SIDECAR", "vision"),
    ] {
        if let Ok(value) = hipfire_config::developer_var(variable) {
            params[key] = json!(value);
            if key == "draft" {
                // The ladder names the draft (or opts out): none is missing.
                draft_missing = None;
            }
        }
    }
    // daemon/main.rs:1813-1820 and :1845-1862 use these process fallbacks
    // for absent/non-numeric params (including an "auto" mtp_k).
    let runtime = hipfire_runtime::config::get();
    if params["mtp_mode"].as_str().is_none() {
        params["mtp_mode"] = json!(runtime.mtp_mode);
    }
    if params["mtp_k"].as_u64().is_none() {
        if let Some(k) = runtime.mtp_k {
            params["mtp_k"] = json!(k);
        }
    }
    if params["ngram_draft"].as_bool().is_none() {
        params["ngram_draft"] = json!(runtime.ngram_draft);
    }
    if params["ngram_k"].as_u64().is_none() {
        params["ngram_k"] = json!(runtime.ngram_k);
    }
    if params["ngram_min_count"].as_u64().is_none() {
        params["ngram_min_count"] = json!(runtime.ngram_min_count);
    }
    // daemon/main.rs:2132-2146: retain the explicit experimental PP gates.
    params["_pp_dflash"] = json!(
        hipfire_config::developer_var("HIPFIRE_PP_DFLASH")
            .ok()
            .as_deref()
            == Some("1")
    );
    params["_pp_pflash"] = json!(
        hipfire_config::developer_var("HIPFIRE_PP_PFLASH")
            .ok()
            .as_deref()
            == Some("1")
    );
    // daemon/main.rs:1270-1289 resolves the process topology before parsing
    // admission inputs. Keep environmental resolution outside the pure mapper.
    let mut message = json!({"params": params});
    hipfire_config::topology::resolve_load_pp(&mut message).map_err(anyhow::Error::msg)?;
    Ok(FitLoad {
        params: message["params"].take(),
        selector,
        draft_missing,
    })
}

/// A registry DFlash draft that the final mode wants and the machine lacks.
/// `on` is a load refusal (the engine's own text), so an installed model
/// does not fit; for a model fit only read the header of, `hipfire pull`
/// fetches the draft with it, so the answer is open. `auto` runs without the
/// draft. Fit never downloads the draft: the size is the registry's.
fn report_missing_draft(result: &mut Analysis, params: &Value, tag: &str, missing: &MissingDraft) {
    let size = missing
        .size_bytes
        .map(|bytes| {
            format!(
                " ({} per the registry; fit does not download it)",
                gib(bytes)
            )
        })
        .unwrap_or_else(|| " (fit does not download it)".into());
    if params["dflash_mode"].as_str() == Some("on") {
        let detail = format!(
            "{}{size}",
            crate::dflash_draft_not_pulled(&missing.file, Some(tag))
        );
        let fix = format!("`hipfire pull {tag}` (it fetches the draft), or choose another --spec");
        result.findings.push(if result.model.installed {
            Finding::fails("draft", detail, fix)
        } else {
            Finding::undetermined("draft", detail, Some(fix))
        });
    } else {
        result.notes.push(format!(
            "DFlash draft {} is not installed{size}; speculation.dflash = auto runs without it, `hipfire pull {tag}` fetches it",
            missing.file
        ));
    }
}

fn source_context(source: &ModelSource) -> Option<usize> {
    let ModelSource::Hfq(hfq) = source else {
        return None;
    };
    let metadata: Value = serde_json::from_str(&hfq.metadata_json).ok()?;
    metadata
        .pointer("/config/text_config/max_position_embeddings")
        .or_else(|| metadata.pointer("/config/max_position_embeddings"))
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

/// The plan's topology string. Qwen4 expert parallelism admits as `Single`
/// with an Ep axis on the admitted mesh; the plan names it `ep<N>` so it is not
/// read (or matched against receipts) as a single-GPU load.
fn plan_topology_key(admitted: &SourceAdmission) -> String {
    ep_axis_key(admitted.topology, &admitted.mesh.mesh)
}

fn ep_axis_key(topology: EffectiveTopology, mesh: &DeviceMesh) -> String {
    let ep = mesh.size_of(DimKind::Ep);
    match topology {
        EffectiveTopology::Single if ep > 1 => format!("ep{ep}"),
        topology => topology_key(topology),
    }
}

fn topology_key(topology: EffectiveTopology) -> String {
    match topology {
        EffectiveTopology::Single => "single".into(),
        EffectiveTopology::Expert(n) => format!("ep{n}"),
        EffectiveTopology::Pipeline(n) => format!("pp{n}"),
    }
}

/// The facts besides the params that decide which route the load takes.
struct DecodeFacts<'a> {
    /// The effective speculation selector, and whether `--spec` set it.
    selector: &'a str,
    from_flag: bool,
    /// The architecture's declared measured autoregressive floor
    /// (`Carrier::mtp_measured_ar_floor`), which lets `auto` MTP attach;
    /// `None` when the carrier is not resolved before load.
    mtp_measured_ar_floor: Option<bool>,
    draft_missing: Option<&'a MissingDraft>,
}

struct DecodePlan {
    route: String,
    reason: String,
}

/// Every decode route the params select and why. The load installs one
/// speculator, the first armed of DSpark, DFlash, MTP, n-gram
/// (`hipfire_loader::spec_build::build_speculator`, and the carrier's DSpark
/// arm before it); with none armed it decodes autoregressively.
fn decode_plan(params: &Value, facts: &DecodeFacts<'_>) -> DecodePlan {
    let dflash_mode = params["dflash_mode"].as_str().unwrap_or("auto");
    let mtp = hipfire_config::MtpMode::parse(params["mtp_mode"].as_str().unwrap_or("auto"));
    // What armed a route: the flag, the config selector, or its own setting.
    let chosen = |name: &str, setting: String| -> String {
        match (facts.selector == name, facts.from_flag) {
            (true, true) => format!("`--spec {name}`"),
            (true, false) => format!("speculation.mode = {name}"),
            (false, _) => setting,
        }
    };
    let mut armed: Vec<(String, String)> = Vec::new();
    if params["dspark_mode"].as_str() == Some("on") {
        armed.push((
            "dspark".into(),
            format!(
                "DSpark drafter ({}); its sidecar and architecture support are checked at load",
                chosen("dspark", "dspark_mode = on".into())
            ),
        ));
    }
    let setting = format!("speculation.dflash = {dflash_mode}");
    match (
        (dflash_mode != "off")
            .then(|| str_param(params, "draft"))
            .flatten(),
        facts.draft_missing.filter(|_| dflash_mode == "on"),
    ) {
        (Some(draft), _) => {
            let file = safe_name(draft);
            armed.push((
                format!("dflash (draft {file})"),
                format!(
                    "DFlash draft {file} attached ({})",
                    chosen("dflash", setting)
                ),
            ));
        }
        (None, Some(missing)) => armed.push((
            format!("dflash (draft {} not installed)", missing.file),
            format!(
                "DFlash is on ({}) but its draft is not installed: the load refuses",
                chosen("dflash", setting)
            ),
        )),
        (None, None) => {}
    }
    match mtp {
        hipfire_config::MtpMode::On => armed.push((
            "mtp".into(),
            format!(
                "MTP head attached ({}), also over host-mapped experts; a missing head refuses at load",
                chosen("mtp", "speculation.mtp = on".into())
            ),
        )),
        hipfire_config::MtpMode::Auto if facts.mtp_measured_ar_floor == Some(true) => {
            armed.push((
                "mtp (auto)".into(),
                format!(
                    "speculation.mtp = auto: this architecture declares a measured AR floor, so the head attaches at load; it stays AR when the head is absent or experts are host-mapped ({})",
                    hipfire_config::mtp_host_spill_reason()
                ),
            ))
        }
        _ => {}
    }
    if params["ngram_draft"].as_bool() == Some(true) {
        armed.push((
            "ngram".into(),
            format!(
                "model-free n-gram drafter ({}); only architectures with an n-gram speculator arm it, the rest stay AR",
                chosen("ngram", "speculation.ngram = on".into())
            ),
        ));
    }
    if mtp == hipfire_config::MtpMode::Auto && facts.mtp_measured_ar_floor.is_none() {
        armed.push((
            "mtp / ar (auto; decided at load)".into(),
            "speculation.mtp = auto attaches the head only where the architecture declares a measured AR floor and experts are not host-mapped; both are decided at load for this topology".into(),
        ));
    }
    let mut armed = armed.into_iter();
    let Some((route, mut reason)) = armed.next() else {
        let reason = if facts.selector == "off" {
            format!(
                "{} disables every speculative route",
                if facts.from_flag {
                    "`--spec off`"
                } else {
                    "speculation.mode = off (config)"
                }
            )
        } else {
            let dflash = match (dflash_mode, facts.draft_missing) {
                ("off", _) => "dflash off".to_owned(),
                (_, Some(missing)) => format!(
                    "dflash {dflash_mode} runs without its draft {} (not installed)",
                    missing.file
                ),
                _ => format!("dflash {dflash_mode} has no draft for this artifact"),
            };
            let mtp = match mtp {
                hipfire_config::MtpMode::Auto => {
                    "mtp auto keeps AR (this architecture declares no measured AR floor; `--spec mtp` opts in)"
                }
                _ => "mtp off",
            };
            format!("no speculative route is enabled: {dflash}, {mtp}, ngram off")
        };
        return DecodePlan {
            route: "ar".into(),
            reason,
        };
    };
    let rest: Vec<String> = armed.map(|(route, _)| route).collect();
    if !rest.is_empty() {
        reason.push_str(&format!(
            "; also enabled, not installed ahead of it: {}",
            rest.join(", ")
        ));
    }
    DecodePlan { route, reason }
}

fn link_routes(device: &FitDevice) -> Vec<Route> {
    let Some(link) = &device.link else {
        return Vec::new();
    };
    let Some(gbps) = link
        .gbps
        .filter(|_| link.calibrated && !device.unified_memory)
    else {
        return Vec::new();
    };
    hipfire_arch_qwen4::host_link::link_route_choices(
        &device.arch,
        hipfire_arch_qwen4::host_link::HostLink::Discrete { gbps },
    )
    .into_iter()
    .map(|r| Route {
        name: r.name.into(),
        value: r.value,
        why: r.why,
    })
    .collect()
}

fn projection_json(p: Qwen35CardProjection) -> Value {
    json!({"free": p.free, "weights": p.weights, "scratch": p.scratch,
        "page_rounding": p.page_rounding, "bytes_per_token": p.bytes_per_token,
        "card_cap": p.card_cap, "model_ctx": p.model_ctx, "kv_mode": p.kv_mode})
}

/// The context finding for a Qwen3.5-family load whose context the daemon
/// does not check before load (no stored admission projection: MoE on a
/// discrete GPU, or the legacy KV backend). Always `Undetermined`: the VMM
/// projection is reported as advisory, never as the load's verdict.
fn unchecked_context_finding(
    projection: Result<Qwen35CardProjection, String>,
    max_seq: usize,
    reason: &str,
) -> Finding {
    match projection {
        Ok(p) => Finding::undetermined(
            "context",
            format!(
                "the daemon does not check this context before load ({reason}); advisory only: \
                 the VMM admission projection would leave room for {} tokens against {max_seq} \
                 (free {} - weights {} - prefill scratch {} - page rounding {})",
                p.card_cap,
                gib(p.free as u64),
                gib(p.weights as u64),
                gib(p.scratch as u64),
                gib(p.page_rounding as u64),
            ),
            (max_seq > p.card_cap).then(|| capacity::context_fix(&p, None)),
        ),
        Err(error) => Finding::undetermined(
            "context",
            format!("the daemon does not check this context before load ({reason}); no projection: {error}"),
            None,
        ),
    }
}

fn admission_fix(message: &str) -> String {
    if message.contains("max_seq") || message.contains("context") {
        "lower --ctx (or `hipfire config set memory.max_seq 512`), free GPU memory, or pick a smaller tier; apply any bound named above".into()
    } else if ["DFlash", "MTP", "DSpark", "n-gram", "speculat"]
        .iter()
        .any(|word| message.contains(word))
    {
        "choose another speculation mode (`--spec off|auto|ngram|mtp|dflash|dspark`, or `hipfire config set speculation.mode off`), or provide the sidecar the message names".into()
    } else if message.contains("reload without") || message.contains("set ") {
        format!("apply the setting or reload change named above: {message}")
    } else if message.contains("kv") || message.contains("KV") {
        "change `--kv-mode`/`--kv-k`/`--kv-v`, or select a KV mode/backend supported by this model and GPU using `hipfire config set memory.kv_cache q8` and `hipfire config set memory.kv_backend vmm`; apply the admission message if it names a different required mode".into()
    } else if message.contains("topology")
        || message.contains("pp=")
        || message.contains("tp=")
        || message.contains("--tp")
    {
        "use a supported topology (`hipfire config set topology.pp 1`, `hipfire config set topology.ep 1`, and no `--tp`) or select the model/topology named above".into()
    } else {
        "see the message; change the named load setting or choose a supported smaller tier with `hipfire ls`, then rerun `hipfire fit`".into()
    }
}

fn receipt_matches(
    r: &receipts::Receipt,
    tag: &str,
    arch: &str,
    topology: &str,
    placement: &str,
) -> bool {
    r.tag == tag
        && r.arch == arch
        && r.topology == topology
        // Never across placement: a receipt whose source does not state its
        // placement ("unstated") matches no plan. Historical rows (an older
        // engine, or an artifact the registry no longer serves) are kept in
        // the table but never applied to this machine's expectation.
        && r.placement == placement
        && !r.historical
}

fn matching_receipts(tag: &str, machine: &FitHost, plan: Option<&Plan>) -> Vec<Value> {
    let (Some(device), Some(plan)) = (machine.devices.first(), plan) else {
        return Vec::new();
    };
    let placement = if plan
        .placement
        .host_mapped_bytes
        .is_some_and(|bytes| bytes > 0)
    {
        "host-experts"
    } else {
        "resident"
    };
    receipts::for_model(tag, &device.arch)
        .into_iter()
        .filter(|r| receipt_matches(r, tag, &device.arch, &plan.topology, placement))
        .map(receipt_json)
        .collect()
}

fn receipt_json(r: &receipts::Receipt) -> Value {
    let class = receipts::class_of(r.metric, r.tok_s);
    let band = receipts::class_band(r.metric, class);
    json!({"tag": r.tag, "arch": r.arch, "topology": r.topology,
        "placement": r.placement, "metric": match r.metric { receipts::Metric::Decode => "decode", receipts::Metric::Prefill => "prefill" },
        "route": r.route, "tokens": r.tokens, "context": r.context, "tok_s": r.tok_s,
        "date": r.date, "source": r.source, "quote": r.quote, "historical": r.historical, "note": r.note,
        "class": class.to_string(), "band": band})
}

fn safe_name(input: &str) -> String {
    if input.contains('/') || input.contains('\\') {
        input
            .rsplit(['/', '\\'])
            .find(|s| !s.is_empty())
            .unwrap_or("model")
            .into()
    } else {
        input.into()
    }
}

/// Error strings from the loader/network may carry a local path. The public
/// JSON contract intentionally contains only basenames, never home paths or
/// PCI addresses; scrub nested findings/notes as well as the explicit file.
fn json_report(report: &Report) -> Result<Value> {
    let mut value = serde_json::to_value(report)?;
    fn scrub_with_home(value: &mut Value, home: Option<&str>) {
        match value {
            Value::String(text) => {
                let mut out = String::new();
                for part in text.split_inclusive(char::is_whitespace) {
                    let word = part.trim_end_matches(char::is_whitespace);
                    let suffix = &part[word.len()..];
                    if word.contains("/home/")
                        || word.contains("/root/")
                        || home.is_some_and(|home| home != "/" && word.contains(home))
                    {
                        out.push_str("<local-file>");
                    } else if word.as_bytes().windows(12).any(|b| {
                        b[4] == b':'
                            && b[7] == b':'
                            && b[10] == b'.'
                            && b.iter()
                                .enumerate()
                                .all(|(i, c)| matches!(i, 4 | 7 | 10) || c.is_ascii_hexdigit())
                    }) {
                        out.push_str("<device>");
                    } else {
                        out.push_str(word);
                    }
                    out.push_str(suffix);
                }
                *text = out;
            }
            Value::Array(items) => items
                .iter_mut()
                .for_each(|item| scrub_with_home(item, home)),
            Value::Object(fields) => fields
                .values_mut()
                .for_each(|item| scrub_with_home(item, home)),
            _ => {}
        }
    }
    let home = std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty() && home != "/");
    scrub_with_home(&mut value, home.as_deref());
    Ok(value)
}

fn render_text(report: &Report, verbose: bool) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "hipfire fit {}", report.input);
    let _ = writeln!(out, "this machine");
    let plan_window = report.plan.as_ref().and_then(|p| p.model_ctx);
    for device in &report.machine.devices {
        let free = device.free.map(gib).unwrap_or_else(|| "unknown".into());
        let total = device
            .vram_total
            .map(gib)
            .unwrap_or_else(|| "unknown".into());
        let link = device.link.as_ref().map(|l| l.detail.as_str()).unwrap_or(
            "not calibrated: run `hipfire calibrate` (the engine probes the link at load)",
        );
        // Per-GPU capacity of the fitted model, only where it tells GPUs apart.
        let cap_text = device
            .capacity
            .filter(|_| report.machine.devices.len() > 1)
            .map(|tokens| {
                format!(
                    "  capacity: {}",
                    capacity::device_capacity_text(tokens, plan_window)
                )
            })
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  GPU{} {}  {free} free of {total}{cap_text}  link: {link}",
            device.index, device.arch
        );
    }
    let bytes = |v: Option<u64>| v.map(gib).unwrap_or_else(|| "unknown".into());
    let _ = writeln!(
        out,
        "  host  RAM {}, {} available; GTT cap {}, {} used",
        bytes(report.machine.ram_total),
        bytes(report.machine.mem_available),
        bytes(report.machine.gtt_limit),
        bytes(report.machine.gtt_used)
    );
    if let Some(preflight) = report.findings.iter().find(|f| f.id == "preflight") {
        let _ = writeln!(
            out,
            "  runtime preflight: {}",
            if preflight.verdict == Verdict::Fits {
                "ok"
            } else {
                "blocked (run `hipfire doctor`)"
            }
        );
    }
    let _ = writeln!(
        out,
        "model  {}  {}, {}, arch {}, {}",
        report.model.tag,
        if report.model.installed {
            "installed"
        } else if report.model.replica {
            "header replica (not installed)"
        } else {
            "not installed"
        },
        bytes(report.model.size_bytes),
        report
            .model
            .arch_id
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".into()),
        report.model.quant.as_deref().unwrap_or("quant unknown")
    );
    if let Some(plan) = &report.plan {
        let _ = writeln!(
            out,
            "plan (what `hipfire run{}` would load on GPU{})",
            report.overrides.flags(),
            plan.device_index
        );
        let _ = writeln!(out, "  basis      {PLAN_BASIS_TEXT}");
        let _ = writeln!(
            out,
            "  topology   {}",
            if plan.topology == "single" {
                "single GPU"
            } else {
                &plan.topology
            }
        );
        let _ = writeln!(out, "  kv         {}, {}", plan.kv_backend, plan.kv_mode);
        if let Some(reason) = &plan.kv_backend_reason {
            let _ = writeln!(out, "               {reason}");
        }
        let _ = writeln!(
            out,
            "  context    {} ({}); trained window {}",
            plan.max_seq,
            plan.bound,
            plan.model_ctx
                .map(|v| v.to_string())
                .unwrap_or_else(|| "decided at load".into())
        );
        if let Some(p) = &plan.projection {
            let device = report
                .machine
                .devices
                .iter()
                .find(|d| d.index == plan.device_index);
            let mut lines = capacity::capacity_lines(p, device).into_iter();
            if let Some(first) = lines.next() {
                let _ = writeln!(out, "  capacity   {first}");
            }
            for line in lines {
                let _ = writeln!(out, "               {line}");
            }
        }
        // The best visible GPU, when it is not the one this plan loads on.
        if let Some(best) = plan
            .best_device
            .as_ref()
            .filter(|best| best.index != plan.device_index)
            .and_then(|best| plan.gpus.iter().find(|g| g.index == best.index))
        {
            if let Some(line) = capacity::best_gpu_line(best, plan.devices_pinned) {
                let _ = writeln!(out, "  best GPU   {line}");
            }
        }
        let _ = writeln!(out, "  placement  {}", plan.placement.summary);
        let _ = writeln!(out, "  decode     {}", plan.decode_route);
        let _ = writeln!(out, "               {}", plan.decode_reason);
        for route in &plan.routes {
            let _ = writeln!(
                out,
                "  routes     {}={} ({})",
                route.name, route.value, route.why
            );
        }
    }
    let _ = writeln!(out, "findings");
    for finding in &report.findings {
        let _ = writeln!(
            out,
            "  {} {}: {}",
            finding.verdict.label(),
            finding.id,
            finding.detail
        );
        if let Some(fix) = &finding.fix {
            let _ = writeln!(out, "    fix: {fix}");
        }
    }
    let context = report
        .plan
        .as_ref()
        .map(|p| {
            if p.max_seq == 0 {
                "automatic context".into()
            } else {
                format!("{} tokens", p.max_seq)
            }
        })
        .unwrap_or_else(|| "requested context".into());
    let _ = writeln!(out, "verdict: {} at {context}", report.verdict.label());
    if !report.tiers.is_empty() {
        let _ = writeln!(
            out,
            "quant tiers (model fit only{}; pull default: {})",
            if report.machine.devices.len() > 1 {
                ", on each tier's best GPU"
            } else {
                ""
            },
            report.model.tag
        );
        for tier in &report.tiers {
            let _ = writeln!(
                out,
                "  {}  {}  {}  {}",
                tier.tag,
                bytes(tier.size_bytes),
                tier.verdict.label(),
                tier.summary
            );
        }
        let largest = report
            .tiers
            .iter()
            .filter(|t| t.verdict == Verdict::Fits)
            .fold(None::<&TierRow>, |best, tier| match best {
                Some(best) if best.size_bytes.unwrap_or(0) >= tier.size_bytes.unwrap_or(0) => {
                    Some(best)
                }
                _ => Some(tier),
            });
        let _ = writeln!(
            out,
            "largest tier that fits at {context}: {}",
            largest
                .map(|t| match t.best_gpu {
                    Some(gpu) if report.machine.devices.len() > 1 => {
                        format!("{} (on GPU{gpu})", t.tag)
                    }
                    _ => t.tag.clone(),
                })
                .unwrap_or_else(|| "none established".into())
        );
    }
    let _ = writeln!(
        out,
        "expected speed (published receipts; same model and GPU arch only)"
    );
    if report.receipts.is_empty() {
        let arch = report
            .machine
            .devices
            .first()
            .map(|d| d.arch.as_str())
            .unwrap_or("unknown GPU arch");
        let _ = writeln!(out, "  no published receipt for {} on {arch} with this placement; measure with `hipfire bench {}`", report.model.tag, report.model.tag);
    }
    for row in &report.receipts {
        let s = |key: &str| row[key].as_str().unwrap_or("");
        let at = row["context"]
            .as_u64()
            .map(|v| format!(", at {v}"))
            .unwrap_or_default();
        let historical = if row["historical"].as_bool() == Some(true) {
            " [historical]"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  {} {} tok/s ({}, {} tokens{at}) on {}, {} — {} ({}){historical} — {}",
            s("metric"),
            row["tok_s"],
            s("route"),
            row["tokens"],
            s("arch"),
            s("date"),
            s("class"),
            s("band"),
            s("source")
        );
        if row["placement"].as_str() == Some("unstated") {
            let _ = writeln!(out, "    (placement not stated by the source)");
        }
        if let Some(note) = row["note"].as_str() {
            let _ = writeln!(out, "    {note}");
        }
    }
    for note in &report.notes {
        let _ = writeln!(out, "note: {note}");
    }
    if verbose {
        for detail in &report.details {
            let _ = writeln!(out, "detail: {detail}");
        }
    } else if !report.details.is_empty() {
        let _ = writeln!(
            out,
            "note: --verbose (or --json) lists how each bound was derived, with engine sources"
        );
    }
    out
}

/// Borrowed controls plus the owned eviction config the load API expects.
/// Environment/process normalization happens before this pure mapper.
struct AdmissionInputs<'a> {
    tp: usize,
    pp: usize,
    max_seq: usize,
    request: KvBackendRequest,
    draft: Option<&'a str>,
    vision: Option<&'a str>,
    vision_mode: &'a str,
    head: Option<&'a str>,
    kv_mode: Option<&'a str>,
    kv_k: Option<&'a str>,
    kv_v: Option<&'a str>,
    kv_adaptive: Option<&'a str>,
    cask: CaskConfig,
    options: SourceAdmissionOptions,
    error: Option<String>,
}

impl AdmissionInputs<'_> {
    fn hints(&self, device: &FitDevice) -> KvBackendHints<'_> {
        KvBackendHints {
            kv_mode: self.kv_mode,
            kv_k: self.kv_k,
            kv_v: self.kv_v,
            kv_adaptive: self.kv_adaptive,
            cask: Some(&self.cask),
            deepseek4_heterogeneous: self.options.non_single_compute,
            qwen_default_q8: qwen_default_q8_enabled(),
            vmm_runtime_available: true,
            free_vram_bytes: device.free.and_then(|n| usize::try_from(n).ok()),
        }
    }

    fn admit(
        &self,
        path: &Path,
        device: &FitDevice,
    ) -> std::result::Result<SourceAdmission, String> {
        self.admit_at(path, device, self.max_seq)
    }

    fn admit_at(
        &self,
        path: &Path,
        device: &FitDevice,
        max_seq: usize,
    ) -> std::result::Result<SourceAdmission, String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let path = path.to_str().ok_or("model path is not valid UTF-8")?;
        admit_source_with_options(
            path,
            self.tp,
            self.pp,
            self.request,
            self.draft,
            &device.arch,
            self.vision,
            self.vision_mode,
            self.head,
            max_seq,
            self.hints(device),
            self.options,
        )
    }

    /// The admission of an automatic context the loader refused because the
    /// projection leaves less than its minimum (`resolve_sequence`), taken at
    /// the minimum explicit context so the projection it refused on is
    /// available; the sequence is then what the automatic load would resolve
    /// (trained window, pending the measured capacity). `None` when this is
    /// not that refusal.
    fn capacity_refusal(&self, path: &Path, device: &FitDevice) -> Option<SourceAdmission> {
        if self.max_seq != 0 {
            return None;
        }
        let mut admitted = self.admit_at(path, device, QWEN35_MIN_AUTO_CONTEXT).ok()?;
        let sequence = admitted.sequence.as_mut()?;
        let projection = sequence.projection?;
        if projection.card_cap >= QWEN35_MIN_AUTO_CONTEXT {
            return None;
        }
        sequence.max_seq = projection.model_ctx;
        sequence.bound = "pending";
        admitted.max_seq = projection.model_ctx;
        Some(admitted)
    }
}

fn str_param<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn usize_param(params: &Value, key: &str) -> Option<usize> {
    params.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

/// The sole pure params -> admission mapping. Cited lines are in
/// crates/hipfire-daemon/src/main.rs; external ladders are sampled by
/// model_params so tests never mutate the environment or a process snapshot.
fn admission_inputs(params: &Value) -> AdmissionInputs<'_> {
    let mut error = None;
    // :1327-1339, :1640-1652: omitted is automatic; explicit zero refuses,
    // large explicit requests clamp to MAX_REQUESTED_SEQ (1 Mi tokens, :469).
    let requested = usize_param(params, "max_seq");
    if requested == Some(0) {
        error = Some("load refused: max_seq 0 below floor 1".into());
    }
    let max_seq = requested.unwrap_or(0).min(1_048_576);
    // :1270-1289 resolves pp outside this pure fn; :2094-2102 reads it and tp.
    let tp = usize_param(params, "tp").unwrap_or(1);
    let pp = usize_param(params, "pp").unwrap_or(1);
    if tp > 1 && pp > 1 {
        error.get_or_insert_with(|| hipfire_loader::bench_topology::PP_EP_EXCLUSIVE_MESSAGE.into());
    }
    // :1665-1693: the developer draft ladder is already sampled in params,
    // but dflash_mode=off still suppresses even an explicit draft.
    let dflash_mode = params["dflash_mode"].as_str().unwrap_or("auto");
    // An `auto` pairing is dropped on tp>1 (DFlash is not served on EP/TP);
    // an explicit `on` keeps the draft and meets the tp>1 refusal below.
    let draft = (dflash_mode != "off" && !(dflash_mode == "auto" && tp > 1))
        .then(|| str_param(params, "draft"))
        .flatten();
    // vision_ladder.rs:43-71, same ladder and hard mode gate.
    let vision_mode = params["vision_mode"]
        .as_str()
        .unwrap_or(hipfire_config::vision_mode_default());
    let vision = (vision_mode != "off")
        .then(|| str_param(params, "vision"))
        .flatten();
    // :1748-1777 and :1799-1804: empty overrides are omitted.
    let head = str_param(params, "head");
    let kv_mode = str_param(params, "kv_mode");
    let kv_k = str_param(params, "kv_k");
    let kv_v = str_param(params, "kv_v");
    let kv_adaptive = str_param(params, "kv_adaptive");
    // :2219-2236: use the engine parser, including automatic omission.
    let request =
        KvBackendRequest::from_override(str_param(params, "kv_backend")).unwrap_or_else(|e| {
            error.get_or_insert(e);
            KvBackendRequest::Automatic
        });
    // :1809-1895: process defaults for mtp/ngram are normalized by caller;
    // all fields otherwise retain exactly the daemon's typed per-load values.
    let (dflash_block, dflash_block_auto) =
        hipfire_runtime::dflash::DflashBlockPolicy::from_param(params.get("dflash_block"))
            .map(|p| {
                (
                    p.resolve(
                        params.get("serve_vmm_batch").is_some(),
                        usize_param(params, "continuous_batch_size").unwrap_or(1),
                    ),
                    p == hipfire_runtime::dflash::DflashBlockPolicy::Auto,
                )
            })
            .unwrap_or_else(|e| {
                error.get_or_insert(e);
                (0, false)
            });
    let spec = SpecLoadCfg {
        ngram_draft: params["ngram_draft"].as_bool(),
        ngram_k: usize_param(params, "ngram_k"),
        ngram_min_count: params["ngram_min_count"].as_u64().map(|v| v as u32),
        ddtree_budget: usize_param(params, "ddtree_budget"),
        ddtree_topk: usize_param(params, "ddtree_topk"),
        dspark: params["dspark_mode"]
            .as_str()
            .and_then(hipfire_config::parse_tri_state),
        dspark_conf_threshold: params["dspark_conf_threshold"].as_f64().map(|v| v as f32),
        dflash: hipfire_config::parse_tri_state(dflash_mode),
        mtp: hipfire_config::parse_tri_state(params["mtp_mode"].as_str().unwrap_or("auto")),
        mtp_k: usize_param(params, "mtp_k"),
        dflash_adaptive_b: Some(params["dflash_adaptive_b"].as_bool().unwrap_or(false)),
        dflash_block: Some(dflash_block),
        dflash_block_auto,
    };
    // :1906-1972: keep a CaskConfig even when eviction is off, exactly as
    // admission's hints do; cask option below is SIDECAR presence, not enabled.
    let enabled = params["cask"].as_bool().unwrap_or(false);
    let cask = CaskConfig {
        sidecar: str_param(params, "cask_sidecar").map(str::to_owned),
        cask_m_folding: enabled && draft.is_none(),
        budget: usize_param(params, "cask_budget").unwrap_or(512),
        beta: usize_param(params, "cask_beta").unwrap_or(128),
        handoff_tokens: usize_param(params, "cask_handoff_tokens").unwrap_or(0),
        core_frac: params["cask_core_frac"].as_f64().unwrap_or(0.5) as f32,
        fold_m: usize_param(params, "cask_fold_m").unwrap_or(2),
    };
    // :2152-2183: parse even an empty placement string (daemon refuses it).
    let placement = params["deepseek4_compute_placement"]
        .as_str()
        .unwrap_or("single")
        .parse::<hipfire_config::Deepseek4ComputePlacement>()
        .unwrap_or_else(|e| {
            error.get_or_insert_with(|| format!("invalid DeepSeek V4 compute placement: {e}"));
            hipfire_config::Deepseek4ComputePlacement::Single
        });
    // :2204-2218: every field of SourceAdmissionOptions is the real load option.
    let options = SourceAdmissionOptions {
        spec,
        kv_adaptive: qwen4_kv_adaptive_requested(kv_adaptive),
        eagle_drafter: str_param(params, "drafter").is_some(),
        cask: cask.sidecar.is_some(),
        state_quant: str_param(params, "state_quant").is_some(),
        non_single_compute: !matches!(placement, hipfire_config::Deepseek4ComputePlacement::Single),
        expert_count_override: usize_param(params, "deepseek4_experts_per_token").is_some(),
        pflash: str_param(params, "prefill_drafter").is_some()
            || params["prefill_compression"].as_str().unwrap_or("off") != "off",
    };
    // :2115-2197: source-independent pre-admission feature refusals. They must
    // not become a successful fit merely because a non-Qwen4 carrier is used.
    let refusal = if tp > 1 && draft.is_some() {
        Some("EP serving (tp>1) does not support DFlash drafters in v1; reload without a draft.")
    } else if (tp > 1 || pp > 1) && options.eagle_drafter {
        Some("EAGLE speculative decode requires a single GPU; reload without params.drafter.")
    } else if pp > 1 && draft.is_some() && params["_pp_dflash"].as_bool() != Some(true) {
        Some(
            "DFlash speculative decode requires pp=1; reload without a draft or set topology.pp=1.",
        )
    } else if pp > 1 && options.cask {
        Some(
            "CASK / TriAttention eviction requires pp=1; reload without CASK or set topology.pp=1.",
        )
    } else if pp > 1 && options.pflash && params["_pp_pflash"].as_bool() != Some(true) {
        Some("PFlash prefill compression requires pp=1; turn off prefill compression or set topology.pp=1.")
    } else if tp > 1 && options.expert_count_override {
        Some("DeepSeek V4 experts-per-token override requires tp=1; reload without the experts-per-token override.")
    } else {
        None
    };
    if let Some(refusal) = refusal {
        error.get_or_insert_with(|| refusal.into());
    }
    AdmissionInputs {
        tp,
        pp,
        max_seq,
        request,
        draft,
        vision,
        vision_mode,
        head,
        kv_mode,
        kv_k,
        kv_v,
        kv_adaptive,
        cask,
        options,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn unchecked_context_is_undetermined_and_advisory() {
        let p = projection();
        let fits = unchecked_context_finding(Ok(p), 4096, "legacy KV backend");
        assert_eq!(fits.verdict, Verdict::Undetermined);
        assert!(fits.detail.contains("advisory only"));
        assert!(fits.detail.contains("legacy KV backend"));
        assert_eq!(fits.fix, None);
        let over = unchecked_context_finding(Ok(p), p.card_cap + 1, "MoE on a discrete GPU");
        assert_eq!(over.verdict, Verdict::Undetermined);
        assert!(over
            .fix
            .as_deref()
            .is_some_and(|fix| fix.contains(&p.card_cap.to_string())));
        let failed = unchecked_context_finding(Err("no HFQ".into()), 4096, "legacy KV backend");
        assert_eq!(failed.verdict, Verdict::Undetermined);
        assert!(failed.detail.contains("no projection: no HFQ"));
    }

    fn projection() -> Qwen35CardProjection {
        Qwen35CardProjection {
            free: 32 << 30,
            weights: 20 << 30,
            scratch: 1 << 30,
            page_rounding: 128 << 20,
            bytes_per_token: 49_152,
            card_cap: 200_000,
            model_ctx: 262_144,
            kv_mode: "q8",
        }
    }

    fn device() -> FitDevice {
        FitDevice {
            index: 0,
            arch: "gfx1201".into(),
            unified_memory: false,
            vram_total: Some(32 << 30),
            free: Some(31 << 30),
            link: None,
            capacity: None,
            capacity_shown: None,
            env_selector: "0".into(),
            config_selector: "GPU-0".into(),
        }
    }

    fn receipt() -> receipts::Receipt {
        receipts::Receipt {
            tag: "example:mq4".into(),
            arch: "gfx1201".into(),
            topology: "single".into(),
            placement: "resident".into(),
            metric: receipts::Metric::Decode,
            route: "ar".into(),
            tokens: 128,
            context: Some(4096),
            tok_s: 203.9,
            date: "2026-10-17".into(),
            source: "registry/v1.json".into(),
            quote: "203.9 tok/s".into(),
            historical: false,
            note: None,
        }
    }

    fn report() -> Report {
        let p = projection();
        Report {
            schema: "hipfire-fit-1",
            hipfire: json!({"version": "0.4.3", "commit": "test"}),
            input: "example:mq4".into(),
            overrides: OverridesEcho {
                kv_mode: None,
                kv_k: None,
                kv_v: None,
                spec: None,
                tp: None,
            },
            model: ModelInfo {
                tag: "example:mq4".into(),
                file: Some("example.mq4".into()),
                installed: true,
                replica: false,
                size_bytes: Some(20 << 30),
                arch_id: Some(6),
                quant: Some("MQ4".into()),
            },
            machine: FitHost {
                ram_total: Some(128 << 30),
                mem_available: Some(64 << 30),
                ttm_pool: 0,
                gtt_limit: Some(64 << 30),
                gtt_used: Some(1 << 30),
                devices: vec![device()],
            },
            plan: Some(Plan {
                basis: PLAN_BASIS,
                device_index: 0,
                topology: "single".into(),
                kv_backend: "vmm".into(),
                kv_backend_reason: None,
                kv_mode: "q8".into(),
                max_seq: 65_536,
                bound: "user".into(),
                model_ctx: Some(p.model_ctx),
                projection: Some(projection_json(p)),
                best_device: None,
                gpus: Vec::new(),
                automatic: false,
                devices_pinned: false,
                placement: PlacementPlan {
                    summary: "fully resident".into(),
                    expert_layers_resident: None,
                    host_mapped_bytes: Some(0),
                    decided_at_load: false,
                },
                decode_route: "ar".into(),
                decode_reason: "no speculative route is enabled".into(),
                routes: vec![Route {
                    name: "prefill_chunk".into(),
                    value: "512".into(),
                    why: "narrow link".into(),
                }],
            }),
            findings: vec![
                Finding::fits("preflight", "runtime preflight ok"),
                capacity::context_finding(p, 65_536, false, &device(), None),
                Finding::fails(
                    "host_ram",
                    "host RAM gate failed",
                    "free host RAM or pick a smaller tier",
                ),
            ],
            verdict: Verdict::DoesNotFit,
            tiers: Vec::new(),
            receipts: vec![receipt_json(&receipt())],
            notes: vec!["source /home/private-user/models/example.mq4 on 0000:03:00.0".into()],
            details: vec!["reserve term (expert_residency.rs:1)".into()],
        }
    }

    #[test]
    fn clap_fit_forms_and_command_wiring() {
        for flags in [
            vec![],
            vec!["--ctx", "65536", "--no-fetch"],
            vec!["-j"],
            vec!["--json"],
        ] {
            let mut argv = vec!["hipfire", "fit", "example:mq4"];
            argv.extend(flags);
            let cli = crate::Cli::try_parse_from(argv).unwrap();
            let Some(crate::Commands::Fit(args)) = cli.command else {
                panic!("fit dispatch variant missing")
            };
            assert_eq!(args.model, "example:mq4");
        }
        let cli = crate::Cli::try_parse_from([
            "hipfire",
            "fit",
            "example",
            "--ctx",
            "65536",
            "--no-fetch",
            "-j",
        ])
        .unwrap();
        let Some(crate::Commands::Fit(args)) = cli.command else {
            panic!("wrong command")
        };
        assert_eq!(args.ctx, Some(65_536));
        assert!(args.no_fetch && args.json);
    }

    #[test]
    fn clap_context_range_and_required_model() {
        for n in ["0", "511", "1048577", "-1", "not-a-number"] {
            assert!(crate::Cli::try_parse_from(["hipfire", "fit", "example", "--ctx", n]).is_err());
        }
        for n in ["512", "1048576"] {
            assert!(crate::Cli::try_parse_from(["hipfire", "fit", "example", "--ctx", n]).is_ok());
        }
        assert!(crate::Cli::try_parse_from(["hipfire", "fit"]).is_err());
    }

    #[test]
    fn admission_inputs_map_kv_context_pp_and_mtp() {
        let params = json!({"max_seq": 65536, "pp": 2, "kv_mode": "q8", "kv_backend": "vmm",
            "kv_k": "fwht3", "kv_v": "q8", "kv_adaptive": "off",
            "mtp_mode": "on", "mtp_k": 3, "ngram_draft": true, "ngram_k": 8,
            "ngram_min_count": 2, "ddtree_budget": 16, "ddtree_topk": 4,
            "dspark_mode": "off", "dflash_mode": "off", "dflash_block": 8,
            "dflash_adaptive_b": true, "head": "head.hfq"});
        let a = admission_inputs(&params);
        assert!(a.error.is_none());
        assert_eq!((a.tp, a.pp, a.max_seq), (1, 2, 65_536));
        assert_eq!(
            a.request,
            KvBackendRequest::Explicit(hipfire_runtime::kv_backend::KvBackend::Vmm)
        );
        assert_eq!(
            (a.kv_mode, a.kv_k, a.kv_v, a.kv_adaptive),
            (Some("q8"), Some("fwht3"), Some("q8"), Some("off"))
        );
        assert!(!a.options.kv_adaptive);
        assert_eq!(a.options.spec.mtp, Some(true));
        assert_eq!(a.options.spec.mtp_k, Some(3));
        assert_eq!(a.options.spec.ngram_draft, Some(true));
        assert_eq!(a.options.spec.ddtree_budget, Some(16));
        assert_eq!(a.options.spec.ddtree_topk, Some(4));
        assert_eq!(a.options.spec.dspark, Some(false));
        assert_eq!(a.options.spec.dflash_block, Some(8));
        assert_eq!(a.options.spec.dflash_adaptive_b, Some(true));
        assert_eq!(a.head, Some("head.hfq"));
    }

    #[test]
    fn admission_inputs_defaults_clamp_and_validate() {
        let empty = json!({});
        let a = admission_inputs(&empty);
        assert_eq!((a.tp, a.pp, a.max_seq), (1, 1, 0));
        assert_eq!(a.request, KvBackendRequest::Automatic);
        assert_eq!(a.vision_mode, hipfire_config::vision_mode_default());
        assert!(a.error.is_none());
        assert_eq!(
            admission_inputs(&json!({"max_seq": 2_000_000})).max_seq,
            1_048_576
        );
        assert!(admission_inputs(&json!({"max_seq": 0})).error.is_some());
        assert!(admission_inputs(&json!({"kv_backend": "invalid"}))
            .error
            .is_some());
        assert!(admission_inputs(&json!({"pp": 2, "tp": 2})).error.is_some());
        assert!(
            admission_inputs(&json!({"deepseek4_compute_placement": ""}))
                .error
                .is_some()
        );
    }

    #[test]
    fn admission_inputs_gates_sidecars_and_maps_cask_flags() {
        let params = json!({"draft": "draft.hfq", "dflash_mode": "off", "vision": "vision.hfq",
            "vision_mode": "off", "cask": true, "cask_sidecar": "eviction.hfq",
            "cask_budget": 768, "cask_beta": 256, "cask_handoff_tokens": 128,
            "cask_core_frac": 0.25, "cask_fold_m": 3, "state_quant": "off",
            "prefill_drafter": "prefill.hfq", "drafter": "eagle.hfq",
            "deepseek4_experts_per_token": 4});
        let a = admission_inputs(&params);
        assert!(a.error.is_none());
        assert!(a.draft.is_none() && a.vision.is_none());
        assert!(a.cask.cask_m_folding);
        assert_eq!(
            (
                a.cask.budget,
                a.cask.beta,
                a.cask.handoff_tokens,
                a.cask.fold_m
            ),
            (768, 256, 128, 3)
        );
        assert_eq!(a.cask.core_frac, 0.25);
        assert!(a.options.cask && a.options.state_quant && a.options.pflash);
        assert!(a.options.eagle_drafter && a.options.expert_count_override);
        let params = json!({"draft": "draft.hfq", "dflash_mode": "auto",
            "vision": "vision.hfq", "vision_mode": "auto", "cask": true});
        let a = admission_inputs(&params);
        assert_eq!(a.draft, Some("draft.hfq"));
        assert_eq!(a.vision, Some("vision.hfq"));
        assert!(!a.cask.cask_m_folding);
        assert!(!a.options.cask); // only a sidecar activates this admission gate
                                  // tp>1: an `auto` pairing is dropped (DFlash is not served on EP/TP),
                                  // an explicit `on` keeps the draft for the tp>1 refusal.
        let auto_tp = json!({"draft": "draft.hfq", "dflash_mode": "auto", "tp": 2});
        assert_eq!(admission_inputs(&auto_tp).draft, None);
        let on_tp = json!({"draft": "draft.hfq", "dflash_mode": "on", "tp": 2});
        assert_eq!(admission_inputs(&on_tp).draft, Some("draft.hfq"));
    }

    #[test]
    fn context_user_fits_and_exceeds() {
        assert_eq!(
            capacity::context_finding(projection(), 65_536, false, &device(), None).verdict,
            Verdict::Fits
        );
        let failure = capacity::context_finding(projection(), 262_144, false, &device(), None);
        assert_eq!(failure.verdict, Verdict::DoesNotFit);
        assert!(failure.detail.contains("49152 bytes"));
        assert!(failure.fix.unwrap().contains("200000"));
    }

    #[test]
    fn context_automatic_and_minimum_floor() {
        let automatic = capacity::context_finding(projection(), 0, true, &device(), None);
        assert_eq!(automatic.verdict, Verdict::Fits);
        assert!(automatic
            .detail
            .contains("automatic context: min(trained window"));
        let mut p = projection();
        p.card_cap = QWEN35_MIN_AUTO_CONTEXT - 1;
        let failure = capacity::context_finding(p, 0, true, &device(), None);
        assert_eq!(failure.verdict, Verdict::DoesNotFit);
        assert!(!failure.fix.unwrap().is_empty());
    }

    #[test]
    fn verdict_fold_and_exit_codes() {
        assert_eq!(fold_findings(&[]), Verdict::Fits);
        assert_eq!(
            fold_findings(&[
                Finding::fits("context", "ok"),
                Finding::undetermined("weights", "unknown", None)
            ]),
            Verdict::Undetermined
        );
        assert_eq!(
            fold_findings(&[
                Finding::undetermined("context", "unknown", None),
                Finding::fails("host_ram", "too much", "free host RAM")
            ]),
            Verdict::DoesNotFit
        );
        assert_eq!(
            (
                exit_code(Verdict::Fits),
                exit_code(Verdict::DoesNotFit),
                exit_code(Verdict::Undetermined)
            ),
            (0, 1, 2)
        );
    }

    #[test]
    fn receipts_require_exact_tag_arch_topology_and_placement() {
        let mut r = receipt();
        assert!(receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq3",
            "gfx1201",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1100",
            "single",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "ep2",
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "host-experts"
        ));
        // A source that does not state its placement never matches a plan.
        r.placement = "unstated".into();
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "single",
            "host-experts"
        ));
        assert!(!receipt_matches(
            &r,
            "example:mq4",
            "gfx1201",
            "pp2",
            "host-experts"
        ));
    }

    #[test]
    fn text_report_has_plan_lines_and_every_fix() {
        let report = report();
        let text = render_text(&report, false);
        for line in [
            "plan (what `hipfire run` would load on GPU0)",
            "topology",
            "kv",
            "context",
            "capacity",
            "placement",
            "decode",
            "routes",
            "verdict: DOES NOT FIT",
            "expected speed",
            "basis      pre-load estimate",
            "(pre-load estimate:",
        ] {
            assert!(text.contains(line), "missing {line}");
        }
        for finding in &report.findings {
            if let Some(fix) = &finding.fix {
                assert!(text.contains(&format!("fix: {fix}")));
            }
            if finding.verdict == Verdict::DoesNotFit {
                assert!(finding.fix.as_ref().is_some_and(|fix| !fix.is_empty()));
            }
        }
    }

    #[test]
    fn json_report_schema_and_privacy() {
        let value = json_report(&report()).unwrap();
        assert_eq!(value["schema"], "hipfire-fit-1");
        assert_eq!(value["model"]["file"], "example.mq4");
        assert_eq!(value["plan"]["topology"], "single");
        assert_eq!(value["verdict"], "does-not-fit");
        assert_eq!(value["plan"]["basis"], "pre-load");
        assert_eq!(value["plan"]["decode_route"], "ar");
        assert!(value["plan"]["decode_reason"].is_string());
        assert_eq!(value["input"], "example:mq4");
        assert_eq!(
            value["overrides"],
            json!({"kv_mode": null, "kv_k": null, "kv_v": null, "spec": null, "tp": null})
        );
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(!encoded.contains("/home/"));
        assert!(!encoded.contains("private-user"));
        assert!(!encoded.contains("0000:03:00.0"));
    }

    #[test]
    fn unstated_and_historical_receipts_never_match() {
        let mut r = receipt();
        assert!(receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            &r.placement.clone()
        ));
        r.placement = "unstated".into();
        assert!(!receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            "resident"
        ));
        assert!(!receipt_matches(
            &r,
            &r.tag.clone(),
            &r.arch.clone(),
            &r.topology.clone(),
            "host-experts"
        ));
        let mut h = receipt();
        h.historical = true;
        assert!(!receipt_matches(
            &h,
            &h.tag.clone(),
            &h.arch.clone(),
            &h.topology.clone(),
            &h.placement.clone()
        ));
    }

    #[test]
    fn unstated_receipt_placement_is_labelled() {
        let mut report = report();
        report.receipts[0]["placement"] = json!("unstated");
        assert!(render_text(&report, false).contains("(placement not stated by the source)"));
    }

    #[test]
    fn missing_header_has_pull_and_fetch_fixes() {
        let finding = missing_header("example:mq4", "not installed; fit needs the model header");
        assert_eq!(finding.verdict, Verdict::Undetermined);
        let fix = finding.fix.unwrap();
        assert!(fix.contains("hipfire pull example:mq4"));
        assert!(fix.contains("without --no-fetch"));
    }

    fn fit_args(flags: &[&str]) -> FitArgs {
        let mut argv = vec!["hipfire", "fit", "example:mq4"];
        argv.extend(flags);
        match crate::Cli::try_parse_from(argv).unwrap().command {
            Some(crate::Commands::Fit(args)) => args,
            _ => panic!("fit dispatch variant missing"),
        }
    }

    /// A private home holding one model file; `model_params` never opens it.
    fn fixture(label: &str) -> (crate::Paths, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("hipfire-fit-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let config = hipfire_config::ConfigPaths::under(&root);
        std::fs::create_dir_all(&config.models).unwrap();
        let model = config.models.join("example.mq4");
        std::fs::write(&model, b"params never open the model file").unwrap();
        let paths = crate::Paths {
            models: config.models.clone(),
            registry: hipfire_registry::RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        };
        (paths, model)
    }

    /// A registry whose one model declares a DFlash draft (0.5 GiB).
    fn declared_registry() -> RegistryV1 {
        RegistryV1::parse(
            r#"{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{
                "example:mq4":{"repo":"x","file":"example.mq4","size_gb":1,"min_vram_gb":1,"desc":"x",
                    "dflash":{"file":"example-dflash.hfq","size_bytes":557265920}}},"aliases":{}}"#,
            "test",
        )
        .unwrap()
    }

    fn analysis(installed: bool) -> Analysis {
        Analysis {
            model: ModelInfo {
                tag: "example:mq4".into(),
                file: None,
                installed,
                replica: !installed,
                size_bytes: None,
                arch_id: None,
                quant: None,
            },
            plan: None,
            findings: Vec::new(),
            notes: Vec::new(),
            details: Vec::new(),
        }
    }

    #[test]
    fn clap_fit_accepts_run_kv_and_spec_overrides() {
        let args = fit_args(&[
            "--kv-mode",
            "q8",
            "--kv-k",
            "fwht3",
            "--kv-v",
            "q8",
            "--spec",
            "mtp",
        ]);
        let o = args.load_overrides();
        assert_eq!(
            (o.kv_mode, o.kv_k, o.kv_v, o.speculation),
            (Some("q8"), Some("fwht3"), Some("q8"), Some("mtp"))
        );
        assert_eq!(
            OverridesEcho::from(&args).flags(),
            " --kv-mode q8 --kv-k fwht3 --kv-v q8 --spec mtp"
        );
        assert_eq!(
            fit_args(&["--speculation", "dflash"]).spec.as_deref(),
            Some("dflash")
        );
        for spec in ["off", "auto", "ngram", "dflash", "mtp", "dspark"] {
            assert!(fit_args(&["--spec", spec]).spec.as_deref() == Some(spec));
        }
        assert!(
            crate::Cli::try_parse_from(["hipfire", "fit", "example", "--spec", "turbo"]).is_err()
        );
        let none = fit_args(&[]);
        assert!(none.kv_mode.is_none() && none.kv_k.is_none() && none.spec.is_none());
        assert_eq!(OverridesEcho::from(&none).flags(), "");
    }

    #[test]
    fn model_params_thread_overrides_exactly_like_run() {
        let (paths, model) = fixture("thread");
        let args = fit_args(&[
            "--kv-mode",
            "q8",
            "--kv-k",
            "fwht3",
            "--kv-v",
            "q8",
            "--spec",
            "mtp",
        ]);
        let load = model_params(&paths, "example:mq4", None, None, &model, &args).unwrap();
        let p = &load.params;
        assert_eq!(load.selector, "mtp");
        assert_eq!(
            (
                p["kv_mode"].as_str(),
                p["kv_k"].as_str(),
                p["kv_v"].as_str()
            ),
            (Some("q8"), Some("fwht3"), Some("q8"))
        );
        assert_eq!(
            (
                p["mtp_mode"].as_str(),
                p["dflash_mode"].as_str(),
                p["ngram_draft"].as_bool(),
                p["dspark_mode"].as_str()
            ),
            (Some("on"), Some("off"), Some(false), Some("off"))
        );
        // The admission inputs carry the overrides, so the engine's own
        // refusal (an unsupported KV pair, MTP without a head) is the finding.
        let a = admission_inputs(p);
        assert!(a.error.is_none());
        assert_eq!(
            (a.kv_mode, a.kv_k, a.kv_v),
            (Some("q8"), Some("fwht3"), Some("q8"))
        );
        assert_eq!(a.options.spec.mtp, Some(true));
        assert_eq!(a.hints(&device()).kv_k, Some("fwht3"));
        // And the same helper `hipfire run` calls builds the same params.
        let resolved = crate::resolved_for_model(&paths, "example:mq4", None, None).unwrap();
        let run = crate::load_params_with_overrides(
            &resolved,
            None,
            &paths.models,
            &model,
            0,
            None,
            &args.load_overrides(),
        )
        .unwrap();
        for key in [
            "kv_mode",
            "kv_k",
            "kv_v",
            "mtp_mode",
            "dflash_mode",
            "ngram_draft",
            "dspark_mode",
        ] {
            assert_eq!(p[key], run[key], "{key}");
        }
    }

    #[test]
    fn tp_flag_matches_run_range_and_reaches_the_admission_inputs() {
        let (paths, model) = fixture("tp");
        let args = fit_args(&["--tp", "2"]);
        assert_eq!(args.tp, Some(2));
        assert_eq!(OverridesEcho::from(&args).flags(), " --tp 2");
        let load = model_params(&paths, "example:mq4", None, None, &model, &args).unwrap();
        assert_eq!(load.params["tp"], 2);
        assert_eq!(admission_inputs(&load.params).tp, 2);
        // One rank is the default: nothing is sent, admission reads tp = 1.
        let one = fit_args(&["--tp", "1"]);
        let load = model_params(&paths, "example:mq4", None, None, &model, &one).unwrap();
        assert!(load.params.get("tp").is_none());
        assert_eq!(admission_inputs(&load.params).tp, 1);
        // The same 1..=64 range as `run --tp`.
        for bad in ["0", "65"] {
            assert!(
                crate::Cli::try_parse_from(["hipfire", "fit", "example", "--tp", bad]).is_err(),
                "--tp {bad}"
            );
        }
    }

    #[test]
    fn qwen4_ep_plans_report_the_ep_degree_not_single_gpu() {
        assert_eq!(topology_key(EffectiveTopology::Single), "single");
        assert_eq!(topology_key(EffectiveTopology::Pipeline(2)), "pp2");
        assert_eq!(topology_key(EffectiveTopology::Expert(4)), "ep4");
        // Qwen4 EP admits as `Single` with an Ep axis on the admitted mesh.
        let mesh = DeviceMesh::rect(&[(DimKind::Ep, 2)]).expect("ep mesh");
        let key = ep_axis_key(EffectiveTopology::Single, &mesh);
        assert_eq!(key, "ep2");
        // The receipt key follows the plan string exactly: an `ep2` receipt
        // matches this plan and a single-GPU receipt no longer does.
        let ep_receipt = receipts::Receipt {
            topology: "ep2".into(),
            ..receipt()
        };
        assert!(receipt_matches(
            &ep_receipt,
            "example:mq4",
            "gfx1201",
            &key,
            "resident"
        ));
        assert!(!receipt_matches(
            &receipt(),
            "example:mq4",
            "gfx1201",
            &key,
            "resident"
        ));
        let single = DeviceMesh::single().unwrap();
        assert_eq!(ep_axis_key(EffectiveTopology::Single, &single), "single");
        assert_eq!(ep_axis_key(EffectiveTopology::Pipeline(2), &single), "pp2");
    }

    #[test]
    fn model_params_without_overrides_follow_the_config_selector() {
        let (paths, model) = fixture("config");
        let resolved = crate::resolved_for_model(&paths, "example:mq4", None, None).unwrap();
        let load = model_params(&paths, "example:mq4", None, None, &model, &fit_args(&[])).unwrap();
        assert_eq!(
            load.selector,
            crate::effective_speculation_selector(&resolved, None).unwrap()
        );
        assert!(load.params.get("kv_k").is_none() && load.params.get("kv_v").is_none());
        assert!(load.draft_missing.is_none());
    }

    #[test]
    fn model_params_refuse_what_run_refuses() {
        let (paths, model) = fixture("refuse");
        let refused = |flags: &[&str]| {
            model_params(&paths, "example:mq4", None, None, &model, &fit_args(flags))
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default()
        };
        assert!(refused(&["--kv-mode", "bogus"]).contains("invalid value for memory.kv_cache"));
        assert!(refused(&["--kv-k", "bogus"]).starts_with("--kv-k bogus:"));
        // `--spec dflash` on an artifact with no declared draft: run's refusal.
        assert!(refused(&["--spec", "dflash"]).starts_with("DFlash needs a draft model:"));
        // Every refusal reaches the plan as an `admission` finding whose fix
        // names the flag.
        assert!(admission_fix("DFlash needs a draft model: x").contains("--spec"));
    }

    #[test]
    fn spec_dflash_reports_a_missing_draft_and_never_fetches_it() {
        let (paths, model) = fixture("draft");
        let registry = declared_registry();
        let (_, entry) = registry.model("example:mq4").unwrap();
        let args = fit_args(&["--spec", "dflash"]);
        let load = || {
            model_params(
                &paths,
                "example:mq4",
                Some("example:mq4"),
                Some(entry),
                &model,
                &args,
            )
            .ok()
            .unwrap()
        };
        let missing_load = load();
        let missing = missing_load
            .draft_missing
            .as_ref()
            .expect("draft not installed");
        assert_eq!(
            (missing.file.as_str(), missing.size_bytes),
            ("example-dflash.hfq", Some(557_265_920))
        );
        assert!(str_param(&missing_load.params, "draft").is_none());
        let facts = |selector, missing| DecodeFacts {
            selector,
            from_flag: true,
            mtp_measured_ar_floor: Some(false),
            draft_missing: missing,
        };
        let plan = decode_plan(&missing_load.params, &facts("dflash", Some(missing)));
        assert_eq!(
            plan.route,
            "dflash (draft example-dflash.hfq not installed)"
        );
        assert!(plan.reason.contains("`--spec dflash`") && plan.reason.contains("refuses"));
        // An installed model that lacks the draft does not load: run's text.
        let mut result = analysis(true);
        report_missing_draft(&mut result, &missing_load.params, "example:mq4", missing);
        let finding = &result.findings[0];
        assert_eq!(
            (finding.id, finding.verdict),
            ("draft", Verdict::DoesNotFit)
        );
        assert!(finding.detail.starts_with(&crate::dflash_draft_not_pulled(
            "example-dflash.hfq",
            Some("example:mq4")
        )));
        assert!(finding.detail.contains("0.5 GiB per the registry"));
        assert!(finding
            .fix
            .as_deref()
            .is_some_and(|fix| fix.contains("hipfire pull example:mq4")));
        // A header replica is not installed: `hipfire pull` brings both.
        let mut replica = analysis(false);
        report_missing_draft(&mut replica, &missing_load.params, "example:mq4", missing);
        assert_eq!(replica.findings[0].verdict, Verdict::Undetermined);
        // `auto` runs without the draft: a note, never a finding.
        let mut auto = analysis(true);
        report_missing_draft(
            &mut auto,
            &json!({"dflash_mode": "auto"}),
            "example:mq4",
            missing,
        );
        assert!(auto.findings.is_empty());
        assert!(auto.notes[0].contains("example-dflash.hfq") && auto.notes[0].contains("auto"));
        // Once the draft is on disk the route names it.
        std::fs::write(paths.models.join("example-dflash.hfq"), b"draft").unwrap();
        let found = load();
        assert!(found.draft_missing.is_none());
        assert!(
            str_param(&found.params, "draft").is_some_and(|d| d.ends_with("example-dflash.hfq"))
        );
        let plan = decode_plan(&found.params, &facts("dflash", None));
        assert_eq!(plan.route, "dflash (draft example-dflash.hfq)");
    }

    fn spec_params(dflash: &str, mtp: &str, ngram: bool, dspark: &str) -> Value {
        json!({"dflash_mode": dflash, "mtp_mode": mtp, "ngram_draft": ngram, "dspark_mode": dspark})
    }

    fn plan_for(
        params: &Value,
        selector: &str,
        from_flag: bool,
        floor: Option<bool>,
    ) -> DecodePlan {
        decode_plan(
            params,
            &DecodeFacts {
                selector,
                from_flag,
                mtp_measured_ar_floor: floor,
                draft_missing: None,
            },
        )
    }

    #[test]
    fn decode_route_ar_says_why() {
        let off = spec_params("off", "off", false, "off");
        let flag = plan_for(&off, "off", true, Some(false));
        assert_eq!(flag.route, "ar");
        assert!(flag.reason.contains("`--spec off`"));
        let config = plan_for(&off, "off", false, Some(false));
        assert!(config.reason.contains("speculation.mode = off (config)"));
        // The default: MTP `auto` on an architecture with no measured AR floor.
        let auto = plan_for(
            &spec_params("off", "auto", false, "auto"),
            "auto",
            false,
            Some(false),
        );
        assert_eq!(auto.route, "ar");
        assert!(auto.reason.contains("no measured AR floor") && auto.reason.contains("--spec mtp"));
        assert!(auto.reason.contains("dflash off") && auto.reason.contains("ngram off"));
    }

    #[test]
    fn decode_route_names_every_armed_route() {
        let mtp = plan_for(
            &spec_params("off", "on", false, "off"),
            "mtp",
            true,
            Some(false),
        );
        assert_eq!(mtp.route, "mtp");
        assert!(mtp.reason.contains("`--spec mtp`") && mtp.reason.contains("host-mapped"));
        let configured = plan_for(
            &spec_params("off", "on", false, "auto"),
            "auto",
            false,
            Some(false),
        );
        assert!(configured.reason.contains("speculation.mtp = on"));
        // MTP `auto` attaches only where the architecture declares an AR floor,
        // and never over host-mapped experts.
        let auto = spec_params("off", "auto", false, "auto");
        let floor = plan_for(&auto, "auto", false, Some(true));
        assert_eq!(floor.route, "mtp (auto)");
        assert!(floor
            .reason
            .contains(hipfire_config::mtp_host_spill_reason()));
        let unknown = plan_for(&auto, "auto", false, None);
        assert!(unknown.route.starts_with("mtp / ar (auto"));
        let ngram = plan_for(
            &spec_params("off", "off", true, "off"),
            "ngram",
            true,
            Some(false),
        );
        assert_eq!(ngram.route, "ngram");
        assert!(ngram.reason.contains("`--spec ngram`"));
        let dspark = plan_for(
            &spec_params("off", "off", false, "on"),
            "dspark",
            true,
            Some(false),
        );
        assert_eq!(dspark.route, "dspark");
        let mut drafted = spec_params("on", "off", true, "off");
        drafted["draft"] = json!("/models/qwen35-9b-dflash-mq4.hfq");
        let dflash = plan_for(&drafted, "auto", false, Some(false));
        assert_eq!(dflash.route, "dflash (draft qwen35-9b-dflash-mq4.hfq)");
        assert!(dflash.reason.contains("also enabled") && dflash.reason.contains("ngram"));
        // `--spec auto` and the registry draft with DFlash off: the mode wins.
        drafted["dflash_mode"] = json!("off");
        assert_eq!(plan_for(&drafted, "off", true, Some(false)).route, "ngram");
    }

    #[test]
    fn text_plan_states_its_basis_overrides_and_decode_reason() {
        let mut report = report();
        report.overrides = OverridesEcho {
            kv_mode: Some("q8".into()),
            kv_k: None,
            kv_v: None,
            spec: Some("mtp".into()),
            tp: None,
        };
        let text = render_text(&report, false);
        assert!(
            text.contains("plan (what `hipfire run --kv-mode q8 --spec mtp` would load on GPU0)")
        );
        assert!(text.contains("  basis      pre-load estimate from file headers + sysfs; the load re-measures free VRAM after weights, drafts and scratch (max_seq fit) and may admit less"));
        assert!(text.contains("  capacity   up to 200000 tokens (pre-load estimate: "));
        assert!(text.contains("  decode     ar\n               no speculative route is enabled"));
        let value = json_report(&report).unwrap();
        assert_eq!(
            value["overrides"],
            json!({"kv_mode": "q8", "kv_k": null, "kv_v": null, "spec": "mtp", "tp": null})
        );
    }

    #[test]
    fn busy_gpu_report_names_the_best_gpu_rows_tiers_and_json() {
        let mut report = report();
        let busy = {
            let mut p = projection();
            p.free = 2 << 30;
            p.card_cap = 0;
            p
        };
        let mut roomy = projection();
        roomy.card_cap = 1_600_000;
        let mut gpu0 = device();
        gpu0.free = Some(2 << 30);
        gpu0.capacity = Some(0);
        let mut gpu2 = device();
        gpu2.index = 2;
        gpu2.capacity = Some(1_600_000);
        gpu2.capacity_shown = Some(262_144);
        gpu2.env_selector = "2".into();
        gpu2.config_selector = "GPU-0000000000000002".into();
        let gpus = vec![
            GpuCapacity::new(&gpu0, Ok(busy)),
            GpuCapacity::new(&gpu2, Ok(roomy)),
        ];
        report.machine.devices = vec![gpu0, gpu2];
        let plan = report.plan.as_mut().unwrap();
        plan.projection = Some(projection_json(busy));
        plan.best_device = capacity::best_gpu(&gpus).and_then(capacity::best_device);
        plan.gpus = gpus;
        report.tiers = vec![TierRow {
            tag: "example:mq4".into(),
            size_bytes: Some(20 << 30),
            verdict: Verdict::Fits,
            summary: "up to 1600000 tokens on GPU2".into(),
            best_gpu: Some(2),
        }];
        let text = render_text(&report, false);
        assert!(text.contains(
            "  GPU0 gfx1201  2.0 GiB free of 32.0 GiB  capacity: weights don't fit  link:"
        ));
        assert!(text.contains(
            "  GPU2 gfx1201  31.0 GiB free of 32.0 GiB  capacity: up to 262144 tokens (model limit; KV room for 1.60M)  link:"
        ));
        assert!(text.contains("  capacity   weights + prefill scratch exceed free memory (2.0 GiB free of 32.0 GiB; 30.0 GiB held by other processes or the desktop)"));
        assert!(text.contains("  best GPU   GPU2 (32.0 GiB free): up to 262144 tokens (model limit; KV room for 1.60M) — select it with HIP_VISIBLE_DEVICES=2 (or `hipfire config set hardware.devices GPU-0000000000000002`)"));
        assert!(text.contains("quant tiers (model fit only, on each tier's best GPU;"));
        assert!(text.contains("  example:mq4  20.0 GiB  FITS  up to 1600000 tokens on GPU2"));
        assert!(text.contains("largest tier that fits at 65536 tokens: example:mq4 (on GPU2)"));
        let value = json_report(&report).unwrap();
        assert_eq!(value["plan"]["best_device"]["index"], 2);
        assert_eq!(value["plan"]["best_device"]["capacity"], 1_600_000);
        assert_eq!(value["plan"]["best_device"]["capacity_shown"], 262_144);
        assert_eq!(value["machine"]["devices"][1]["capacity_shown"], 262_144);
        assert!(value["plan"]["best_device"]["free_bytes"]
            .as_u64()
            .is_some());
        assert_eq!(value["machine"]["devices"][0]["capacity"], 0);
        assert_eq!(value["machine"]["devices"][1]["capacity"], 1_600_000);
        assert_eq!(value["tiers"][0]["best_gpu"], 2);
        assert!(value["machine"]["devices"][0].get("env_selector").is_none());
    }
}
