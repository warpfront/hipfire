// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire emu` and the `--emu` plumbing for `hipfire bench` (EMU-DESIGN.md §1, §6).
//!
//! SKU emulation is an application-capacity profile plus a measured-input
//! derating model. There is NO CU-count emulation on RDNA (CU masks are refused
//! at runtime init). Nothing here sets a CU mask, touches `HSA_CU_MASK`, or
//! mutates the process environment: the profile travels to the daemon only as a
//! `developer.emu` value in the `ProcessConfig` snapshot sent with the
//! `configure` message, because a late `std::env::set_var` is invisible once the
//! snapshot is taken.
//!
//! - `emu list` and `emu estimate` are CPU-only: they never start a daemon or
//!   touch HIP.
//! - `emu validate` spawns one fresh configured daemon and sends the model-free
//!   `emu_validate` verb.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand};
use hipfire_client::Engine;
use hipfire_config::emu::{self, EmuProfile, Fidelity, MemoryKind};
use hipfire_config::emu_model::{self, DeratingInput};
use hipfire_config::ProcessConfig;

use crate::Paths;

/// `ProcessConfig` key carrying the profile id (compat name `HIPFIRE_EMU`).
const PROFILE_KEY: &str = "developer.emu";
/// UMA scenario override keys (compat names `HIPFIRE_EMU_*_BYTES`). Only valid
/// next to an active profile, so `--emu off` drops them with the profile.
const UMA_KEYS: [&str; 4] = [
    "developer.emu_host_bytes",
    "developer.emu_carveout_bytes",
    "developer.emu_host_reserve_bytes",
    "developer.emu_gtt_bytes",
];

#[derive(Args, Debug)]
pub(crate) struct EmuArgs {
    #[command(subcommand)]
    action: EmuAction,
}

#[derive(Subcommand, Debug)]
enum EmuAction {
    /// Print the SKU emulation registry (profiles, targets, fidelity classes).
    List(EmuListArgs),
    /// Run the model-free validation on a fresh daemon with the profile installed.
    Validate(EmuValidateArgs),
    /// CPU-only measured-input derating estimate; starts no daemon and never touches HIP.
    Estimate(EmuEstimateArgs),
}

#[derive(Args, Debug)]
struct EmuListArgs {
    /// Emit the registry as JSON.
    #[arg(short, long)]
    json: bool,
}

#[derive(Args, Debug)]
struct EmuValidateArgs {
    /// Profile id from `hipfire emu list` (exact lower-case; not `off`).
    id: String,
    /// Print the raw receipt JSON (one line) instead of the summary.
    #[arg(short, long)]
    json: bool,
    /// Traffic buffer size in MiB (1..=2048); repeat for several sizes
    /// (daemon default 256, at most 8). Omitted: the validator's default.
    #[arg(long = "mib", value_name = "N", value_parser = clap::value_parser!(u32).range(1..=2048))]
    mib: Vec<u32>,
}

#[derive(Args, Debug)]
struct EmuEstimateArgs {
    /// Target profile id from `hipfire emu list` (labels the report).
    id: String,
    /// DeratingInput JSON (schema v1: identity, host/target rates, kernel observations).
    #[arg(long, value_name = "FILE")]
    input: PathBuf,
    /// Emit the estimate report as JSON.
    #[arg(short, long)]
    json: bool,
}

pub(crate) fn emu_command(paths: &Paths, args: EmuArgs) -> Result<()> {
    match args.action {
        EmuAction::List(args) => list_command(args),
        EmuAction::Validate(args) => validate_command(paths, args),
        EmuAction::Estimate(args) => estimate_command(args),
    }
}

// ---------------------------------------------------------------------------
// `--emu` / process-config plumbing
// ---------------------------------------------------------------------------

/// Where the effective emulation value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmuSource {
    /// An explicit `--emu` flag (overrides environment and config, including `off`).
    Cli,
    /// `HIPFIRE_EMU` / `developer.emu` already in the resolved configuration.
    Environment,
    /// Nothing requested.
    Unset,
}

/// The emulation state a spawned daemon will see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EffectiveEmu {
    /// `None` = emulation off.
    pub(crate) profile: Option<String>,
    pub(crate) source: EmuSource,
}

impl EffectiveEmu {
    /// One log line, or `None` when nothing was requested (no-profile runs stay
    /// byte-identical).
    pub(crate) fn log_line(&self) -> Option<String> {
        match (self.source, self.profile.as_deref()) {
            (EmuSource::Unset, _) => None,
            (EmuSource::Cli, Some(id)) => Some(format!(
                "emu: effective profile `{id}` (from --emu; overrides HIPFIRE_EMU)"
            )),
            (EmuSource::Cli, None) => Some(
                "emu: effective profile `off` (from --emu off; overrides HIPFIRE_EMU)".to_owned(),
            ),
            (EmuSource::Environment, Some(id)) => Some(format!(
                "emu: effective profile `{id}` (from HIPFIRE_EMU / developer.emu)"
            )),
            (EmuSource::Environment, None) => None,
        }
    }
}

/// Validate an explicit `--emu` value: `off` or a registry id. Unknown ids
/// error here, before any daemon is spawned.
pub(crate) fn validate_cli_value(value: &str) -> Result<()> {
    if value == "off" {
        return Ok(());
    }
    emu::profile(value)
        .map(|_| ())
        .map_err(|e| anyhow!("--emu: {e}"))
}

/// Apply an explicit `--emu` value to the daemon's process snapshot and return
/// the effective state. `Some(value)` overrides whatever the environment or
/// config resolved, including `off` (which also drops the UMA override knobs,
/// since they are invalid without a profile). `None` leaves the config
/// untouched. The resulting configuration is checked with
/// `hipfire_config::emu::from_process_config`, so unknown ids and UMA knob
/// conflicts fail here, not as a late daemon error. The config is not modified
/// when validation fails.
pub(crate) fn apply_cli_override(
    config: &mut ProcessConfig,
    cli: Option<&str>,
) -> Result<EffectiveEmu> {
    let Some(value) = cli else {
        let options = emu::from_process_config(config).map_err(|e| anyhow!("{e}"))?;
        return Ok(match options {
            Some(options) => EffectiveEmu {
                profile: Some(options.profile_id),
                source: EmuSource::Environment,
            },
            None => EffectiveEmu {
                profile: None,
                source: EmuSource::Unset,
            },
        });
    };
    validate_cli_value(value)?;
    let mut candidate = config.clone();
    candidate.values.set_cli(PROFILE_KEY, value)?;
    if value == "off" {
        for key in UMA_KEYS {
            candidate.values.remove(key)?;
        }
    }
    let options =
        emu::from_process_config(&candidate).map_err(|e| anyhow!("--emu {value}: {e}"))?;
    *config = candidate;
    Ok(EffectiveEmu {
        profile: options.map(|o| o.profile_id),
        source: EmuSource::Cli,
    })
}

/// Fail-early check for `bench --emu`: resolve the same configuration a bench
/// daemon would get and apply the override to a scratch copy. No daemon, no
/// model pull, no HIP.
pub(crate) fn preflight_bench(paths: &Paths, cli: Option<&str>) -> Result<()> {
    let Some(value) = cli else {
        return Ok(());
    };
    validate_cli_value(value)?;
    let (_, resolved) = crate::resolved_global(paths, true)?;
    let mut scratch = ProcessConfig::from_resolved(&resolved)?;
    apply_cli_override(&mut scratch, Some(value))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// emu list
// ---------------------------------------------------------------------------

fn fidelity_label(fidelity: Fidelity) -> &'static str {
    match fidelity {
        Fidelity::CapacityOnly => "capacity-only",
        Fidelity::CapacityEstimate => "capacity+estimate",
        Fidelity::CrossDieEstimate => "cross-die estimate",
        Fidelity::UmaScenario => "UMA scenario",
    }
}

fn memory_label(memory: MemoryKind) -> String {
    match memory {
        MemoryKind::Discrete { capacity_bytes } => {
            format!("{} GiB", capacity_bytes >> 30)
        }
        MemoryKind::Uma => "UMA (scenario)".to_owned(),
    }
}

/// The human registry table. Pure so the layout is testable.
pub(crate) fn render_list() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<13} {:<8} {:<8} {:>4} {:<15} {:>8} {:>9} {:>8}  {}\n",
        "ID", "HOST", "TARGET", "CUS", "MEMORY", "BW GB/s", "IC/L2 MiB", "VGPR KiB", "FIDELITY"
    ));
    for p in emu::PROFILES {
        let ic = if p.infinity_cache_bytes == 0 {
            "none".to_owned()
        } else {
            (p.infinity_cache_bytes >> 20).to_string()
        };
        out.push_str(&format!(
            "{:<13} {:<8} {:<8} {:>4} {:<15} {:>8} {:>9} {:>8}  {}\n",
            p.id,
            p.host_gfx,
            p.target_gfx,
            p.target_cus,
            memory_label(p.memory),
            p.peak_bw_bytes_s / 1_000_000_000,
            format!("{ic}/{}", p.l2_bytes >> 20),
            p.vgpr_file_bytes >> 10,
            fidelity_label(p.fidelity),
        ));
    }
    out.push_str(
        "\nCapacity-only: the host runs at its own speed with the target's application memory cap.\n\
         Target gfx/CUs/caches are metadata for reports and `hipfire emu estimate`; they are not executed.\n\
         There is no CU-count emulation on RDNA.\n",
    );
    out
}

fn list_command(args: EmuListArgs) -> Result<()> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(emu::PROFILES)?);
    } else {
        print!("{}", render_list());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// emu estimate (CPU only)
// ---------------------------------------------------------------------------

/// Build the estimate report. Every section says what it is: `measured` is
/// input data, `estimate` is model output, `target_profile` is registry metadata.
pub(crate) fn estimate_report(
    profile: &EmuProfile,
    input: &DeratingInput,
) -> Result<serde_json::Value> {
    let estimate =
        emu_model::estimate(input).map_err(|e| anyhow!("estimate input rejected: {e}"))?;
    Ok(serde_json::json!({
        "type": "emu_estimate",
        "schema_version": emu_model::SCHEMA_VERSION,
        "cpu_only": true,
        "target_profile": {
            "kind": "registry_metadata",
            "id": profile.id,
            "target_gfx": profile.target_gfx,
            "target_cus": profile.target_cus,
            "fidelity": fidelity_label(profile.fidelity),
        },
        "measured": {
            "kind": "input_measurement",
            "measured_wall_ns": input.measured_wall_ns,
            "unscaled_wall_ns": input.unscaled_wall_ns,
            "physical_gfx": input.identity.physical_gfx,
            "physical_cus": input.identity.physical_cus,
            "capacity_profile_during_measurement": input.identity.emu_id,
        },
        "estimate": {
            "kind": "model_estimate_not_measured",
            "quality": estimate.quality,
            "estimated_wall_ns": estimate.estimated_wall_ns,
            "unclassified_time_fraction": estimate.unclassified_time_fraction,
            "coverage_error_fraction": estimate.coverage_error_fraction,
            "assumptions": estimate.assumptions,
            "unavailable_reasons": estimate.unavailable_reasons,
            "buckets": estimate.buckets,
        },
    }))
}

fn render_estimate(report: &serde_json::Value) -> String {
    fn get<'a>(report: &'a serde_json::Value, pointer: &str) -> Option<&'a serde_json::Value> {
        report.pointer(pointer)
    }
    let mut out = String::new();
    let id = get(report, "/target_profile/id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    out.push_str(&format!(
        "hipfire emu estimate: target `{id}` (ESTIMATE from measured inputs, not a measurement)\n"
    ));
    out.push_str(&format!(
        "  measured (input): wall {} ns on physical {} ({} CUs)\n",
        get(report, "/measured/measured_wall_ns").map_or_else(|| "?".to_owned(), |v| v.to_string()),
        get(report, "/measured/physical_gfx")
            .and_then(|v| v.as_str())
            .unwrap_or("?"),
        get(report, "/measured/physical_cus").map_or_else(|| "?".to_owned(), |v| v.to_string()),
    ));
    out.push_str(&format!(
        "  estimate: quality {}, wall {}\n",
        get(report, "/estimate/quality")
            .and_then(|v| v.as_str())
            .unwrap_or("?"),
        match get(report, "/estimate/estimated_wall_ns").and_then(|v| v.as_f64()) {
            Some(ns) => format!("{ns:.0} ns (estimated)"),
            None => "unavailable".to_owned(),
        }
    ));
    for key in ["assumptions", "unavailable_reasons"] {
        if let Some(items) = get(report, &format!("/estimate/{key}")).and_then(|v| v.as_array()) {
            for item in items.iter().filter_map(|v| v.as_str()) {
                out.push_str(&format!("  {key}: {item}\n"));
            }
        }
    }
    out
}

fn estimate_command(args: EmuEstimateArgs) -> Result<()> {
    let profile = emu::profile(&args.id).map_err(|e| anyhow!("{e}"))?;
    let raw = std::fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let input: DeratingInput = serde_json::from_str(&raw).with_context(|| {
        format!(
            "{} is not a DeratingInput (schema v1)",
            args.input.display()
        )
    })?;
    let report = estimate_report(profile, &input)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_estimate(&report));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// emu validate (fresh configured daemon)
// ---------------------------------------------------------------------------

/// The `emu_validate` request body (contract with the daemon verb).
pub(crate) fn validate_request(id: &str, mib: &[u32]) -> serde_json::Value {
    let mut request = serde_json::json!({ "type": "emu_validate", "profile_id": id });
    if !mib.is_empty() {
        request["buffer_mib"] = serde_json::json!(mib);
    }
    request
}

fn section_status<'a>(receipt: &'a serde_json::Value, section: &str) -> Option<&'a str> {
    receipt
        .get(section)
        .and_then(|s| s.get("status"))
        .and_then(|s| s.as_str())
}

/// Required failures: capacity, host topology or graph `fail`, or a top-level
/// `fail`. `unavailable` is reported, not failed.
pub(crate) fn receipt_failures(receipt: &serde_json::Value) -> Vec<String> {
    let mut failures = Vec::new();
    for section in ["capacity", "host_topology", "graph"] {
        if section_status(receipt, section) == Some("fail") {
            failures.push(format!("{section} failed"));
        }
    }
    if failures.is_empty() && receipt.get("status").and_then(|s| s.as_str()) == Some("fail") {
        failures.push("receipt status is fail".to_owned());
    }
    failures
}

fn render_receipt(id: &str, receipt: &serde_json::Value) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "hipfire emu validate {id}: status {}\n",
        receipt
            .get("status")
            .and_then(|s| s.as_str())
            .unwrap_or("?")
    ));
    for section in [
        "capacity",
        "host_topology",
        "bandwidth",
        "graph",
        "estimate",
    ] {
        let Some(node) = receipt.get(section) else {
            continue;
        };
        let status = node.get("status").and_then(|s| s.as_str()).unwrap_or("?");
        out.push_str(&format!("  {section:<14} {status}"));
        if let Some(reason) = node.get("reason").and_then(|r| r.as_str()) {
            out.push_str(&format!("  ({reason})"));
        }
        out.push('\n');
    }
    out.push_str(
        "\nReading the receipt: raw/physical fields are hardware facts of this host; budgeted/\n\
         emulated fields are this process's managed application charge against the profile cap\n\
         (accounting_kind=managed_application_charge); estimate fields are model output, never a\n\
         measured target speed. cu_emulation is `none`: all host CUs execute.\n\n",
    );
    out.push_str(&serde_json::to_string_pretty(receipt).unwrap_or_default());
    out.push('\n');
    out
}

fn validate_command(paths: &Paths, args: EmuValidateArgs) -> Result<()> {
    // Registry check first: an unknown id (or `off`) is refused before anything spawns.
    emu::profile(&args.id).map_err(|e| anyhow!("{e}"))?;
    let (_, resolved) = crate::resolved_global(paths, true)?;
    let mut process_config = ProcessConfig::from_resolved(&resolved)?;
    let effective = apply_cli_override(&mut process_config, Some(&args.id))?;
    if let Some(line) = effective.log_line() {
        eprintln!("{line}");
    }
    let daemon = crate::find_daemon(paths).ok_or_else(|| crate::daemon_not_found_error(paths))?;
    let engine = Engine::spawn_configured(&daemon, &BTreeMap::new(), &process_config, None)
        .with_context(|| format!("failed to start a daemon with emu profile `{}`", args.id))?;
    engine.ping()?;
    let response = engine.request(&validate_request(&args.id, &args.mib))?;
    match response.get("type").and_then(|t| t.as_str()) {
        Some("emu_validate") => {}
        Some("error") => bail!(
            "emu validate refused: {}",
            response
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown daemon error")
        ),
        other => bail!(
            "unexpected daemon response to emu_validate: {}",
            other.unwrap_or("missing type")
        ),
    }
    let receipt = response
        .get("receipt")
        .ok_or_else(|| anyhow!("emu_validate response has no receipt"))?;
    if args.json {
        println!("{}", serde_json::to_string(receipt)?);
    } else {
        print!("{}", render_receipt(&args.id, receipt));
    }
    let failures = receipt_failures(receipt);
    if !failures.is_empty() {
        bail!("emu validate {}: {}", args.id, failures.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BenchArgs, Commands};
    use clap::Parser;
    use hipfire_config::emu_model::{
        EmuLimitsRecord, KernelObservation, RatesSource, Reading, ResourceRates, RouteConfig,
        RunIdentity, Telemetry,
    };

    fn parse(args: &[&str]) -> Commands {
        let mut argv = vec!["hipfire"];
        argv.extend_from_slice(args);
        crate::Cli::try_parse_from(argv)
            .expect("parse")
            .command
            .expect("subcommand")
    }

    fn parse_bench(args: &[&str]) -> BenchArgs {
        match parse(args) {
            Commands::Bench(bench) => bench,
            other => panic!("expected bench, got {other:?}"),
        }
    }

    fn empty_config() -> ProcessConfig {
        ProcessConfig {
            schema_version: hipfire_config::CONFIG_SCHEMA_VERSION,
            values: Default::default(),
        }
    }

    fn effective_id(config: &ProcessConfig) -> Option<String> {
        emu::from_process_config(config)
            .expect("config")
            .map(|o| o.profile_id)
    }

    #[test]
    fn bench_parses_emu_flag() {
        assert_eq!(
            parse_bench(&["bench", "model", "--emu", "rx9070xt"])
                .emu
                .as_deref(),
            Some("rx9070xt")
        );
        assert_eq!(
            parse_bench(&["bench", "model", "--emu", "off"])
                .emu
                .as_deref(),
            Some("off")
        );
        assert_eq!(parse_bench(&["bench", "model"]).emu, None);
    }

    #[test]
    fn emu_subcommands_parse() {
        let Commands::Emu(list) = parse(&["emu", "list", "--json"]) else {
            panic!("emu list");
        };
        assert!(matches!(
            list.action,
            EmuAction::List(EmuListArgs { json: true })
        ));

        let Commands::Emu(validate) = parse(&[
            "emu", "validate", "rx9070xt", "--json", "--mib", "256", "--mib", "1024",
        ]) else {
            panic!("emu validate");
        };
        let EmuAction::Validate(v) = validate.action else {
            panic!("validate action");
        };
        assert_eq!(v.id, "rx9070xt");
        assert!(v.json);
        assert_eq!(v.mib, vec![256, 1024]);

        let Commands::Emu(estimate) = parse(&[
            "emu",
            "estimate",
            "rx9060xt",
            "--input",
            "/tmp/in.json",
            "--json",
        ]) else {
            panic!("emu estimate");
        };
        let EmuAction::Estimate(e) = estimate.action else {
            panic!("estimate action");
        };
        assert_eq!(e.id, "rx9060xt");
        assert_eq!(e.input, PathBuf::from("/tmp/in.json"));
        assert!(e.json);
    }

    #[test]
    fn emu_validate_bounds_and_required_args() {
        for argv in [
            vec!["hipfire", "emu", "validate"],
            vec!["hipfire", "emu", "validate", "rx9070xt", "--mib", "0"],
            vec!["hipfire", "emu", "validate", "rx9070xt", "--mib", "2049"],
            vec!["hipfire", "emu", "estimate", "rx9070xt"],
        ] {
            assert!(
                crate::Cli::try_parse_from(argv.clone()).is_err(),
                "{argv:?}"
            );
        }
    }

    #[test]
    fn cli_overrides_env_value() {
        let mut config = empty_config();
        config.values.set_cli(PROFILE_KEY, "rx7900xt").unwrap();
        assert_eq!(effective_id(&config).as_deref(), Some("rx7900xt"));

        let effective = apply_cli_override(&mut config, Some("rx9070xt")).unwrap();
        assert_eq!(effective.source, EmuSource::Cli);
        assert_eq!(effective.profile.as_deref(), Some("rx9070xt"));
        assert_eq!(effective_id(&config).as_deref(), Some("rx9070xt"));
        // The value reaches the child exactly as the daemon's `configure` message
        // carries it: serialized ProcessConfig, read through the compat name.
        let wire = serde_json::to_string(&config).unwrap();
        let received: ProcessConfig = serde_json::from_str(&wire).unwrap();
        assert_eq!(
            received.legacy_value("HIPFIRE_EMU").as_deref(),
            Some("rx9070xt")
        );
    }

    #[test]
    fn cli_off_overrides_env_and_drops_uma_knobs() {
        let mut config = empty_config();
        config.values.set_cli(PROFILE_KEY, "strix-point").unwrap();
        config
            .values
            .set_cli("developer.emu_gtt_bytes", "8589934592")
            .unwrap();
        let effective = apply_cli_override(&mut config, Some("off")).unwrap();
        assert_eq!(effective.source, EmuSource::Cli);
        assert_eq!(effective.profile, None);
        assert_eq!(effective_id(&config), None);
        assert!(config.legacy_value("HIPFIRE_EMU_GTT_BYTES").is_none());
        assert_eq!(
            effective.log_line().as_deref(),
            Some("emu: effective profile `off` (from --emu off; overrides HIPFIRE_EMU)")
        );
    }

    #[test]
    fn env_value_is_kept_and_logged_when_no_flag() {
        let mut config = empty_config();
        config.values.set_cli(PROFILE_KEY, "rx9070").unwrap();
        let effective = apply_cli_override(&mut config, None).unwrap();
        assert_eq!(effective.source, EmuSource::Environment);
        assert_eq!(effective.profile.as_deref(), Some("rx9070"));
        assert!(effective.log_line().unwrap().contains("HIPFIRE_EMU"));
    }

    #[test]
    fn no_profile_leaves_config_and_log_untouched() {
        let mut config = empty_config();
        let before = config.clone();
        let effective = apply_cli_override(&mut config, None).unwrap();
        assert_eq!(config, before);
        assert_eq!(effective.source, EmuSource::Unset);
        assert_eq!(effective.log_line(), None);
    }

    #[test]
    fn unknown_id_is_refused_and_config_unchanged() {
        let mut config = empty_config();
        config.values.set_cli(PROFILE_KEY, "rx7900xt").unwrap();
        let before = config.clone();
        let err = apply_cli_override(&mut config, Some("rx9999")).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("rx9999") && text.contains("rx9070xt"),
            "{text}"
        );
        assert_eq!(config, before);
        assert!(
            validate_cli_value("Rx9070XT").is_err(),
            "ids are exact lower-case"
        );
        assert!(validate_cli_value("").is_err());
    }

    #[test]
    fn uma_knob_on_discrete_profile_is_refused_before_spawn() {
        let mut config = empty_config();
        config
            .values
            .set_cli("developer.emu_gtt_bytes", "8589934592")
            .unwrap();
        let before = config.clone();
        assert!(apply_cli_override(&mut config, Some("rx9070xt")).is_err());
        assert_eq!(config, before);
    }

    #[test]
    fn bench_with_unknown_emu_fails_before_any_spawn() {
        // `bench_command` validates `--emu` first: no model lookup, no pull, no
        // daemon. A daemon-less, model-less environment still yields the
        // registry error rather than "model not found"/"daemon binary not found".
        let args = parse_bench(&["bench", "no-such-model", "--emu", "rx9999"]);
        let err = crate::bench_command(&Paths::discover().unwrap(), args).unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("rx9999") && text.contains("unknown"),
            "{text}"
        );
    }

    #[test]
    fn list_has_all_ten_profiles_and_the_no_cu_emulation_note() {
        let table = render_list();
        for profile in emu::PROFILES {
            assert!(table.contains(profile.id), "{}", profile.id);
        }
        assert_eq!(emu::PROFILES.len(), 10);
        assert!(table.contains("no CU-count emulation"));
    }

    #[test]
    fn validate_request_shape() {
        assert_eq!(
            validate_request("rx9070xt", &[]),
            serde_json::json!({ "type": "emu_validate", "profile_id": "rx9070xt" })
        );
        assert_eq!(
            validate_request("rx9070xt", &[256, 1024]),
            serde_json::json!({
                "type": "emu_validate", "profile_id": "rx9070xt", "buffer_mib": [256, 1024]
            })
        );
    }

    #[test]
    fn receipt_failures_follow_required_sections() {
        let ok = serde_json::json!({
            "status": "pass",
            "capacity": {"status": "pass"}, "host_topology": {"status": "pass"},
            "bandwidth": {"status": "unavailable"}, "graph": {"status": "pass"},
            "estimate": {"status": "unavailable"},
        });
        assert!(receipt_failures(&ok).is_empty());
        let mut bad = ok.clone();
        bad["graph"]["status"] = "fail".into();
        bad["status"] = "fail".into();
        assert_eq!(receipt_failures(&bad), vec!["graph failed".to_owned()]);
        let mut top = ok.clone();
        top["status"] = "fail".into();
        assert_eq!(
            receipt_failures(&top),
            vec!["receipt status is fail".to_owned()]
        );
        assert!(render_receipt("rx9070xt", &ok).contains("cu_emulation is `none`"));
    }

    fn identity() -> RunIdentity {
        RunIdentity {
            source_revision: "abc".into(),
            source_dirty_note: "clean".into(),
            binary_md5: "0".repeat(32),
            model_md5: Reading::known("m".into()),
            sidecar_md5: Reading::unknown("no sidecar"),
            prompt_md5: Reading::unknown("no text prompt"),
            token_digest: "t".into(),
            pci_bdf: "0000:03:00.0".into(),
            physical_gfx: "gfx1201".into(),
            physical_cus: 64,
            hip_mp_count: 32,
            rocm_version: "7.0".into(),
            driver_version: Reading::unknown("unread"),
            emu_id: Some("rx9070xt".into()),
            emu_limits: Some(EmuLimitsRecord {
                device_bytes: Some(16 << 30),
                host_pinned_gtt_bytes: None,
                host_pinned_userptr_bytes: None,
                host_application_bytes: None,
            }),
            route: RouteConfig {
                graph: true,
                replay: "off".into(),
                spec: "off".into(),
                quant: "mq4".into(),
                context_tokens: 128,
                rows: 1,
            },
            telemetry: Telemetry {
                sclk_mhz: Reading::unknown("unread"),
                mclk_mhz: Reading::unknown("unread"),
                power_w: Reading::unknown("unread"),
                temperature_c: Reading::unknown("unread"),
            },
            calibration_receipt_digest: Reading::unknown("none"),
        }
    }

    fn sample_input() -> DeratingInput {
        DeratingInput {
            schema_version: emu_model::SCHEMA_VERSION,
            identity: identity(),
            host_calibration: identity(),
            measured_wall_ns: 1_000_000,
            unscaled_wall_ns: 100_000,
            host_rates: ResourceRates {
                compute_ops_s: Some(1.0e13),
                read_bytes_s: Some(6.0e11),
                copy_rw_bytes_s: Some(5.5e11),
                host_read_bytes_s: None,
            },
            target_rates: ResourceRates {
                compute_ops_s: Some(1.0e13),
                read_bytes_s: Some(3.0e11),
                copy_rw_bytes_s: Some(2.75e11),
                host_read_bytes_s: None,
            },
            target_rates_source: RatesSource::SpecRatioScenario,
            kernels: vec![KernelObservation {
                symbol: "gemv".into(),
                shape_key: "mq4/4096".into(),
                calls: 100,
                total_ns: 900_000,
                fixed_ns: Some(0),
                arithmetic_ops: Some(1_000_000),
                local_dram_bytes: Some(450_000_000),
                host_mapped_bytes: Some(0),
                byte_basis: "measured_dram".into(),
            }],
        }
    }

    #[test]
    fn estimate_reads_an_input_file_and_labels_every_section() {
        let dir = std::env::temp_dir().join(format!("hipfire-emu-estimate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("input.json");
        std::fs::write(&path, serde_json::to_string(&sample_input()).unwrap()).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        let input: DeratingInput = serde_json::from_str(&raw).unwrap();
        let profile = emu::profile("rx9060xt").unwrap();
        let report = estimate_report(profile, &input).unwrap();
        assert_eq!(report["type"], "emu_estimate");
        assert_eq!(report["cpu_only"], true);
        assert_eq!(report["target_profile"]["id"], "rx9060xt");
        assert_eq!(report["target_profile"]["kind"], "registry_metadata");
        assert_eq!(report["measured"]["measured_wall_ns"], 1_000_000);
        assert_eq!(
            report["measured"]["capacity_profile_during_measurement"],
            "rx9070xt"
        );
        assert_eq!(report["estimate"]["kind"], "model_estimate_not_measured");
        assert_eq!(report["estimate"]["quality"], "topology_scenario");
        // Halving bandwidth on a bandwidth-bound kernel lengthens it: the
        // 900 us kernel becomes ~1.8 ms, so the estimate exceeds the measured wall.
        let estimated = report["estimate"]["estimated_wall_ns"]
            .as_f64()
            .expect("scalar estimate");
        assert!(estimated > 1_000_000.0, "{estimated}");
        let human = render_estimate(&report);
        assert!(human.contains("ESTIMATE from measured inputs"), "{human}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn estimate_rejects_a_bad_schema_version() {
        let mut input = sample_input();
        input.schema_version = 99;
        let err = estimate_report(emu::profile("rx9070xt").unwrap(), &input).unwrap_err();
        assert!(format!("{err}").contains("schema_version"), "{err}");
    }
}
