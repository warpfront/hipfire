// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire doctor`: with no flag, a GPU-free runtime/driver/version preflight
//! ([`runtime`]) that names one fix per failed check before any model loads.
//!
//! `hipfire doctor --npu`: GPU-free fitness checks for the Strix Halo (gfx1151 +
//! XDNA2) NPU trunk prefill route, a tier, then — after confirmation — a short
//! calibration A/B against the GPU-only route and a greedy exactness check, or a
//! long `--soak`. `--apply` only ever sets `npu_trunk off`.
//!
//! Exit status: 0 ok, 1 blocked (or a failed calibration/soak), 2 calibration
//! refused because stdin is not a terminal and `--yes` was not given (`--npu`
//! only; the runtime preflight exits 0 ok or 1 blocked).

mod calibrate;
mod checks;
mod power;
mod probe;
mod profiles;
mod report;
pub(crate) mod runtime;
mod soak;

use crate::Paths;
use anyhow::Result;
use checks::{HostProbe, Target};
use clap::Args;
use probe::SystemProbe;
use report::{Action, Calibration, CheckStatus, ExactnessVerdict, Recommendation, Report};
use report::{SizeVerdict, Tier};
use std::fmt::Write as _;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

const DEFAULT_SOAK_HOURS: f64 = 4.0;
const SET_OFF: &str = "hipfire config set npu_trunk off";
const SET_AUTO: &str = "hipfire config set npu_trunk auto";
const MISMATCH_REASON: &str =
    "on and emulate disagree: please report a bug and attach this report (--json)";
const SKIPPED_NOT_A_TTY: &str = "not confirmed: stdin is not a terminal (pass --yes)";

#[derive(Args, Debug, Clone, PartialEq)]
pub(crate) struct DoctorArgs {
    /// Check the Strix Halo (gfx1151 + XDNA2) NPU trunk prefill route instead of
    /// running the runtime preflight.
    #[arg(long)]
    npu: bool,
    /// Run the calibration (or soak) without asking first. Needs `--npu`.
    #[arg(long, requires = "npu")]
    yes: bool,
    /// Print the report as one JSON object (no hostname, user, path or PCI address).
    #[arg(short, long)]
    json: bool,
    /// Set `npu_trunk` to `off` when the calibration measures a loss or an
    /// exactness mismatch. Changes nothing else. Needs `--npu`.
    #[arg(long, requires = "npu", conflicts_with = "soak")]
    apply: bool,
    /// Run a long stability soak instead of the calibration. Needs `--npu`.
    #[arg(long, requires = "npu")]
    soak: bool,
    /// Soak duration in hours (default 4).
    #[arg(long, value_name = "H", requires_all = ["soak", "npu"], value_parser = parse_hours)]
    hours: Option<f64>,
    /// Soak output directory (default: a new `npu-soak-<UTC timestamp>` here).
    #[arg(long, value_name = "DIR", requires_all = ["soak", "npu"])]
    out: Option<PathBuf>,
    /// Summarize a finished soak (including the previous boot's kernel log after a freeze).
    /// Needs `--npu`.
    #[arg(long, value_name = "DIR", requires = "npu", conflicts_with_all = ["soak", "apply"])]
    soak_report: Option<PathBuf>,
}

impl DoctorArgs {
    fn hours(&self) -> f64 {
        self.hours.unwrap_or(DEFAULT_SOAK_HOURS)
    }
}

fn parse_hours(value: &str) -> Result<f64, String> {
    let hours = value
        .parse::<f64>()
        .map_err(|_| format!("'{value}' is not a number of hours"))?;
    if hours.is_finite() && hours > 0.0 {
        Ok(hours)
    } else {
        Err("hours must be a positive number".to_owned())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Confirm {
    Run,
    Declined,
    Refused,
}

/// `--yes` runs without reading input; otherwise a terminal is asked
/// `Run it now? [y/N]` and only `y`/`yes` (any case) runs. A non-terminal
/// without `--yes` is refused, never silently run.
fn confirm(stdin_is_tty: bool, yes: bool, input: &mut dyn BufRead, out: &mut dyn Write) -> Confirm {
    if yes {
        return Confirm::Run;
    }
    if !stdin_is_tty {
        return Confirm::Refused;
    }
    let _ = write!(out, "Run it now? [y/N] ");
    let _ = out.flush();
    let mut answer = String::new();
    if input.read_line(&mut answer).is_err() {
        return Confirm::Declined;
    }
    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Confirm::Run,
        _ => Confirm::Declined,
    }
}

fn none_recommendation(reason: impl Into<String>) -> Recommendation {
    Recommendation {
        action: Action::None,
        reason: reason.into(),
        command: None,
    }
}

/// What to do with `fusions.npu_trunk` given the measurement. `npu_trunk` is the
/// effective config value.
pub(crate) fn recommend(npu_trunk: &str, cal: Option<&Calibration>) -> Recommendation {
    let Some(cal) = cal else {
        return none_recommendation("no calibration ran, so there is nothing to recommend from");
    };
    let already_off = npu_trunk == "off";
    let set_off = |reason: String| {
        if already_off {
            // Nothing to switch: report the finding, change nothing.
            none_recommendation(format!("npu_trunk is already off; {reason}"))
        } else {
            Recommendation {
                action: Action::SetOff,
                reason,
                command: Some(SET_OFF.to_owned()),
            }
        }
    };
    if cal
        .exactness
        .as_ref()
        .is_some_and(|exactness| exactness.verdict == ExactnessVerdict::Mismatch)
    {
        return set_off(MISMATCH_REASON.to_owned());
    }
    if !cal.armed {
        let why = cal.off_reason.as_deref().unwrap_or("no reason reported");
        return none_recommendation(format!("auto did not arm: {why}"));
    }
    let sizes = &cal.sizes;
    let loss = sizes.iter().any(|size| size.verdict == SizeVerdict::Loss);
    let gain = sizes.iter().any(|size| size.verdict == SizeVerdict::Gain);
    let extreme = |wanted: SizeVerdict, sign: f64| {
        sizes
            .iter()
            .filter(|size| size.verdict == wanted)
            .map(|size| (sign * size.saved_pct, size.rows))
            .max_by(|a, b| a.0.total_cmp(&b.0))
    };
    if loss && !gain {
        let (percent, rows) = extreme(SizeVerdict::Loss, -1.0).unwrap_or_default();
        return set_off(format!(
            "auto is slower than the GPU-only route (worst {percent:.1}% at {rows} rows) \
             and faster at no size"
        ));
    }
    if gain && !loss {
        let (percent, rows) = extreme(SizeVerdict::Gain, 1.0).unwrap_or_default();
        let reason =
            format!("auto is faster than the GPU-only route (best {percent:.1}% at {rows} rows)");
        return if already_off {
            Recommendation {
                action: Action::SetAuto,
                reason,
                command: Some(SET_AUTO.to_owned()),
            }
        } else {
            Recommendation {
                action: Action::KeepAuto,
                reason,
                command: None,
            }
        };
    }
    let reason = if loss {
        "gain at some sizes and loss at others; auto's guard bounds any loss"
    } else {
        "no measurable difference; auto's guard bounds any loss"
    };
    if already_off {
        none_recommendation(format!("npu_trunk stays off: {reason}"))
    } else {
        Recommendation {
            action: Action::KeepAuto,
            reason: reason.to_owned(),
            command: None,
        }
    }
}

/// `npu-soak-<UTC timestamp>` for a Unix time.
fn default_out_dir(unix_secs: u64) -> PathBuf {
    PathBuf::from(format!("npu-soak-{}", utc_stamp(unix_secs)))
}

/// `YYYYMMDDTHHMMSSZ`.
fn utc_stamp(unix_secs: u64) -> String {
    let secs = unix_secs as i64;
    let (year, month, day) = soak::civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn build_report(outcome: &checks::Outcome, assessment: &profiles::Assessment) -> Report {
    let commit = crate::BUILD_COMMIT;
    Report {
        schema: report::SCHEMA,
        hipfire: report::Build {
            // Semver only: the build string also names the source branch.
            version: env!("CARGO_PKG_VERSION"),
            commit: commit.get(..12).unwrap_or(commit).to_owned(),
        },
        tier: assessment.tier,
        profile: assessment.profile,
        profile_diff: assessment.diff.clone(),
        host: outcome.host.clone(),
        checks: outcome.checks.clone(),
        model: outcome.model.clone(),
        npu_trunk: outcome.npu_trunk.clone(),
        calibration: None,
        calibration_skipped: None,
        recommendation: none_recommendation("no calibration ran"),
        applied: false,
    }
}

fn tier_text(tier: Tier) -> &'static str {
    match tier {
        Tier::Reference => "reference",
        Tier::CompatibleUnvalidated => "compatible-unvalidated",
        Tier::Blocked => "blocked",
    }
}

fn verdict_text(verdict: SizeVerdict) -> &'static str {
    match verdict {
        SizeVerdict::Gain => "gain",
        SizeVerdict::Loss => "loss",
        SizeVerdict::NoMeasurableGain => "no measurable gain",
    }
}

fn action_text(action: Action) -> &'static str {
    match action {
        Action::KeepAuto => "keep auto",
        Action::SetOff => "set off",
        Action::SetAuto => "set auto",
        Action::None => "none",
    }
}

/// Tier, host facts, one line per check and the profile differences.
fn render_head(report: &Report) -> String {
    let mut text = String::new();
    let _ = writeln!(
        text,
        "hipfire doctor --npu (hipfire {} {})",
        report.hipfire.version, report.hipfire.commit
    );
    let profile = report
        .profile
        .map(|id| format!(" (profile {id})"))
        .unwrap_or_default();
    let _ = writeln!(text, "tier: {}{profile}", tier_text(report.tier));
    let host = &report.host;
    let _ = writeln!(
        text,
        "host: kernel={} iommu=gpu:{},npu:{} amdxdna={} npu_fw={} kernels={}",
        host.kernel, host.iommu.gpu, host.iommu.npu, host.amdxdna, host.npu_fw, host.kernels
    );
    text.push('\n');
    for check in &report.checks {
        let status = match check.status {
            CheckStatus::Pass => "PASS",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Info => "INFO",
            CheckStatus::Warn => "WARN",
        };
        let _ = writeln!(text, "{status} {}: {}", check.id, check.detail);
        if let Some(fix) = &check.fix {
            for (index, line) in fix.lines().enumerate() {
                let label = if index == 0 { "fix:" } else { "    " };
                let _ = writeln!(text, "       {label} {line}");
            }
        }
    }
    if !report.profile_diff.is_empty() {
        let id = report.profile.unwrap_or("?");
        let _ = writeln!(text, "\ndiffers from validated profile {id}:");
        for diff in &report.profile_diff {
            let _ = writeln!(
                text,
                "  {}: found {} (reference {})",
                diff.field, diff.found, diff.reference
            );
        }
    }
    text
}

/// Calibration table, exactness and the recommendation.
fn render_tail(report: &Report) -> String {
    let mut text = String::new();
    match (&report.calibration, &report.calibration_skipped) {
        (Some(cal), _) => render_calibration(&mut text, cal),
        (None, Some(why)) => {
            let _ = writeln!(text, "\ncalibration: not run ({why})");
        }
        (None, None) => {
            let _ = writeln!(text, "\ncalibration: not run");
        }
    }
    let recommendation = &report.recommendation;
    let _ = writeln!(
        text,
        "\nrecommendation: {}: {}",
        action_text(recommendation.action),
        recommendation.reason
    );
    if let Some(command) = &recommendation.command {
        let _ = writeln!(text, "  {command}");
    }
    if report.applied {
        let _ = writeln!(text, "applied: npu_trunk is now off");
    }
    text
}

fn render_calibration(text: &mut String, cal: &Calibration) {
    let _ = writeln!(text, "\ncalibration ({:.0} s):", cal.duration_s);
    if !cal.armed {
        let why = cal.off_reason.as_deref().unwrap_or("no reason reported");
        let _ = writeln!(text, "  auto did not arm: {why}");
    }
    if let Some(route) = &cal.trunk_route {
        let _ = writeln!(text, "  trunk route: {route}");
    }
    if !cal.sizes.is_empty() {
        let _ = writeln!(
            text,
            "  {:>6}  {:>9}  {:>9}  {:>8}  {:>9}  verdict",
            "rows", "off ms", "auto ms", "saved %", "spread %"
        );
        for size in &cal.sizes {
            let _ = writeln!(
                text,
                "  {:>6}  {:>9.1}  {:>9.1}  {:>8.1}  {:>9.1}  {}",
                size.rows,
                size.off.median_ms,
                size.auto.median_ms,
                size.saved_pct,
                size.spread_pct,
                verdict_text(size.verdict)
            );
        }
    }
    if let Some(rates) = &cal.rate_scales {
        let _ = writeln!(
            text,
            "  rate scales: gpu_x {:.2}, npu_x {:.2}{}, generation {}",
            rates.gpu_x,
            rates.npu_x,
            if rates.frozen { " (frozen)" } else { "" },
            rates.generation
        );
    }
    if let Some(wait) = &cal.npu_wait_per_pair {
        let _ = writeln!(
            text,
            "  NPU wait per pair: median {:.1} ms, max {:.1} ms over {} pairs",
            wait.median_ms, wait.max_ms, wait.pairs
        );
    }
    let _ = writeln!(text, "  guard trips: {}", cal.guard_trips);
    if let Some(exactness) = &cal.exactness {
        let line = match exactness.verdict {
            ExactnessVerdict::Exact => format!(
                "exact: on and emulate agree ({} rows, {} greedy tokens)",
                exactness.prompt_rows, exactness.tokens
            ),
            ExactnessVerdict::Mismatch => format!(
                "MISMATCH: on and emulate differ{} ({} rows, {} greedy tokens)",
                exactness
                    .first_difference
                    .map(|index| format!(" at token {index}"))
                    .unwrap_or_default(),
                exactness.prompt_rows,
                exactness.tokens
            ),
            ExactnessVerdict::NotExercised => {
                "not exercised: on ran no NPU pair, so the comparison proves nothing".to_owned()
            }
        };
        let _ = writeln!(text, "  exactness: {line}");
    }
}

/// Everything the flow does that is not a host read; faked in tests.
trait Actions {
    fn plan_lines(&self, soak_hours: Option<f64>) -> Vec<String>;
    fn calibrate(&mut self, target: &Target, progress: &mut dyn FnMut(&str))
        -> Result<Calibration>;
    fn soak(&mut self, target: &Target, hours: f64, out: &Path) -> Result<bool>;
    fn soak_report(&mut self, dir: &Path) -> Result<()>;
    /// `hipfire config set npu_trunk off`; returns the line that command prints.
    fn set_off(&mut self) -> Result<String>;
}

struct SystemActions<'a> {
    paths: &'a Paths,
}

impl Actions for SystemActions<'_> {
    fn plan_lines(&self, soak_hours: Option<f64>) -> Vec<String> {
        match soak_hours {
            Some(hours) => soak::plan_lines(hours),
            None => calibrate::plan_lines(),
        }
    }

    fn calibrate(
        &mut self,
        target: &Target,
        progress: &mut dyn FnMut(&str),
    ) -> Result<Calibration> {
        calibrate::run(self.paths, target, progress)
    }

    fn soak(&mut self, target: &Target, hours: f64, out: &Path) -> Result<bool> {
        soak::run(self.paths, target, hours, out)
    }

    fn soak_report(&mut self, dir: &Path) -> Result<()> {
        soak::report(dir)
    }

    fn set_off(&mut self) -> Result<String> {
        crate::set_global_config(self.paths, "npu_trunk", "off")
    }
}

struct Io<'a> {
    stdin_is_tty: bool,
    input: &'a mut dyn BufRead,
    out: &'a mut dyn Write,
    err: &'a mut dyn Write,
}

struct Ctx<'a> {
    probe: &'a dyn HostProbe,
    profiles: &'a [profiles::Profile],
    /// Unix time, for the default soak directory name.
    now: u64,
}

fn print_report(io: &mut Io<'_>, report: &Report, json: bool, head_printed: bool) -> Result<()> {
    if json {
        writeln!(io.out, "{}", serde_json::to_string_pretty(report)?)?;
        return Ok(());
    }
    if !head_printed {
        write!(io.out, "{}", render_head(report))?;
    }
    write!(io.out, "{}", render_tail(report))?;
    Ok(())
}

/// `--apply` after the recommendation: switch to `off` only for `SetOff`.
/// Returns the line to print.
fn apply(actions: &mut dyn Actions, report: &mut Report) -> String {
    if report.recommendation.action != Action::SetOff {
        return format!("--apply: no change ({})", report.recommendation.reason);
    }
    match actions.set_off() {
        Ok(line) => {
            report.applied = true;
            line
        }
        Err(error) => format!("--apply: could not set npu_trunk off: {error:#}"),
    }
}

/// The whole flow; returns the exit status.
fn execute(
    ctx: &Ctx<'_>,
    actions: &mut dyn Actions,
    args: &DoctorArgs,
    io: &mut Io<'_>,
) -> Result<i32> {
    if let Some(dir) = &args.soak_report {
        actions.soak_report(dir)?;
        return Ok(0);
    }
    let json = args.json;
    let outcome = checks::run(ctx.probe);
    let assessment = profiles::assess(
        ctx.profiles,
        outcome.blocked(),
        &outcome.host,
        outcome.model_sha(),
    );
    let mut report = build_report(&outcome, &assessment);

    let Some(target) = outcome.target.as_ref() else {
        let why = "blocked: fix the failed checks (see the fix lines)";
        report.calibration_skipped = Some(why.to_owned());
        report.recommendation = none_recommendation(why);
        let note = args.apply.then(|| apply(actions, &mut report));
        print_report(io, &report, json, false)?;
        if let Some(note) = note {
            side(io, json).write_all(format!("{note}\n").as_bytes())?;
        }
        return Ok(1);
    };

    if !json {
        write!(io.out, "{}", render_head(&report))?;
    }
    let soak_hours = args.soak.then(|| args.hours());
    {
        let channel = side(io, json);
        writeln!(channel)?;
        for line in actions.plan_lines(soak_hours) {
            writeln!(channel, "{line}")?;
        }
        writeln!(channel)?;
    }
    let decision = {
        let channel: &mut dyn Write = if json { &mut *io.err } else { &mut *io.out };
        confirm(io.stdin_is_tty, args.yes, &mut *io.input, channel)
    };
    let (skipped, status) = match decision {
        Confirm::Run => (None, 0),
        Confirm::Declined => (Some("declined"), 0),
        Confirm::Refused => (Some(SKIPPED_NOT_A_TTY), 2),
    };
    if let Some(why) = skipped {
        report.calibration_skipped = Some(why.to_owned());
        report.recommendation = none_recommendation(format!("calibration skipped: {why}"));
        let note = args.apply.then(|| apply(actions, &mut report));
        print_report(io, &report, json, !json)?;
        if let Some(note) = note {
            side(io, json).write_all(format!("{note}\n").as_bytes())?;
        }
        return Ok(status);
    }

    if args.soak {
        let out = args.out.clone().unwrap_or_else(|| default_out_dir(ctx.now));
        report.calibration_skipped = Some("soak mode: no calibration".to_owned());
        report.recommendation = none_recommendation("soak mode makes no recommendation");
        print_report(io, &report, json, !json)?;
        let passed = actions.soak(target, args.hours(), &out)?;
        writeln!(
            side(io, json),
            "soak: {}",
            if passed { "PASS" } else { "FAIL" }
        )?;
        return Ok(if passed { 0 } else { 1 });
    }

    let result = {
        let err = &mut *io.err;
        actions.calibrate(target, &mut |line| {
            let _ = writeln!(err, "{line}");
        })
    };
    let mut status = status;
    match result {
        Ok(calibration) => {
            report.recommendation = recommend(&report.npu_trunk, Some(&calibration));
            report.calibration = Some(calibration);
        }
        Err(error) => {
            // The error may carry paths; it goes to stderr, never into the report.
            writeln!(io.err, "hipfire: calibration failed: {error:#}")?;
            report.calibration_skipped = Some("calibration failed (see stderr)".to_owned());
            report.recommendation = none_recommendation("calibration failed");
            status = 1;
        }
    }
    let note = args.apply.then(|| apply(actions, &mut report));
    print_report(io, &report, json, !json)?;
    if let Some(note) = note {
        side(io, json).write_all(format!("{note}\n").as_bytes())?;
    }
    Ok(status)
}

/// Where prose goes: stdout, except under `--json` where stdout carries only
/// the report.
fn side<'a>(io: &'a mut Io<'_>, json: bool) -> &'a mut dyn Write {
    if json {
        &mut *io.err
    } else {
        &mut *io.out
    }
}

pub(crate) fn doctor_command(paths: &Paths, args: DoctorArgs) -> Result<()> {
    if !args.npu {
        let status = runtime::runtime_command(paths, args.json)?;
        let _ = io::stdout().flush();
        if status != 0 {
            std::process::exit(status);
        }
        return Ok(());
    }
    let probe = SystemProbe::new(paths);
    let mut actions = SystemActions { paths };
    let ctx = Ctx {
        probe: &probe,
        profiles: profiles::PROFILES,
        now: crate::unix_timestamp(),
    };
    let stdin = io::stdin();
    let mut input = stdin.lock();
    // Unlocked handles: the engine's reader threads write to stderr too.
    let mut out = io::stdout();
    let mut err = io::stderr();
    let mut streams = Io {
        stdin_is_tty: io::stdin().is_terminal(),
        input: &mut input,
        out: &mut out,
        err: &mut err,
    };
    let status = execute(&ctx, &mut actions, &args, &mut streams)?;
    let _ = io::stdout().flush();
    if status != 0 {
        std::process::exit(status);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::checks::testing::{FakeProbe, FAKE_USER, REFERENCE_INDEX};
    use super::report::{Exactness, PairWait, RateScales, Samples, SizeResult};
    use super::*;
    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Wrapper {
        #[command(flatten)]
        doctor: DoctorArgs,
    }

    fn parse(args: &[&str]) -> Result<DoctorArgs, clap::Error> {
        let argv = std::iter::once("doctor").chain(args.iter().copied());
        Wrapper::try_parse_from(argv).map(|wrapper| wrapper.doctor)
    }

    // ---- clap -------------------------------------------------------------

    #[test]
    fn bare_doctor_is_the_runtime_preflight_and_npu_flags_need_npu() {
        let args = parse(&[]).unwrap();
        assert!(!args.npu && !args.yes && !args.json && !args.apply && !args.soak);
        assert!(parse(&["--json"]).unwrap().json);
        assert!(parse(&["-j"]).unwrap().json);
        assert!(parse(&["--yes"]).is_err());
        let args = parse(&["--npu"]).unwrap();
        assert!(args.npu && !args.yes && !args.json && !args.apply && !args.soak);
        assert_eq!(args.hours(), 4.0);
        assert_eq!(args.out, None);
    }

    #[test]
    fn flags_parse() {
        let args = parse(&["--npu", "--yes", "-j", "--apply"]).unwrap();
        assert!(args.yes && args.json && args.apply);
        let args = parse(&["--npu", "--json"]).unwrap();
        assert!(args.json);
        let args = parse(&["--npu", "--soak", "--hours", "1.5", "--out", "run1"]).unwrap();
        assert!(args.soak);
        assert_eq!(args.hours(), 1.5);
        assert_eq!(args.out, Some(PathBuf::from("run1")));
        let args = parse(&["--npu", "--soak-report", "run1"]).unwrap();
        assert_eq!(args.soak_report, Some(PathBuf::from("run1")));
    }

    #[test]
    fn soak_options_require_soak_and_conflicts_are_rejected() {
        // Every NPU-only flag is an error without `--npu`.
        let npu_only: [&[&str]; 6] = [
            &["--yes"],
            &["--apply"],
            &["--soak"],
            &["--soak", "--hours", "2"],
            &["--soak", "--out", "x"],
            &["--soak-report", "x"],
        ];
        for flags in npu_only {
            assert!(parse(flags).is_err(), "{flags:?} must need --npu");
        }
        assert!(parse(&["--npu", "--hours", "2"]).is_err());
        assert!(parse(&["--npu", "--out", "x"]).is_err());
        assert!(parse(&["--npu", "--apply", "--soak"]).is_err());
        assert!(parse(&["--npu", "--soak-report", "x", "--soak"]).is_err());
        assert!(parse(&["--npu", "--soak-report", "x", "--apply"]).is_err());
        assert!(parse(&["--npu", "--soak", "--hours", "0"]).is_err());
        assert!(parse(&["--npu", "--soak", "--hours", "-1"]).is_err());
        assert!(parse(&["--npu", "--soak", "--hours", "abc"]).is_err());
    }

    #[test]
    fn cli_wires_the_doctor_subcommand() {
        let cli = crate::Cli::try_parse_from(["hipfire", "doctor", "--npu", "--yes"]).unwrap();
        assert!(matches!(cli.command, Some(crate::Commands::Doctor(_))));
        let cli = crate::Cli::try_parse_from(["hipfire", "doctor"]).unwrap();
        assert!(matches!(cli.command, Some(crate::Commands::Doctor(_))));
        let cli = crate::Cli::try_parse_from(["hipfire", "doctor", "--json"]).unwrap();
        assert!(matches!(cli.command, Some(crate::Commands::Doctor(_))));
        assert!(crate::Cli::try_parse_from(["hipfire", "doctor", "--yes"]).is_err());
    }

    // ---- confirmation -----------------------------------------------------

    /// Input that fails the test if it is read at all.
    struct NoRead;
    impl io::Read for NoRead {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            panic!("input must not be read");
        }
    }
    impl BufRead for NoRead {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            panic!("input must not be read");
        }
        fn consume(&mut self, _: usize) {}
    }

    fn ask(tty: bool, yes: bool, answer: &str) -> (Confirm, String) {
        let mut out = Vec::new();
        let mut input = io::Cursor::new(answer.as_bytes().to_vec());
        let decision = confirm(tty, yes, &mut input, &mut out);
        (decision, String::from_utf8(out).unwrap())
    }

    #[test]
    fn confirm_on_a_terminal_runs_only_on_y_or_yes() {
        for answer in ["y\n", "Y\n", "yes\n", "YES\n", " y \n"] {
            let (decision, out) = ask(true, false, answer);
            assert_eq!(decision, Confirm::Run, "{answer:?}");
            assert_eq!(out, "Run it now? [y/N] ");
        }
        for answer in ["n\n", "N\n", "no\n", "\n", "", "yep\n", "ye\n"] {
            assert_eq!(ask(true, false, answer).0, Confirm::Declined, "{answer:?}");
        }
    }

    #[test]
    fn confirm_refuses_without_a_terminal_and_never_prompts() {
        let (decision, out) = ask(false, false, "y\n");
        assert_eq!(decision, Confirm::Refused);
        assert!(out.is_empty());
    }

    #[test]
    fn confirm_with_yes_runs_without_reading_input() {
        let mut out = Vec::new();
        assert_eq!(confirm(true, true, &mut NoRead, &mut out), Confirm::Run);
        assert_eq!(confirm(false, true, &mut NoRead, &mut out), Confirm::Run);
        assert!(out.is_empty());
    }

    // ---- recommend --------------------------------------------------------

    fn size(rows: u64, off: f64, auto: f64, verdict: SizeVerdict) -> SizeResult {
        SizeResult {
            rows,
            off: Samples {
                samples_ms: vec![off, off],
                median_ms: off,
            },
            auto: Samples {
                samples_ms: vec![auto, auto],
                median_ms: auto,
            },
            saved_pct: (off - auto) / off * 100.0,
            spread_pct: 1.0,
            verdict,
        }
    }

    fn calibration(sizes: Vec<SizeResult>) -> Calibration {
        Calibration {
            armed: true,
            off_reason: None,
            trunk_route: Some("overlay:gfx1151-zab-v1".to_owned()),
            sizes,
            rate_scales: Some(RateScales {
                gpu_x: 1.02,
                npu_x: 0.97,
                frozen: true,
                generation: 9,
            }),
            npu_wait_per_pair: Some(PairWait {
                pairs: 40,
                median_ms: 3.1,
                max_ms: 7.9,
            }),
            guard_trips: 0,
            exactness: Some(Exactness {
                verdict: ExactnessVerdict::Exact,
                prompt_rows: 9319,
                tokens: 16,
                first_difference: None,
            }),
            duration_s: 390.0,
        }
    }

    fn gain() -> SizeResult {
        size(16343, 9000.0, 8200.0, SizeVerdict::Gain)
    }
    fn loss() -> SizeResult {
        size(32687, 20000.0, 21000.0, SizeVerdict::Loss)
    }
    fn flat() -> SizeResult {
        size(8000, 4000.0, 3995.0, SizeVerdict::NoMeasurableGain)
    }

    #[test]
    fn recommend_loss_sets_off() {
        let rec = recommend("auto", Some(&calibration(vec![flat(), loss()])));
        assert_eq!(rec.action, Action::SetOff);
        assert_eq!(
            rec.command.as_deref(),
            Some("hipfire config set npu_trunk off")
        );
        assert!(rec.reason.contains("5.0% at 32687 rows"), "{}", rec.reason);
    }

    #[test]
    fn recommend_gain_keeps_auto_or_sets_auto_when_off() {
        let cal = calibration(vec![gain(), flat()]);
        let rec = recommend("auto", Some(&cal));
        assert_eq!(rec.action, Action::KeepAuto);
        assert_eq!(rec.command, None);
        assert!(rec.reason.contains("8.9% at 16343 rows"), "{}", rec.reason);
        let rec = recommend("off", Some(&cal));
        assert_eq!(rec.action, Action::SetAuto);
        assert_eq!(
            rec.command.as_deref(),
            Some("hipfire config set npu_trunk auto")
        );
    }

    #[test]
    fn recommend_mixed_and_flat_keep_auto() {
        let rec = recommend("auto", Some(&calibration(vec![gain(), loss()])));
        assert_eq!(rec.action, Action::KeepAuto);
        assert!(rec.reason.contains("auto's guard bounds any loss"));
        let rec = recommend("auto", Some(&calibration(vec![flat()])));
        assert_eq!(rec.action, Action::KeepAuto);
        assert_eq!(
            rec.reason,
            "no measurable difference; auto's guard bounds any loss"
        );
        // Off stays off when nothing was gained.
        assert_eq!(
            recommend("off", Some(&calibration(vec![flat()]))).action,
            Action::None
        );
        assert_eq!(
            recommend("off", Some(&calibration(vec![gain(), loss()]))).action,
            Action::None
        );
    }

    #[test]
    fn recommend_mismatch_sets_off_and_asks_for_a_bug_report() {
        let mut cal = calibration(vec![gain()]);
        cal.exactness = Some(Exactness {
            verdict: ExactnessVerdict::Mismatch,
            prompt_rows: 9319,
            tokens: 16,
            first_difference: Some(3),
        });
        let rec = recommend("auto", Some(&cal));
        assert_eq!(rec.action, Action::SetOff);
        assert_eq!(
            rec.reason,
            "on and emulate disagree: please report a bug and attach this report (--json)"
        );
        // Mismatch outranks a measured gain, and a config that is already off
        // has nothing to switch.
        assert_eq!(recommend("off", Some(&cal)).action, Action::None);
    }

    #[test]
    fn recommend_not_exercised_exactness_does_not_force_off() {
        let mut cal = calibration(vec![gain()]);
        cal.exactness = Some(Exactness {
            verdict: ExactnessVerdict::NotExercised,
            prompt_rows: 9319,
            tokens: 16,
            first_difference: None,
        });
        assert_eq!(recommend("auto", Some(&cal)).action, Action::KeepAuto);
    }

    #[test]
    fn recommend_not_armed_and_not_calibrated_say_none() {
        let mut cal = calibration(Vec::new());
        cal.armed = false;
        cal.off_reason = Some("off (host not admitted)".to_owned());
        let rec = recommend("auto", Some(&cal));
        assert_eq!(rec.action, Action::None);
        assert!(rec.reason.contains("off (host not admitted)"));
        assert_eq!(rec.command, None);
        assert_eq!(recommend("auto", None).action, Action::None);
    }

    // ---- default soak directory ------------------------------------------

    #[test]
    fn default_out_dir_is_a_utc_timestamp() {
        assert_eq!(
            default_out_dir(0),
            PathBuf::from("npu-soak-19700101T000000Z")
        );
        assert_eq!(
            default_out_dir(1_760_000_000),
            PathBuf::from("npu-soak-20251009T085320Z")
        );
        // Leap day.
        assert_eq!(utc_stamp(1_709_164_800), "20240229T000000Z");
    }

    // ---- flow -------------------------------------------------------------

    #[derive(Default)]
    struct FakeActions {
        calibration: Option<Calibration>,
        calibrate_error: Option<&'static str>,
        soak_passes: bool,
        calls: Vec<String>,
        soak_out: Option<PathBuf>,
    }

    impl Actions for FakeActions {
        fn plan_lines(&self, soak_hours: Option<f64>) -> Vec<String> {
            match soak_hours {
                Some(hours) => vec![format!("soak plan {hours} h")],
                None => vec!["calibration plan about 6-7 minutes".to_owned()],
            }
        }
        fn calibrate(&mut self, _: &Target, progress: &mut dyn FnMut(&str)) -> Result<Calibration> {
            self.calls.push("calibrate".to_owned());
            progress("progress line");
            match self.calibrate_error {
                Some(error) => Err(anyhow::anyhow!(error)),
                None => Ok(self.calibration.clone().unwrap()),
            }
        }
        fn soak(&mut self, _: &Target, hours: f64, out: &Path) -> Result<bool> {
            self.calls.push(format!("soak {hours}"));
            self.soak_out = Some(out.to_owned());
            Ok(self.soak_passes)
        }
        fn soak_report(&mut self, dir: &Path) -> Result<()> {
            self.calls.push(format!("soak_report {}", dir.display()));
            Ok(())
        }
        fn set_off(&mut self) -> Result<String> {
            self.calls.push("set_off".to_owned());
            Ok("fusions.npu_trunk = off".to_owned())
        }
    }

    struct Run {
        status: i32,
        stdout: String,
        stderr: String,
        calls: Vec<String>,
        soak_out: Option<PathBuf>,
    }

    static MATCHING: [profiles::Profile; 1] = [profiles::Profile {
        id: "fake",
        kernel: "7.3.0-rc7-perfopt+",
        iommu_gpu: "identity",
        iommu_npu: "identity",
        amdxdna: "in-tree",
        npu_fw: "1.1.2.65",
        model_trunk_index_sha256: REFERENCE_INDEX,
        kernels: &["pack:v0.4.3@fedcba987654"],
    }];

    /// Same row, but no kernel set is validated yet.
    static NO_KERNEL_SET: [profiles::Profile; 1] = [profiles::Profile {
        id: "fake",
        kernel: "7.3.0-rc7-perfopt+",
        iommu_gpu: "identity",
        iommu_npu: "identity",
        amdxdna: "in-tree",
        npu_fw: "1.1.2.65",
        model_trunk_index_sha256: REFERENCE_INDEX,
        kernels: &[],
    }];

    /// A profile row equal to the healthy fake host, with the fake kernel set.
    fn matching_profiles() -> &'static [profiles::Profile] {
        &MATCHING
    }

    fn no_kernel_row() -> &'static [profiles::Profile] {
        &NO_KERNEL_SET
    }

    fn drive(
        probe: &FakeProbe,
        table: &[profiles::Profile],
        actions: &mut FakeActions,
        argv: &[&str],
        tty: bool,
        input: &str,
    ) -> Run {
        let args = parse(argv).unwrap();
        let ctx = Ctx {
            probe,
            profiles: table,
            now: 1_760_000_000,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let mut reader = io::Cursor::new(input.as_bytes().to_vec());
        let status = {
            let mut io = Io {
                stdin_is_tty: tty,
                input: &mut reader,
                out: &mut out,
                err: &mut err,
            };
            execute(&ctx, actions, &args, &mut io).unwrap()
        };
        Run {
            status,
            stdout: String::from_utf8(out).unwrap(),
            stderr: String::from_utf8(err).unwrap(),
            calls: std::mem::take(&mut actions.calls),
            soak_out: actions.soak_out.take(),
        }
    }

    fn actions_with(cal: Calibration) -> FakeActions {
        FakeActions {
            calibration: Some(cal),
            ..FakeActions::default()
        }
    }

    fn json(run: &Run) -> serde_json::Value {
        serde_json::from_str(&run.stdout).expect("stdout is one JSON object")
    }

    #[test]
    fn reference_tier_runs_the_calibration_and_prints_one_json_object() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        assert_eq!(run.status, 0);
        assert_eq!(run.calls, ["calibrate"]);
        let report = json(&run);
        assert_eq!(report["schema"], "hipfire-doctor-npu-1");
        assert_eq!(report["tier"], "reference");
        assert_eq!(report["profile"], "fake");
        assert_eq!(report["profile_diff"], serde_json::json!([]));
        assert_eq!(report["recommendation"]["action"], "keep-auto");
        assert_eq!(report["applied"], false);
        // Plan and progress stay off stdout under --json.
        assert!(run.stderr.contains("calibration plan about 6-7 minutes"));
        assert!(run.stderr.contains("progress line"));
    }

    #[test]
    fn kernel_set_difference_alone_is_compatible_unvalidated() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            no_kernel_row(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        assert_eq!(run.status, 0);
        let report = json(&run);
        assert_eq!(report["tier"], "compatible-unvalidated");
        assert_eq!(report["profile"], "fake");
        let diff = report["profile_diff"].as_array().unwrap();
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0]["field"], "kernels");
        assert_eq!(diff[0]["found"], "pack:v0.4.3@fedcba987654");
        assert_eq!(run.calls, ["calibrate"]);
    }

    #[test]
    fn other_differences_are_listed_per_field() {
        let probe = FakeProbe::healthy();
        probe.put("proc/sys/kernel/osrelease", "6.17.0-stock\n");
        probe.put("sys/class/accel/accel0/device/fw_version", "1.1.9.9\n");
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        let report = json(&run);
        assert_eq!(report["tier"], "compatible-unvalidated");
        let fields = report["profile_diff"]
            .as_array()
            .unwrap()
            .iter()
            .map(|diff| diff["field"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(fields, ["kernel", "npu_fw"]);
    }

    #[test]
    fn blocked_prints_fixes_exits_1_and_never_calibrates() {
        let mut probe = FakeProbe::healthy();
        probe.daemon = None;
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes"],
            true,
            "y\n",
        );
        assert_eq!(run.status, 1);
        assert!(run.calls.is_empty());
        assert!(run.stdout.contains("tier: blocked"), "{}", run.stdout);
        assert!(run.stdout.contains("FAIL daemon: hipfire-daemon not found"));
        assert!(run.stdout.contains("fix: hipfire setup"));
        assert!(!run.stdout.contains("Run it now?"));

        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--json"],
            false,
            "",
        );
        assert_eq!(run.status, 1);
        let report = json(&run);
        assert_eq!(report["tier"], "blocked");
        assert_eq!(report["recommendation"]["action"], "none");
    }

    #[test]
    fn low_power_warns_in_text_and_json_but_still_calibrates() {
        let probe = FakeProbe::healthy();
        probe.put("sys/firmware/acpi/platform_profile", "low-power\n");
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes"],
            true,
            "",
        );
        assert_eq!(run.status, 0, "{}", run.stdout);
        assert!(run.stdout.contains("WARN power: "), "{}", run.stdout);
        assert!(run.stdout.contains("assume full power"));
        assert!(run
            .stdout
            .contains("fix: Switch the system to its performance power mode"));
        assert!(run.stdout.contains("tier: reference"), "{}", run.stdout);
        assert!(!run.calls.is_empty());

        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        let report = json(&run);
        let power = report["checks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|check| check["id"] == "power")
            .unwrap();
        assert_eq!(power["status"], "warn");
        assert_eq!(report["tier"], "reference");
    }

    #[test]
    fn terminal_yes_runs_and_prints_the_text_report() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![gain(), loss()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu"],
            true,
            "yes\n",
        );
        assert_eq!(run.status, 0);
        assert_eq!(run.calls, ["calibrate"]);
        let text = &run.stdout;
        assert!(text.contains("tier: reference (profile fake)"), "{text}");
        assert!(text.contains("PASS gpu_arch: gfx1151 (Strix Halo)"));
        assert!(text.contains("INFO fclk:"));
        assert!(text.contains("calibration plan about 6-7 minutes"));
        assert!(text.contains("Run it now? [y/N] "));
        assert!(
            text.lines().any(|line| line
                .split_whitespace()
                .eq(["rows", "off", "ms", "auto", "ms", "saved", "%", "spread", "%", "verdict"])),
            "{text}"
        );
        assert!(text.contains("16343"));
        assert!(text.contains("gain"));
        assert!(text.contains("loss"));
        assert!(text.contains("rate scales: gpu_x 1.02, npu_x 0.97 (frozen), generation 9"));
        assert!(text.contains("NPU wait per pair: median 3.1 ms, max 7.9 ms over 40 pairs"));
        assert!(text.contains("exactness: exact"));
        assert!(text.contains("recommendation: keep auto:"));
        assert!(run.stderr.contains("progress line"));
    }

    #[test]
    fn declined_skips_the_calibration_with_exit_0() {
        let probe = FakeProbe::healthy();
        for input in ["n\n", "\n", ""] {
            let mut actions = actions_with(calibration(vec![gain()]));
            let run = drive(
                &probe,
                matching_profiles(),
                &mut actions,
                &["--npu", "--json"],
                true,
                input,
            );
            assert_eq!(run.status, 0, "{input:?}");
            assert!(run.calls.is_empty());
            let report = json(&run);
            assert_eq!(report["calibration_skipped"], "declined");
            assert!(report.get("calibration").is_none());
            // The prompt is on stderr so stdout stays one JSON object.
            assert!(run.stderr.contains("Run it now? [y/N] "));
        }
    }

    #[test]
    fn non_terminal_without_yes_is_refused_with_exit_2_and_still_reports() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![gain()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--json"],
            false,
            "y\n",
        );
        assert_eq!(run.status, 2);
        assert!(run.calls.is_empty());
        let report = json(&run);
        assert_eq!(
            report["calibration_skipped"],
            "not confirmed: stdin is not a terminal (pass --yes)"
        );
        assert_eq!(report["tier"], "reference");

        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu"],
            false,
            "",
        );
        assert_eq!(run.status, 2);
        assert!(run.stdout.contains(
            "calibration: not run (not confirmed: stdin is not a terminal (pass --yes))"
        ));
        assert!(!run.stdout.contains("Run it now?"));
    }

    #[test]
    fn apply_sets_off_on_a_measured_loss() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![loss()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--apply", "--json"],
            false,
            "",
        );
        assert_eq!(run.calls, ["calibrate", "set_off"]);
        let report = json(&run);
        assert_eq!(report["recommendation"]["action"], "set-off");
        assert_eq!(report["applied"], true);
        assert!(run.stderr.contains("fusions.npu_trunk = off"));
    }

    #[test]
    fn apply_sets_off_on_an_exactness_mismatch() {
        let probe = FakeProbe::healthy();
        let mut cal = calibration(vec![gain()]);
        cal.exactness = Some(Exactness {
            verdict: ExactnessVerdict::Mismatch,
            prompt_rows: 9319,
            tokens: 16,
            first_difference: Some(4),
        });
        let mut actions = actions_with(cal);
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--apply"],
            false,
            "",
        );
        assert_eq!(run.calls, ["calibrate", "set_off"]);
        assert!(run
            .stdout
            .contains("MISMATCH: on and emulate differ at token 4"));
        assert!(run.stdout.contains("applied: npu_trunk is now off"));
        assert!(run.stdout.contains("fusions.npu_trunk = off"));
    }

    #[test]
    fn apply_changes_nothing_otherwise() {
        let probe = FakeProbe::healthy();
        // Gain, flat, mixed, not armed, declined, refused, blocked: never set_off.
        let mut not_armed = calibration(Vec::new());
        not_armed.armed = false;
        not_armed.off_reason = Some("off (reason)".to_owned());
        for cal in [
            calibration(vec![gain()]),
            calibration(vec![flat()]),
            calibration(vec![gain(), loss()]),
            not_armed,
        ] {
            let mut actions = actions_with(cal);
            let run = drive(
                &probe,
                matching_profiles(),
                &mut actions,
                &["--npu", "--yes", "--apply"],
                false,
                "",
            );
            assert_eq!(run.calls, ["calibrate"]);
            assert!(
                run.stdout.contains("--apply: no change ("),
                "{}",
                run.stdout
            );
            assert!(!run.stdout.contains("applied:"));
        }
        let mut actions = actions_with(calibration(vec![loss()]));
        for (tty, input) in [(true, "n\n"), (false, "")] {
            let run = drive(
                &probe,
                matching_profiles(),
                &mut actions,
                &["--npu", "--apply"],
                tty,
                input,
            );
            assert!(run.calls.is_empty());
            assert!(run
                .stdout
                .contains("--apply: no change (calibration skipped"));
        }
        let mut blocked = FakeProbe::healthy();
        blocked.serve = true;
        let run = drive(
            &blocked,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--apply"],
            false,
            "",
        );
        assert!(run.calls.is_empty());
        assert!(run.stdout.contains("--apply: no change (blocked"));
    }

    #[test]
    fn failed_calibration_exits_1_and_keeps_the_error_out_of_the_report() {
        let probe = FakeProbe::healthy();
        let mut actions = FakeActions {
            calibrate_error: Some("daemon crashed at /home/someone/.hipfire/bin/daemon"),
            ..FakeActions::default()
        };
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        assert_eq!(run.status, 1);
        assert!(run.stderr.contains("daemon crashed"));
        assert!(!run.stdout.contains("someone"));
        assert_eq!(
            json(&run)["calibration_skipped"],
            "calibration failed (see stderr)"
        );
    }

    #[test]
    fn soak_runs_with_hours_and_a_default_out_dir() {
        let probe = FakeProbe::healthy();
        let mut actions = FakeActions {
            soak_passes: true,
            ..FakeActions::default()
        };
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--soak", "--hours", "2", "--yes"],
            false,
            "",
        );
        assert_eq!(run.status, 0);
        assert_eq!(run.calls, ["soak 2"]);
        assert_eq!(
            run.soak_out,
            Some(PathBuf::from("npu-soak-20251009T085320Z"))
        );
        assert!(run.stdout.contains("soak plan 2 h"));
        assert!(run.stdout.contains("soak: PASS"));

        let mut failing = FakeActions::default();
        let run = drive(
            &probe,
            matching_profiles(),
            &mut failing,
            &["--npu", "--soak", "--yes", "--out", "mine"],
            false,
            "",
        );
        assert_eq!(run.status, 1);
        assert_eq!(run.calls, ["soak 4"]);
        assert_eq!(run.soak_out, Some(PathBuf::from("mine")));
        assert!(run.stdout.contains("soak: FAIL"));
    }

    #[test]
    fn soak_is_refused_without_confirmation_and_never_starts() {
        let probe = FakeProbe::healthy();
        let mut actions = FakeActions::default();
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--soak"],
            false,
            "",
        );
        assert_eq!(run.status, 2);
        assert!(run.calls.is_empty());
    }

    #[test]
    fn soak_report_skips_the_checks() {
        let mut probe = FakeProbe::healthy();
        probe.daemon = None;
        let mut actions = FakeActions::default();
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--soak-report", "prev"],
            false,
            "",
        );
        assert_eq!(run.status, 0);
        assert_eq!(run.calls, ["soak_report prev"]);
        assert!(run.stdout.is_empty());
    }

    #[test]
    fn json_report_carries_no_path_user_host_or_pci_address() {
        let probe = FakeProbe::healthy();
        let mut actions = actions_with(calibration(vec![gain(), flat()]));
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu", "--yes", "--json"],
            false,
            "",
        );
        let text = &run.stdout;
        assert!(!text.contains('/'), "{text}");
        assert!(!text.contains(FAKE_USER), "{text}");
        assert!(!text.contains("0000:"), "{text}");
        assert!(!text.contains(probe_root_name(&probe)), "{text}");
        // Only a tag and the trunk index digest identify the model.
        let report = json(&run);
        assert_eq!(report["model"]["tag"], "qwen3.8:flash-next");
        assert_eq!(report["model"]["model_trunk_index_sha256"], REFERENCE_INDEX);
        assert_eq!(report["model"]["calibration"], "builtin");
        assert_eq!(report["npu_trunk"], "auto");
    }

    fn probe_root_name(probe: &FakeProbe) -> &str {
        probe
            .root()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap()
    }

    #[test]
    fn blocked_text_report_has_one_fix_per_failed_check() {
        let mut probe = FakeProbe::healthy();
        probe.accessible = false;
        probe.memlock = (Some(8 << 20), Some(8 << 20));
        let mut actions = FakeActions::default();
        let run = drive(
            &probe,
            matching_profiles(),
            &mut actions,
            &["--npu"],
            true,
            "",
        );
        assert_eq!(run.status, 1);
        let text = &run.stdout;
        assert_eq!(text.matches("FAIL ").count(), 2, "{text}");
        assert!(text.contains("fix: sudo usermod -aG render $USER, then log out and in."));
        assert!(text.contains("LimitMEMLOCK="));
        assert!(text.contains("<user> soft memlock"));
    }
}
