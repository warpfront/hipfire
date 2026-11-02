// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire config tuning`: preview, apply and undo tuning presets.
//!
//! A thin renderer over [`hipfire_config::tuning`]: the plan, the numbers and
//! the write/undo algorithms all live there and are shared with the Settings
//! TUI. This module only detects the local facts the plan needs (GPU arch,
//! #902 schema state, `.ple-q8` files), prints the preview and asks before
//! writing.

use std::io::{BufRead, Write};

use anyhow::{bail, Result};
use hipfire_config::effective::{install_gpu_arch, A8Prefill, ArchCtx};
use hipfire_config::tuning::{
    self, current_value, display_value, evidence_summary, find_conflicts, Conflict, PlanEnv,
    Preset, TuningPlan, TuningState, TuningStatus,
};
use hipfire_config::{
    load_catalog, load_env_layer, load_global, ConfigFormat, ConfigLayer, ConfigLayerDiff,
};
use hipfire_registry::{load as load_registry, tuning_plan_env};

use crate::{ConfigTuningAction, OutputArgs, Paths};

/// Facts about this machine and process; explicit so tests drive every case.
pub(crate) struct TuningRuntime {
    /// Exact GPU arch from `install.json`.
    pub arch: Option<String>,
    pub a8: A8Prefill,
    pub plan_env: PlanEnv,
    /// stdin is a terminal: `[y/N]` may be asked.
    pub interactive: bool,
    /// Environment layer (`HIPFIRE_*`), for the conflict check.
    pub env: ConfigLayer,
    /// Seconds since the Unix epoch, recorded as `applied_at`.
    pub now: u64,
}

impl TuningRuntime {
    fn detect(paths: &Paths) -> Result<Self> {
        use std::io::IsTerminal;
        let registry = load_registry(&paths.registry).registry;
        Ok(Self {
            arch: install_gpu_arch(&paths.config.root),
            a8: A8Prefill::detect(),
            plan_env: tuning_plan_env(&paths.models, &registry),
            interactive: std::io::stdin().is_terminal(),
            env: load_env_layer()?,
            now: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Done,
    /// A write needed confirmation that could not be given; nothing was
    /// written. The process exits 2.
    Refused,
}

/// Bare `hipfire config tuning` opens the TUI on the Tuning chooser (like
/// `config profile` opens its wizard).
pub(crate) fn tui_args() -> Vec<String> {
    vec!["--tuning".to_owned()]
}

pub(crate) fn config_tuning_command(paths: &Paths, action: ConfigTuningAction) -> Result<()> {
    let runtime = TuningRuntime::detect(paths)?;
    let stdin = std::io::stdin();
    let outcome = run(
        paths,
        action,
        &runtime,
        &mut stdin.lock(),
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )?;
    if outcome == Outcome::Refused {
        std::process::exit(2);
    }
    Ok(())
}

pub(crate) fn run(
    paths: &Paths,
    action: ConfigTuningAction,
    runtime: &TuningRuntime,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<Outcome> {
    match action {
        ConfigTuningAction::Status(output) => {
            status_command(paths, runtime, output.json, out)?;
            Ok(Outcome::Done)
        }
        ConfigTuningAction::Show {
            preset,
            arch,
            output,
        } => {
            let ctx = arch_ctx(runtime, arch.as_deref())?;
            let plan = tuning::plan(preset, &ctx, &runtime.plan_env);
            let ticked = plan.default_ticked();
            let loaded = load_global(&paths.config)?;
            let state = TuningState::load(&paths.config)?;
            let view = View::new(paths, runtime, &plan, &ticked, &loaded.layer, &state)?;
            if output.json {
                writeln!(out, "{}", serde_json::to_string_pretty(&view.json(&ctx))?)?;
            } else {
                view.write_text(out, &ctx)?;
            }
            Ok(Outcome::Done)
        }
        ConfigTuningAction::Set {
            preset,
            skip,
            arch,
            yes,
            output,
        } => {
            if let Some(requested) = arch.as_deref() {
                if runtime.arch.as_deref() != Some(requested) {
                    bail!(
                        "refusing `set --arch {requested}`: this machine's GPU is {}; \
                         use `show --arch {requested}` to preview another GPU's plan",
                        runtime.arch.as_deref().unwrap_or("unknown")
                    );
                }
            }
            let ctx = arch_ctx(runtime, None)?;
            let plan = tuning::plan(preset, &ctx, &runtime.plan_env);
            let mut ticked = plan.default_ticked();
            for id in &skip {
                if plan.line(id).is_none() {
                    bail!(
                        "no line '{id}' in the {} plan for {}; lines: {}",
                        preset.title(),
                        runtime.arch.as_deref().unwrap_or("this GPU"),
                        plan.lines
                            .iter()
                            .map(|line| line.id)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                ticked.retain(|ticked| ticked != id);
            }
            let loaded = load_global(&paths.config)?;
            let state = TuningState::load(&paths.config)?;
            let view = View::new(paths, runtime, &plan, &ticked, &loaded.layer, &state)?;
            if !output.json {
                view.write_text(out, &ctx)?;
            }
            let outcome = tuning::apply(&loaded.layer, &state, &plan, &ticked, runtime.now)?;
            if !outcome.changed {
                if output.json {
                    let mut doc = view.json(&ctx);
                    doc["saved"] = serde_json::json!(false);
                    writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
                } else {
                    writeln!(
                        out,
                        "nothing to change: {} is already in effect",
                        preset.title()
                    )?;
                }
                return Ok(Outcome::Done);
            }
            let changes = outcome.net.len();
            let question = format!("Save {changes} {}?", plural(changes, "change"));
            match confirm(&question, yes, runtime, input, out, err)? {
                Confirm::Yes => {}
                Confirm::Refused => return Ok(Outcome::Refused),
                Confirm::No => {
                    writeln!(out, "aborted; nothing written")?;
                    return Ok(Outcome::Done);
                }
            }
            tuning::commit(&paths.config, &outcome.layer, &outcome.state)?;
            if output.json {
                let mut doc = view.json(&ctx);
                doc["saved"] = serde_json::json!(true);
                writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
            } else {
                writeln!(out, "{}", saved_message(preset, &outcome.net, &view))?;
                if loaded.format == ConfigFormat::LegacyJson {
                    writeln!(
                        out,
                        "migrated active configuration to {}; preserved {} as a rollback copy",
                        paths.config.config_toml.display(),
                        paths.config.config_json.display()
                    )?;
                }
            }
            Ok(Outcome::Done)
        }
        ConfigTuningAction::Undo { yes } => {
            let loaded = load_global(&paths.config)?;
            let state = TuningState::load(&paths.config)?;
            let Some(outcome) = tuning::undo(&loaded.layer, &state) else {
                writeln!(out, "nothing to undo")?;
                return Ok(Outcome::Done);
            };
            writeln!(
                out,
                "Undo the last tuning action: back to {} · writes to {}",
                outcome.restored_preset.title(),
                paths.config.config_toml.display()
            )?;
            write_net(out, &outcome.net)?;
            for key in &outcome.kept {
                writeln!(out, "  kept (you changed it since): {key}")?;
            }
            if outcome.net.is_empty() {
                writeln!(out, "  no change to config.toml; tuning.toml state only")?;
            }
            let changes = outcome.net.len();
            let question = format!("Undo {changes} {}?", plural(changes, "change"));
            match confirm(&question, yes, runtime, input, out, err)? {
                Confirm::Yes => {}
                Confirm::Refused => return Ok(Outcome::Refused),
                Confirm::No => {
                    writeln!(out, "aborted; nothing written")?;
                    return Ok(Outcome::Done);
                }
            }
            tuning::commit(&paths.config, &outcome.layer, &outcome.state)?;
            writeln!(
                out,
                "undone ({changes} {}); now {}",
                plural(changes, "change"),
                outcome.restored_preset.title()
            )?;
            Ok(Outcome::Done)
        }
    }
}

fn plural(count: usize, noun: &str) -> String {
    if count == 1 {
        noun.to_owned()
    } else {
        format!("{noun}s")
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Confirm {
    Yes,
    No,
    Refused,
}

/// `--yes`, else `[y/N]` on a terminal, else the refusal message (exit 2,
/// nothing written).
fn confirm(
    question: &str,
    yes: bool,
    runtime: &TuningRuntime,
    input: &mut dyn BufRead,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<Confirm> {
    if yes {
        return Ok(Confirm::Yes);
    }
    if !runtime.interactive {
        writeln!(
            err,
            "error: refusing to change config without confirmation; pass --yes"
        )?;
        return Ok(Confirm::Refused);
    }
    write!(out, "{question} [y/N] ")?;
    out.flush()?;
    let mut answer = String::new();
    input.read_line(&mut answer)?;
    Ok(
        if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            Confirm::Yes
        } else {
            Confirm::No
        },
    )
}

fn arch_ctx(runtime: &TuningRuntime, arch_override: Option<&str>) -> Result<ArchCtx> {
    let gpu_arch = match arch_override {
        Some(arch) => {
            let arch = arch.trim().to_ascii_lowercase();
            if !arch.starts_with("gfx") || arch.len() < 6 {
                bail!("--arch expects a gfx target such as gfx1151, got '{arch}'");
            }
            Some(arch)
        }
        None => runtime.arch.clone(),
    };
    Ok(ArchCtx {
        gpu_arch,
        gpu_count: 1,
        model_arch_id: None,
        registry_tag: None,
        a8_prefill: runtime.a8,
    })
}

// ------------------------------------------------------------------ status

fn status_command(
    paths: &Paths,
    runtime: &TuningRuntime,
    json: bool,
    out: &mut dyn Write,
) -> Result<()> {
    let loaded = load_global(&paths.config)?;
    let state = TuningState::load(&paths.config)?;
    let ctx = arch_ctx(runtime, None)?;
    let status = tuning::status(&loaded.layer, &state, &ctx);
    let arch_detail = match &runtime.arch {
        Some(arch) => format!(
            "{arch}, from {}",
            paths.config.root.join("install.json").display()
        ),
        None => "GPU arch unknown; run `hipfire setup`".to_owned(),
    };
    let config_path = paths.config.config_toml.display();
    let owned = state.applied.as_ref().map(|applied| &applied.set);
    let edited: Vec<&String> = owned
        .into_iter()
        .flatten()
        .filter(|(key, value)| loaded.layer.get(key.as_str()) != Some(*value))
        .map(|(key, _)| key)
        .collect();
    if json {
        let doc = serde_json::json!({
            "status": match status {
                TuningStatus::Balanced => "balanced",
                TuningStatus::Preset(_) => "preset",
                TuningStatus::PresetEdited(_) => "preset-edited",
                TuningStatus::Custom => "custom",
            },
            "label": status.label(),
            "preset": state.preset().id(),
            "arch": runtime.arch,
            "applied_arch": state.applied.as_ref().map(|applied| applied.arch.clone()),
            "applied_at": state.applied.as_ref().map(|applied| applied.applied_at),
            "config_path": paths.config.config_toml.display().to_string(),
            "owned": owned
                .map(|set| set.iter()
                    .map(|(key, value)| (key.clone(), serde_json::to_value(value).unwrap_or_default()))
                    .collect::<serde_json::Map<_, _>>())
                .unwrap_or_default(),
            "edited": edited,
            "undo": state.undo.as_ref().map(|undo| serde_json::json!({
                "previous_preset": undo.previous_preset.id(),
            })),
        });
        writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
        return Ok(());
    }
    writeln!(out, "Tuning: {} ({arch_detail})", status.label())?;
    match owned.filter(|set| !set.is_empty()) {
        Some(set) => {
            writeln!(out, "Preset-owned settings in {config_path}:")?;
            for (key, value) in set {
                let mark = if edited.iter().any(|edited| *edited == key) {
                    "  (changed since; left as you set it)"
                } else {
                    ""
                };
                writeln!(out, "  {key} = {}{mark}", value.to_toml_string())?;
            }
        }
        None if status == TuningStatus::Custom => writeln!(
            out,
            "No preset owns settings; some tuning keys in {config_path} are set by hand."
        )?,
        None => writeln!(out, "No preset-owned settings in {config_path}.")?,
    }
    if let Some(undo) = &state.undo {
        writeln!(
            out,
            "Undo: hipfire config tuning undo (back to {})",
            undo.previous_preset.title()
        )?;
    }
    Ok(())
}

// -------------------------------------------------------------------- show

/// Everything `show`/`set` print, computed once.
struct View<'a> {
    plan: &'a TuningPlan,
    ticked: &'a [&'static str],
    global: &'a ConfigLayer,
    preview: tuning::TuningPreview,
    config_path: String,
    takes_effect: Option<&'static str>,
}

impl<'a> View<'a> {
    fn new(
        paths: &Paths,
        runtime: &TuningRuntime,
        plan: &'a TuningPlan,
        ticked: &'a [&'static str],
        global: &'a ConfigLayer,
        state: &TuningState,
    ) -> Result<Self> {
        let mut preview = tuning::preview(global, state, plan, ticked)?;
        let catalog = load_catalog(&paths.config)?.catalog;
        preview.conflicts = find_conflicts(plan, ticked, &runtime.env, &catalog);
        let keys: Vec<&str> = preview
            .written
            .iter()
            .chain(&preview.reverted)
            .map(String::as_str)
            .collect();
        Ok(Self {
            plan,
            ticked,
            global,
            takes_effect: tuning::takes_effect(keys).map(|applies| applies.label()),
            preview,
            config_path: paths.config.config_toml.display().to_string(),
        })
    }

    fn json(&self, ctx: &ArchCtx) -> serde_json::Value {
        serde_json::json!({
            "preset": self.plan.preset.id(),
            "arch": ctx.gpu_arch,
            "config_path": self.config_path,
            "scope": "all models",
            "lines": tuning::lines_json(self.plan, self.ticked, self.global),
            "notes": self.plan.notes,
            "not_changed": self.plan.not_changed.iter()
                .map(|item| serde_json::json!({ "what": item.what, "why": item.why }))
                .collect::<Vec<_>>(),
            "net_diff": tuning::diff_json(&self.preview.net),
            "reverted": self.preview.reverted,
            "kept": self.preview.kept,
            "conflicts": tuning::conflicts_json(&self.preview.conflicts),
            "takes_effect": self.takes_effect,
        })
    }

    fn write_text(&self, out: &mut dyn Write, ctx: &ArchCtx) -> Result<()> {
        writeln!(
            out,
            "{} · {} · writes to {} (all models)",
            self.plan.preset.title(),
            ctx.gpu_arch.as_deref().unwrap_or("unknown GPU"),
            self.config_path
        )?;
        for line in &self.plan.lines {
            let ticked = line.requirement.met && self.ticked.contains(&line.id);
            let changes = line
                .writes
                .iter()
                .map(|(key, value)| {
                    let from = current_value(self.global, key);
                    let to = display_value(Some(value));
                    if from.as_ref() == Some(value) {
                        format!("{key}  {to} (already set)")
                    } else {
                        let from =
                            from.map_or_else(|| "unset".to_owned(), |value| value.to_string());
                        format!("{key}  {from} -> {to}")
                    }
                })
                .collect::<Vec<_>>()
                .join("; ");
            writeln!(
                out,
                "  [{}] {:<19} {changes}",
                if ticked { 'x' } else { ' ' },
                line.id
            )?;
            if !line.requirement.met {
                let reason = line
                    .requirement
                    .reason
                    .as_deref()
                    .unwrap_or("requirement unmet");
                match line.requirement.fix.as_deref() {
                    Some(fix) => writeln!(out, "        skipped: {reason} ({fix})")?,
                    None => writeln!(out, "        skipped: {reason}")?,
                }
            } else if !ticked {
                writeln!(out, "        skipped by --skip")?;
            }
            match line.evidence {
                Some(row) => {
                    writeln!(out, "        {}", evidence_summary(row))?;
                    if let Some(source) = row.source {
                        writeln!(out, "        measured: {source}")?;
                    }
                    if let Some(caveat) = row.caveat {
                        writeln!(out, "        note: {caveat}")?;
                    }
                }
                None => writeln!(out, "        no measurement for this GPU yet")?,
            }
        }
        for note in &self.plan.notes {
            writeln!(out, "  note: {note}")?;
        }
        if self.plan.lines.is_empty() && self.plan.preset == Preset::Balanced {
            writeln!(
                out,
                "  writes nothing; gives back the settings the applied preset still owns"
            )?;
        }
        write_net(out, &self.preview.net)?;
        for key in &self.preview.kept {
            writeln!(out, "  kept (you changed it since): {key}")?;
        }
        for conflict in &self.preview.conflicts {
            match conflict {
                Conflict::Env { key, var } => writeln!(
                    out,
                    "  warning: {var} is set in the environment and overrides {key}"
                )?,
                Conflict::PerModel { model, key } => writeln!(
                    out,
                    "  warning: model {model} sets {key} itself; the preset does not change that model"
                )?,
            }
        }
        let unchanged = self
            .plan
            .not_changed
            .iter()
            .map(|item| format!("{} ({})", item.what, item.why))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(out, "  unchanged: {unchanged}")?;
        if let Some(takes_effect) = self.takes_effect {
            writeln!(out, "  takes effect: {takes_effect}")?;
        }
        Ok(())
    }
}

/// The net `config.toml` change, key by key.
fn write_net(out: &mut dyn Write, net: &ConfigLayerDiff) -> Result<()> {
    if net.is_empty() {
        return Ok(());
    }
    writeln!(out, "  config.toml changes:")?;
    for (key, value) in &net.added {
        writeln!(out, "    + {key} = {}", value.to_toml_string())?;
    }
    for (key, from, to) in &net.changed {
        writeln!(
            out,
            "    ~ {key}: {} -> {}",
            from.to_toml_string(),
            to.to_toml_string()
        )?;
    }
    for (key, value) in &net.removed {
        writeln!(out, "    - {key} (was {})", value.to_toml_string())?;
    }
    Ok(())
}

fn saved_message(preset: Preset, net: &ConfigLayerDiff, view: &View<'_>) -> String {
    let count = net.len();
    let undo = "undo: hipfire config tuning undo";
    match preset {
        Preset::Balanced => {
            format!(
                "back to Balanced ({count} {} removed); {undo}",
                plural(count, "setting")
            )
        }
        _ => {
            let restart = if view.takes_effect == Some("serve restart") {
                "; restart serve to use them"
            } else {
                ""
            };
            format!(
                "saved {} ({count} {}){restart}; {undo}",
                preset.title(),
                plural(count, "change")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ConfigAction, OutputArgs};
    use hipfire_config::{write_global_toml, ConfigPaths, ConfigValue};
    use hipfire_registry::RegistryPaths;
    use std::fs;

    fn test_paths(label: &str) -> Paths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hipfire-cli-tuning-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let config = ConfigPaths::under(&root);
        Paths {
            models: config.models.clone(),
            registry: RegistryPaths {
                cache: root.join("registry.cache.json"),
            },
            root,
            config,
        }
    }

    fn runtime(arch: Option<&str>, interactive: bool) -> TuningRuntime {
        TuningRuntime {
            arch: arch.map(str::to_owned),
            a8: A8Prefill::NotLanded,
            plan_env: PlanEnv::default(),
            interactive,
            env: ConfigLayer::default(),
            now: 1_760_630_000,
        }
    }

    struct Ran {
        outcome: Outcome,
        out: String,
        err: String,
    }

    fn exec(
        paths: &Paths,
        runtime: &TuningRuntime,
        action: ConfigTuningAction,
        stdin: &str,
    ) -> Ran {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let outcome = run(
            paths,
            action,
            runtime,
            &mut stdin.as_bytes(),
            &mut out,
            &mut err,
        )
        .unwrap();
        Ran {
            outcome,
            out: String::from_utf8(out).unwrap(),
            err: String::from_utf8(err).unwrap(),
        }
    }

    fn set(preset: Preset, yes: bool) -> ConfigTuningAction {
        ConfigTuningAction::Set {
            preset,
            skip: Vec::new(),
            arch: None,
            yes,
            output: OutputArgs { json: false },
        }
    }

    fn seed_config(paths: &Paths) -> Vec<u8> {
        let mut layer = ConfigLayer::default();
        layer
            .set("generation.max_tokens", ConfigValue::Integer(1234))
            .unwrap();
        write_global_toml(&paths.config, &layer).unwrap();
        fs::read(&paths.config.config_toml).unwrap()
    }

    #[test]
    fn show_json_for_another_arch_is_stable() {
        let paths = test_paths("show-golden");
        seed_config(&paths);
        let ran = exec(
            &paths,
            &runtime(None, false),
            ConfigTuningAction::Show {
                preset: Preset::QualityFirst,
                arch: Some("gfx1201".into()),
                output: OutputArgs { json: true },
            },
            "",
        );
        assert_eq!(ran.outcome, Outcome::Done);
        let root = paths.root.display().to_string();
        let transcript = ran.out.replace(&root, "<ROOT>");
        fs::remove_dir_all(&paths.root).unwrap();
        assert_eq!(
            transcript,
            include_str!("../tests/fixtures/config_tuning_show_quality_first_gfx1201.json")
        );
        // `show` never writes.
    }

    #[test]
    fn show_text_follows_the_c1_mockup_shape() {
        let paths = test_paths("show-text");
        let mut rt = runtime(Some("gfx1151"), false);
        rt.plan_env = PlanEnv {
            ple_q8_present: false,
            flash_next_installed: true,
            ple_q8_missing: vec!["qwen3.8-flash-next-base.mq3-xts".into()],
            pull_tag: Some("qwen3.8:flash-next-mq3-xts".into()),
        };
        let ran = exec(
            &paths,
            &rt,
            ConfigTuningAction::Show {
                preset: Preset::QualityFirst,
                arch: None,
                output: OutputArgs { json: false },
            },
            "",
        );
        fs::remove_dir_all(&paths.root).unwrap();
        assert!(
            ran.out.contains("Quality-first · gfx1151 · writes to "),
            "{}",
            ran.out
        );
        assert!(ran.out.contains("(all models)"), "{}", ran.out);
        assert!(
            ran.out
                .contains("[x] prompt-processing   kernel.iu4_prefill  true -> false"),
            "{}",
            ran.out
        );
        assert!(
            ran.out.contains(
                "42% closer (WT2 KLD 0.0695 -> 0.0403), prompts 2.8× slower, typing unchanged"
            ),
            "{}",
            ran.out
        );
        assert!(
            ran.out
                .contains("[x] flash-next-experts  developer.qwen4_moe_sym_iu4  unset -> false"),
            "{}",
            ran.out
        );
        assert!(ran.out.contains("[ ] flash-next-ple"), "{}", ran.out);
        assert!(
            ran.out.contains(
                "skipped: qwen3.8-flash-next-base.mq3-xts has no .ple-q8 (not downloaded) \
                 (hipfire pull qwen3.8:flash-next-mq3-xts --ple q8)"
            ),
            "{}",
            ran.out
        );
        assert!(
            ran.out.contains(
                "unchanged: KV cache (fp8 vs exact within noise), speculation (designed lossless)"
            ),
            "{}",
            ran.out
        );
        assert!(
            ran.out.contains("takes effect: serve restart"),
            "{}",
            ran.out
        );
    }

    #[test]
    fn set_without_a_tty_or_yes_refuses_and_writes_nothing() {
        let paths = test_paths("refuse");
        let before = seed_config(&paths);
        let ran = exec(
            &paths,
            &runtime(Some("gfx1201"), false),
            set(Preset::QualityFirst, false),
            "",
        );
        assert_eq!(ran.outcome, Outcome::Refused);
        assert!(ran.out.contains("Quality-first · gfx1201"), "{}", ran.out);
        assert_eq!(
            ran.err,
            "error: refusing to change config without confirmation; pass --yes\n"
        );
        assert_eq!(fs::read(&paths.config.config_toml).unwrap(), before);
        assert!(!paths.config.tuning_toml.exists());
        let undo = exec(
            &paths,
            &runtime(Some("gfx1201"), false),
            ConfigTuningAction::Undo { yes: false },
            "",
        );
        assert_eq!(undo.out, "nothing to undo\n");
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn set_yes_then_undo_yes_restores_the_bytes() {
        let paths = test_paths("roundtrip");
        let before = seed_config(&paths);
        let rt = runtime(Some("gfx1201"), false);
        let saved = exec(&paths, &rt, set(Preset::QualityFirst, true), "");
        assert_eq!(saved.outcome, Outcome::Done);
        assert!(
            saved.out.contains(
                "saved Quality-first (2 changes); restart serve to use them; \
                 undo: hipfire config tuning undo"
            ),
            "{}",
            saved.out
        );
        let after = fs::read_to_string(&paths.config.config_toml).unwrap();
        assert!(after.contains("iu4_prefill = false"), "{after}");
        assert!(after.contains("qwen4_moe_sym_iu4 = false"), "{after}");
        assert!(after.contains("max_tokens = 1234"), "{after}");

        let status = exec(
            &paths,
            &rt,
            ConfigTuningAction::Status(OutputArgs { json: false }),
            "",
        );
        assert!(
            status
                .out
                .starts_with("Tuning: Quality-first (gfx1201, from "),
            "{}",
            status.out
        );
        assert!(
            status.out.contains("kernel.iu4_prefill = false"),
            "{}",
            status.out
        );
        let json = exec(
            &paths,
            &rt,
            ConfigTuningAction::Status(OutputArgs { json: true }),
            "",
        );
        let doc: serde_json::Value = serde_json::from_str(&json.out).unwrap();
        assert_eq!(doc["status"], "preset");
        assert_eq!(doc["preset"], "quality-first");

        // `config list` attributes the preset-owned keys to the preset.
        fs::write(
            paths.config.root.join("install.json"),
            r#"{"gpu_arch":"gfx1201"}"#,
        )
        .unwrap();
        let mut listed = Vec::new();
        crate::config_view::list_global_to(
            &paths,
            &crate::ConfigListArgs {
                group: Some("quality".into()),
                ..Default::default()
            },
            &mut listed,
        )
        .unwrap();
        let listed = String::from_utf8(listed).unwrap();
        let iu4 = listed
            .lines()
            .find(|line| line.trim_start().starts_with("kernel.iu4_prefill "))
            .unwrap_or_else(|| panic!("{listed}"));
        assert!(iu4.contains(" false ") && iu4.contains(" preset "), "{iu4}");

        let again = exec(&paths, &rt, set(Preset::QualityFirst, true), "");
        assert!(again.out.contains("nothing to change"), "{}", again.out);

        let undone = exec(&paths, &rt, ConfigTuningAction::Undo { yes: true }, "");
        assert_eq!(undone.outcome, Outcome::Done);
        assert!(
            undone.out.contains("undone (2 changes); now Balanced"),
            "{}",
            undone.out
        );
        assert_eq!(fs::read(&paths.config.config_toml).unwrap(), before);
        assert!(!paths.config.tuning_toml.exists());
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn interactive_set_asks_and_honours_the_answer() {
        let paths = test_paths("interactive");
        let before = seed_config(&paths);
        let rt = runtime(Some("gfx1201"), true);
        let declined = exec(&paths, &rt, set(Preset::QualityFirst, false), "n\n");
        assert_eq!(declined.outcome, Outcome::Done);
        assert!(
            declined.out.contains("Save 2 changes? [y/N] "),
            "{}",
            declined.out
        );
        assert!(
            declined.out.contains("aborted; nothing written"),
            "{}",
            declined.out
        );
        assert_eq!(fs::read(&paths.config.config_toml).unwrap(), before);
        let accepted = exec(&paths, &rt, set(Preset::QualityFirst, false), "y\n");
        assert!(
            accepted.out.contains("saved Quality-first (2 changes)"),
            "{}",
            accepted.out
        );
        assert_ne!(fs::read(&paths.config.config_toml).unwrap(), before);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn set_refuses_another_arch_and_unknown_line_ids() {
        let paths = test_paths("refusals");
        let before = seed_config(&paths);
        let rt = runtime(Some("gfx1201"), false);
        let mismatch = run(
            &paths,
            ConfigTuningAction::Set {
                preset: Preset::QualityFirst,
                skip: Vec::new(),
                arch: Some("gfx1151".into()),
                yes: true,
                output: OutputArgs { json: false },
            },
            &rt,
            &mut "".as_bytes(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            mismatch.contains("this machine's GPU is gfx1201"),
            "{mismatch}"
        );
        let unknown = run(
            &paths,
            ConfigTuningAction::Set {
                preset: Preset::QualityFirst,
                skip: vec!["no-such-line".into()],
                arch: None,
                yes: true,
                output: OutputArgs { json: false },
            },
            &rt,
            &mut "".as_bytes(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(unknown.contains("no line 'no-such-line'"), "{unknown}");
        assert!(
            unknown.contains("prompt-processing, flash-next-experts"),
            "{unknown}"
        );
        assert_eq!(fs::read(&paths.config.config_toml).unwrap(), before);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn skip_leaves_a_line_out_and_balanced_gives_back_only_preset_keys() {
        let paths = test_paths("skip");
        let before = seed_config(&paths);
        let rt = runtime(Some("gfx1201"), false);
        let skipped = exec(
            &paths,
            &rt,
            ConfigTuningAction::Set {
                preset: Preset::QualityFirst,
                skip: vec!["flash-next-experts".into()],
                arch: None,
                yes: true,
                output: OutputArgs { json: false },
            },
            "",
        );
        assert!(
            skipped.out.contains("[ ] flash-next-experts"),
            "{}",
            skipped.out
        );
        assert!(
            skipped.out.contains("saved Quality-first (1 change)"),
            "{}",
            skipped.out
        );
        let after = fs::read_to_string(&paths.config.config_toml).unwrap();
        assert!(!after.contains("qwen4_moe_sym_iu4"), "{after}");
        let balanced = exec(&paths, &rt, set(Preset::Balanced, true), "");
        assert!(
            balanced
                .out
                .contains("back to Balanced (1 setting removed)"),
            "{}",
            balanced.out
        );
        assert_eq!(fs::read(&paths.config.config_toml).unwrap(), before);
        fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn env_and_per_model_values_are_flagged_as_conflicts() {
        let paths = test_paths("conflicts");
        seed_config(&paths);
        let mut rt = runtime(Some("gfx1201"), false);
        rt.env
            .set("kernel.iu4_prefill", ConfigValue::Bool(true))
            .unwrap();
        let ran = exec(
            &paths,
            &rt,
            ConfigTuningAction::Show {
                preset: Preset::QualityFirst,
                arch: None,
                output: OutputArgs { json: false },
            },
            "",
        );
        fs::remove_dir_all(&paths.root).unwrap();
        assert!(
            ran.out.contains(
                "warning: HIPFIRE_IU4_PREFILL is set in the environment and overrides kernel.iu4_prefill"
            ),
            "{}",
            ran.out
        );
    }

    #[test]
    fn bare_config_tuning_launches_the_tui_on_the_chooser() {
        use clap::Parser;
        let parsed = crate::Cli::try_parse_from(["hipfire", "config", "tuning"]).unwrap();
        let Some(crate::Commands::Config(args)) = parsed.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Tuning { action: None })
        ));
        assert_eq!(tui_args(), ["--tuning"]);
        // A subcommand still runs headless.
        let parsed = crate::Cli::try_parse_from(["hipfire", "config", "tuning", "status"]).unwrap();
        let Some(crate::Commands::Config(args)) = parsed.command else {
            panic!("expected config command")
        };
        assert!(matches!(
            args.action,
            Some(ConfigAction::Tuning { action: Some(_) })
        ));
    }

    #[test]
    fn model_argument_is_rejected_for_tuning() {
        // `hipfire config MODEL tuning` is global-only like profile/migrate.
        let paths = test_paths("model-tuning");
        let registry = hipfire_registry::RegistryV1 {
            schema_version: 1,
            generated_at: "2099-01-01T00:00:00Z".to_owned(),
            _comment: None,
            models: Default::default(),
            aliases: Default::default(),
        };
        let error = crate::model_config_command_to(
            &paths,
            &registry,
            "qwen3:8b",
            Some(ConfigAction::Tuning { action: None }),
            &mut Vec::new(),
        )
        .unwrap_err()
        .to_string();
        fs::remove_dir_all(&paths.root).ok();
        assert!(error.contains("are global"), "{error}");
    }
}
