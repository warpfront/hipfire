// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tuning presets: which keys a preset writes, what it costs, and how to take
//! it back.
//!
//! A preset is a named bundle of global (`config.toml`) writes chosen per GPU
//! arch. [`plan`] is the single place that knows which keys and spellings a
//! preset writes on which arch; the prompt-processing line is read from
//! [`quality_choices_for`] so the Settings virtual row and the preset cannot
//! disagree, and every number shown next to a line comes from [`evidence`].
//!
//! Presets only ever merge their ticked lines into the global layer. What a
//! preset wrote is recorded in `~/.hipfire/tuning.toml` ([`TuningState`]) so
//! that
//!
//! * switching presets (or choosing Balanced) reverts exactly the keys the old
//!   preset still owns, keeping any key you edited since,
//! * [`undo`] restores the bytes of the keys the last action touched, and
//! * the Settings "From" column can say `preset` ([`attribute`]).
//!
//! Everything except [`TuningState::load`] and [`commit`] is pure; callers
//! pass the A8 state, the PLE-file facts and the clock in explicitly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::effective::{
    origin_of, quality_choices_for, A8Prefill, Applies, ArchCtx, Effective, Origin,
};
use crate::evidence::{
    evidence_for, format_closer, format_slower, EvidenceRow, A8_TO_IU4, FLASH_NEXT_EXPERTS,
    FLASH_NEXT_PLE, PROMPT_PROCESSING, SHORTER_THINKING,
};
use crate::{
    canonical_config_key, developer_env_for_key, diff_config_layers, field, write_global_toml,
    ConfigError, ConfigLayer, ConfigLayerDiff, ConfigPaths, ConfigSource, ConfigValue,
    ModelCatalog, ResolvedConfig, Result,
};

/// Schema version of `tuning.toml`.
pub const TUNING_SCHEMA_VERSION: i64 = 1;

/// Registry tag the unmet-requirement fix names when no installed Flash-Next
/// entry supplies one.
pub const DEFAULT_FLASH_NEXT_TAG: &str = "qwen3.8:flash-next";

const SYM_IU4_KEY: &str = "developer.qwen4_moe_sym_iu4";
const PLE_FORMAT_KEY: &str = "memory.ple_format";
const EFFORT_KEY: &str = "reasoning.effort";

/// Line ids, in the order a plan lists them.
pub const LINE_IDS: [&str; 4] = [
    PROMPT_PROCESSING,
    FLASH_NEXT_EXPERTS,
    FLASH_NEXT_PLE,
    SHORTER_THINKING,
];

// ---------------------------------------------------------------- presets

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    Balanced,
    QualityFirst,
    LowPower,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Self::Balanced, Self::QualityFirst, Self::LowPower];

    /// Stable machine name (`tuning.toml`, CLI, JSON).
    pub fn id(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::QualityFirst => "quality-first",
            Self::LowPower => "low-power",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Balanced => "Balanced",
            Self::QualityFirst => "Quality-first",
            Self::LowPower => "Low-power",
        }
    }

    /// Parse `balanced`, `quality-first` (alias `quality`), `low-power`;
    /// ASCII case-insensitive, `_` accepted for `-`.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "balanced" => Some(Self::Balanced),
            "quality-first" | "quality" => Some(Self::QualityFirst),
            "low-power" => Some(Self::LowPower),
            _ => None,
        }
    }

    /// Chooser one-liner (spec §2.6).
    pub fn one_liner(self) -> &'static str {
        match self {
            Self::Balanced => "What hipfire ships for this GPU. Fast, good answers.",
            Self::QualityFirst => "More accurate answers. Long prompts take longer to read.",
            Self::LowPower => "Less work per answer: shorter thinking.",
        }
    }

    /// Chooser second line (spec §2.6).
    pub fn second_line(self) -> &'static str {
        match self {
            Self::Balanced => "Removes settings a preset added; keeps yours.",
            Self::QualityFirst => "Typing speed after that is about the same.",
            Self::LowPower => {
                "Keeps the fastest prompt math. Does not change your GPU's power limit."
            }
        }
    }
}

impl std::fmt::Display for Preset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.id())
    }
}

impl std::str::FromStr for Preset {
    type Err = String;

    fn from_str(raw: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(raw).ok_or_else(|| {
            format!("unknown preset '{raw}'; expected balanced, quality-first or low-power")
        })
    }
}

// ------------------------------------------------------------------- plan

/// Facts about the local install that decide which lines can be ticked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanEnv {
    /// A Flash-Next split base is installed and every one has its `.ple-q8`
    /// (or the shared q8 file).
    pub ple_q8_present: bool,
    /// At least one Flash-Next model is installed.
    pub flash_next_installed: bool,
    /// File names of installed Flash-Next bases that lack a q8 PLE file.
    pub ple_q8_missing: Vec<String>,
    /// Registry tag `hipfire pull <tag> --ple q8` should name.
    pub pull_tag: Option<String>,
}

impl PlanEnv {
    /// Every requirement met (used to enumerate a preset's keys).
    fn everything() -> Self {
        Self {
            ple_q8_present: true,
            flash_next_installed: true,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    pub met: bool,
    /// Why the line cannot be ticked.
    pub reason: Option<String>,
    /// Command that fixes it.
    pub fix: Option<String>,
}

impl Requirement {
    fn met() -> Self {
        Self {
            met: true,
            reason: None,
            fix: None,
        }
    }

    fn unmet(reason: String, fix: String) -> Self {
        Self {
            met: false,
            reason: Some(reason),
            fix: Some(fix),
        }
    }
}

/// One tickable line of a preset preview.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanLine {
    /// One of [`LINE_IDS`].
    pub id: &'static str,
    /// Plain title (`Prompt processing`).
    pub title: &'static str,
    /// `Fast (4-bit math) → Accurate (8-bit math)`.
    pub summary: String,
    /// What the line applies to (`27B-class dense models`, `Flash-Next only`).
    pub scope: Option<&'static str>,
    /// Keys and values written when the line is ticked.
    pub writes: Vec<(&'static str, ConfigValue)>,
    pub requirement: Requirement,
    /// Ticked unless the requirement is unmet.
    pub ticked: bool,
    pub evidence: Option<&'static EvidenceRow>,
}

/// A setting no preset touches, and why (spec §2.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotChanged {
    pub what: String,
    pub why: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TuningPlan {
    pub preset: Preset,
    pub arch: Option<String>,
    pub a8: A8Prefill,
    pub lines: Vec<PlanLine>,
    /// Free-text notes (`no measured quality switches for gfx1030 yet`).
    pub notes: Vec<String>,
    pub not_changed: Vec<NotChanged>,
}

impl TuningPlan {
    pub fn line(&self, id: &str) -> Option<&PlanLine> {
        self.lines.iter().find(|line| line.id == id)
    }

    /// Ids of the lines ticked by default.
    pub fn default_ticked(&self) -> Vec<&'static str> {
        self.lines
            .iter()
            .filter(|line| line.ticked)
            .map(|line| line.id)
            .collect()
    }

    /// Every key any line writes, ticked or not.
    pub fn keys(&self) -> BTreeSet<&'static str> {
        self.lines
            .iter()
            .flat_map(|line| line.writes.iter().map(|(key, _)| *key))
            .collect()
    }

    /// Writes of the lines in `ticked`. A line whose requirement is unmet is
    /// never written, ticked or not.
    pub fn writes_for(&self, ticked: &[&str]) -> BTreeMap<String, ConfigValue> {
        self.lines
            .iter()
            .filter(|line| line.requirement.met && ticked.contains(&line.id))
            .flat_map(|line| line.writes.iter())
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect()
    }
}

fn measured_arch(arch: &str) -> bool {
    matches!(arch, "gfx1151" | "gfx1201" | "gfx1100")
}

fn standard_not_changed() -> Vec<NotChanged> {
    let item = |what: &str, why: &str| NotChanged {
        what: what.to_owned(),
        why: why.to_owned(),
    };
    vec![
        item("KV cache", "fp8 vs exact within noise"),
        item("speculation", "designed lossless"),
        item(
            "context, devices, offload, model",
            "yours; presets never change them",
        ),
    ]
}

fn all_set(
    writes: &[(&'static str, Option<ConfigValue>)],
) -> Option<Vec<(&'static str, ConfigValue)>> {
    writes
        .iter()
        .map(|(key, value)| value.clone().map(|value| (*key, value)))
        .collect()
}

/// The Prompt processing line moving to `target` (`accurate` or `fast`) when
/// that choice is not what the built-in defaults already give.
fn prompt_line(
    ctx: &ArchCtx,
    arch: &str,
    a8: A8Prefill,
    target: &str,
    evidence_line: &str,
) -> Option<PlanLine> {
    let choices = quality_choices_for(ctx, a8);
    let chosen = choices.iter().find(|choice| choice.id == target)?;
    if chosen.builtin_default {
        return None;
    }
    let writes = all_set(&chosen.writes)?;
    let other = choices.iter().find(|choice| choice.id != target)?;
    Some(PlanLine {
        id: PROMPT_PROCESSING,
        title: "Prompt processing",
        summary: format!("{} → {}", other.label, chosen.label),
        scope: Some("27B-class dense models"),
        writes,
        requirement: Requirement::met(),
        ticked: true,
        evidence: evidence_for(evidence_line, arch, a8),
    })
}

fn experts_line(arch: &str, a8: A8Prefill) -> PlanLine {
    PlanLine {
        id: FLASH_NEXT_EXPERTS,
        title: "Flash-Next experts",
        summary: "Fast (4-bit) → Accurate (16-bit)".to_owned(),
        scope: Some("Flash-Next only"),
        // R1: typed boolean, never the legacy "0" string.
        writes: vec![(SYM_IU4_KEY, ConfigValue::Bool(false))],
        requirement: Requirement::met(),
        ticked: true,
        evidence: evidence_for(FLASH_NEXT_EXPERTS, arch, a8),
    }
}

fn ple_line(arch: &str, a8: A8Prefill, env: &PlanEnv) -> PlanLine {
    let requirement = if env.ple_q8_present {
        Requirement::met()
    } else {
        let tag = env.pull_tag.as_deref().unwrap_or(DEFAULT_FLASH_NEXT_TAG);
        let reason = if !env.flash_next_installed {
            "no Flash-Next model is installed, so there is no .ple-q8 file".to_owned()
        } else if env.ple_q8_missing.is_empty() {
            "the .ple-q8 file is not on disk".to_owned()
        } else {
            format!(
                "{} has no .ple-q8 (not downloaded)",
                env.ple_q8_missing.join(", ")
            )
        };
        Requirement::unmet(reason, format!("hipfire pull {tag} --ple q8"))
    };
    PlanLine {
        id: FLASH_NEXT_PLE,
        title: "Flash-Next word table",
        summary: "4-bit → 8-bit".to_owned(),
        scope: Some("Flash-Next only"),
        writes: vec![(PLE_FORMAT_KEY, ConfigValue::String("q8".to_owned()))],
        ticked: requirement.met,
        requirement,
        evidence: evidence_for(FLASH_NEXT_PLE, arch, a8),
    }
}

fn thinking_line(arch: &str, a8: A8Prefill) -> PlanLine {
    PlanLine {
        id: SHORTER_THINKING,
        title: "Shorter thinking",
        summary: "reasoning effort → low".to_owned(),
        scope: None,
        writes: vec![(EFFORT_KEY, ConfigValue::String("low".to_owned()))],
        requirement: Requirement::met(),
        ticked: true,
        evidence: evidence_for(SHORTER_THINKING, arch, a8),
    }
}

/// The plan for `preset` on the GPU in `ctx` (spec §2.2). The #902 state comes
/// from `ctx.a8_prefill`; `env` says which sidecar files exist.
pub fn plan(preset: Preset, ctx: &ArchCtx, env: &PlanEnv) -> TuningPlan {
    let arch = ctx.gpu_arch.as_deref().unwrap_or("");
    let a8 = ctx.a8_prefill;
    let mut lines = Vec::new();
    let mut notes = Vec::new();
    let mut not_changed = standard_not_changed();
    match preset {
        Preset::Balanced => {}
        Preset::QualityFirst => {
            match prompt_line(ctx, arch, a8, "accurate", PROMPT_PROCESSING) {
                Some(line) => lines.push(line),
                None if arch == "gfx1151" && a8 == A8Prefill::DefaultOn => {
                    not_changed.insert(
                        0,
                        NotChanged {
                            what: "prompt processing".to_owned(),
                            why: "already Accurate (8-bit) by default on gfx1151".to_owned(),
                        },
                    );
                }
                None => {}
            }
            if matches!(arch, "gfx1151" | "gfx1201" | "gfx1100") {
                lines.push(experts_line(arch, a8));
            }
            if arch == "gfx1151" {
                lines.push(ple_line(arch, a8, env));
            }
            if !measured_arch(arch) {
                notes.push(format!(
                    "no measured quality switches for {} yet",
                    if arch.is_empty() { "this GPU" } else { arch }
                ));
            }
        }
        Preset::LowPower => {
            if let Some(line) = prompt_line(ctx, arch, a8, "fast", A8_TO_IU4) {
                lines.push(line);
            }
            lines.push(thinking_line(arch, a8));
        }
    }
    TuningPlan {
        preset,
        arch: ctx.gpu_arch.clone(),
        a8,
        lines,
        notes,
        not_changed,
    }
}

// ------------------------------------------------------------------ state

/// A key's value before an action: `None` = the key was absent.
pub type Prior = Option<ConfigValue>;

/// What the applied preset wrote, with the values to restore when it leaves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Applied {
    pub preset: Preset,
    pub arch: String,
    pub applied_at: u64,
    /// Keys this preset wrote (ownership), exact values.
    #[serde(default)]
    pub set: BTreeMap<String, ConfigValue>,
    /// Per owned key: what to restore when the preset is reverted (the value
    /// before any preset owned it, or unset).
    #[serde(default, with = "prior_map")]
    pub base: BTreeMap<String, Prior>,
}

/// Restores the state before the last tuning action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Undo {
    pub previous_preset: Preset,
    /// Per touched key: its value before the action.
    #[serde(default, with = "prior_map")]
    pub prior: BTreeMap<String, Prior>,
    /// Per touched key: its value right after the action (undo only restores
    /// a key that still has this value).
    #[serde(default, with = "prior_map")]
    pub after: BTreeMap<String, Prior>,
    /// Ownership before the action, restored by undo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<Applied>,
}

/// Contents of `~/.hipfire/tuning.toml`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TuningState {
    pub schema_version: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub applied: Option<Applied>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub undo: Option<Undo>,
}

impl Default for TuningState {
    fn default() -> Self {
        Self {
            schema_version: TUNING_SCHEMA_VERSION,
            applied: None,
            undo: None,
        }
    }
}

impl TuningState {
    /// Nothing recorded (no preset owns keys, nothing to undo).
    pub fn is_empty(&self) -> bool {
        self.applied.is_none() && self.undo.is_none()
    }

    /// The preset in effect (`Balanced` when none is recorded).
    pub fn preset(&self) -> Preset {
        self.applied
            .as_ref()
            .map_or(Preset::Balanced, |applied| applied.preset)
    }

    pub fn from_toml(raw: &str, origin: &Path) -> Result<Self> {
        let state: Self = toml::from_str(raw).map_err(|source| ConfigError::Parse {
            path: origin.to_owned(),
            message: source.to_string(),
        })?;
        if state.schema_version != TUNING_SCHEMA_VERSION {
            return Err(ConfigError::Parse {
                path: origin.to_owned(),
                message: format!("unsupported schema_version {}", state.schema_version),
            });
        }
        Ok(state)
    }

    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|source| ConfigError::Parse {
            path: "tuning.toml".into(),
            message: source.to_string(),
        })
    }

    /// Read `tuning.toml`; a missing file is the empty state.
    pub fn load(paths: &ConfigPaths) -> Result<Self> {
        match std::fs::read_to_string(&paths.tuning_toml) {
            Ok(raw) => Self::from_toml(&raw, &paths.tuning_toml),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Read {
                path: paths.tuning_toml.clone(),
                source,
            }),
        }
    }
}

/// `{ unset = true }` / `{ value = … }` encoding of a [`Prior`].
mod prior_map {
    use super::{ConfigValue, Prior};
    use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    #[derive(Serialize, Deserialize)]
    struct Repr {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unset: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<ConfigValue>,
    }

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<String, Prior>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        map.iter()
            .map(|(key, prior)| {
                let repr = match prior {
                    Some(value) => Repr {
                        unset: None,
                        value: Some(value.clone()),
                    },
                    None => Repr {
                        unset: Some(true),
                        value: None,
                    },
                };
                (key.as_str(), repr)
            })
            .collect::<BTreeMap<_, _>>()
            .serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<String, Prior>, D::Error> {
        BTreeMap::<String, Repr>::deserialize(deserializer)?
            .into_iter()
            .map(|(key, repr)| match (repr.unset, repr.value) {
                (Some(true), None) => Ok((key, None)),
                (_, Some(value)) => Ok((key, Some(value))),
                (_, None) => Err(D::Error::custom(format!(
                    "{key}: expected `unset = true` or `value = …`"
                ))),
            })
            .collect()
    }
}

/// Write the new `config.toml` and `tuning.toml` (spec §2.5 order): the state
/// goes to a temp file, then `config.toml` is written, then the state is
/// renamed into place. An empty state removes `tuning.toml`.
pub fn commit(paths: &ConfigPaths, layer: &ConfigLayer, state: &TuningState) -> Result<()> {
    let tmp = paths
        .tuning_toml
        .with_extension(format!("tmp.{}", std::process::id()));
    let write_err = |path: &Path, source| ConfigError::Write {
        path: path.to_owned(),
        source,
    };
    if !state.is_empty() {
        let rendered = state.to_toml_string()?;
        if let Some(parent) = tmp.parent() {
            std::fs::create_dir_all(parent).map_err(|source| write_err(parent, source))?;
        }
        std::fs::write(&tmp, rendered).map_err(|source| write_err(&tmp, source))?;
    }
    if let Err(error) = write_global_toml(paths, layer) {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    if state.is_empty() {
        match std::fs::remove_file(&paths.tuning_toml) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => return Err(write_err(&paths.tuning_toml, source)),
        }
    } else {
        std::fs::rename(&tmp, &paths.tuning_toml)
            .map_err(|source| write_err(&paths.tuning_toml, source))?;
    }
    Ok(())
}

// ------------------------------------------------------------ apply / undo

/// Result of [`apply`]: the new layer and state, and what changed.
#[derive(Clone, Debug, PartialEq)]
pub struct ApplyOutcome {
    pub layer: ConfigLayer,
    pub state: TuningState,
    /// Net key-level change against the layer passed in.
    pub net: ConfigLayerDiff,
    /// Keys of the old preset left alone because you edited them since.
    pub kept: Vec<String>,
    /// Keys the old preset owned that were given back.
    pub reverted: Vec<String>,
    /// Keys the new preset wrote.
    pub written: Vec<String>,
    /// False when nothing would change and ownership is the same: the state
    /// (and its undo) are returned untouched.
    pub changed: bool,
}

/// Apply `plan` with the lines in `ticked`: revert what the applied preset
/// still owns, then merge the new writes (spec §2.5). Balanced is an empty
/// plan.
pub fn apply(
    global: &ConfigLayer,
    state: &TuningState,
    plan: &TuningPlan,
    ticked: &[&str],
    now: u64,
) -> Result<ApplyOutcome> {
    let mut layer = global.clone();
    let mut kept = Vec::new();
    let mut reverted = Vec::new();
    let mut carried_base: BTreeMap<String, Prior> = BTreeMap::new();
    if let Some(applied) = &state.applied {
        for (key, written) in &applied.set {
            if layer.values.get(key) == Some(written) {
                match applied.base.get(key).cloned().flatten() {
                    Some(value) => {
                        layer.values.insert(key.clone(), value);
                    }
                    None => {
                        layer.values.remove(key);
                    }
                }
                carried_base.insert(key.clone(), applied.base.get(key).cloned().flatten());
                reverted.push(key.clone());
            } else {
                kept.push(key.clone());
            }
        }
    }

    let writes = if plan.preset == Preset::Balanced {
        BTreeMap::new()
    } else {
        plan.writes_for(ticked)
    };
    let mut set = BTreeMap::new();
    let mut base = BTreeMap::new();
    for (key, value) in &writes {
        let prior = match carried_base.get(key) {
            Some(prior) => prior.clone(),
            None => global.values.get(key).cloned(),
        };
        layer.set(key, value.clone())?;
        // `set` canonicalises; the plan's keys are already canonical.
        set.insert(key.clone(), value.clone());
        base.insert(key.clone(), prior);
    }
    kept.retain(|key| !writes.contains_key(key));
    let written: Vec<String> = writes.keys().cloned().collect();

    let net = diff_config_layers(global, &layer);
    let same_ownership = match &state.applied {
        Some(applied) => applied.preset == plan.preset && applied.set == set,
        None => plan.preset == Preset::Balanced && set.is_empty(),
    };
    if net.is_empty() && same_ownership {
        return Ok(ApplyOutcome {
            layer,
            state: state.clone(),
            net,
            kept,
            reverted: Vec::new(),
            written: Vec::new(),
            changed: false,
        });
    }

    let touched: BTreeSet<&String> = reverted.iter().chain(writes.keys()).collect();
    let prior = touched
        .iter()
        .map(|key| ((*key).clone(), global.values.get(*key).cloned()))
        .collect();
    let after = touched
        .iter()
        .map(|key| ((*key).clone(), layer.values.get(*key).cloned()))
        .collect();
    let new_state = TuningState {
        schema_version: TUNING_SCHEMA_VERSION,
        applied: Some(Applied {
            preset: plan.preset,
            arch: plan.arch.clone().unwrap_or_default(),
            applied_at: now,
            set,
            base,
        }),
        undo: Some(Undo {
            previous_preset: state.preset(),
            prior,
            after,
            previous: state.applied.clone(),
        }),
    };
    Ok(ApplyOutcome {
        layer,
        state: new_state,
        net,
        kept,
        reverted,
        written,
        changed: true,
    })
}

/// Result of [`undo`].
#[derive(Clone, Debug, PartialEq)]
pub struct UndoOutcome {
    pub layer: ConfigLayer,
    pub state: TuningState,
    pub net: ConfigLayerDiff,
    /// Touched keys you edited since the action; left as they are.
    pub kept: Vec<String>,
    /// The preset in effect after the undo.
    pub restored_preset: Preset,
}

/// Undo the last tuning action: every touched key that still has the value the
/// action left goes back to its prior value (or is removed). Single level:
/// `None` means there is nothing to undo.
pub fn undo(global: &ConfigLayer, state: &TuningState) -> Option<UndoOutcome> {
    let record = state.undo.as_ref()?;
    let mut layer = global.clone();
    let mut kept = Vec::new();
    for (key, prior) in &record.prior {
        let expected = record.after.get(key).cloned().flatten();
        if global.values.get(key) != expected.as_ref() {
            kept.push(key.clone());
            continue;
        }
        match prior {
            Some(value) => {
                layer.values.insert(key.clone(), value.clone());
            }
            None => {
                layer.values.remove(key);
            }
        }
    }
    let net = diff_config_layers(global, &layer);
    let new_state = TuningState {
        schema_version: TUNING_SCHEMA_VERSION,
        applied: record.previous.clone(),
        undo: None,
    };
    Some(UndoOutcome {
        layer,
        restored_preset: new_state.preset(),
        state: new_state,
        net,
        kept,
    })
}

// ------------------------------------------------------- preview / checks

/// A setting that will override or shadow what the preset writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conflict {
    /// An environment variable sets the key; it wins over config.toml.
    Env { key: String, var: String },
    /// A `models.toml` entry sets the key for one model.
    PerModel { model: String, key: String },
}

/// What applying a plan would do (the preview screen / `show`).
#[derive(Clone, Debug, PartialEq)]
pub struct TuningPreview {
    pub net: ConfigLayerDiff,
    pub kept: Vec<String>,
    pub reverted: Vec<String>,
    pub written: Vec<String>,
    /// Filled by [`find_conflicts`]; [`preview`] cannot see the environment.
    pub conflicts: Vec<Conflict>,
}

/// Dry-run of [`apply`]. Conflicts are left empty; see [`find_conflicts`].
pub fn preview(
    global: &ConfigLayer,
    state: &TuningState,
    plan: &TuningPlan,
    ticked: &[&str],
) -> Result<TuningPreview> {
    let outcome = apply(global, state, plan, ticked, 0)?;
    Ok(TuningPreview {
        net: outcome.net,
        kept: outcome.kept,
        reverted: outcome.reverted,
        written: outcome.written,
        conflicts: Vec::new(),
    })
}

/// Environment variable that sets `key`, as `origin_of` names it.
fn env_var_for(key: &str) -> Option<String> {
    field(key)
        .and_then(|schema| schema.env_compat.map(str::to_owned))
        .or_else(|| developer_env_for_key(key))
}

/// Conflicts for the keys the ticked lines write: the environment layer sets
/// one (it wins over config.toml) or a `models.toml` entry sets one (it wins
/// for that model). The running-serve check belongs to the caller.
pub fn find_conflicts(
    plan: &TuningPlan,
    ticked: &[&str],
    env: &ConfigLayer,
    catalog: &ModelCatalog,
) -> Vec<Conflict> {
    let mut conflicts = Vec::new();
    for key in plan.writes_for(ticked).keys() {
        if env.get(key).is_some() {
            conflicts.push(Conflict::Env {
                key: key.clone(),
                var: env_var_for(key).unwrap_or_else(|| key.clone()),
            });
        }
    }
    for (model, record) in &catalog.models {
        for key in plan.writes_for(ticked).keys() {
            if record.overrides.get(key).is_some() {
                conflicts.push(Conflict::PerModel {
                    model: model.clone(),
                    key: key.clone(),
                });
            }
        }
    }
    conflicts
}

// ----------------------------------------------------------------- status

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuningStatus {
    Balanced,
    Preset(Preset),
    PresetEdited(Preset),
    Custom,
}

impl TuningStatus {
    /// `Balanced`, `Quality-first`, `Quality-first (edited)`, `Custom`.
    pub fn label(self) -> String {
        match self {
            Self::Balanced => "Balanced".to_owned(),
            Self::Preset(preset) => preset.title().to_owned(),
            Self::PresetEdited(preset) => format!("{} (edited)", preset.title()),
            Self::Custom => "Custom".to_owned(),
        }
    }
}

/// Where the global config stands against the tuning state (spec §2.5):
/// `P` when every owned key still has the value it wrote, `P (edited)` when
/// one differs or is gone, `Custom` when no preset owns anything but a key a
/// preset writes holds a value that is neither its default nor what any
/// preset writes for it on this arch (a hand-set `reasoning.effort = "low"`
/// is still `Balanced`; `"high"` is `Custom`), `Balanced` otherwise. `ctx`
/// picks the arch whose keys count.
pub fn status(global: &ConfigLayer, state: &TuningState, ctx: &ArchCtx) -> TuningStatus {
    if let Some(applied) = &state.applied {
        if !applied.set.is_empty() {
            let intact = applied
                .set
                .iter()
                .all(|(key, value)| global.values.get(key) == Some(value));
            return if intact {
                TuningStatus::Preset(applied.preset)
            } else {
                TuningStatus::PresetEdited(applied.preset)
            };
        }
    }
    let env = PlanEnv::everything();
    let mut written: BTreeMap<String, Vec<ConfigValue>> = BTreeMap::new();
    for preset in [Preset::QualityFirst, Preset::LowPower] {
        let plan = plan(preset, ctx, &env);
        for (key, value) in plan.writes_for(&plan.default_ticked()) {
            written.entry(key).or_default().push(value);
        }
    }
    let custom = written
        .iter()
        .any(|(key, values)| match global.values.get(key) {
            None => false,
            Some(value) => {
                field(key).is_none_or(|schema| schema.default.to_value() != *value)
                    && !values.contains(value)
            }
        });
    if custom {
        TuningStatus::Custom
    } else {
        TuningStatus::Balanced
    }
}

/// Origin for the "From" column: [`Origin::Preset`] when the applied preset
/// owns `key` and config.toml still holds exactly what it wrote, otherwise
/// whatever [`origin_of`] says.
pub fn attribute(
    key: &str,
    resolved: &ResolvedConfig,
    effective: &Effective,
    ctx: &ArchCtx,
    state: &TuningState,
    profile: Option<(&str, &ConfigLayer)>,
) -> Origin {
    let canonical = canonical_config_key(key).unwrap_or_else(|| key.to_owned());
    if let (Some(applied), Some(slot)) = (&state.applied, resolved.get(&canonical)) {
        if let Some(owned) = applied.set.get(&canonical) {
            if matches!(slot.source, ConfigSource::GlobalUser { .. }) && slot.value == *owned {
                return Origin::Preset;
            }
        }
    }
    origin_of(key, resolved, effective, ctx, profile)
}

// ----------------------------------------------------------- presentation

/// A key's current value as the layer has it, else its schema default;
/// `None` for an unset key without a default (developer keys).
pub fn current_value(global: &ConfigLayer, key: &str) -> Option<ConfigValue> {
    global
        .get(key)
        .cloned()
        .or_else(|| field(key).map(|schema| schema.default.to_value()))
}

/// `"q8"`, `false`, `unset`.
pub fn display_value(value: Option<&ConfigValue>) -> String {
    value.map_or_else(|| "unset".to_owned(), ConfigValue::to_toml_string)
}

/// The slowest-applying scope among `keys` (developer keys count as a serve
/// restart). `None` for no keys.
pub fn takes_effect<'a>(keys: impl IntoIterator<Item = &'a str>) -> Option<Applies> {
    let rank = |applies: Applies| match applies {
        Applies::NextMessage => 0,
        Applies::NewChat => 1,
        Applies::NextModelLoad => 2,
        Applies::ServeRestart => 3,
    };
    keys.into_iter()
        .map(|key| field(key).map_or(Applies::ServeRestart, |s| Applies::from_scope(s.scope)))
        .max_by_key(|applies| rank(*applies))
}

fn kld(value: f64) -> String {
    format!("{value:.4}")
}

/// One-line trade-off summary: `43% closer (WT2 KLD 0.0695 -> 0.0394),
/// prompts 1.6× slower, typing unchanged`. Unknowns are said, not guessed.
pub fn evidence_summary(row: &EvidenceRow) -> String {
    let mut parts = Vec::new();
    match (format_closer(row), row.kld_from, row.kld_to) {
        (Some(closer), Some(from), Some(to)) => parts.push(format!(
            "{closer} ({} {} -> {})",
            row.kld_metric.unwrap_or("KLD"),
            kld(from),
            kld(to)
        )),
        _ => parts.push("quality gain: not measured".to_owned()),
    }
    match format_slower(row) {
        Some(slower) => parts.push(format!("prompts {slower}")),
        None => parts.push("prompt speed: not measured".to_owned()),
    }
    match row.typing {
        Some(typing) if typing.starts_with("unchanged") => {
            parts.push("typing unchanged".to_owned())
        }
        Some(typing) => parts.push(format!("typing {typing}")),
        None => {}
    }
    parts.join(", ")
}

fn json_value(value: Option<&ConfigValue>) -> serde_json::Value {
    value.map_or(serde_json::Value::Null, |value| {
        serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
    })
}

/// `lines[]` of the `show --json` document: id, keys, from, to, ticked,
/// requirement and evidence. `from` is the current value (schema default when
/// unset, `null` when there is none), `to` what the line writes.
pub fn lines_json(plan: &TuningPlan, ticked: &[&str], global: &ConfigLayer) -> serde_json::Value {
    let lines = plan
        .lines
        .iter()
        .map(|line| {
            let keys: Vec<&str> = line.writes.iter().map(|(key, _)| *key).collect();
            let from: serde_json::Map<_, _> = line
                .writes
                .iter()
                .map(|(key, _)| {
                    (
                        (*key).to_owned(),
                        json_value(current_value(global, key).as_ref()),
                    )
                })
                .collect();
            let to: serde_json::Map<_, _> = line
                .writes
                .iter()
                .map(|(key, value)| ((*key).to_owned(), json_value(Some(value))))
                .collect();
            let evidence = line.evidence.map(|row| {
                serde_json::json!({
                    "kld_from": row.kld_from,
                    "kld_to": row.kld_to,
                    "metric": row.kld_metric,
                    "speed_from": row.speed_from,
                    "speed_to": row.speed_to,
                    "speed_metric": row.speed_metric,
                    "closer": format_closer(row),
                    "slower": format_slower(row),
                    "typing": row.typing,
                    "caveat": row.caveat,
                    "source": row.source,
                })
            });
            serde_json::json!({
                "id": line.id,
                "title": line.title,
                "summary": line.summary,
                "scope": line.scope,
                "keys": keys,
                "from": from,
                "to": to,
                "ticked": line.requirement.met && ticked.contains(&line.id),
                "requirement": {
                    "met": line.requirement.met,
                    "reason": line.requirement.reason,
                    "fix": line.requirement.fix,
                },
                "evidence": evidence,
            })
        })
        .collect::<Vec<_>>();
    serde_json::Value::Array(lines)
}

/// `net_diff` of the `show --json` document.
pub fn diff_json(diff: &ConfigLayerDiff) -> serde_json::Value {
    serde_json::json!({
        "added": diff.added.iter()
            .map(|(key, value)| serde_json::json!({ "key": key, "value": json_value(Some(value)) }))
            .collect::<Vec<_>>(),
        "changed": diff.changed.iter()
            .map(|(key, from, to)| serde_json::json!({
                "key": key,
                "from": json_value(Some(from)),
                "to": json_value(Some(to)),
            }))
            .collect::<Vec<_>>(),
        "removed": diff.removed.iter()
            .map(|(key, value)| serde_json::json!({ "key": key, "value": json_value(Some(value)) }))
            .collect::<Vec<_>>(),
    })
}

/// `conflicts` of the `show --json` document.
pub fn conflicts_json(conflicts: &[Conflict]) -> serde_json::Value {
    conflicts
        .iter()
        .map(|conflict| match conflict {
            Conflict::Env { key, var } => {
                serde_json::json!({ "kind": "env", "key": key, "var": var })
            }
            Conflict::PerModel { model, key } => {
                serde_json::json!({ "kind": "per-model", "model": model, "key": key })
            }
        })
        .collect()
}

/// Names of the evidence rows a plan cites (for tests and docs tooling).
pub fn evidence_lines(plan: &TuningPlan) -> Vec<&'static str> {
    plan.lines
        .iter()
        .filter(|line| line.evidence.is_some())
        .map(|line| line.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_typed_bool;

    fn ctx(arch: &str, a8: A8Prefill) -> ArchCtx {
        ArchCtx {
            gpu_arch: Some(arch.to_owned()),
            gpu_count: 1,
            model_arch_id: None,
            registry_tag: Some("qwen3.8:27b".to_owned()),
            a8_prefill: a8,
        }
    }

    fn env_ok() -> PlanEnv {
        PlanEnv {
            ple_q8_present: true,
            flash_next_installed: true,
            ..PlanEnv::default()
        }
    }

    fn keys(plan: &TuningPlan) -> Vec<&'static str> {
        plan.keys().into_iter().collect()
    }

    fn sorted(list: &[&'static str]) -> Vec<&'static str> {
        let mut list = list.to_vec();
        list.sort_unstable();
        list
    }

    fn temp_paths(label: &str) -> ConfigPaths {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hipfire-tuning-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        ConfigPaths::under(root)
    }

    fn layer(pairs: &[(&str, ConfigValue)]) -> ConfigLayer {
        let mut layer = ConfigLayer::default();
        for (key, value) in pairs {
            layer.set(key, value.clone()).unwrap();
        }
        layer
    }

    #[test]
    fn preset_names_parse_with_the_quality_alias() {
        assert_eq!(Preset::parse("balanced"), Some(Preset::Balanced));
        assert_eq!(Preset::parse("quality-first"), Some(Preset::QualityFirst));
        assert_eq!(Preset::parse("quality"), Some(Preset::QualityFirst));
        assert_eq!(Preset::parse("Low-Power"), Some(Preset::LowPower));
        assert_eq!(Preset::parse("fast"), None);
    }

    #[test]
    fn plan_key_sets_match_the_spec_table() {
        use A8Prefill::{DefaultOn, NotLanded, OptIn};
        let sym = SYM_IU4_KEY;
        let cases: &[(&str, A8Prefill, &[&str], &[&str])] = &[
            // (arch, a8, Quality-first keys, Low-power keys)
            (
                "gfx1151",
                NotLanded,
                &["kernel.iu4_prefill", sym, "memory.ple_format"],
                &[EFFORT_KEY],
            ),
            (
                "gfx1151",
                OptIn,
                &["kernel.gfx11_a8_prefill", sym, "memory.ple_format"],
                &[EFFORT_KEY],
            ),
            (
                "gfx1151",
                DefaultOn,
                &[sym, "memory.ple_format"],
                &["kernel.gfx11_a8_prefill", EFFORT_KEY],
            ),
            (
                "gfx1201",
                NotLanded,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            (
                "gfx1201",
                OptIn,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            (
                "gfx1201",
                DefaultOn,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            (
                "gfx1100",
                NotLanded,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            (
                "gfx1100",
                OptIn,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            (
                "gfx1100",
                DefaultOn,
                &["kernel.iu4_prefill", sym],
                &[EFFORT_KEY],
            ),
            ("gfx1030", NotLanded, &[], &[EFFORT_KEY]),
            ("gfx1030", OptIn, &[], &[EFFORT_KEY]),
            ("gfx1030", DefaultOn, &[], &[EFFORT_KEY]),
        ];
        for (arch, a8, quality, low_power) in cases {
            let context = ctx(arch, *a8);
            let q = plan(Preset::QualityFirst, &context, &env_ok());
            let lp = plan(Preset::LowPower, &context, &env_ok());
            assert_eq!(keys(&q), sorted(quality), "quality-first {arch} {a8:?}");
            assert_eq!(keys(&lp), sorted(low_power), "low-power {arch} {a8:?}");
            assert!(plan(Preset::Balanced, &context, &env_ok()).lines.is_empty());
        }
    }

    #[test]
    fn opt_in_quality_first_writes_the_a8_key_true() {
        let q = plan(
            Preset::QualityFirst,
            &ctx("gfx1151", A8Prefill::OptIn),
            &env_ok(),
        );
        let line = q.line(PROMPT_PROCESSING).unwrap();
        assert_eq!(
            line.writes,
            vec![("kernel.gfx11_a8_prefill", ConfigValue::Bool(true))]
        );
        let lp = plan(
            Preset::LowPower,
            &ctx("gfx1151", A8Prefill::DefaultOn),
            &env_ok(),
        );
        assert_eq!(
            lp.line(PROMPT_PROCESSING).unwrap().writes,
            vec![("kernel.gfx11_a8_prefill", ConfigValue::Bool(false))]
        );
    }

    #[test]
    fn unknown_arch_gets_only_neutral_lines_and_a_note() {
        let q = plan(
            Preset::QualityFirst,
            &ctx("gfx1030", A8Prefill::NotLanded),
            &env_ok(),
        );
        assert!(q.lines.is_empty());
        assert_eq!(
            q.notes,
            vec!["no measured quality switches for gfx1030 yet"]
        );
        let lp = plan(
            Preset::LowPower,
            &ctx("gfx1030", A8Prefill::NotLanded),
            &env_ok(),
        );
        assert_eq!(lp.default_ticked(), vec![SHORTER_THINKING]);
    }

    #[test]
    fn sym_iu4_line_writes_a_typed_false_that_round_trips() {
        let q = plan(
            Preset::QualityFirst,
            &ctx("gfx1201", A8Prefill::NotLanded),
            &env_ok(),
        );
        let line = q.line(FLASH_NEXT_EXPERTS).unwrap();
        assert_eq!(line.writes, vec![(SYM_IU4_KEY, ConfigValue::Bool(false))]);
        let written = ConfigValue::Bool(false).to_toml_string();
        assert_eq!(written, "false");
        assert_eq!(parse_typed_bool(&written), Some(false));
    }

    #[test]
    fn word_table_needs_the_q8_file_and_says_how_to_get_it() {
        let context = ctx("gfx1151", A8Prefill::NotLanded);
        let env = PlanEnv {
            ple_q8_present: false,
            flash_next_installed: true,
            ple_q8_missing: vec!["qwen3.8-flash-next-gptq3".to_owned()],
            pull_tag: Some("qwen3.8:flash-next".to_owned()),
        };
        let q = plan(Preset::QualityFirst, &context, &env);
        let line = q.line("flash-next-ple").unwrap();
        assert!(!line.ticked);
        assert!(!line.requirement.met);
        assert!(line
            .requirement
            .reason
            .as_deref()
            .unwrap()
            .contains("qwen3.8-flash-next-gptq3 has no .ple-q8"));
        assert_eq!(
            line.requirement.fix.as_deref(),
            Some("hipfire pull qwen3.8:flash-next --ple q8")
        );
        // An unmet line is never written, even when named as ticked.
        let writes = q.writes_for(&["flash-next-ple", PROMPT_PROCESSING]);
        assert!(!writes.contains_key("memory.ple_format"));
        assert!(writes.contains_key("kernel.iu4_prefill"));
        let met = plan(Preset::QualityFirst, &context, &env_ok());
        assert!(met.line("flash-next-ple").unwrap().ticked);
    }

    #[test]
    fn lines_cite_the_evidence_table() {
        let q = plan(
            Preset::QualityFirst,
            &ctx("gfx1151", A8Prefill::OptIn),
            &env_ok(),
        );
        let row = q.line(PROMPT_PROCESSING).unwrap().evidence.unwrap();
        assert_eq!(format_closer(row).as_deref(), Some("43% closer"));
        assert_eq!(evidence_lines(&q), LINE_IDS[..3].to_vec());
        assert!(evidence_summary(row).starts_with("43% closer (WT2 KLD 0.0695 -> 0.0394)"));
    }

    #[test]
    fn merge_keeps_unrelated_keys_and_balanced_only_removes() {
        let global = layer(&[
            ("generation.max_tokens", ConfigValue::Integer(1234)),
            ("memory.max_seq", ConfigValue::Integer(4096)),
        ]);
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let ticked = q.default_ticked();
        let applied = apply(&global, &TuningState::default(), &q, &ticked, 10).unwrap();
        assert_eq!(
            applied.layer.get("generation.max_tokens"),
            Some(&ConfigValue::Integer(1234))
        );
        assert_eq!(
            applied.layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(4096))
        );
        assert_eq!(applied.net.added.len(), 2);

        let balanced = plan(Preset::Balanced, &context, &env_ok());
        let back = apply(&applied.layer, &applied.state, &balanced, &[], 11).unwrap();
        assert!(back.net.added.is_empty());
        assert!(back.net.changed.is_empty());
        assert_eq!(back.net.removed.len(), 2);
        assert_eq!(back.layer, global);
        assert_eq!(
            status(&back.layer, &back.state, &context),
            TuningStatus::Balanced
        );
    }

    #[test]
    fn apply_then_undo_leaves_config_toml_bytes_identical() {
        let paths = temp_paths("roundtrip");
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let seed = layer(&[
            ("generation.max_tokens", ConfigValue::Integer(1234)),
            ("kernel.iu4_prefill", ConfigValue::Bool(true)),
        ]);
        write_global_toml(&paths, &seed).unwrap();
        let before = std::fs::read(&paths.config_toml).unwrap();

        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let state = TuningState::load(&paths).unwrap();
        assert!(state.is_empty());
        let global = crate::load_global(&paths).unwrap().layer;
        let applied = apply(&global, &state, &q, &q.default_ticked(), 100).unwrap();
        commit(&paths, &applied.layer, &applied.state).unwrap();
        assert_ne!(std::fs::read(&paths.config_toml).unwrap(), before);
        assert!(paths.tuning_toml.exists());

        let state = TuningState::load(&paths).unwrap();
        assert_eq!(state, applied.state);
        assert_eq!(
            status(&applied.layer, &state, &context),
            TuningStatus::Preset(Preset::QualityFirst)
        );
        let global = crate::load_global(&paths).unwrap().layer;
        let undone = undo(&global, &state).unwrap();
        assert!(undone.kept.is_empty());
        assert_eq!(undone.restored_preset, Preset::Balanced);
        commit(&paths, &undone.layer, &undone.state).unwrap();
        assert_eq!(std::fs::read(&paths.config_toml).unwrap(), before);
        assert!(!paths.tuning_toml.exists());
        // Single level.
        assert!(undo(&undone.layer, &undone.state).is_none());
        std::fs::remove_dir_all(&paths.root).unwrap();
    }

    #[test]
    fn switching_presets_reverts_owned_keys_and_keeps_hand_edits() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let global = layer(&[("generation.max_tokens", ConfigValue::Integer(1234))]);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let first = apply(&global, &TuningState::default(), &q, &q.default_ticked(), 1).unwrap();

        // Hand-edit one owned key (sym IU4 back to true).
        let mut edited = first.layer.clone();
        edited.set(SYM_IU4_KEY, ConfigValue::Bool(true)).unwrap();
        assert_eq!(
            status(&edited, &first.state, &context),
            TuningStatus::PresetEdited(Preset::QualityFirst)
        );

        let lp = plan(Preset::LowPower, &context, &env_ok());
        let second = apply(&edited, &first.state, &lp, &lp.default_ticked(), 2).unwrap();
        assert_eq!(second.kept, vec![SYM_IU4_KEY.to_owned()]);
        assert_eq!(second.reverted, vec!["kernel.iu4_prefill".to_owned()]);
        assert_eq!(second.layer.get("kernel.iu4_prefill"), None);
        assert_eq!(
            second.layer.get(SYM_IU4_KEY),
            Some(&ConfigValue::Bool(true))
        );
        assert_eq!(
            second.layer.get(EFFORT_KEY),
            Some(&ConfigValue::String("low".into()))
        );
        assert_eq!(second.state.preset(), Preset::LowPower);
        assert_eq!(
            status(&second.layer, &second.state, &context),
            TuningStatus::Preset(Preset::LowPower)
        );
        // The kept key is yours now: it is not owned any more.
        assert!(!second
            .state
            .applied
            .as_ref()
            .unwrap()
            .set
            .contains_key(SYM_IU4_KEY));

        // Undo the switch: Quality-first's iu4 value returns and ownership
        // goes back to Quality-first.
        let undone = undo(&second.layer, &second.state).unwrap();
        assert_eq!(undone.restored_preset, Preset::QualityFirst);
        assert_eq!(
            undone.layer.get("kernel.iu4_prefill"),
            Some(&ConfigValue::Bool(false))
        );
        assert_eq!(undone.layer.get(EFFORT_KEY), None);
        assert_eq!(
            undone.layer.get(SYM_IU4_KEY),
            Some(&ConfigValue::Bool(true))
        );
    }

    #[test]
    fn reapplying_the_same_preset_is_a_no_op_that_keeps_undo() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let first = apply(
            &ConfigLayer::default(),
            &TuningState::default(),
            &q,
            &q.default_ticked(),
            1,
        )
        .unwrap();
        let again = apply(&first.layer, &first.state, &q, &q.default_ticked(), 2).unwrap();
        assert!(!again.changed);
        assert!(again.net.is_empty());
        assert_eq!(again.state, first.state);
        // Balanced with nothing recorded is also a no-op.
        let balanced = plan(Preset::Balanced, &context, &env_ok());
        let none = apply(
            &ConfigLayer::default(),
            &TuningState::default(),
            &balanced,
            &[],
            3,
        )
        .unwrap();
        assert!(!none.changed);
        assert!(none.state.is_empty());
    }

    #[test]
    fn a_preset_over_a_hand_set_key_restores_your_value_not_the_default() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let global = layer(&[("kernel.iu4_prefill", ConfigValue::Bool(true))]);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let on = apply(&global, &TuningState::default(), &q, &q.default_ticked(), 1).unwrap();
        // Re-apply with a line skipped: iu4 stays owned and its base survives.
        let narrower = apply(&on.layer, &on.state, &q, &[PROMPT_PROCESSING], 2).unwrap();
        assert_eq!(narrower.layer.get(SYM_IU4_KEY), None);
        let balanced = plan(Preset::Balanced, &context, &env_ok());
        let off = apply(&narrower.layer, &narrower.state, &balanced, &[], 3).unwrap();
        assert_eq!(off.layer, global);
    }

    #[test]
    fn status_reports_custom_when_plan_keys_are_hand_set() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let state = TuningState::default();
        assert_eq!(
            status(&ConfigLayer::default(), &state, &context),
            TuningStatus::Balanced
        );
        // A value equal to the schema default is not a deviation.
        let at_default = layer(&[("kernel.iu4_prefill", ConfigValue::Bool(true))]);
        assert_eq!(
            status(&at_default, &state, &context),
            TuningStatus::Balanced
        );
        // A hand-set value some preset writes is not a deviation either
        // (spec S1 mockup: `reasoning.effort = "low"` by hand reads Balanced).
        for preset_value in [
            layer(&[("kernel.iu4_prefill", ConfigValue::Bool(false))]),
            layer(&[("reasoning.effort", ConfigValue::String("low".into()))]),
        ] {
            assert_eq!(
                status(&preset_value, &state, &context),
                TuningStatus::Balanced
            );
        }
        let custom = layer(&[("reasoning.effort", ConfigValue::String("high".into()))]);
        assert_eq!(status(&custom, &state, &context), TuningStatus::Custom);
        assert_eq!(TuningStatus::Custom.label(), "Custom");
        assert_eq!(
            TuningStatus::PresetEdited(Preset::LowPower).label(),
            "Low-power (edited)"
        );
    }

    #[test]
    fn attribute_names_the_preset_only_while_the_value_matches() {
        use crate::effective::effective;
        use crate::{resolve, NamedLayer};
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let out = apply(
            &ConfigLayer::default(),
            &TuningState::default(),
            &q,
            &q.default_ticked(),
            1,
        )
        .unwrap();
        let global_source = |layer: ConfigLayer| NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        };
        let resolved = resolve([global_source(out.layer.clone())]).unwrap();
        let eff = effective("kernel.iu4_prefill", &resolved, &context);
        assert_eq!(
            attribute(
                "kernel.iu4_prefill",
                &resolved,
                &eff,
                &context,
                &out.state,
                None
            ),
            Origin::Preset
        );
        let mut edited = out.layer.clone();
        edited
            .set("kernel.iu4_prefill", ConfigValue::Bool(true))
            .unwrap();
        let resolved = resolve([global_source(edited)]).unwrap();
        let eff = effective("kernel.iu4_prefill", &resolved, &context);
        assert_eq!(
            attribute(
                "kernel.iu4_prefill",
                &resolved,
                &eff,
                &context,
                &out.state,
                None
            ),
            Origin::You
        );
    }

    #[test]
    fn conflicts_name_env_and_per_model_values() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let ticked = q.default_ticked();
        let env = layer(&[("kernel.iu4_prefill", ConfigValue::Bool(true))]);
        let mut catalog = ModelCatalog::default();
        let mut record = crate::LocalModelConfig::default();
        record
            .overrides
            .values
            .insert(EFFORT_KEY.to_owned(), ConfigValue::String("high".into()));
        catalog.models.insert("qwen3:8b".to_owned(), record);
        let found = find_conflicts(&q, &ticked, &env, &catalog);
        assert_eq!(
            found,
            vec![Conflict::Env {
                key: "kernel.iu4_prefill".into(),
                var: "HIPFIRE_IU4_PREFILL".into()
            }]
        );
        let lp = plan(Preset::LowPower, &context, &env_ok());
        let found = find_conflicts(&lp, &lp.default_ticked(), &ConfigLayer::default(), &catalog);
        assert_eq!(
            found,
            vec![Conflict::PerModel {
                model: "qwen3:8b".into(),
                key: EFFORT_KEY.into()
            }]
        );
    }

    #[test]
    fn tuning_state_round_trips_through_toml() {
        let context = ctx("gfx1201", A8Prefill::NotLanded);
        let global = layer(&[("kernel.iu4_prefill", ConfigValue::Bool(true))]);
        let q = plan(Preset::QualityFirst, &context, &env_ok());
        let first = apply(
            &global,
            &TuningState::default(),
            &q,
            &q.default_ticked(),
            1760630000,
        )
        .unwrap();
        let lp = plan(Preset::LowPower, &context, &env_ok());
        let second = apply(
            &first.layer,
            &first.state,
            &lp,
            &lp.default_ticked(),
            1760630001,
        )
        .unwrap();
        for state in [&first.state, &second.state] {
            let text = state.to_toml_string().unwrap();
            let parsed = TuningState::from_toml(&text, Path::new("tuning.toml")).unwrap();
            assert_eq!(&parsed, state, "{text}");
        }
        let text = first.state.to_toml_string().unwrap();
        assert!(text.contains("schema_version = 1"), "{text}");
        assert!(text.contains("preset = \"quality-first\""), "{text}");
        assert!(text.contains("unset = true"), "{text}");
        assert!(text.contains("value = true"), "{text}");
        assert!(TuningState::from_toml("schema_version = 9\n", Path::new("t")).is_err());
    }

    #[test]
    fn takes_effect_picks_the_slowest_scope() {
        assert_eq!(takes_effect([EFFORT_KEY]), Some(Applies::NextMessage));
        assert_eq!(
            takes_effect([EFFORT_KEY, "kernel.iu4_prefill"]),
            Some(Applies::ServeRestart)
        );
        assert_eq!(takes_effect([SYM_IU4_KEY]), Some(Applies::ServeRestart));
        assert_eq!(takes_effect([]), None);
    }
}
