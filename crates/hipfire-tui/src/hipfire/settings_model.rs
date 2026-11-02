// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! The Settings tab's row model: intent groups, "Using now" values, origins,
//! locks, search and filters (spec §3.1–3.5).
//!
//! Everything here consumes `hipfire_config` (`effective`, `intent`,
//! `evidence`) and `hipfire_registry::resolve_for_model`; no rule is
//! re-implemented. The model is pure data built from explicit inputs (paths,
//! active model, registry, [`ArchCtx`], env layer), so tests never read the
//! real `~/.hipfire` or the process environment.

use hipfire_config::tuning::{self, TuningState, TuningStatus};
use hipfire_config::{
    canonical_config_key, detect_config_profile,
    effective::{active_choice, quality_choices, ArchCtx, Choice, Effective, Origin},
    field, fields,
    intent::{intent_of, Intent},
    is_deprecated_key, is_developer_key, load_config_profile, load_global, model_override_allowed,
    ConfigCategory, ConfigField, ConfigLayer, ConfigPaths, ConfigSource, ConfigValue,
    ResolvedConfig,
};
use hipfire_registry::RegistryV1;

use super::config::{layers_label, offload_exec_label, passback_share_label};
use super::dashboard::LoadedConfig;
use super::knobs::{self, ExplainCtx, PROMPT_PROCESSING_ID};
use super::registry::RegistryState;
use super::search::{match_tier, row_matches};
use super::writer::SYM_IU4_KEY;
use super::HipfirePaths;

const IU4_KEY: &str = "kernel.iu4_prefill";
const A8_KEY: &str = "kernel.gfx11_a8_prefill";

/// The raw keys the Prompt processing virtual row replaces in Quality. They
/// stay editable from Advanced.
const RAW_PREFILL_KEYS: [&str; 2] = [IU4_KEY, A8_KEY];

/// Why a row cannot be edited for one model (spec §3.5).
pub const MODEL_SCOPE_LOCK: &str = "all models only (locked)";

/// One step of the "why this value" chain (spec §4 L1): the layer, the value
/// it held, and whether it is the one in use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainEntry {
    pub label: String,
    pub value: String,
    pub used: bool,
}

const NOT_USED_ON_GPU: &str = "— not used on this GPU";

/// A virtual row: one row whose options map to per-arch key writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtualRow {
    PromptProcessing,
}

/// What a row edits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowId {
    /// One canonical config key.
    Key(String),
    Virtual(VirtualRow),
}

impl RowId {
    /// The canonical key of a key row.
    pub fn key(&self) -> Option<&str> {
        match self {
            Self::Key(key) => Some(key),
            Self::Virtual(_) => None,
        }
    }

    /// Id into the curated knob table (`knobs::curated_knob`).
    pub fn knob_id(&self) -> &str {
        match self {
            Self::Key(key) => key,
            Self::Virtual(VirtualRow::PromptProcessing) => PROMPT_PROCESSING_ID,
        }
    }
}

/// One Settings row.
#[derive(Clone, Debug)]
pub struct SettingsRow {
    pub id: RowId,
    /// Plain title when `knobs.rs` has one, else the canonical key.
    pub title: String,
    pub group: Intent,
    /// The resolved layer value (`Null` when no layer, built-in included, has
    /// the key, e.g. an absent `developer.*` key). For a virtual row, the id of
    /// the active choice as a string, or `Null`.
    pub resolved: ConfigValue,
    pub effective: Effective,
    pub origin: Origin,
    /// How the value was reached, lowest priority first, the used entry last
    /// (the "why this value" pane). Empty when no layer has the key.
    pub chain: Vec<ChainEntry>,
    /// The "Using now" text.
    pub display: String,
    /// Why the row cannot be edited right now (an environment variable owns it).
    pub locked: Option<String>,
    pub deprecated: bool,
    /// Canonical keys behind the row (virtual: every key its choices write).
    pub keys: Vec<String>,
    /// Virtual rows only: the options for this GPU. Empty = no route here.
    pub choices: Vec<Choice>,
    /// Virtual rows only: the choice the global `config.toml` selects, the
    /// starting point for ←/→.
    pub current_choice: Option<usize>,
}

impl SettingsRow {
    /// The active model or this GPU never reads the row.
    pub fn is_unused(&self) -> bool {
        self.effective.used_by_model == Some(false) || !self.effective.used_on_gpu
    }

    /// Set by something other than the defaults or the model card (the ● mark).
    pub fn is_changed(&self) -> bool {
        self.origin.marked()
    }

    #[cfg(test)]
    pub fn is_virtual(&self) -> bool {
        matches!(self.id, RowId::Virtual(_))
    }

    /// Why this row cannot be edited for one model (it is process-wide), or
    /// `None` when a per-model value is allowed.
    pub fn model_scope_lock(&self) -> Option<&'static str> {
        self.keys
            .iter()
            .any(|key| model_override_allowed(key).is_err())
            .then_some(MODEL_SCOPE_LOCK)
    }
}

/// Search and filters applied over the rows (spec §3.4).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowFilter {
    pub search: String,
    pub changed_only: bool,
    pub hide_unused: bool,
}

impl RowFilter {
    pub fn matches(&self, row: &SettingsRow) -> bool {
        if self.changed_only && !row.is_changed() {
            return false;
        }
        if self.hide_unused && row.is_unused() {
            return false;
        }
        let keys: Vec<&str> = row.keys.iter().map(String::as_str).collect();
        row_matches(&row.title, &keys, &self.search)
    }
}

/// Every Settings row for one model on one GPU, in group order.
#[derive(Clone, Debug)]
pub struct SettingsModel {
    pub rows: Vec<SettingsRow>,
    pub arch: ArchCtx,
    /// Name of the profile whose bundle the global config contains, if any.
    pub profile: Option<String>,
    /// Where the global config stands against `tuning.toml` (header item).
    pub tuning: TuningStatus,
    /// `config.toml` / resolve warnings the screen can show.
    pub warnings: Vec<String>,
}

struct Builder<'a> {
    resolved: &'a ResolvedConfig,
    arch: &'a ArchCtx,
    profile: Option<(&'a str, &'a ConfigLayer)>,
    global: &'a ConfigLayer,
    /// Which keys the applied tuning preset owns (the From column's `preset`).
    tuning: &'a TuningState,
    model_label: String,
}

impl SettingsModel {
    /// Build every row. A config that does not load degrades to the built-in
    /// values with a warning instead of failing the screen.
    pub fn build(
        paths: &HipfirePaths,
        active_model: &str,
        registry: &RegistryV1,
        arch: &ArchCtx,
        env: &ConfigLayer,
    ) -> Self {
        let config_paths = ConfigPaths::under(&paths.root);
        let mut warnings = Vec::new();
        let resolved = match hipfire_registry::resolve_for_model(
            &config_paths,
            active_model,
            registry.model(active_model),
            env.clone(),
        ) {
            Ok(resolved) => {
                warnings.extend(resolved.warnings);
                resolved.resolved
            }
            Err(error) => {
                warnings.push(format!("config: {error}"));
                hipfire_config::resolve(Vec::new()).expect("built-in config schema resolves")
            }
        };
        let global = load_global(&config_paths)
            .map(|loaded| loaded.layer)
            .unwrap_or_default();
        let profile = detect_config_profile(&config_paths, &global).and_then(|name| {
            load_config_profile(&config_paths, &name)
                .ok()
                .map(|bundle| (name, bundle))
        });
        // An unreadable `tuning.toml` reads as no preset applied.
        let tuning_state = TuningState::load(&config_paths).unwrap_or_default();
        let tuning = tuning::status(&global, &tuning_state, arch);
        let model_label = short_model_name(arch.registry_tag.as_deref().unwrap_or(active_model));
        let builder = Builder {
            resolved: &resolved,
            arch,
            profile: profile
                .as_ref()
                .map(|(name, bundle)| (name.as_str(), bundle)),
            global: &global,
            tuning: &tuning_state,
            model_label,
        };

        let mut rows = Vec::new();
        for group in Intent::ALL {
            if group == Intent::Quality {
                rows.push(builder.prompt_processing_row());
            }
            for key in group_keys(group, &resolved) {
                rows.push(builder.key_row(&key, group));
            }
        }
        Self {
            rows,
            arch: arch.clone(),
            profile: profile.map(|(name, _)| name),
            tuning,
            warnings,
        }
    }

    /// Every row of `group`, unfiltered, in display order.
    pub fn rows_in(&self, group: Intent) -> impl Iterator<Item = &SettingsRow> {
        self.rows.iter().filter(move |row| row.group == group)
    }

    /// The rows of `group` that pass `filter` (search included).
    pub fn visible(&self, group: Intent, filter: &RowFilter) -> Vec<&SettingsRow> {
        self.rows_in(group)
            .filter(|row| filter.matches(row))
            .collect()
    }

    /// Rows of every group that pass `filter`: the search results view, where
    /// the Group column tells them apart. Title and key matches come first,
    /// then environment-variable matches, then help-text-only matches; group
    /// order holds within each tier.
    pub fn search(&self, filter: &RowFilter) -> Vec<&SettingsRow> {
        let mut hits: Vec<&SettingsRow> =
            self.rows.iter().filter(|row| filter.matches(row)).collect();
        hits.sort_by_key(|row| {
            let keys: Vec<&str> = row.keys.iter().map(String::as_str).collect();
            match_tier(&row.title, &keys, &filter.search)
        });
        hits
    }

    /// Rows per group after `filter`, in [`Intent::ALL`] order.
    pub fn group_counts(&self, filter: &RowFilter) -> [(Intent, usize); 6] {
        Intent::ALL.map(|group| {
            (
                group,
                self.rows_in(group)
                    .filter(|row| filter.matches(row))
                    .count(),
            )
        })
    }

    pub fn row(&self, id: &RowId) -> Option<&SettingsRow> {
        self.rows.iter().find(|row| &row.id == id)
    }

    /// The four-line explainer of `row` (spec §3.7): the curated entry, or
    /// the schema help for a row with none (the Advanced rows). `live` is the
    /// value serve reports for the resident model, `restart` is set when
    /// `config.toml` is newer than serve.
    pub fn explainer(&self, row: &SettingsRow, live: Option<&str>, restart: bool) -> [String; 4] {
        let ctx = ExplainCtx {
            arch: &self.arch,
            effective: &row.effective,
            origin: &row.origin,
            locked: row.locked.as_deref(),
            display: &row.display,
            live,
            restart,
        };
        match row.id.key() {
            Some(key) if knobs::curated_knob(key).is_none() => knobs::schema_explainer(key, &ctx),
            _ => knobs::explainer(row.id.knob_id(), &ctx).expect("a curated row has a knob entry"),
        }
    }
}

/// The registry the TUI already loaded, as the library type
/// `resolve_for_model` looks models up in. Local files outside the registry
/// are left out: they have no card.
pub fn registry_v1(state: &RegistryState) -> RegistryV1 {
    RegistryV1 {
        schema_version: 1,
        generated_at: state.generated_at.clone(),
        _comment: None,
        models: state
            .models
            .iter()
            .filter(|row| !row.entry.repo.is_empty())
            .map(|row| (row.tag.clone(), row.entry.clone()))
            .collect(),
        aliases: state.aliases.clone(),
    }
}

/// The GPU and model context for `active_model`: arch from `install.json`,
/// model fields from its registry card.
pub fn arch_ctx(root: &std::path::Path, registry: &RegistryV1, active_model: &str) -> ArchCtx {
    let card = registry.model(active_model);
    ArchCtx {
        gpu_arch: hipfire_config::effective::install_gpu_arch(root),
        gpu_count: 1,
        model_arch_id: card.and_then(|(_, entry)| entry.arch_id),
        registry_tag: card.map(|(tag, _)| tag.to_owned()),
        ..ArchCtx::default()
    }
}

/// `qwen3.8:27b` → `27B`; a tag with no size reads whole.
pub fn short_model_name(tag: &str) -> String {
    let size = tag
        .split_once(':')
        .map(|(_, rest)| rest.split('-').next().unwrap_or(rest));
    match size {
        Some(size)
            if size.len() > 1
                && size.to_ascii_lowercase().ends_with('b')
                && size[..size.len() - 1]
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.') =>
        {
            size.to_ascii_uppercase()
        }
        _ => tag.to_owned(),
    }
}

fn category_rank(category: ConfigCategory) -> u8 {
    match category {
        ConfigCategory::Generation => 0,
        ConfigCategory::Reasoning => 1,
        ConfigCategory::Hardware => 2,
        ConfigCategory::Kernel => 3,
        ConfigCategory::Memory => 4,
        ConfigCategory::Attention => 5,
        ConfigCategory::Speculation => 6,
        ConfigCategory::Vision => 7,
        ConfigCategory::Replay => 8,
        ConfigCategory::Fusions => 9,
        ConfigCategory::Prompt => 10,
        ConfigCategory::Serve => 11,
        ConfigCategory::Diagnostic => 12,
        ConfigCategory::Experimental => 13,
    }
}

/// Canonical keys of `group`'s key rows, in display order (the Prompt
/// processing virtual row is added by the caller).
fn group_keys(group: Intent, resolved: &ResolvedConfig) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    if group == Intent::Advanced {
        let mut live: Vec<&ConfigField> = fields()
            .iter()
            .filter(|schema| {
                !is_deprecated_key(schema.key)
                    && (intent_of(schema.key) == Intent::Advanced
                        || RAW_PREFILL_KEYS.contains(&schema.key))
            })
            .collect();
        live.sort_by_key(|schema| (category_rank(schema.category), schema.key));
        keys.extend(live.into_iter().map(|schema| schema.key.to_owned()));
        // `developer.*` keys exist only when some layer sets them; the curated
        // one lives in Quality.
        keys.extend(
            resolved
                .values
                .keys()
                .filter(|key| is_developer_key(key) && key.as_str() != SYM_IU4_KEY)
                .cloned(),
        );
        let mut deprecated: Vec<&ConfigField> = fields()
            .iter()
            .filter(|schema| {
                is_deprecated_key(schema.key) && intent_of(schema.key) == Intent::Advanced
            })
            .collect();
        deprecated.sort_by_key(|schema| (category_rank(schema.category), schema.key));
        keys.extend(deprecated.into_iter().map(|schema| schema.key.to_owned()));
        return keys;
    }

    let curated: Vec<&'static str> = group
        .curated_present()
        .into_iter()
        .filter(|key| !RAW_PREFILL_KEYS.contains(key))
        .collect();
    keys.extend(curated.iter().map(|key| (*key).to_owned()));
    let mut rest: Vec<&ConfigField> = fields()
        .iter()
        .filter(|schema| {
            intent_of(schema.key) == group
                && !curated.contains(&schema.key)
                && !RAW_PREFILL_KEYS.contains(&schema.key)
        })
        .collect();
    // Category-default keys, deprecated ones last (the existing Advanced rule).
    rest.sort_by_key(|schema| (is_deprecated_key(schema.key), schema.key));
    keys.extend(rest.into_iter().map(|schema| schema.key.to_owned()));
    keys
}

impl Builder<'_> {
    /// Which layer set `key`, with the applied tuning preset's keys read as
    /// [`Origin::Preset`].
    fn origin(&self, key: &str, effective: &Effective) -> Origin {
        tuning::attribute(
            key,
            self.resolved,
            effective,
            self.arch,
            self.tuning,
            self.profile,
        )
    }

    fn key_row(&self, key: &str, group: Intent) -> SettingsRow {
        let canonical = canonical_config_key(key).unwrap_or_else(|| key.to_owned());
        let resolved = self
            .resolved
            .get(&canonical)
            .map_or(ConfigValue::Null, |slot| slot.value.clone());
        let effective = hipfire_config::effective::effective(&canonical, self.resolved, self.arch);
        let origin = self.origin(&canonical, &effective);
        let display = self.display(&canonical, &effective);
        let locked = lock_text(&origin);
        let title = row_title(&canonical);
        let chain = self.chain(&canonical, &effective, &display);
        SettingsRow {
            id: RowId::Key(canonical.clone()),
            title,
            group,
            resolved,
            effective,
            origin,
            chain,
            display,
            locked,
            deprecated: is_deprecated_key(&canonical),
            keys: vec![canonical],
            choices: Vec::new(),
            current_choice: None,
        }
    }

    /// The Quality group's virtual Prompt processing row over
    /// `kernel.iu4_prefill` (and `kernel.gfx11_a8_prefill` once it is in the
    /// schema).
    fn prompt_processing_row(&self) -> SettingsRow {
        let choices = quality_choices(self.arch);
        let mut keys: Vec<String> = Vec::new();
        for choice in &choices {
            for (key, _) in &choice.writes {
                if !keys.iter().any(|known| known == key) {
                    keys.push((*key).to_owned());
                }
            }
        }
        if keys.is_empty() {
            keys.push(IU4_KEY.to_owned());
        }
        // The key the effective value and origin are read from: the first one
        // this build's schema has.
        let primary = keys
            .iter()
            .find(|key| field(key).is_some())
            .cloned()
            .unwrap_or_else(|| IU4_KEY.to_owned());
        let effective = hipfire_config::effective::effective(&primary, self.resolved, self.arch);

        let origin = {
            let primary_origin = self.origin(&primary, &effective);
            let marked = keys
                .iter()
                .filter(|key| field(key).is_some())
                .map(|key| {
                    let eff = hipfire_config::effective::effective(key, self.resolved, self.arch);
                    self.origin(key, &eff)
                })
                .find(|origin| origin.marked() || matches!(origin, Origin::Registry));
            match (marked, &self.arch.gpu_arch) {
                (Some(origin), _) => origin,
                // Nothing set it: the built-in route for this GPU.
                (None, Some(arch)) if !choices.is_empty() => Origin::DefaultArch(arch.clone()),
                (None, _) => primary_origin,
            }
        };

        // What the layers resolve to (env and registry included) for the label;
        // the global layer alone decides where ←/→ starts.
        let in_effect = active_choice(&choices, |key| {
            self.resolved
                .get(key)
                .filter(|slot| !matches!(slot.source, ConfigSource::BuiltIn))
                .map(|slot| slot.value.clone())
        });
        let current_choice = active_choice(&choices, |key| self.global.get(key).cloned());
        let display = if choices.is_empty() {
            if self.arch.gpu_arch.is_some() {
                NOT_USED_ON_GPU.to_owned()
            } else {
                "— GPU not detected".to_owned()
            }
        } else {
            match in_effect {
                Some(index) => choices[index].label.to_owned(),
                None => "Custom (see Advanced)".to_owned(),
            }
        };
        let resolved = in_effect.map_or(ConfigValue::Null, |index| {
            ConfigValue::String(choices[index].id.to_owned())
        });
        let locked = keys.iter().find_map(|key| {
            let eff = hipfire_config::effective::effective(key, self.resolved, self.arch);
            lock_text(&self.origin(key, &eff))
        });
        let chain = self.virtual_chain(&choices, &origin, &display);
        // No route on this GPU: report it through `effective` too, so the
        // hide-unused filter drops the row.
        let mut effective = effective;
        if choices.is_empty() && self.arch.gpu_arch.is_some() {
            effective.used_on_gpu = false;
        }
        SettingsRow {
            id: RowId::Virtual(VirtualRow::PromptProcessing),
            title: knobs::curated_knob(PROMPT_PROCESSING_ID)
                .map_or("Prompt processing", |knob| knob.title)
                .to_owned(),
            group: Intent::Quality,
            resolved,
            effective,
            origin,
            chain,
            display,
            locked,
            deprecated: false,
            keys,
            choices,
            current_choice,
        }
    }

    /// The resolution chain of a key row, from `ResolvedValue.shadowed`
    /// (lowest priority first) ending with the value in use.
    fn chain(&self, key: &str, effective: &Effective, display: &str) -> Vec<ChainEntry> {
        let Some(slot) = self.resolved.get(key) else {
            return Vec::new();
        };
        let arch = self.arch.gpu_arch.as_deref();
        let mut chain: Vec<ChainEntry> = slot
            .shadowed
            .iter()
            .map(|candidate| ChainEntry {
                label: source_label(&candidate.source, false, arch),
                value: value_text(key, &candidate.value),
                used: false,
            })
            .collect();
        let builtin = matches!(slot.source, ConfigSource::BuiltIn);
        chain.push(ChainEntry {
            label: source_label(&slot.source, effective.by_arch_rule && builtin, arch),
            // A rule-derived built-in (`auto` → `fp8`) shows what it became.
            value: if builtin {
                display.to_owned()
            } else {
                value_text(key, &slot.value)
            },
            used: true,
        });
        chain
    }

    /// The Prompt processing chain: the built-in route for this GPU, then the
    /// layer that changed it, if any.
    fn virtual_chain(&self, choices: &[Choice], origin: &Origin, display: &str) -> Vec<ChainEntry> {
        if choices.is_empty() {
            return Vec::new();
        }
        let arch = self.arch.gpu_arch.as_deref();
        let builtin_label = choices
            .iter()
            .find(|choice| choice.builtin_default)
            .map_or("", |choice| choice.label)
            .to_owned();
        let default_origin = matches!(origin, Origin::Default | Origin::DefaultArch(_));
        let mut chain = vec![ChainEntry {
            label: match arch {
                Some(arch) => format!("built-in default · {arch}"),
                None => "built-in default".to_owned(),
            },
            value: builtin_label,
            used: default_origin,
        }];
        if !default_origin {
            chain.push(ChainEntry {
                label: match origin {
                    Origin::You => "config.toml".to_owned(),
                    other => other.label(),
                },
                value: display.to_owned(),
                used: true,
            });
        }
        chain
    }

    /// The "Using now" text of a key row (spec §3.2).
    fn display(&self, key: &str, effective: &Effective) -> String {
        if !effective.used_on_gpu {
            return NOT_USED_ON_GPU.to_owned();
        }
        if effective.used_by_model == Some(false) {
            // A tag with no size reads whole; keep the cell short.
            let model = if self.model_label.chars().count() > 8 {
                "this model"
            } else {
                self.model_label.as_str()
            };
            return format!("— not used by {model}");
        }
        let text = value_text(key, &effective.value);
        let auto = effective
            .reason
            .as_deref()
            .is_some_and(|reason| reason.starts_with("auto"));
        if auto {
            format!("{text} (auto)")
        } else {
            text
        }
    }
}

/// The layer a value came from, in words (the "why this value" chain).
/// `by_arch_rule` appends the GPU for a built-in the arch rules resolved.
fn source_label(source: &ConfigSource, by_arch_rule: bool, arch: Option<&str>) -> String {
    match source {
        ConfigSource::BuiltIn => match arch {
            Some(arch) if by_arch_rule => format!("built-in default · {arch}"),
            _ => "built-in default".to_owned(),
        },
        ConfigSource::RegistryModel { tag, .. } => format!("model card · {tag}"),
        ConfigSource::RegistryTarget { tag, arch, .. } => format!("model card · {tag} · {arch}"),
        ConfigSource::GlobalUser { .. } => "config.toml".to_owned(),
        ConfigSource::ModelUser { model, .. } => format!("models.toml · {model}"),
        ConfigSource::LegacyEnv { name } => format!("env {name}"),
        ConfigSource::OneShot { .. } => "command line".to_owned(),
    }
}

/// A config value as the text the editors and the enum cycle start from
/// (`Null` reads empty).
pub fn value_string(value: &ConfigValue) -> String {
    match value {
        ConfigValue::Null => String::new(),
        ConfigValue::String(text) => text.clone(),
        ConfigValue::Integer(number) => number.to_string(),
        ConfigValue::Float(number) => number.to_string(),
        ConfigValue::Bool(flag) => flag.to_string(),
    }
}

/// What the resident model was loaded with for `key`, from serve's
/// `/stats` `loaded_config` (KV mode, KV backend, context); `None` for any
/// other key or a field serve did not report.
pub fn live_text(key: &str, loaded: &LoadedConfig) -> Option<String> {
    match key {
        "memory.kv_cache" => loaded.kv_mode.clone(),
        "memory.kv_backend" => loaded.kv_backend.clone(),
        "memory.max_seq" => loaded
            .max_seq
            .and_then(|tokens| i64::try_from(tokens).ok())
            .map(|tokens| format!("{} tokens", thousands(tokens))),
        _ => None,
    }
}

fn lock_text(origin: &Origin) -> Option<String> {
    match origin {
        Origin::Env(name) => Some(format!("set by {name}; unset it to change")),
        _ => None,
    }
}

/// Plain title when the knob tables have one, else the canonical key.
pub(crate) fn row_title(key: &str) -> String {
    if let Some(knob) = knobs::curated_knob(key) {
        return knob.title.to_owned();
    }
    field(key)
        .and_then(|schema| knobs::knob_info(schema.legacy_key))
        .map_or_else(|| key.to_owned(), |info| info.title.to_owned())
}

/// A value in plain words. The offload and context labels are the easy view's
/// own ([`super::config`]).
pub(crate) fn value_text(key: &str, value: &ConfigValue) -> String {
    let raw = match value {
        ConfigValue::Null => String::new(),
        ConfigValue::String(text) => text.clone(),
        ConfigValue::Integer(number) => number.to_string(),
        ConfigValue::Float(number) => number.to_string(),
        ConfigValue::Bool(flag) => flag.to_string(),
    };
    match key {
        "memory.max_seq" => match value {
            ConfigValue::Integer(tokens) => format!("{} tokens", thousands(*tokens)),
            _ => raw,
        },
        "memory.gpu_layer_budget" | "memory.moe_expert_budget" => layers_label(&raw),
        "memory.offload_exec" => offload_exec_label(&raw),
        "memory.offload_passback_share" => passback_share_label(&raw),
        // An empty K/V axis takes its half of `memory.kv_cache`'s pair.
        "memory.kv_k" | "memory.kv_v" if matches!(value, ConfigValue::String(text) if text.is_empty()) => {
            "(follows KV)".to_owned()
        }
        _ => match value {
            ConfigValue::Bool(flag) => knobs::curated_knob(key)
                .and_then(|knob| {
                    knob.options
                        .iter()
                        .find(|(option, _)| *option == flag.to_string())
                        .map(|(_, label)| (*label).to_owned())
                })
                .unwrap_or_else(|| if *flag { "on" } else { "off" }.to_owned()),
            ConfigValue::Null => "not set".to_owned(),
            ConfigValue::String(text) if text.is_empty() => "(empty)".to_owned(),
            _ => raw,
        },
    }
}

/// `32768` → `32,768`.
fn thousands(number: i64) -> String {
    let digits = number.unsigned_abs().to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    if number < 0 {
        out.insert(0, '-');
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    use hipfire_config::effective::A8Prefill;

    use super::*;

    fn fixture_home(label: &str) -> (HipfirePaths, PathBuf) {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "hipfire-settings-model-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        (HipfirePaths::under(root.clone()), root)
    }

    fn write_config(paths: &HipfirePaths, text: &str) {
        std::fs::write(&paths.config, text).unwrap();
    }

    fn registry(entries: &str) -> RegistryV1 {
        RegistryV1::parse(
            &format!(
                r#"{{"schema_version":1,"generated_at":"2026-09-01T00:00:00Z","models":{{{entries}}},"aliases":{{}}}}"#
            ),
            "test",
        )
        .unwrap()
    }

    const CARD: &str = r#""qwen3.8:27b":{"repo":"r","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","arch_id":5}"#;
    const FLASH: &str = r#""qwen3.8:flash-next":{"repo":"r","file":"flash.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","arch_id":16}"#;

    fn arch(gpu: &str, a8: A8Prefill) -> ArchCtx {
        ArchCtx {
            gpu_arch: Some(gpu.into()),
            gpu_count: 1,
            model_arch_id: Some(5),
            registry_tag: Some("qwen3.8:27b".into()),
            a8_prefill: a8,
        }
    }

    fn build(
        paths: &HipfirePaths,
        reg: &RegistryV1,
        arch: &ArchCtx,
        env: &ConfigLayer,
    ) -> SettingsModel {
        SettingsModel::build(paths, "qwen3.8:27b", reg, arch, env)
    }

    fn row<'a>(model: &'a SettingsModel, key: &str) -> &'a SettingsRow {
        model
            .row(&RowId::Key(key.into()))
            .unwrap_or_else(|| panic!("no row {key}"))
    }

    fn string(value: &str) -> ConfigValue {
        ConfigValue::String(value.into())
    }

    #[test]
    fn gfx1151_quality_group_titles_order_and_effort_override() {
        let (paths, root) = fixture_home("quality");
        write_config(&paths, "[reasoning]\neffort = \"low\"\n");
        let reg = registry(CARD);
        let model = build(
            &paths,
            &reg,
            &arch("gfx1151", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let rows: Vec<&SettingsRow> = model.rows_in(Intent::Quality).collect();
        let titles: Vec<&str> = rows.iter().map(|row| row.title.as_str()).collect();
        assert_eq!(
            &titles[..6],
            [
                "Prompt processing",
                "Flash-Next experts",
                "Flash-Next word table",
                "Thinking",
                "Thinking effort",
                "Thinking cap",
            ],
            "{titles:?}"
        );
        // The raw prefill keys are replaced by the virtual row, not listed twice.
        assert!(rows
            .iter()
            .all(|row| !row.keys.iter().any(|key| key == IU4_KEY) || row.is_virtual()));

        let effort = row(&model, "reasoning.effort");
        assert_eq!(effort.display, "low");
        assert_eq!(effort.origin, Origin::You);
        assert!(effort.origin.marked() && effort.is_changed());
        assert_eq!(effort.locked, None);

        // Flash-Next rows are not used by a dense 27B model.
        assert_eq!(row(&model, SYM_IU4_KEY).display, "— not used by 27B");
        assert_eq!(
            row(&model, "memory.ple_format").display,
            "— not used by 27B"
        );
        assert_eq!(row(&model, "reasoning.mode").display, "on");
        assert_eq!(row(&model, "reasoning.mode").origin, Origin::Default);

        let prompt = &rows[0];
        assert_eq!(prompt.display, "Fast (4-bit math)");
        assert_eq!(prompt.origin, Origin::DefaultArch("gfx1151".into()));
        assert_eq!(prompt.choices.len(), 2);
        assert_eq!(prompt.current_choice, Some(0));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn flash_next_model_uses_its_rows() {
        let (paths, root) = fixture_home("flash");
        let reg = registry(&format!("{CARD},{FLASH}"));
        let mut ctx = arch("gfx1151", A8Prefill::NotLanded);
        ctx.model_arch_id = Some(16);
        ctx.registry_tag = Some("qwen3.8:flash-next".into());
        let model = SettingsModel::build(
            &paths,
            "qwen3.8:flash-next",
            &reg,
            &ctx,
            &ConfigLayer::default(),
        );
        assert_eq!(row(&model, SYM_IU4_KEY).display, "Fast");
        assert_eq!(row(&model, "memory.ple_format").display, "q4 (auto)");
        assert_eq!(
            row(&model, "memory.ple_format").origin,
            Origin::DefaultArch("gfx1151".into())
        );
        assert_eq!(row(&model, "speculation.mtp_k").display, "7 (auto)");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn env_layer_owns_and_locks_the_row() {
        let (paths, root) = fixture_home("env");
        write_config(&paths, "[memory]\nkv_cache = \"fwht3\"\n");
        let mut env = ConfigLayer::default();
        env.set("memory.kv_cache", string("q8")).unwrap();
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1201", A8Prefill::NotLanded),
            &env,
        );
        let kv = row(&model, "memory.kv_cache");
        assert_eq!(kv.origin, Origin::Env("HIPFIRE_KV_MODE".into()));
        assert_eq!(
            kv.locked.as_deref(),
            Some("set by HIPFIRE_KV_MODE; unset it to change")
        );
        assert_eq!(kv.display, "q8");
        assert!(kv.is_changed());
        assert_eq!(row(&model, "memory.max_seq").locked, None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn registry_card_default_reads_registry() {
        let (paths, root) = fixture_home("card");
        let reg = registry(
            r#""qwen3.8:27b":{"repo":"r","file":"qwen3.8-27b.mq4","size_gb":1,"min_vram_gb":1,"desc":"d","arch_id":5,"default_kv_mode":"fwht3"}"#,
        );
        let model = build(
            &paths,
            &reg,
            &arch("gfx1201", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let kv = row(&model, "memory.kv_cache");
        assert_eq!(kv.origin, Origin::Registry);
        assert_eq!(kv.display, "fwht3");
        assert_eq!(kv.resolved, string("fwht3"));
        assert!(!kv.is_changed(), "a model-card value is not a ● override");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn auto_kv_reads_fp8_auto_from_the_gpu_rule() {
        let (paths, root) = fixture_home("kvauto");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1201", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let kv = row(&model, "memory.kv_cache");
        assert_eq!(kv.display, "fp8 (auto)");
        assert_eq!(kv.origin, Origin::DefaultArch("gfx1201".into()));
        assert_eq!(row(&model, "memory.max_seq").display, "32,768 tokens");
        assert_eq!(row(&model, "memory.gpu_layer_budget").display, "all on GPU");
        assert_eq!(row(&model, "memory.offload_exec").display, "over PCIe");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn gfx1030_has_no_prompt_processing_route() {
        let (paths, root) = fixture_home("gfx1030");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1030", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let prompt = model
            .row(&RowId::Virtual(VirtualRow::PromptProcessing))
            .unwrap();
        assert_eq!(prompt.display, "— not used on this GPU");
        assert!(prompt.choices.is_empty());
        assert!(prompt.is_unused());
        let hidden = model.visible(
            Intent::Quality,
            &RowFilter {
                hide_unused: true,
                ..RowFilter::default()
            },
        );
        assert!(hidden.iter().all(|row| !row.is_virtual()));
        // The raw key stays reachable in Advanced.
        let raw = row(&model, IU4_KEY);
        assert_eq!(raw.group, Intent::Advanced);
        assert_eq!(raw.display, "— not used on this GPU");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a8_opt_in_routes_accurate_to_the_a8_key_without_writing_it() {
        let (paths, root) = fixture_home("a8");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1151", A8Prefill::OptIn),
            &ConfigLayer::default(),
        );
        let prompt = model
            .row(&RowId::Virtual(VirtualRow::PromptProcessing))
            .unwrap();
        assert_eq!(prompt.choices.len(), 2);
        assert_eq!(prompt.choices[1].id, "accurate");
        assert_eq!(
            prompt.choices[1].writes,
            vec![(A8_KEY, Some(ConfigValue::Bool(true)))]
        );
        assert_eq!(prompt.choices[0].writes, vec![(A8_KEY, None)]);
        assert!(prompt.keys.iter().any(|key| key == A8_KEY));
        // Nothing was written, and a missing schema key does not break the row.
        assert!(!paths.config.exists());
        assert_eq!(prompt.display, "Fast (4-bit math)");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn prompt_processing_reads_the_config_it_was_written_with() {
        let (paths, root) = fixture_home("iu4off");
        write_config(&paths, "[kernel]\niu4_prefill = false\n");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1201", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let prompt = model
            .row(&RowId::Virtual(VirtualRow::PromptProcessing))
            .unwrap();
        assert_eq!(prompt.display, "Accurate (fp8)");
        assert_eq!(prompt.current_choice, Some(1));
        assert_eq!(prompt.origin, Origin::You);
        assert!(prompt.is_changed());
        assert_eq!(prompt.model_scope_lock(), Some(MODEL_SCOPE_LOCK));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn search_finds_kv_rows_in_memory_and_advanced() {
        let (paths, root) = fixture_home("search");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1201", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let filter = RowFilter {
            search: "kv".into(),
            ..RowFilter::default()
        };
        let hits = model.search(&filter);
        let in_memory = hits.iter().any(|row| {
            row.group == Intent::MemoryContext && row.id == RowId::Key("memory.kv_cache".into())
        });
        assert!(in_memory, "KV cache precision is in Memory & context");
        assert!(
            hits.iter().any(|row| row.group == Intent::Advanced),
            "an Advanced row mentions KV"
        );
        // Title, key, env var and search is case-insensitive.
        let by_title = RowFilter {
            search: "KV CACHE PREC".into(),
            ..RowFilter::default()
        };
        assert!(model
            .search(&by_title)
            .iter()
            .any(|row| row.title == "KV cache precision"));
        let by_env = RowFilter {
            search: "hipfire_kv_mode".into(),
            ..RowFilter::default()
        };
        assert!(model
            .search(&by_env)
            .iter()
            .any(|row| row.id == RowId::Key("memory.kv_cache".into())));
        let counts = model.group_counts(&filter);
        assert!(counts[Intent::MemoryContext as usize].1 >= 1);
        assert_eq!(
            counts.iter().map(|(_, n)| n).sum::<usize>(),
            hits.len(),
            "group counts add up to the search results"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn changed_only_and_hide_unused_filters() {
        let (paths, root) = fixture_home("filters");
        write_config(&paths, "[reasoning]\neffort = \"low\"\n");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1151", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let changed = model.search(&RowFilter {
            changed_only: true,
            ..RowFilter::default()
        });
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].title, "Thinking effort");

        let quality = |hide| {
            model
                .visible(
                    Intent::Quality,
                    &RowFilter {
                        hide_unused: hide,
                        ..RowFilter::default()
                    },
                )
                .into_iter()
                .map(|row| row.title.clone())
                .collect::<Vec<_>>()
        };
        let all = quality(false);
        let used = quality(true);
        assert!(all.contains(&"Flash-Next experts".to_owned()));
        assert!(!used.contains(&"Flash-Next experts".to_owned()));
        assert!(used.contains(&"Prompt processing".to_owned()));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn groups_partition_the_rows_and_advanced_orders_deprecated_last() {
        let (paths, root) = fixture_home("groups");
        write_config(&paths, "[developer]\nsome_flag = \"1\"\n");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1151", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        // Every key once, plus the one virtual row.
        let mut keys: Vec<&str> = model.rows.iter().filter_map(|row| row.id.key()).collect();
        let total = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), total, "a key appears in two groups");
        let advanced: Vec<&SettingsRow> = model.rows_in(Intent::Advanced).collect();
        let first_deprecated = advanced
            .iter()
            .position(|row| row.deprecated)
            .expect("deprecated keys are in Advanced");
        assert!(advanced[first_deprecated..]
            .iter()
            .all(|row| row.deprecated));
        assert!(advanced[..first_deprecated]
            .iter()
            .all(|row| !row.deprecated));
        assert!(advanced
            .iter()
            .any(|row| row.id == RowId::Key("developer.some_flag".into())));
        assert!(
            advanced
                .iter()
                .any(|row| row.id == RowId::Key(A8_KEY.into()))
                == hipfire_config::field(A8_KEY).is_some()
        );
        // Curated rows keep §3.1 order within their group.
        let memory: Vec<&str> = model
            .rows_in(Intent::MemoryContext)
            .filter_map(|row| row.id.key())
            .collect();
        let curated = Intent::MemoryContext.curated_present();
        assert_eq!(&memory[..curated.len()], curated.as_slice());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn per_model_scope_locks_process_wide_rows() {
        let (paths, root) = fixture_home("scope");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1151", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        assert_eq!(
            row(&model, SYM_IU4_KEY).model_scope_lock(),
            Some(MODEL_SCOPE_LOCK)
        );
        assert_eq!(
            row(&model, "memory.ple_format").model_scope_lock(),
            Some(MODEL_SCOPE_LOCK)
        );
        assert_eq!(row(&model, "memory.kv_cache").model_scope_lock(), None);
        assert_eq!(row(&model, "reasoning.effort").model_scope_lock(), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explainer_reaches_line_three_with_the_row_state() {
        let (paths, root) = fixture_home("explain");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1151", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let prompt = model
            .row(&RowId::Virtual(VirtualRow::PromptProcessing))
            .unwrap();
        let lines = model.explainer(prompt, None, false);
        assert!(
            lines[1].starts_with("Accurate: 42% closer to the full model"),
            "{lines:?}"
        );
        assert!(lines[3].contains("kernel.iu4_prefill"), "{lines:?}");
        // A row with no curated entry reads the schema help instead.
        let raw = model.explainer(row(&model, "kernel.iu4_prefill"), None, false);
        assert!(raw[3].starts_with("kernel.iu4_prefill ·"), "{raw:?}");
        assert!(raw.iter().all(|line| line.chars().count() <= 76), "{raw:?}");
        // Live and restart states reach line three.
        let kv = row(&model, "memory.kv_cache");
        let live = model.explainer(kv, Some("q8"), false);
        assert!(live[2].starts_with("Live from serve: q8;"), "{live:?}");
        let restart = model.explainer(kv, None, true);
        assert!(restart[2].contains("↻ restart to apply"), "{restart:?}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn chain_ends_with_the_used_layer_and_live_text_maps_loaded_config() {
        let (paths, root) = fixture_home("chain");
        write_config(&paths, "[reasoning]\neffort = \"low\"\n");
        let model = build(
            &paths,
            &registry(CARD),
            &arch("gfx1201", A8Prefill::NotLanded),
            &ConfigLayer::default(),
        );
        let effort = row(&model, "reasoning.effort");
        assert!(effort.chain.len() >= 2, "{:?}", effort.chain);
        let last = effort.chain.last().unwrap();
        assert_eq!(
            (last.label.as_str(), last.value.as_str(), last.used),
            ("config.toml", "low", true)
        );
        assert!(effort.chain[..effort.chain.len() - 1]
            .iter()
            .all(|entry| !entry.used));
        assert_eq!(effort.chain[0].label, "built-in default");
        let kv = row(&model, "memory.kv_cache");
        let kv_last = kv.chain.last().unwrap();
        assert!(kv_last.used && kv_last.label.starts_with("built-in default · gfx1201"));

        let loaded = LoadedConfig {
            model: "qwen3.8:27b".into(),
            kv_mode: Some("q8".into()),
            kv_backend: Some("vmm".into()),
            max_seq: Some(32768),
        };
        assert_eq!(live_text("memory.kv_cache", &loaded).as_deref(), Some("q8"));
        assert_eq!(
            live_text("memory.kv_backend", &loaded).as_deref(),
            Some("vmm")
        );
        assert_eq!(
            live_text("memory.max_seq", &loaded).as_deref(),
            Some("32,768 tokens")
        );
        assert_eq!(live_text("reasoning.effort", &loaded), None);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn short_names_and_number_grouping() {
        assert_eq!(short_model_name("qwen3.8:27b"), "27B");
        assert_eq!(short_model_name("qwen3.8:27b-fast"), "27B");
        assert_eq!(short_model_name("qwen3.5:0.8b"), "0.8B");
        assert_eq!(short_model_name("qwen3.8:flash-next"), "qwen3.8:flash-next");
        assert_eq!(short_model_name("local.mq4"), "local.mq4");
        assert_eq!(thousands(32768), "32,768");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_048_576), "1,048,576");
    }
}
