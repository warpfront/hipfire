// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire config list | explain | overrides`: the settings view shared with
//! the TUI. Every value shown here comes from `hipfire_config::effective`
//! (predicted value, origin) and `hipfire_config::intent` (group, search); this
//! module only gathers inputs and formats rows.

use std::collections::BTreeSet;
use std::io::Write;

use anyhow::{anyhow, bail, Result};
use hipfire_config::effective::{
    effective, install_gpu_arch, A8Prefill, ArchCtx, Effective, Origin,
};
use hipfire_config::env_audit::EnvEntry;
use hipfire_config::intent::{field_matches, intent_of, key_matches, Intent};
use hipfire_config::tuning::{attribute, TuningState};
use hipfire_config::{
    canonical_config_key, detect_config_profile, developer_env_for_key, field, fields,
    is_developer_key, load_catalog, load_config_profile, load_global, unknown_key, ConfigField,
    ConfigFormat, ConfigLayer, ConfigValue, LoadedCatalog, ResolvedConfig, ResolvedValue,
};
use hipfire_registry::RegistryV1;

use crate::{format_default, ConfigListArgs, Paths};

const VALID_GROUPS: &str = "quality, speed, memory, speculation, hardware, advanced";

/// Parse a `--group` argument (case-insensitive; `memory-context` and
/// `hardware-devices` are aliases).
pub(crate) fn parse_group(name: &str) -> Result<Intent> {
    match name.trim().to_ascii_lowercase().as_str() {
        "quality" => Ok(Intent::Quality),
        "speed" => Ok(Intent::Speed),
        "memory" | "memory-context" => Ok(Intent::MemoryContext),
        "speculation" => Ok(Intent::Speculation),
        "hardware" | "hardware-devices" => Ok(Intent::HardwareDevices),
        "advanced" => Ok(Intent::Advanced),
        _ => bail!("unknown group '{name}'; valid groups: {VALID_GROUPS}"),
    }
}

/// Row filters of `config list`.
pub(crate) struct Filter {
    group: Option<Intent>,
    changed: bool,
    search: Option<String>,
}

impl Filter {
    pub(crate) fn from_args(args: &ConfigListArgs) -> Result<Self> {
        Ok(Self {
            group: args.group.as_deref().map(parse_group).transpose()?,
            changed: args.changed,
            search: args.search.clone().filter(|query| !query.is_empty()),
        })
    }

    /// No filtering: every row.
    fn all() -> Self {
        Self {
            group: None,
            changed: false,
            search: None,
        }
    }

    fn keeps(&self, row: &Row<'_>) -> bool {
        if self.changed && !row.origin.marked() {
            return false;
        }
        match (&self.search, row.schema) {
            (None, _) => true,
            (Some(query), Some(schema)) => field_matches(schema, query),
            (Some(query), None) => key_matches(&row.key, query),
        }
    }
}

/// One key as the settings view shows it.
pub(crate) struct Row<'a> {
    key: String,
    schema: Option<&'static ConfigField>,
    intent: Intent,
    item: Option<&'a ResolvedValue>,
    effective: Effective,
    origin: Origin,
    overridden: bool,
}

impl Row<'_> {
    /// The "Using now" cell: the predicted value, `(auto)` when an arch rule or
    /// an `auto` setting produced it.
    fn using_now(&self) -> String {
        let mut text = self.effective.value.to_string();
        let resolved_auto = self
            .item
            .is_some_and(|item| matches!(&item.value, ConfigValue::String(s) if s == "auto"));
        let already_auto = matches!(&self.effective.value, ConfigValue::String(s) if s == "auto");
        if (self.effective.by_arch_rule || resolved_auto) && !already_auto {
            text.push_str(" (auto)");
        }
        text
    }

    fn text_line(&self) -> String {
        let mut line = format!(
            "  {:<43} {:<16} {:<18} {}",
            self.key,
            self.using_now(),
            self.origin.label(),
            self.effective.applies.label()
        );
        if self.effective.used_by_model == Some(false) {
            line.push_str("   (not used by this model)");
        }
        if !self.effective.used_on_gpu {
            line.push_str("   (not used on this GPU)");
        }
        line
    }

    /// `group`, `effective`, `origin`, `from`, `applies` — shared by list and
    /// explain JSON.
    fn view_json(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        map.insert("group".into(), serde_json::json!(self.intent));
        map.insert(
            "effective".into(),
            serde_json::json!({
                "value": self.effective.value,
                "reason": self.effective.reason,
            }),
        );
        map.insert("origin".into(), serde_json::json!(self.origin.kind()));
        map.insert("from".into(), serde_json::json!(self.origin.label()));
        map.insert(
            "applies".into(),
            serde_json::json!(self.effective.applies.label()),
        );
        map
    }

    /// Extra `explain` JSON fields.
    pub(crate) fn explain_json(&self) -> serde_json::Map<String, serde_json::Value> {
        self.view_json()
    }

    /// Extra `explain` text lines: effective (only when it differs from the
    /// resolved value or carries a reason), from, group, applies.
    pub(crate) fn explain_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let differs = self
            .item
            .is_none_or(|item| item.value != self.effective.value);
        if differs || self.effective.reason.is_some() {
            let reason = self
                .effective
                .reason
                .as_ref()
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default();
            lines.push(format!("  effective:   {}{reason}", self.effective.value));
        }
        lines.push(format!("  from:        {}", self.origin.label()));
        lines.push(format!("  group:       {}", self.intent.title()));
        lines.push(format!("  applies:     {}", self.effective.applies.label()));
        lines
    }
}

/// Inputs shared by every row of one scope (global, or one model).
pub(crate) struct View<'a> {
    ctx: ArchCtx,
    resolved: &'a ResolvedConfig,
    /// The layer whose keys count as "overridden" in this scope.
    overridden_in: &'a ConfigLayer,
    profile: Option<(String, ConfigLayer)>,
    /// `~/.hipfire/tuning.toml`: keys the applied preset owns read `preset`.
    tuning: TuningState,
}

impl<'a> View<'a> {
    /// `model` is `(registry arch_id, canonical registry tag)` for a model scope.
    pub(crate) fn new(
        paths: &Paths,
        resolved: &'a ResolvedConfig,
        global_layer: &ConfigLayer,
        overridden_in: &'a ConfigLayer,
        model: Option<(Option<u32>, Option<String>)>,
    ) -> Self {
        let (model_arch_id, registry_tag) = model.unwrap_or((None, None));
        let gpu_count = match resolved.get("hardware.devices").map(|item| &item.value) {
            Some(ConfigValue::String(list)) => list
                .split(',')
                .filter(|part| !part.trim().is_empty())
                .count()
                .max(1),
            _ => 1,
        };
        let ctx = ArchCtx {
            gpu_arch: install_gpu_arch(&paths.config.root),
            gpu_count,
            model_arch_id,
            registry_tag,
            a8_prefill: A8Prefill::detect(),
        };
        let profile = detect_config_profile(&paths.config, global_layer).and_then(|name| {
            let bundle = load_config_profile(&paths.config, &name).ok()?;
            Some((name, bundle))
        });
        // An unreadable tuning.toml only loses the `preset` attribution; the
        // listing itself must not fail on it.
        let tuning = TuningState::load(&paths.config).unwrap_or_default();
        Self {
            ctx,
            resolved,
            overridden_in,
            profile,
            tuning,
        }
    }

    pub(crate) fn gpu_arch(&self) -> Option<&str> {
        self.ctx.gpu_arch.as_deref()
    }

    pub(crate) fn row(&self, key: &str) -> Row<'a> {
        let canonical = hipfire_config::canonical_config_key(key).unwrap_or_else(|| key.to_owned());
        let effective = effective(&canonical, self.resolved, &self.ctx);
        let origin = attribute(
            &canonical,
            self.resolved,
            &effective,
            &self.ctx,
            &self.tuning,
            self.profile
                .as_ref()
                .map(|(name, bundle)| (name.as_str(), bundle)),
        );
        Row {
            schema: field(&canonical),
            intent: intent_of(&canonical),
            item: self.resolved.get(&canonical),
            overridden: self.overridden_in.get(&canonical).is_some(),
            key: canonical,
            effective,
            origin,
        }
    }

    /// Every row passing `filter`, in `Intent::ALL` order: curated keys first
    /// (their curated order), then the rest by canonical key. Developer keys
    /// appear when present in the config; the curated one always does.
    pub(crate) fn rows(&self, filter: &Filter) -> Vec<Row<'a>> {
        let mut keys: BTreeSet<String> = fields().iter().map(|f| f.key.to_owned()).collect();
        keys.extend(
            self.resolved
                .values
                .keys()
                .filter(|key| is_developer_key(key))
                .cloned(),
        );
        let mut rows = Vec::new();
        for intent in Intent::ALL {
            if filter.group.is_some_and(|group| group != intent) {
                continue;
            }
            let mut ordered: Vec<String> = intent
                .curated_present()
                .into_iter()
                .map(str::to_owned)
                .collect();
            for key in &keys {
                if intent_of(key) == intent && !ordered.contains(key) {
                    ordered.push(key.clone());
                }
            }
            for key in ordered {
                let row = self.row(&key);
                if filter.keeps(&row) {
                    rows.push(row);
                }
            }
        }
        rows
    }
}

/// A model scope's inputs, loaded once.
pub(crate) struct ModelScope {
    pub(crate) tag: Option<String>,
    arch_id: Option<u32>,
    pub(crate) resolved: ResolvedConfig,
    global_layer: ConfigLayer,
    overrides: ConfigLayer,
    pub(crate) catalog: LoadedCatalog,
}

impl ModelScope {
    pub(crate) fn load(paths: &Paths, registry: &RegistryV1, model: &str) -> Result<Self> {
        let card = crate::model_card_by_name(registry, model);
        let (tag, entry) = (card.tag().map(str::to_owned), card.entry());
        let catalog = load_catalog(&paths.config)?;
        let overrides = catalog
            .catalog
            .model(model)
            .or_else(|| tag.as_deref().and_then(|tag| catalog.catalog.model(tag)))
            .map(|(_, record)| record.overrides.clone())
            .unwrap_or_default();
        let resolved = crate::resolved_for_model(paths, model, tag.as_deref(), entry)?;
        Ok(Self {
            arch_id: entry.and_then(|entry| entry.arch_id),
            global_layer: load_global(&paths.config)?.layer,
            tag,
            resolved,
            overrides,
            catalog,
        })
    }

    pub(crate) fn view<'a>(&'a self, paths: &Paths) -> View<'a> {
        View::new(
            paths,
            &self.resolved,
            &self.global_layer,
            &self.overrides,
            Some((self.arch_id, self.tag.clone())),
        )
    }
}

fn gpu_line(arch: Option<&str>) -> String {
    match arch {
        Some(arch) => format!("GPU: {arch} (install.json)"),
        None => "GPU: unknown (no gpu_arch in install.json)".to_owned(),
    }
}

fn row_json(row: &Row<'_>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "legacy_key".into(),
        serde_json::json!(row.schema.map(|schema| schema.legacy_key)),
    );
    map.insert(
        "value".into(),
        serde_json::json!(row.item.map(|i| &i.value)),
    );
    map.insert(
        "default".into(),
        serde_json::json!(row.schema.map(format_default)),
    );
    map.insert(
        "source".into(),
        serde_json::json!(row.item.map(|i| &i.source)),
    );
    map.insert("overridden".into(), serde_json::json!(row.overridden));
    map.extend(row.view_json());
    serde_json::Value::Object(map)
}

fn write_rows(
    out: &mut dyn Write,
    header: &[String],
    rows: &[Row<'_>],
    json: bool,
    mut head: serde_json::Map<String, serde_json::Value>,
    gpu_arch: Option<&str>,
) -> Result<()> {
    if json {
        let values = rows
            .iter()
            .map(|row| (row.key.clone(), row_json(row)))
            .collect::<serde_json::Map<_, _>>();
        head.insert("gpu_arch".into(), serde_json::json!(gpu_arch));
        head.insert("values".into(), serde_json::Value::Object(values));
        writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&serde_json::Value::Object(head))?
        )?;
        return Ok(());
    }
    for line in header {
        writeln!(out, "{line}")?;
    }
    writeln!(out, "{}", gpu_line(gpu_arch))?;
    writeln!(out)?;
    if rows.is_empty() {
        writeln!(out, "No matching settings.")?;
        return Ok(());
    }
    let mut current: Option<Intent> = None;
    for row in rows {
        if current != Some(row.intent) {
            if current.is_some() {
                writeln!(out)?;
            }
            writeln!(out, "{}", row.intent.title())?;
            current = Some(row.intent);
        }
        writeln!(out, "{}", row.text_line())?;
    }
    Ok(())
}

/// `hipfire config list`.
pub(crate) fn list_global_to(
    paths: &Paths,
    args: &ConfigListArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let filter = Filter::from_args(args)?;
    let (loaded, resolved) = crate::resolved_global(paths, true)?;
    let view = View::new(paths, &resolved, &loaded.layer, &loaded.layer, None);
    let rows = view.rows(&filter);
    let mut header = vec![format!("Global configuration: {}", loaded.path.display())];
    if loaded.format == ConfigFormat::LegacyJson {
        header.push("  legacy JSON is active; the next write will create config.toml".into());
    }
    let mut head = serde_json::Map::new();
    head.insert("scope".into(), "global".into());
    head.insert("path".into(), serde_json::json!(loaded.path));
    head.insert(
        "format".into(),
        format!("{:?}", loaded.format).to_lowercase().into(),
    );
    head.insert("warnings".into(), serde_json::json!(loaded.warnings));
    write_rows(out, &header, &rows, args.json, head, view.gpu_arch())?;
    if !args.json {
        for warning in &loaded.warnings {
            eprintln!("warning: {warning}");
        }
    }
    Ok(())
}

/// `hipfire config MODEL list`.
pub(crate) fn list_model_to(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
    args: &ConfigListArgs,
    out: &mut dyn Write,
) -> Result<()> {
    let filter = Filter::from_args(args)?;
    let scope = ModelScope::load(paths, registry, model)?;
    let view = scope.view(paths);
    let rows = view.rows(&filter);
    let catalog = &scope.catalog;
    let header = vec![
        format!("Model configuration: {model}"),
        format!("Catalog: {} ({:?})", catalog.path.display(), catalog.format),
    ];
    let mut head = serde_json::Map::new();
    head.insert("scope".into(), "model".into());
    head.insert("model".into(), model.into());
    head.insert("canonical_tag".into(), serde_json::json!(scope.tag));
    head.insert("path".into(), serde_json::json!(catalog.path));
    head.insert(
        "format".into(),
        format!("{:?}", catalog.format).to_lowercase().into(),
    );
    head.insert("warnings".into(), serde_json::json!(catalog.warnings));
    write_rows(out, &header, &rows, args.json, head, view.gpu_arch())?;
    if !args.json {
        for warning in &catalog.warnings {
            eprintln!("warning: {warning}");
        }
    }
    Ok(())
}

/// `hipfire config overrides`: every models.toml value next to the
/// all-models value.
pub(crate) fn overrides_to(paths: &Paths, json: bool, out: &mut dyn Write) -> Result<()> {
    let catalog = load_catalog(&paths.config)?;
    let (_, resolved) = crate::resolved_global(paths, true)?;
    let mut entries = Vec::new();
    let mut models = 0usize;
    for (id, record) in &catalog.catalog.models {
        let before = entries.len();
        for (key, value) in &record.overrides.values {
            let global = resolved
                .get(key)
                .map(|item| item.value.clone())
                .ok_or_else(|| anyhow!("configuration key '{key}' is not set"))?;
            entries.push((id.clone(), key.clone(), value.clone(), global));
        }
        if entries.len() > before {
            models += 1;
        }
    }
    if json {
        let overrides = entries
            .iter()
            .map(|(model, key, value, global)| {
                serde_json::json!({
                    "model": model,
                    "key": key,
                    "value": value,
                    "all_models": global,
                })
            })
            .collect::<Vec<_>>();
        writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": catalog.path,
                "format": format!("{:?}", catalog.format).to_lowercase(),
                "count": entries.len(),
                "models": models,
                "overrides": overrides,
                "warnings": catalog.warnings,
            }))?
        )?;
        return Ok(());
    }
    if entries.is_empty() {
        writeln!(
            out,
            "No per-model overrides (~/.hipfire/models.toml has none)."
        )?;
    } else {
        for (model, key, value, global) in &entries {
            writeln!(
                out,
                "  {model:<22} {key:<32} {:<10} (all models: {global})",
                value.to_string()
            )?;
        }
        writeln!(out)?;
        writeln!(
            out,
            "{} override{} on {models} model{}.",
            entries.len(),
            if entries.len() == 1 { "" } else { "s" },
            if models == 1 { "" } else { "s" },
        )?;
    }
    for warning in &catalog.warnings {
        eprintln!("warning: {warning}");
    }
    Ok(())
}

fn write_shadowed(out: &mut dyn Write, value: &ResolvedValue) -> Result<()> {
    if !value.shadowed.is_empty() {
        writeln!(out, "  shadowed:")?;
        for candidate in value.shadowed.iter().rev() {
            writeln!(
                out,
                "    {:<16} {}",
                candidate.value,
                crate::source_label(&candidate.source)
            )?;
        }
    }
    Ok(())
}

/// `hipfire config explain KEY` (global scope).
pub(crate) fn explain_global_to(
    paths: &Paths,
    key: &str,
    json: bool,
    out: &mut dyn Write,
) -> Result<()> {
    let (loaded, resolved) = crate::resolved_global(paths, true)?;
    let canonical = canonical_config_key(key).ok_or_else(|| unknown_key(key))?;
    let value = resolved
        .get(&canonical)
        .ok_or_else(|| anyhow!("configuration key '{canonical}' is not set"))?;
    let view = View::new(paths, &resolved, &loaded.layer, &loaded.layer, None);
    let row = view.row(&canonical);
    if is_developer_key(&canonical) {
        let env_compat = developer_env_for_key(&canonical)
            .ok_or_else(|| anyhow!("developer key '{canonical}' has no legacy env spelling"))?;
        if json {
            let mut doc = serde_json::json!({
                "key": canonical,
                "legacy_key": null,
                "value": value.value,
                "source": value.source,
                "shadowed": value.shadowed,
                "default": null,
                "category": "diagnostic",
                "scope": "process",
                "registry_allowed": false,
                "experimental": true,
                "env_compat": env_compat,
                "help": "Experimental process-scoped override. Prefer a typed field when one exists.",
                "config_path": loaded.path,
            });
            doc.as_object_mut()
                .expect("object literal")
                .extend(row.explain_json());
            writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
        } else {
            writeln!(out, "{canonical}")?;
            writeln!(out, "  value:       {}", value.value)?;
            writeln!(out, "  source:      {}", crate::source_label(&value.source))?;
            for line in row.explain_lines() {
                writeln!(out, "{line}")?;
            }
            writeln!(out, "  default:     unset")?;
            writeln!(out, "  category:    Diagnostic")?;
            writeln!(out, "  scope:       Process")?;
            writeln!(out, "  registry:    false")?;
            writeln!(out, "  legacy env:  {env_compat}")?;
            writeln!(
                out,
                "  about:       Experimental process-scoped override. Prefer a typed field when one exists."
            )?;
            write_shadowed(out, value)?;
        }
        return Ok(());
    }
    let schema = field(&canonical).ok_or_else(|| unknown_key(&canonical))?;
    if json {
        let mut doc = serde_json::json!({
            "key": schema.key,
            "legacy_key": schema.legacy_key,
            "value": value.value,
            "source": value.source,
            "shadowed": value.shadowed,
            "default": format_default(schema),
            "category": schema.category,
            "scope": schema.scope,
            "registry_allowed": schema.registry_allowed,
            "experimental": schema.experimental,
            "env_compat": schema.env_compat,
            "help": schema.help,
            "config_path": loaded.path,
        });
        doc.as_object_mut()
            .expect("object literal")
            .extend(row.explain_json());
        writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
    } else {
        writeln!(out, "{}", schema.key)?;
        writeln!(out, "  value:       {}", value.value)?;
        writeln!(out, "  source:      {}", crate::source_label(&value.source))?;
        for line in row.explain_lines() {
            writeln!(out, "{line}")?;
        }
        writeln!(out, "  default:     {}", format_default(schema))?;
        writeln!(out, "  category:    {:?}", schema.category)?;
        writeln!(out, "  scope:       {:?}", schema.scope)?;
        writeln!(out, "  registry:    {}", schema.registry_allowed)?;
        if let Some(env) = schema.env_compat {
            writeln!(out, "  legacy env:  {env}")?;
        }
        writeln!(out, "  about:       {}", schema.help)?;
        write_shadowed(out, value)?;
    }
    Ok(())
}

/// `hipfire config MODEL explain KEY`.
pub(crate) fn explain_model_to(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
    key: &str,
    json: bool,
    out: &mut dyn Write,
) -> Result<()> {
    if is_developer_key(key) {
        bail!("developer configuration is global process policy; omit the model argument");
    }
    let scope = ModelScope::load(paths, registry, model)?;
    let schema = field(key).ok_or_else(|| unknown_key(key))?;
    let value = scope
        .resolved
        .get(schema.key)
        .ok_or_else(|| anyhow!("configuration key '{}' is not set", schema.key))?;
    let row = scope.view(paths).row(schema.key);
    if json {
        let mut doc = serde_json::json!({
            "model": model,
            "key": schema.key,
            "value": value.value,
            "source": value.source,
            "shadowed": value.shadowed,
            "scope": schema.scope,
            "help": schema.help,
        });
        doc.as_object_mut()
            .expect("object literal")
            .extend(row.explain_json());
        writeln!(out, "{}", serde_json::to_string_pretty(&doc)?)?;
    } else {
        writeln!(out, "{}", schema.key)?;
        writeln!(out, "  model:       {model}")?;
        writeln!(out, "  value:       {}", value.value)?;
        writeln!(out, "  source:      {}", crate::source_label(&value.source))?;
        for line in row.explain_lines() {
            writeln!(out, "{line}")?;
        }
        writeln!(out, "  scope:       {:?}", schema.scope)?;
        writeln!(out, "  about:       {}", schema.help)?;
        write_shadowed(out, value)?;
    }
    Ok(())
}

/// Where a keyless-explain row's value came from, in the user's vocabulary.
struct ExplainSource {
    kind: &'static str,
    label: String,
    env: Option<String>,
    path: Option<std::path::PathBuf>,
}

/// Files and model of the scope a keyless explain runs in.
struct ExplainScope<'a> {
    config_toml: std::path::PathBuf,
    /// `(model name, models.toml path)` in a model scope.
    model: Option<(&'a str, std::path::PathBuf)>,
}

fn explain_source(row: &Row<'_>, scope: &ExplainScope<'_>) -> ExplainSource {
    let simple = |kind, label: String| ExplainSource {
        kind,
        label,
        env: None,
        path: None,
    };
    match &row.origin {
        Origin::Default => simple("default", "default".into()),
        Origin::DefaultArch(arch) => simple("default-arch", format!("default · {arch}")),
        Origin::Registry => {
            let tag = row.item.and_then(|item| match &item.source {
                hipfire_config::ConfigSource::RegistryModel { tag, .. }
                | hipfire_config::ConfigSource::RegistryTarget { tag, .. } => Some(tag.as_str()),
                _ => None,
            });
            simple(
                "registry",
                tag.map_or_else(|| "registry".into(), |tag| format!("registry {tag}")),
            )
        }
        Origin::Profile(name) => ExplainSource {
            path: Some(scope.config_toml.clone()),
            ..simple("profile", format!("profile {name}"))
        },
        Origin::Preset => ExplainSource {
            path: Some(scope.config_toml.clone()),
            ..simple("preset", "preset".into())
        },
        Origin::You => ExplainSource {
            path: Some(scope.config_toml.clone()),
            ..simple("config", "config.toml".into())
        },
        Origin::ThisModel => match &scope.model {
            Some((model, path)) => {
                let file = path
                    .file_name()
                    .map_or_else(|| "models.toml".into(), |name| name.to_string_lossy());
                ExplainSource {
                    path: Some(path.clone()),
                    ..simple("models", format!("{file} [{model}]"))
                }
            }
            None => simple("models", "models.toml".into()),
        },
        Origin::Env(name) => ExplainSource {
            env: Some(name.clone()),
            ..simple("env", format!("env {name}"))
        },
        Origin::OneShot => simple("cli", "cli".into()),
    }
}

fn write_explain_all(
    out: &mut dyn Write,
    header: &[String],
    rows: &[Row<'_>],
    scope: &ExplainScope<'_>,
    json: bool,
    mut head: serde_json::Map<String, serde_json::Value>,
    env: &[EnvEntry],
    gpu_arch: Option<&str>,
) -> Result<()> {
    if json {
        let settings = rows
            .iter()
            .map(|row| {
                let source = explain_source(row, scope);
                let mut src = serde_json::Map::new();
                src.insert("kind".into(), source.kind.into());
                src.insert("label".into(), source.label.into());
                if let Some(env) = source.env {
                    src.insert("env".into(), env.into());
                }
                if let Some(path) = source.path {
                    src.insert("path".into(), serde_json::json!(path));
                }
                serde_json::json!({
                    "key": row.key,
                    "value": row.item.map(|item| &item.value),
                    "effective": row.effective.value,
                    "source": src,
                })
            })
            .collect::<Vec<_>>();
        head.insert("gpu_arch".into(), serde_json::json!(gpu_arch));
        head.insert("settings".into(), settings.into());
        head.insert(
            "env".into(),
            env.iter()
                .map(|entry| {
                    serde_json::json!({
                        "name": entry.name,
                        "value": entry.value,
                        "status": entry.status,
                        "label": entry.status.label(),
                    })
                })
                .collect::<Vec<_>>()
                .into(),
        );
        writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&serde_json::Value::Object(head))?
        )?;
        return Ok(());
    }
    for line in header {
        writeln!(out, "{line}")?;
    }
    writeln!(out, "{}", gpu_line(gpu_arch))?;
    writeln!(out)?;
    let mut current: Option<Intent> = None;
    for row in rows {
        if current != Some(row.intent) {
            if current.is_some() {
                writeln!(out)?;
            }
            writeln!(out, "{}", row.intent.title())?;
            current = Some(row.intent);
        }
        let source = explain_source(row, scope);
        writeln!(
            out,
            "  {:<43} {:<16} {}",
            row.key,
            row.using_now(),
            source.label
        )?;
    }
    writeln!(out)?;
    writeln!(out, "environment")?;
    if env.is_empty() {
        writeln!(out, "  no HIPFIRE_* variables set")?;
    }
    for entry in env {
        let assignment = format!("{}={}", entry.name, entry.value);
        writeln!(out, "  {assignment:<60} {}", entry.status.label())?;
    }
    writeln!(out)?;
    writeln!(out, "explain one key: hipfire config explain <key>")?;
    Ok(())
}

/// `hipfire config explain` with no key (global scope): every effective
/// setting with its source, then the `HIPFIRE_*` environment audit.
pub(crate) fn explain_all_global_to(
    paths: &Paths,
    json: bool,
    env: &[EnvEntry],
    out: &mut dyn Write,
) -> Result<()> {
    let (loaded, resolved) = crate::resolved_global(paths, true)?;
    let view = View::new(paths, &resolved, &loaded.layer, &loaded.layer, None);
    let rows = view.rows(&Filter::all());
    let scope = ExplainScope {
        config_toml: loaded.path.clone(),
        model: None,
    };
    let header = vec![format!(
        "Effective configuration (config.toml: {})",
        loaded.path.display()
    )];
    let mut head = serde_json::Map::new();
    head.insert("scope".into(), "global".into());
    head.insert("path".into(), serde_json::json!(loaded.path));
    write_explain_all(
        out,
        &header,
        &rows,
        &scope,
        json,
        head,
        env,
        view.gpu_arch(),
    )
}

/// `hipfire config explain --model MODEL` with no key: as
/// [`explain_all_global_to`], resolved in the model's scope.
pub(crate) fn explain_all_model_to(
    paths: &Paths,
    registry: &RegistryV1,
    model: &str,
    json: bool,
    env: &[EnvEntry],
    out: &mut dyn Write,
) -> Result<()> {
    let scope_data = ModelScope::load(paths, registry, model)?;
    let view = scope_data.view(paths);
    let rows = view.rows(&Filter::all());
    let global_path = paths.config.config_toml.clone();
    let models_path = scope_data.catalog.path.clone();
    let scope = ExplainScope {
        config_toml: global_path.clone(),
        model: Some((model, models_path.clone())),
    };
    let header = vec![
        format!("Effective configuration for {model}"),
        format!("config.toml: {}", global_path.display()),
        format!("per-model catalog: {}", models_path.display()),
    ];
    let mut head = serde_json::Map::new();
    head.insert("scope".into(), "model".into());
    head.insert("model".into(), model.into());
    head.insert("canonical_tag".into(), serde_json::json!(scope_data.tag));
    head.insert("path".into(), serde_json::json!(global_path));
    head.insert("models_path".into(), serde_json::json!(models_path));
    write_explain_all(
        out,
        &header,
        &rows,
        &scope,
        json,
        head,
        env,
        view.gpu_arch(),
    )
}
