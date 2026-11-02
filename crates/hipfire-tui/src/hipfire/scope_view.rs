// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Per-model scope (spec §3.5): which model the Settings tab edits, the model
//! picker and the per-model overrides list.
//!
//! Pure state and selection logic. The values come from `models.toml`
//! (`hipfire_config::load_catalog`) and the writes go through
//! `hipfire_config::{set_model_override, remove_model_override}`, the CLI's own
//! functions; nothing here touches the terminal.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use hipfire_config::{
    resolve, ConfigLayer, ConfigPaths, ConfigSource, ConfigValue, ModelCatalog, NamedLayer,
    ResolvedConfig,
};

use super::registry::RegistryState;
use super::settings_model::value_string;

/// Where a model's files are, for the State column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelState {
    /// The resident model of the running serve.
    Loaded,
    /// The model Settings predicts for (the default model).
    Active,
    OnDisk,
    /// Listed in `models.toml` (or the registry) but no file is installed.
    NotOnDisk,
}

impl ModelState {
    pub fn label(self) -> &'static str {
        match self {
            Self::Loaded => "loaded",
            Self::Active => "active",
            Self::OnDisk => "on disk",
            Self::NotOnDisk => "not on disk",
        }
    }
}

/// One value a model holds in `models.toml`.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnValue {
    /// Canonical key.
    pub key: String,
    pub value: ConfigValue,
}

impl OwnValue {
    /// The value as `models.toml` stores it.
    pub fn text(&self) -> String {
        value_string(&self.value)
    }
}

/// One line of the scope list: the All models row or a model.
#[derive(Clone, Debug, PartialEq)]
pub struct ScopeEntry {
    /// `None` is the All models row.
    pub model: Option<String>,
    pub state: Option<ModelState>,
    pub own: Vec<OwnValue>,
}

impl ScopeEntry {
    pub fn name(&self) -> &str {
        self.model.as_deref().unwrap_or("All models")
    }
}

/// The record of `tag` (or of its file) in the catalog, as the resolver finds
/// it.
fn record_of<'a>(
    catalog: &'a ModelCatalog,
    tag: &str,
    file: &str,
) -> Option<(&'a str, &'a hipfire_config::LocalModelConfig)> {
    catalog
        .model(tag)
        .or_else(|| (!file.is_empty()).then(|| catalog.model(file)).flatten())
}

fn own_values(record: &hipfire_config::LocalModelConfig) -> Vec<OwnValue> {
    record
        .overrides
        .values
        .iter()
        .map(|(key, value)| OwnValue {
            key: key.clone(),
            value: value.clone(),
        })
        .collect()
}

/// How many values the model `tag` (file `file`) holds of its own.
pub fn own_count(catalog: &ModelCatalog, tag: &str, file: &str) -> usize {
    record_of(catalog, tag, file).map_or(0, |(_, record)| record.overrides.values.len())
}

/// `1 own value`, `2 own values`.
pub fn own_values_text(count: usize) -> String {
    format!("{count} own value{}", if count == 1 { "" } else { "s" })
}

/// The All models row, the active model, then every other installed model
/// and every model `models.toml` holds values for.
pub fn entries(
    catalog: &ModelCatalog,
    registry: &RegistryState,
    active: &str,
    loaded: Option<&str>,
) -> Vec<ScopeEntry> {
    let is_loaded = |tag: &str, file: &str| {
        loaded.is_some_and(|name| name == tag || (!file.is_empty() && name == file))
    };
    // A catalog record belongs to the registry row whose tag finds it; tags
    // that only share its file (`qwen3.6:35b-a3b` and `-mq4p`) resolve the same
    // values but are not listed twice.
    let mut owner: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, row) in registry.models.iter().enumerate() {
        if let Some((id, _)) = catalog.model(&row.tag) {
            owner.entry(id).or_insert(index);
        }
    }
    for (index, row) in registry.models.iter().enumerate() {
        if let Some((id, _)) = record_of(catalog, &row.tag, &row.entry.file) {
            owner.entry(id).or_insert(index);
        }
    }
    let mut claimed: BTreeSet<String> = BTreeSet::new();
    let mut models: Vec<ScopeEntry> = Vec::new();
    for (index, row) in registry.models.iter().enumerate() {
        let record = record_of(catalog, &row.tag, &row.entry.file);
        if let Some((id, _)) = record {
            claimed.insert(id.to_owned());
        }
        let own = record
            .filter(|(id, _)| owner.get(id) == Some(&index))
            .map(|(_, record)| own_values(record))
            .unwrap_or_default();
        if !(row.downloaded || row.tag == active || !own.is_empty()) {
            continue;
        }
        let state = if is_loaded(&row.tag, &row.entry.file) {
            ModelState::Loaded
        } else if row.tag == active {
            ModelState::Active
        } else if row.downloaded {
            ModelState::OnDisk
        } else {
            ModelState::NotOnDisk
        };
        models.push(ScopeEntry {
            model: Some(row.tag.clone()),
            state: Some(state),
            own,
        });
    }
    // The active model may be a file the registry does not list.
    if !models
        .iter()
        .any(|entry| entry.model.as_deref() == Some(active))
    {
        let record = catalog.model(active);
        if let Some((id, _)) = record {
            claimed.insert(id.to_owned());
        }
        models.push(ScopeEntry {
            model: Some(active.to_owned()),
            state: Some(if is_loaded(active, "") {
                ModelState::Loaded
            } else {
                ModelState::Active
            }),
            own: record
                .map(|(_, record)| own_values(record))
                .unwrap_or_default(),
        });
    }
    // Catalog models the registry does not know (a custom file, a removed model).
    for (id, record) in &catalog.models {
        if claimed.contains(id) {
            continue;
        }
        let on_disk = record.path.as_ref().is_some_and(|path| path.is_file());
        models.push(ScopeEntry {
            model: Some(id.clone()),
            state: Some(if is_loaded(id, "") {
                ModelState::Loaded
            } else if on_disk {
                ModelState::OnDisk
            } else {
                ModelState::NotOnDisk
            }),
            own: own_values(record),
        });
    }
    models.sort_by(|a, b| {
        let rank = |entry: &ScopeEntry| usize::from(entry.model.as_deref() != Some(active));
        (rank(a), a.name()).cmp(&(rank(b), b.name()))
    });
    let mut all = vec![ScopeEntry {
        model: None,
        state: None,
        own: Vec::new(),
    }];
    all.extend(models);
    all
}

/// Which screen is open over the tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenKind {
    /// `m`: choose what Settings edits.
    Picker,
    /// `o`: every model's own values.
    Overrides,
}

/// The open picker or overrides list.
#[derive(Clone, Debug)]
pub struct ScopeScreen {
    pub kind: ScreenKind,
    /// The All models row first, then the models.
    pub entries: Vec<ScopeEntry>,
    /// Selected entry (index into `entries`).
    pub entry: usize,
    /// Selected own value of that entry.
    pub value: usize,
    /// `D` is waiting for y / n.
    pub confirm_remove_all: bool,
    /// The renderer drew the 160-column layout (a models list with a values
    /// pane) last frame, so ↑↓ move over models rather than over value rows.
    pub wide: Cell<bool>,
}

impl ScopeScreen {
    /// The overrides list opens on its first value; the picker on All models.
    pub fn new(kind: ScreenKind, entries: Vec<ScopeEntry>) -> Self {
        let mut screen = Self {
            kind,
            entries,
            entry: 0,
            value: 0,
            confirm_remove_all: false,
            wide: Cell::new(false),
        };
        if kind == ScreenKind::Overrides {
            if let Some(&(entry, value)) = screen.flat_rows().first() {
                (screen.entry, screen.value) = (entry, value);
            }
        }
        screen
    }

    pub fn selected_entry(&self) -> Option<&ScopeEntry> {
        self.entries.get(self.entry)
    }

    pub fn selected_value(&self) -> Option<&OwnValue> {
        self.selected_entry()?.own.get(self.value)
    }

    /// The compact overrides list is one row per value; every other layout
    /// walks the models.
    pub fn is_flat(&self) -> bool {
        self.kind == ScreenKind::Overrides && !self.wide.get()
    }

    /// `(entry, value)` of every row of the flat list, in order.
    pub fn flat_rows(&self) -> Vec<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .flat_map(|(entry, item)| (0..item.own.len()).map(move |value| (entry, value)))
            .collect()
    }

    /// How many values and models the overrides list shows.
    pub fn totals(&self) -> (usize, usize) {
        let values = self.entries.iter().map(|entry| entry.own.len()).sum();
        let models = self.entries.iter().filter(|e| !e.own.is_empty()).count();
        (values, models)
    }

    pub fn move_selection(&mut self, delta: isize) {
        if self.is_flat() {
            let rows = self.flat_rows();
            if rows.is_empty() {
                return;
            }
            let at = rows
                .iter()
                .position(|row| *row == (self.entry, self.value))
                .unwrap_or(0);
            let next = at.saturating_add_signed(delta).min(rows.len() - 1);
            (self.entry, self.value) = rows[next];
        } else {
            let next = self
                .entry
                .saturating_add_signed(delta)
                .min(self.entries.len().saturating_sub(1));
            if next != self.entry {
                self.entry = next;
                self.value = 0;
            }
        }
    }

    pub fn move_value(&mut self, delta: isize) {
        let count = self.selected_entry().map_or(0, |entry| entry.own.len());
        if count == 0 {
            return;
        }
        self.value = self.value.saturating_add_signed(delta).min(count - 1);
    }

    pub fn jump_to_end(&mut self, last: bool) {
        self.move_selection(if last { isize::MAX } else { isize::MIN });
    }

    /// Put the cursor on `model` (its first value); `false` when it is not
    /// listed. With no value of its own the flat list keeps its first row.
    pub fn focus(&mut self, model: &str) -> bool {
        let Some(entry) = self
            .entries
            .iter()
            .position(|item| item.model.as_deref() == Some(model))
        else {
            return false;
        };
        self.value = 0;
        if self.is_flat() && self.entries[entry].own.is_empty() {
            if let Some(&(first, value)) = self.flat_rows().first() {
                (self.entry, self.value) = (first, value);
            }
        } else {
            self.entry = entry;
        }
        true
    }

    /// New contents after a write: keep the cursor on the same model where it
    /// still exists, and in range.
    pub fn refresh(&mut self, entries: Vec<ScopeEntry>) {
        let model = self.selected_entry().and_then(|entry| entry.model.clone());
        let value = self.value;
        self.entries = entries;
        self.entry = model
            .and_then(|model| {
                self.entries
                    .iter()
                    .position(|item| item.model.as_deref() == Some(model.as_str()))
            })
            .unwrap_or(0)
            .min(self.entries.len().saturating_sub(1));
        let count = self.selected_entry().map_or(0, |entry| entry.own.len());
        self.value = value.min(count.saturating_sub(1));
        if self.is_flat() && count == 0 {
            if let Some(&(entry, value)) = self.flat_rows().first() {
                (self.entry, self.value) = (entry, value);
            }
        }
    }
}

/// What Settings edits, plus what the scope screens read from `models.toml`.
#[derive(Clone, Debug, Default)]
pub struct ScopeState {
    /// `None`: all models. `Some(name)`: one model's own values.
    pub model: Option<String>,
    /// `models.toml` as last read.
    pub catalog: ModelCatalog,
    /// The values every model without its own value gets: built-in defaults,
    /// `config.toml` and the environment. The CLI's `config overrides` compares
    /// against the same thing.
    pub all_models: Option<ResolvedConfig>,
    pub screen: Option<ScopeScreen>,
}

impl ScopeState {
    /// Re-read `models.toml` and the all-models values.
    pub fn reload(&mut self, paths: &ConfigPaths, env: &ConfigLayer) {
        self.catalog = hipfire_config::load_catalog(paths)
            .map(|loaded| loaded.catalog)
            .unwrap_or_default();
        self.all_models = resolve_all_models(paths, env);
    }

    /// The value `key` takes for every model without its own, in the words
    /// of the explainer and the overrides list.
    pub fn all_models_value(&self, key: &str) -> Option<(String, String)> {
        let slot = self.all_models.as_ref()?.get(key)?;
        let source = match &slot.source {
            ConfigSource::GlobalUser { .. } => "you, config.toml".to_owned(),
            ConfigSource::LegacyEnv { name } => format!("env {name}"),
            _ => "default".to_owned(),
        };
        Some((value_string(&slot.value), source))
    }

    /// The model holds its own value for `key`.
    pub fn holds(&self, model: &str, key: &str) -> bool {
        self.catalog
            .model(model)
            .is_some_and(|(_, record)| record.overrides.get(key).is_some())
    }

    /// How many values `model` holds of its own.
    pub fn own_count(&self, model: &str) -> usize {
        self.catalog
            .model(model)
            .map_or(0, |(_, record)| record.overrides.values.len())
    }
}

fn resolve_all_models(paths: &ConfigPaths, env: &ConfigLayer) -> Option<ResolvedConfig> {
    let loaded = hipfire_config::load_global(paths).ok()?;
    let mut layers = vec![NamedLayer {
        source: ConfigSource::GlobalUser { path: loaded.path },
        layer: loaded.layer,
    }];
    if !env.values.is_empty() {
        layers.push(NamedLayer {
            source: ConfigSource::LegacyEnv {
                name: "HIPFIRE_*".into(),
            },
            layer: env.clone(),
        });
    }
    resolve(layers).ok()
}

#[cfg(test)]
mod tests {
    use hipfire_config::ModelCatalog;

    use super::*;

    fn entry(model: Option<&str>, keys: &[&str]) -> ScopeEntry {
        ScopeEntry {
            model: model.map(str::to_owned),
            state: model.map(|_| ModelState::OnDisk),
            own: keys
                .iter()
                .map(|key| OwnValue {
                    key: (*key).to_owned(),
                    value: ConfigValue::Integer(1),
                })
                .collect(),
        }
    }

    fn screen(kind: ScreenKind) -> ScopeScreen {
        ScopeScreen::new(
            kind,
            vec![
                entry(None, &[]),
                entry(Some("a"), &["k1", "k2"]),
                entry(Some("b"), &[]),
                entry(Some("c"), &["k3"]),
            ],
        )
    }

    #[test]
    fn flat_list_walks_value_rows_and_skips_models_without_values() {
        let mut view = screen(ScreenKind::Overrides);
        assert!(view.focus("c"));
        assert_eq!((view.entry, view.value), (3, 0));
        view.move_selection(-1);
        assert_eq!((view.entry, view.value), (1, 1));
        view.move_selection(-1);
        assert_eq!((view.entry, view.value), (1, 0));
        view.move_selection(-1);
        assert_eq!((view.entry, view.value), (1, 0), "stops at the first row");
        view.jump_to_end(true);
        assert_eq!((view.entry, view.value), (3, 0));
        assert_eq!(view.totals(), (3, 2));
    }

    #[test]
    fn wide_list_walks_models_and_values_separately() {
        let mut view = screen(ScreenKind::Overrides);
        view.wide.set(true);
        view.entry = 0;
        view.move_selection(1);
        view.move_value(1);
        assert_eq!((view.entry, view.value), (1, 1));
        view.move_value(5);
        assert_eq!(view.value, 1, "stays on the last value");
        view.move_selection(1);
        assert_eq!(
            (view.entry, view.value),
            (2, 0),
            "a new model resets the value"
        );
    }

    #[test]
    fn focusing_a_model_with_no_values_keeps_the_flat_cursor_on_a_row() {
        let mut view = screen(ScreenKind::Overrides);
        assert!(view.focus("b"));
        assert_eq!((view.entry, view.value), (1, 0));
        assert!(!view.focus("zz"));
    }

    #[test]
    fn refresh_keeps_the_model_and_clamps_the_value() {
        let mut view = screen(ScreenKind::Overrides);
        view.wide.set(true);
        (view.entry, view.value) = (1, 1);
        assert_eq!((view.entry, view.value), (1, 1));
        view.refresh(vec![entry(None, &[]), entry(Some("a"), &["k1"])]);
        assert_eq!((view.entry, view.value), (1, 0));
        view.refresh(vec![entry(None, &[])]);
        assert_eq!((view.entry, view.value), (0, 0));
    }

    #[test]
    fn own_count_is_zero_for_an_unknown_model() {
        assert_eq!(own_count(&ModelCatalog::default(), "x", "x.mq4"), 0);
        assert_eq!(own_values_text(1), "1 own value");
        assert_eq!(own_values_text(0), "0 own values");
    }

    #[test]
    fn tags_sharing_a_file_list_the_record_once_under_the_tag_that_owns_it() {
        use crate::hipfire::registry::ModelRow;
        use hipfire_config::LocalModelConfig;
        use hipfire_registry::ModelEntry;
        use std::path::PathBuf;

        let row = |tag: &str| ModelRow {
            tag: tag.to_owned(),
            entry: ModelEntry {
                file: "a.mq4p".into(),
                ..ModelEntry::default()
            },
            downloaded: true,
            has_mtp: false,
        };
        let mut overrides = ConfigLayer::default();
        overrides
            .set("reasoning.effort", ConfigValue::String("high".into()))
            .unwrap();
        let mut catalog = ModelCatalog::default();
        catalog.models.insert(
            "a:big".into(),
            LocalModelConfig {
                path: Some(PathBuf::from("/models/a.mq4p")),
                registry_tag: Some("a:big".into()),
                overrides,
            },
        );
        let registry = RegistryState {
            models: vec![row("a:big-mq4p"), row("a:big")],
            aliases: Default::default(),
            local_files: Vec::new(),
            selected: 0,
            expanded_groups: Default::default(),
            loaded_path: None,
            bundled: true,
            generated_at: String::new(),
            warning: None,
        };
        let listed = entries(&catalog, &registry, "a:big", None);
        let named: Vec<(&str, usize)> = listed
            .iter()
            .map(|entry| (entry.name(), entry.own.len()))
            .collect();
        assert_eq!(
            named,
            [("All models", 0), ("a:big", 1), ("a:big-mq4p", 0)],
            "the owner carries the value; its file-sharing alias is a plain model"
        );
        assert_eq!(
            own_count(&catalog, "a:big-mq4p", "a.mq4p"),
            1,
            "it still resolves it"
        );
    }
}
