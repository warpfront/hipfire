// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

pub use hipfire_registry::ModelEntry;

use super::HipfirePaths;

#[derive(Clone, Debug)]
pub struct RegistryState {
    pub models: Vec<ModelRow>,
    pub aliases: BTreeMap<String, String>,
    pub local_files: Vec<LocalModel>,
    pub selected: usize,
    pub expanded_groups: BTreeSet<String>,
    pub loaded_path: Option<PathBuf>,
    pub bundled: bool,
    pub generated_at: String,
    pub warning: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ModelRow {
    pub tag: String,
    pub entry: ModelEntry,
    pub downloaded: bool,
    pub has_mtp: bool,
}

#[derive(Clone, Debug)]
pub struct LocalModel {
    pub file: String,
    pub size: String,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub enum ModelListItem {
    Group {
        name: String,
        count: usize,
        downloaded: usize,
        expanded: bool,
    },
    Model {
        model_index: usize,
    },
}

#[derive(Clone, Debug)]
pub enum RegistryAction {
    ToggledGroup { name: String, expanded: bool },
    SelectedModel { tag: String },
}

impl RegistryState {
    pub fn load(paths: &HipfirePaths) -> Self {
        let registry_paths = hipfire_registry::RegistryPaths {
            cache: paths.root.join("registry.cache.json"),
        };
        let loaded = hipfire_registry::load(&registry_paths);
        Self::from_loaded(paths, registry_paths.cache, loaded)
    }

    /// The bundled registry only: no `HIPFIRE_REGISTRY_URL` /
    /// `HIPFIRE_NO_REGISTRY_FETCH` process env, no cache read, no network.
    /// Local model files under `paths` still apply.
    pub fn load_bundled(paths: &HipfirePaths) -> Self {
        let loaded = hipfire_registry::LoadedRegistry {
            registry: hipfire_registry::bundled()
                .expect("checked-in registry/v1.json must validate"),
            source: hipfire_registry::RegistrySource::Bundled,
            warnings: Vec::new(),
        };
        Self::from_loaded(paths, paths.root.join("registry.cache.json"), loaded)
    }

    fn from_loaded(
        paths: &HipfirePaths,
        cache_path: PathBuf,
        loaded: hipfire_registry::LoadedRegistry,
    ) -> Self {
        let mut warning = None;
        let local_files = list_local_models(paths);
        // Keep sidecar presence available separately from the selectable model list.
        let local_names = local_files
            .iter()
            .map(|m| m.file.clone())
            .collect::<BTreeSet<_>>();
        if !loaded.warnings.is_empty() {
            warning = Some(loaded.warnings.join("; "));
        }
        let loaded_path = matches!(
            loaded.source,
            hipfire_registry::RegistrySource::Cache | hipfire_registry::RegistrySource::StaleCache
        )
        .then_some(cache_path);
        let registry = loaded.registry;
        let bundled = loaded.source == hipfire_registry::RegistrySource::Bundled;
        let generated_at = registry.generated_at.clone();
        let local_files = local_files
            .into_iter()
            .filter(|local| {
                hipfire_registry::is_standalone_model(
                    &paths.models.join(&local.file),
                    &local.file,
                    &registry,
                    hipfire_registry::GRANDFATHERED_MONOLITHICS,
                )
            })
            .collect::<Vec<_>>();

        let registry_file_keys = registry
            .models
            .values()
            .map(|entry| normalized_file_key(&entry.file))
            .collect::<BTreeSet<_>>();

        let mut models = registry
            .models
            .into_iter()
            .map(|(tag, entry)| {
                let downloaded = local_names.contains(entry.file.as_str());
                let has_mtp = entry
                    .mtp
                    .as_ref()
                    .map(|s| local_names.contains(s.file.as_str()))
                    .unwrap_or(false);
                ModelRow {
                    tag,
                    entry,
                    downloaded,
                    has_mtp,
                }
            })
            .collect::<Vec<_>>();
        models.extend(
            local_files
                .iter()
                .filter(|local| !registry_file_keys.contains(&normalized_file_key(&local.file)))
                .map(|local| ModelRow {
                    tag: local.file.clone(),
                    entry: ModelEntry {
                        file: local.file.clone(),
                        size_gb: local.bytes as f64 / 1_000_000_000.0,
                        desc: "local model file outside registry".into(),
                        ..ModelEntry::default()
                    },
                    downloaded: true,
                    has_mtp: false,
                }),
        );
        models.sort_by(|a, b| a.tag.cmp(&b.tag));

        Self {
            models,
            aliases: registry.aliases,
            local_files,
            selected: 0,
            expanded_groups: BTreeSet::new(),
            loaded_path,
            bundled,
            generated_at,
            warning,
        }
    }

    pub fn downloaded_count(&self) -> usize {
        self.models.iter().filter(|m| m.downloaded).count()
    }

    /// `(tag, downloaded)` of the currently-selected MODEL row, or `None` when a
    /// group header (or nothing) is selected. Used by the Models tab to decide
    /// between pull (remote) and delete (local).
    pub fn selected_model(&self) -> Option<(String, bool)> {
        match self.visible_items().get(self.selected)? {
            ModelListItem::Model { model_index } => {
                let m = self.models.get(*model_index)?;
                Some((m.tag.clone(), m.downloaded))
            }
            ModelListItem::Group { .. } => None,
        }
    }

    /// The registry entry for `tag`, if listed.
    pub fn entry(&self, tag: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.tag == tag).map(|m| &m.entry)
    }

    /// Whether `tag` resolves to a known model (a registry/local entry or an
    /// alias). Used by the Chat `/model <tag>` command.
    pub fn has_model(&self, tag: &str) -> bool {
        self.models.iter().any(|m| m.tag == tag) || self.aliases.contains_key(tag)
    }

    pub fn visible_items(&self) -> Vec<ModelListItem> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, row) in self.models.iter().enumerate() {
            groups.entry(group_key(&row.tag)).or_default().push(idx);
        }

        let mut out = Vec::new();
        for (name, indices) in groups {
            let expanded = self.expanded_groups.contains(&name);
            let downloaded = indices
                .iter()
                .filter(|idx| self.models[**idx].downloaded)
                .count();
            out.push(ModelListItem::Group {
                name: name.clone(),
                count: indices.len(),
                downloaded,
                expanded,
            });
            if expanded {
                out.extend(
                    indices
                        .into_iter()
                        .map(|model_index| ModelListItem::Model { model_index }),
                );
            }
        }
        out
    }

    pub fn visible_len(&self) -> usize {
        self.visible_items().len()
    }

    pub fn clamp_selected(&mut self) {
        let len = self.visible_len().max(1);
        if self.selected >= len {
            self.selected = len - 1;
        }
    }

    pub fn activate_selected(&mut self) -> Option<RegistryAction> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, expanded, .. } => {
                if expanded {
                    self.expanded_groups.remove(&name);
                    self.clamp_selected();
                    Some(RegistryAction::ToggledGroup {
                        name,
                        expanded: false,
                    })
                } else {
                    self.expanded_groups.insert(name.clone());
                    Some(RegistryAction::ToggledGroup {
                        name,
                        expanded: true,
                    })
                }
            }
            ModelListItem::Model { model_index } => {
                self.models
                    .get(model_index)
                    .map(|row| RegistryAction::SelectedModel {
                        tag: row.tag.clone(),
                    })
            }
        }
    }

    pub fn expand_selected_group(&mut self) -> Option<String> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, .. } => {
                self.expanded_groups.insert(name.clone());
                Some(name)
            }
            ModelListItem::Model { .. } => None,
        }
    }

    pub fn collapse_selected_group(&mut self) -> Option<String> {
        match self.visible_items().get(self.selected).cloned()? {
            ModelListItem::Group { name, .. } => {
                self.expanded_groups.remove(&name);
                self.clamp_selected();
                Some(name)
            }
            ModelListItem::Model { model_index } => {
                let name = group_key(&self.models.get(model_index)?.tag);
                self.expanded_groups.remove(&name);
                self.selected = self
                    .visible_items()
                    .iter()
                    .position(
                        |item| matches!(item, ModelListItem::Group { name: n, .. } if n == &name),
                    )
                    .unwrap_or(0);
                Some(name)
            }
        }
    }
}

fn group_key(tag: &str) -> String {
    if let Some((family, _)) = tag.split_once(':') {
        return family.to_string();
    }
    if let Some((family, _)) = tag.split_once('-') {
        return family.to_string();
    }
    tag.to_string()
}

fn normalized_file_key(file: &str) -> String {
    file.replace(".q4.hfq", ".hf4")
        .replace(".hfq6.hfq", ".hf6")
        .replace("-hfq4.hfq", ".hf4")
        .replace(".hfq", ".hf4")
}

fn list_local_models(paths: &HipfirePaths) -> Vec<LocalModel> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(&paths.models) else {
        return out;
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let file = entry.file_name().to_string_lossy().to_string();
        out.push(LocalModel {
            file,
            size: format_bytes(meta.len()),
            bytes: meta.len(),
        });
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

fn format_bytes(bytes: u64) -> String {
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    if bytes as f64 >= GB {
        format!("{:.1} GiB", bytes as f64 / GB)
    } else if bytes as f64 >= MB {
        format!("{:.0} MiB", bytes as f64 / MB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::{group_key, normalized_file_key};

    #[test]
    fn groups_registry_tags_and_local_files_by_family() {
        assert_eq!(group_key("qwopus:27b-mq6"), "qwopus");
        assert_eq!(group_key("qwopus-27b.mq6"), "qwopus");
        assert_eq!(group_key("qwen3.5:9b"), "qwen3.5");
        assert_eq!(group_key("deepseek-v4-flash"), "deepseek");
    }

    #[test]
    fn local_models_exclude_sidecars_and_non_trunk_hfq_files() {
        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-model-classification-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let paths = super::HipfirePaths::under(root.clone());
        std::fs::create_dir_all(&paths.models).unwrap();
        for file in [
            "x.mq4",
            "x.mq4.ple-q4",
            "x.mq4.expert-hotset",
            "y-dflash.hfq",
        ] {
            std::fs::write(paths.models.join(file), []).unwrap();
        }
        for (file, arch) in [("z.hfq", 20_u32), ("w.hfq", 5_u32)] {
            let mut header = [0_u8; 32];
            header[..4].copy_from_slice(b"HFQM");
            header[4..8].copy_from_slice(&1_u32.to_le_bytes());
            header[8..12].copy_from_slice(&arch.to_le_bytes());
            std::fs::write(paths.models.join(file), header).unwrap();
        }
        let registry = hipfire_registry::RegistryV1::parse(
            r#"{
                "schema_version":1,
                "generated_at":"2026-09-01T00:00:00Z",
                "models":{"x:small":{
                    "repo":"test/models","file":"x.mq4",
                    "size_gb":1,"min_vram_gb":1,"desc":"test",
                    "dflash":{"file":"y-dflash.hfq"}
                }},
                "aliases":{}
            }"#,
            "test registry",
        )
        .unwrap();
        let state = super::RegistryState::from_loaded(
            &paths,
            root.join("registry.cache.json"),
            hipfire_registry::LoadedRegistry {
                registry,
                source: hipfire_registry::RegistrySource::Bundled,
                warnings: Vec::new(),
            },
        );
        assert_eq!(
            state
                .local_files
                .iter()
                .map(|m| m.file.as_str())
                .collect::<Vec<_>>(),
            ["w.hfq", "x.mq4"]
        );
        assert_eq!(state.models.len(), 2);
        assert!(state.models.iter().all(|model| model.downloaded));
        assert!(state.bundled);
        assert_eq!(state.generated_at, "2026-09-01T00:00:00Z");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn normalizes_legacy_hfq_names_for_duplicate_detection() {
        assert_eq!(
            normalized_file_key("qwen35-27b-dflash-mq4.hfq"),
            "qwen35-27b-dflash-mq4.hf4"
        );
        assert_eq!(normalized_file_key("qwen3.5-9b.q4.hfq"), "qwen3.5-9b.hf4");
    }
}
