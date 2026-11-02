// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! Hermetic fixtures for the Settings tests: a temporary state root with its
//! own `config.toml`, a controlled registry card, an explicit GPU/model
//! context and an injected environment layer. Nothing reads the real
//! `~/.hipfire`, the process environment or the network.
//!
//! Tests build their [`App`] through [`hermetic_app`] /
//! [`hermetic_app_with_config`] or [`FixtureBuilder`], never the production
//! loader (it reads the real `~/.hipfire`, `install.json` and `HIPFIRE_*` env).

use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU32, Ordering},
};

use hipfire_config::{
    effective::{A8Prefill, ArchCtx},
    intent::Intent,
    ConfigLayer, ConfigValue,
};
use hipfire_registry::ModelEntry;

use crate::{
    app::{App, Tab},
    hipfire::{registry::ModelRow, settings_model::RowId, HipfirePaths},
};

/// A fresh temporary state root, removed (recursively) when dropped. Derefs to
/// its path.
pub struct TempHome(PathBuf);

impl TempHome {
    pub fn new(tag: &str) -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("create temp home");
        Self(root)
    }
}

impl std::ops::Deref for TempHome {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempHome {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempHome {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// An [`App`] over an empty fresh home: unknown GPU and model, no env layer,
/// bundled registry. Keep the returned [`TempHome`] alive as long as the app.
pub fn hermetic_app() -> (App, TempHome) {
    hermetic_app_with_config("")
}

/// [`hermetic_app`] whose `config.toml` holds `config_toml` (written BEFORE
/// the app loads); an empty string writes no file.
pub fn hermetic_app_with_config(config_toml: &str) -> (App, TempHome) {
    let home = TempHome::new("app");
    if !config_toml.is_empty() {
        std::fs::write(home.join("config.toml"), config_toml).expect("write hermetic config");
    }
    let arch = ArchCtx {
        gpu_arch: None,
        gpu_count: 1,
        model_arch_id: None,
        registry_tag: None,
        a8_prefill: A8Prefill::detect(),
    };
    let app = App::load_from_paths(
        HipfirePaths::under(home.to_path_buf()),
        arch,
        ConfigLayer::default(),
    )
    .expect("load hermetic app");
    (app, home)
}

/// The model every fixture serves (a dense 27B, `model_arch` 5).
pub const ACTIVE_MODEL: &str = "qwen3.8:27b";

/// A temporary state root plus the [`App`] loaded from it. The root is removed
/// when the fixture drops.
pub struct Fixture {
    pub app: App,
    root: PathBuf,
}

impl Fixture {
    pub fn config_path(&self) -> PathBuf {
        self.root.join("config.toml")
    }

    pub fn config_bytes(&self) -> Vec<u8> {
        std::fs::read(self.config_path()).unwrap_or_default()
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn models_toml_path(&self) -> PathBuf {
        self.root.join("models.toml")
    }

    pub fn models_toml_bytes(&self) -> Vec<u8> {
        std::fs::read(self.models_toml_path()).unwrap_or_default()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Builder for a Settings fixture.
pub struct FixtureBuilder {
    gpu: String,
    a8: A8Prefill,
    config: String,
    env: ConfigLayer,
    card_kv_mode: Option<String>,
    flash_next_split: bool,
    /// Further registry models `(tag, downloaded)` besides the active one.
    extra_models: Vec<(String, bool)>,
    /// Per-model values written to `models.toml` before the app loads.
    model_values: Vec<(String, String, String)>,
}

/// Split-base file of the Flash-Next fixture card (no `.ple-q8` beside it).
pub const FLASH_NEXT_BASE_FILE: &str = "qwen3.8-flash-next-base.mq4-xts";

impl FixtureBuilder {
    pub fn new(gpu: &str) -> Self {
        Self {
            gpu: gpu.to_owned(),
            a8: A8Prefill::NotLanded,
            config: String::new(),
            env: ConfigLayer::default(),
            card_kv_mode: None,
            flash_next_split: false,
            extra_models: Vec::new(),
            model_values: Vec::new(),
        }
    }

    /// A registry card `qwen3.8:flash-next` (`arch_id` 16, split base) whose
    /// placeholder base file exists in the fixture models dir. No `.ple-q8`
    /// exists, so the Quality-first word-table line is unmet.
    pub fn flash_next_split_base(mut self) -> Self {
        self.flash_next_split = true;
        self
    }

    /// Another model in the registry (`downloaded`: its file is installed).
    pub fn model(mut self, tag: &str, downloaded: bool) -> Self {
        self.extra_models.push((tag.to_owned(), downloaded));
        self
    }

    /// One per-model value in `models.toml`, written through the CLI's own
    /// `set_model_override` (a catalog id outside the registry works too).
    pub fn model_value(mut self, model: &str, key: &str, raw: &str) -> Self {
        self.model_values
            .push((model.to_owned(), key.to_owned(), raw.to_owned()));
        self
    }
    /// The PR #902 state to inject (a gfx1151 fixture mirroring mockup S1
    /// injects [`A8Prefill::OptIn`]).
    pub fn a8(mut self, a8: A8Prefill) -> Self {
        self.a8 = a8;
        self
    }

    /// TOML tables appended to `config.toml` (no `[serve]` table).
    pub fn config(mut self, toml: &str) -> Self {
        self.config.push_str(toml);
        self
    }

    /// One key of the injected `HIPFIRE_*` environment layer.
    pub fn env(mut self, key: &str, value: &str) -> Self {
        self.env
            .set(key, ConfigValue::String(value.to_owned()))
            .expect("fixture env key is valid");
        self
    }

    /// One key of the injected environment layer, parsed like the env loader
    /// does (`HIPFIRE_IU4_PREFILL=1` → `kernel.iu4_prefill`).
    pub fn env_cli(mut self, key: &str, raw: &str) -> Self {
        self.env
            .set_cli(key, raw)
            .expect("fixture env key and value are valid");
        self
    }

    /// The active model's registry card sets `default_kv_mode`.
    pub fn card_kv_mode(mut self, mode: &str) -> Self {
        self.card_kv_mode = Some(mode.to_owned());
        self
    }

    pub fn build(self) -> Fixture {
        static NEXT: AtomicU32 = AtomicU32::new(0);

        let root = std::env::temp_dir().join(format!(
            "hipfire-tui-settings-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).expect("create fixture home");
        std::fs::write(
            root.join("config.toml"),
            format!(
                "schema_version = 1\n{}\n[serve]\ndefault_model = \"{ACTIVE_MODEL}\"\n",
                self.config
            ),
        )
        .expect("write fixture config");

        for (model, key, raw) in &self.model_values {
            hipfire_config::set_model_override(
                &hipfire_config::ConfigPaths::under(&root),
                Some(model),
                None,
                model,
                key,
                raw,
            )
            .expect("write fixture models.toml");
        }

        let arch = ArchCtx {
            gpu_arch: Some(self.gpu),
            gpu_count: 1,
            model_arch_id: Some(5),
            registry_tag: Some(ACTIVE_MODEL.to_owned()),
            a8_prefill: self.a8,
        };
        let mut app = App::load_from_paths(HipfirePaths::under(root.clone()), arch, self.env)
            .expect("load fixture app");
        app.registry.models = vec![ModelRow {
            tag: ACTIVE_MODEL.to_owned(),
            entry: ModelEntry {
                repo: "hipfire-models/qwen3.8-27b".into(),
                file: "qwen3.8-27b.mq4".into(),
                size_gb: 15.66,
                min_vram_gb: 17.0,
                desc: "Qwen3.8 27B".into(),
                arch_id: Some(5),
                default_kv_mode: self.card_kv_mode,
                ..ModelEntry::default()
            },
            downloaded: false,
            has_mtp: false,
        }];
        if self.flash_next_split {
            std::fs::create_dir_all(&app.paths.models).expect("create fixture models dir");
            std::fs::write(app.paths.models.join(FLASH_NEXT_BASE_FILE), b"")
                .expect("write placeholder split base");
            app.registry.models.push(ModelRow {
                tag: "qwen3.8:flash-next".to_owned(),
                entry: ModelEntry {
                    repo: "hipfire-models/qwen3.8-flash-next".into(),
                    file: FLASH_NEXT_BASE_FILE.into(),
                    size_gb: 70.9,
                    min_vram_gb: 8.0,
                    desc: "Qwen3.8 Flash-Next (split base)".into(),
                    arch_id: Some(16),
                    ..ModelEntry::default()
                },
                downloaded: true,
                has_mtp: false,
            });
        }
        for (tag, downloaded) in &self.extra_models {
            app.registry.models.push(ModelRow {
                tag: tag.clone(),
                entry: ModelEntry {
                    repo: format!("hipfire-models/{}", tag.replace(':', "-")),
                    file: format!("{}.mq4", tag.replace(':', "-")),
                    size_gb: 5.0,
                    min_vram_gb: 6.0,
                    desc: tag.clone(),
                    ..ModelEntry::default()
                },
                downloaded: *downloaded,
                has_mtp: false,
            });
        }
        app.rebuild_settings();
        app.tab = Tab::Settings;
        // Serve offline, not "checking": the health poller never ran.
        app.status.health_known = true;
        Fixture { app, root }
    }
}

/// Send one key to the Settings handler.
pub fn press(app: &mut App, code: crossterm::event::KeyCode) {
    app.handle_settings_key(crossterm::event::KeyEvent::new(
        code,
        crossterm::event::KeyModifiers::NONE,
    ));
}

/// Type `text` into the Settings handler, one key per char.
pub fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        press(app, crossterm::event::KeyCode::Char(c));
    }
}

/// Put the cursor on the row `id` through the real keys: jump to its group,
/// then step down to it.
pub fn select_row(app: &mut App, id: &RowId) {
    use crossterm::event::KeyCode;
    let group = app.settings.row(id).expect("row exists").group;
    let index = Intent::ALL
        .iter()
        .position(|candidate| *candidate == group)
        .expect("group listed");
    press(app, KeyCode::Char(char::from(b'1' + index as u8)));
    let at = app
        .settings_visible()
        .iter()
        .position(|row| &row.id == id)
        .expect("row visible in its group");
    for _ in 0..at {
        press(app, KeyCode::Down);
    }
}

/// [`select_row`] for a canonical key row.
pub fn select_key(app: &mut App, key: &str) {
    select_row(app, &RowId::Key(key.to_owned()));
}

/// A Dashboard snapshot for a serve that is online with `model` resident and
/// `uptime_s` seconds up. `loaded` is its `/stats` `loaded_config`.
pub fn online_dashboard(
    loaded: Option<crate::hipfire::dashboard::LoadedConfig>,
    uptime_s: u64,
) -> crate::hipfire::dashboard::Dashboard {
    use crate::hipfire::dashboard::{Dashboard, LoadState, ServeStats, VramState};
    Dashboard {
        serve_up: true,
        endpoint: "127.0.0.1:11435".into(),
        health_text: String::new(),
        model: loaded.as_ref().map(|loaded| loaded.model.clone()),
        stats: Some(ServeStats {
            model: loaded.as_ref().map(|loaded| loaded.model.clone()),
            uptime_s,
            queue_depth: 0,
            requests_served: 0,
            recent_tok_s: None,
            loaded_config: loaded,
        }),
        model_ids: Vec::new(),
        vram: VramState::Unavailable("test".into()),
        gpu_load: LoadState::Unavailable("test".into()),
        offline_hint: None,
        system: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_root() -> PathBuf {
        let _guard = crate::hipfire::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        HipfirePaths::discover().root
    }

    /// No test-constructed App may sit on the real `~/.hipfire`.
    #[test]
    fn test_apps_never_use_the_real_home() {
        let real = real_root();
        let (app, home) = hermetic_app();
        assert_ne!(app.paths.root, real);
        assert_eq!(app.paths.root, home.to_path_buf());
        let (app, _home) = hermetic_app_with_config("schema_version = 1\n");
        assert_ne!(app.paths.root, real);
        let fixture = FixtureBuilder::new("gfx1151").build();
        assert_ne!(fixture.app.paths.root, real);
        assert!(fixture.app.paths.root.starts_with(std::env::temp_dir()));
    }

    /// A hermetic app drops its home with it.
    #[test]
    fn temp_home_is_removed_on_drop() {
        let (app, home) = hermetic_app();
        let root = app.paths.root.clone();
        assert!(root.is_dir());
        drop(app);
        drop(home);
        assert!(!root.exists());
    }

    /// Source guard: no test module may build an App from the real home or
    /// mutate process env outside `ENV_LOCK`. Test code is everything after
    /// the file's first `#[cfg(test)]` module.
    #[test]
    fn test_code_never_loads_the_real_app_or_mutates_env_unlocked() {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let load = ["App::", "load()"].concat();
        let set_env = ["env::set_", "var("].concat();
        let lock = ["ENV_", "LOCK"].concat();
        let mut files = Vec::new();
        walk(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let mut offenders = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let is_support = file
                .file_name()
                .is_some_and(|name| name == "test_support.rs");
            let has_tests = text.contains("#[test]");
            // The first inline `#[cfg(test)] mod name {` (not `mod name;`).
            let in_test_module = text
                .match_indices("#[cfg(test)]\nmod ")
                .map(|(at, marker)| &text[at + marker.len()..])
                .find(|rest| rest.lines().next().is_some_and(|line| line.ends_with('{')))
                .or(is_support.then_some(text.as_str()))
                .unwrap_or("");
            if has_tests && in_test_module.is_empty() {
                offenders.push(format!(
                    "{}: has #[test] but no `#[cfg(test)] mod` marker",
                    file.display()
                ));
            }
            if in_test_module.contains(&load) {
                offenders.push(format!(
                    "{}: test code builds an App from the real home",
                    file.display()
                ));
            }
            if in_test_module.contains(&set_env) && !in_test_module.contains(&lock) {
                offenders.push(format!(
                    "{}: test code sets env without ENV_LOCK",
                    file.display()
                ));
            }
        }
        assert!(offenders.is_empty(), "{offenders:#?}");
    }
}
