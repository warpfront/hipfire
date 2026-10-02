// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

pub mod chat;
pub mod chat_history;
pub mod config;
pub mod dashboard;
pub mod doctor;
pub mod knobs;
pub mod log_tail;
pub mod model_actions;
pub mod profile_wizard;
pub mod registry;
pub mod serve_ctrl;
pub mod status;
pub mod ui_state;
pub mod writer;

use std::{env, path::PathBuf, process::Command};

/// Resolve the native Rust `hipfire` control-plane binary.
pub fn native_cli_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("HIPFIRE_CLI_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    // Tests never fall back to a real build: a test reaching
    // `start_background_serve` would exec a detached `hipfire serve` that
    // outlives the test and holds the GPU and its per-card locks.
    if cfg!(test) {
        return None;
    }
    let home = env::var_os("HIPFIRE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
        .unwrap_or_else(|| PathBuf::from(".hipfire"));
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
    let sibling = env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("hipfire")));
    sibling
        .into_iter()
        .chain([
            home.join("bin/hipfire"),
            workspace.join("release/hipfire"),
            workspace.join("debug/hipfire"),
        ])
        .find(|path| path.is_file())
}

pub fn native_cli_command() -> Option<Command> {
    native_cli_path().map(Command::new)
}

#[derive(Clone, Debug)]
pub struct HipfirePaths {
    pub root: PathBuf,
    pub models: PathBuf,
    pub config: PathBuf,
    pub legacy_config: PathBuf,
    pub models_catalog: PathBuf,
    pub legacy_models_catalog: PathBuf,
    pub legacy_per_model_config: PathBuf,
    pub serve_pid: PathBuf,
    pub serve_log: PathBuf,
    pub ui_state: PathBuf,
    pub chat_history: PathBuf,
}

impl HipfirePaths {
    pub fn discover() -> Self {
        let root = env::var_os("HIPFIRE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".hipfire")))
            .unwrap_or_else(|| PathBuf::from(".hipfire"));
        let models = env::var_os("HIPFIRE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("models"));
        Self {
            models,
            config: root.join("config.toml"),
            legacy_config: root.join("config.json"),
            models_catalog: root.join("models.toml"),
            legacy_models_catalog: root.join("models.json"),
            legacy_per_model_config: root.join("per_model_config.json"),
            serve_pid: root.join("serve.pid"),
            serve_log: root.join("serve.log"),
            ui_state: root.join("ui_state.json"),
            chat_history: root.join("chat_history.json"),
            root,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialises the tests that mutate HIPFIRE_CLI_BIN.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn native_cli_path_never_falls_back_to_a_real_build_under_test() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        env::remove_var("HIPFIRE_CLI_BIN");
        assert_eq!(native_cli_path(), None);
        // The deliberate spawn: with a built `target/*/hipfire` on disk this
        // used to exec a detached `hipfire serve -d`.
        assert!(status::start_background_serve().is_err());
    }

    #[test]
    fn native_cli_path_resolves_env_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = env::temp_dir().join(format!("hipfire-native-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let binary = dir.join("hipfire");
        std::fs::write(&binary, b"test binary").unwrap();
        env::set_var("HIPFIRE_CLI_BIN", &binary);
        assert_eq!(native_cli_path().as_deref(), Some(binary.as_path()));
        env::remove_var("HIPFIRE_CLI_BIN");
        let _ = std::fs::remove_dir_all(dir);
    }
}
