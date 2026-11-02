// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

use std::{fs, process::Stdio};

use anyhow::{anyhow, Result};

use super::{
    dashboard::{Dashboard, HealthSnapshot},
    HipfirePaths,
};

#[derive(Clone, Debug)]
pub struct StatusState {
    pub serve_pid: Option<u32>,
    pub serve_pid_alive: bool,
    /// `/health` answered. Owned by the always-on health poller
    /// ([`StatusState::overlay_health`]); the full dashboard snapshot never
    /// writes it, so a stale one cannot override fresher health.
    pub serve_http_ok: bool,
    pub health_text: String,
    /// At least one `/health` probe has completed (`serve_http_ok == false`
    /// before that means "unknown", not "down").
    pub health_known: bool,
    /// Why the serve cannot be probed (bad key file, rejected bearer) — a serve
    /// may be answering, so Chat must not start another one.
    pub health_failure: Option<String>,
    pub gpu_lines: Vec<String>,
    pub paths_ok: Vec<(String, bool)>,
}

impl StatusState {
    /// Load ONLY the fast local state — pure file I/O (serve.pid read + `~/.hipfire`
    /// path-existence checks). Performs NO network (/health) or hardware (lspci)
    /// probe, so it is safe to call synchronously on the UI thread (startup and
    /// the `r` reload). The live fields (`serve_http_ok`, `health_text`,
    /// `gpu_lines`) start empty and are filled in by [`StatusState::overlay_live`]
    /// from the background `DashboardWorker` snapshot.
    pub fn load_local(paths: &HipfirePaths) -> Self {
        let (serve_pid, serve_pid_alive) = read_serve_pid(paths);
        let paths_ok = vec![
            ("~/.hipfire".into(), paths.root.exists()),
            ("models".into(), paths.models.exists()),
            ("config.toml".into(), paths.config.exists()),
            ("legacy config.json".into(), paths.legacy_config.exists()),
            ("models.toml".into(), paths.models_catalog.exists()),
            (
                "legacy models.json".into(),
                paths.legacy_models_catalog.exists(),
            ),
            (
                "legacy per_model_config.json".into(),
                paths.legacy_per_model_config.exists(),
            ),
            ("serve.log".into(), paths.serve_log.exists()),
        ];
        Self {
            serve_pid,
            serve_pid_alive,
            serve_http_ok: false,
            health_text: String::new(),
            health_known: false,
            health_failure: None,
            gpu_lines: Vec::new(),
            paths_ok,
        }
    }

    /// Re-read serve.pid (pure file I/O) so a serve started since the last
    /// reload is seen. Call sparingly, not every frame.
    pub fn refresh_pid(&mut self, paths: &HipfirePaths) {
        let (pid, alive) = read_serve_pid(paths);
        self.serve_pid = pid;
        self.serve_pid_alive = alive;
    }

    /// Fold the latest background snapshot into the live fields. Only the
    /// system-derived `gpu_lines` come from the full [`Dashboard`]; serve health
    /// comes from [`StatusState::overlay_health`]. A cheap in-memory copy with no
    /// I/O, called every frame from `App::sync_dashboard`.
    pub fn overlay_live(&mut self, dash: &Dashboard) {
        self.gpu_lines = dash
            .system
            .as_ref()
            .map(gpu_lines_from_system)
            .unwrap_or_default();
    }

    /// Fold the always-on `/health` poller result (every tab) into
    /// `serve_http_ok` / `health_text`.
    pub fn overlay_health(&mut self, health: &HealthSnapshot) {
        self.health_known = health.checked;
        self.serve_http_ok = health.serve_up;
        self.health_text = health.text.clone();
        self.health_failure = health.failure.clone();
    }

    pub fn serve_label(&self) -> String {
        if self.serve_http_ok {
            "online".into()
        } else if !self.health_known {
            "checking...".into()
        } else if self.serve_pid_alive {
            "pid alive, HTTP not ready".into()
        } else if self.serve_pid.is_some() {
            "stale pid".into()
        } else {
            "offline".into()
        }
    }
}

pub fn start_background_serve() -> Result<()> {
    let mut cmd = super::native_cli_command().ok_or_else(|| {
        anyhow!("native hipfire binary not found (set HIPFIRE_CLI_BIN or install hipfire)")
    })?;
    cmd.arg("serve")
        .arg("-d")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| anyhow!("failed to launch `hipfire serve -d`: {err}"))?;
    Ok(())
}

fn read_serve_pid(paths: &HipfirePaths) -> (Option<u32>, bool) {
    let serve_pid = fs::read_to_string(&paths.serve_pid)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok());
    let alive = serve_pid
        .map(|pid| std::path::Path::new(&format!("/proc/{pid}")).exists())
        .unwrap_or(false);
    (serve_pid, alive)
}

/// Derive the Home-tab GPU summary lines from the background-probed
/// [`SystemInfo`] (amd-smi / rocm-smi product name + gfx arch) rather than
/// running `lspci` synchronously on the UI thread. The worker probed this OFF
/// the UI thread, so this is a pure in-memory format. When neither field probed
/// cleanly it reports WHY (e.g. the stats tool is missing) and never claims "no
/// GPU" — a missing stats tool says nothing about the PCI devices.
fn gpu_lines_from_system(system: &super::dashboard::SystemInfo) -> Vec<String> {
    let mut lines = Vec::new();
    if system.gpu_name.is_available() {
        lines.push(system.gpu_name.display().to_string());
    }
    if system.gpu_arch.is_available() {
        lines.push(format!("arch  {}", system.gpu_arch.display()));
    }
    if lines.is_empty() {
        lines.push(format!(
            "{}. Run hipfire diag for the full probe.",
            system.gpu_name.display().trim_end_matches('.')
        ));
    }
    lines
}

#[cfg(test)]
mod gpu_lines_tests {
    use super::*;
    use crate::hipfire::dashboard::{Probe, SystemInfo};

    fn system(gpu_name: Probe, gpu_arch: Probe) -> SystemInfo {
        SystemInfo {
            gpu_name,
            gpu_arch,
            hip_version: Probe::Unavailable("unavailable".into()),
            kernel_cache: Probe::Unavailable("unavailable".into()),
            loaded_model: Probe::Unavailable("no model loaded".into()),
            model_checksum: Probe::Unavailable("no model loaded".into()),
        }
    }

    #[test]
    fn missing_stats_tool_is_reported_without_claiming_no_gpu() {
        let reason = "GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: none)";
        let lines = gpu_lines_from_system(&system(
            Probe::Unavailable(reason.into()),
            Probe::Unavailable("unavailable: rocminfo not found".into()),
        ));
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with(reason), "{}", lines[0]);
        assert!(!lines[0].contains("No GPU detected"), "{}", lines[0]);
    }

    #[test]
    fn probed_name_and_arch_are_listed() {
        let lines = gpu_lines_from_system(&system(
            Probe::Value("AMD Radeon AI PRO R9700".into()),
            Probe::Value("gfx1201".into()),
        ));
        assert_eq!(lines, vec!["AMD Radeon AI PRO R9700", "arch  gfx1201"]);
    }
}
