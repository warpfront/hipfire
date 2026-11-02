// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire - see LICENSE and NOTICE in the project root.

//! O3c-1 live serve Dashboard data layer.
//!
//! The PARSE logic (HTTP JSON → struct, amd-smi / rocm-smi output → VRAM) is factored into
//! pure, I/O-free helpers so it can be unit-tested on CPU without a TTY, a
//! running serve, or a GPU. The fetchers ([`fetch_dashboard`]) do the network /
//! subprocess I/O and then hand raw text to those pure helpers.
//!
//! Honesty contract: every field is either REAL data parsed from the live
//! serve / GPU stats tool, or an explicit unavailable/offline state. We NEVER
//! fabricate placeholder numbers and present them as live.

use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use hipfire_client::{service_url, ServiceEndpoint};
use serde_json::Value;

use super::config::ConfigState;
use super::metrics::{self, MetricsRing, MetricsView};

/// Hard wall-clock bound on a single GPU-tool invocation (`amd-smi`,
/// `rocm-smi`, `rocminfo`, `hipconfig`). Even on the background thread a
/// zombie / hung tool must not accumulate forever, so we spawn it, poll for
/// completion, and `kill()` it past this deadline.
const ROCM_SMI_TIMEOUT: Duration = Duration::from_millis(2000);

/// Per-GPU VRAM reading parsed from `amd-smi` / `rocm-smi`. Values are bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMem {
    /// GPU index (amd-smi `gpu` / the rocm-smi `card0` ordinal).
    pub index: u32,
    pub used_bytes: u64,
    pub total_bytes: u64,
}

impl GpuMem {
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / 1e9
    }
    pub fn free_gb(&self) -> f64 {
        self.free_bytes() as f64 / 1e9
    }
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1e9
    }
}

/// VRAM availability state — distinguishes "we have real per-GPU numbers" from
/// the honest "no stats tool / not Linux / returned garbage" case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VramState {
    /// Real per-GPU readings parsed from amd-smi / rocm-smi.
    Available(Vec<GpuMem>),
    /// Stats tool missing, non-Linux, or unparseable. Carries an honest reason.
    Unavailable(String),
}

/// Per-GPU live load parsed from `amd-smi metric` (or `rocm-smi --showuse
/// --showtemp --showpower`). Each field is optional — key names vary by tool
/// and version, so a
/// missing reading is honestly `None`, never fabricated.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuLoad {
    pub index: u32,
    pub util_pct: Option<f64>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
}

/// GPU-load availability — real per-GPU readings, or an honest reason.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadState {
    Available(Vec<GpuLoad>),
    Unavailable(String),
}

/// Values the resident model was actually loaded with, as reported by the
/// serve's load ack (`/stats` `loaded_config`).
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedConfig {
    pub model: String,
    pub kv_mode: Option<String>,
    pub kv_backend: Option<String>,
    pub max_seq: Option<u64>,
}

/// Stats parsed from the serve's GET /stats endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeStats {
    pub model: Option<String>,
    pub uptime_s: u64,
    pub queue_depth: u64,
    pub requests_served: u64,
    pub recent_tok_s: Option<f64>,
    /// `None` on an older serve without the field, or while nothing is resident.
    pub loaded_config: Option<LoadedConfig>,
}

/// A full Dashboard snapshot. `serve_up` is the single source of truth for the
/// online/offline split the UI renders.
#[derive(Clone, Debug)]
pub struct Dashboard {
    pub serve_up: bool,
    pub endpoint: String,
    /// Raw GET /health body (or the transport-error string when serve is down).
    /// Mirrored into `StatusState.health_text` for the System tab so the UI
    /// thread never re-probes /health synchronously.
    pub health_text: String,
    /// Loaded model tag, from /stats or /health (None when idle / serve down).
    pub model: Option<String>,
    pub stats: Option<ServeStats>,
    /// Models advertised by GET /v1/models (ids).
    pub model_ids: Vec<String>,
    pub vram: VramState,
    /// Live per-GPU util / temp / power from amd-smi / rocm-smi (independent of serve).
    pub gpu_load: LoadState,
    /// When serve is down: an honest hint, e.g. how to start it.
    pub offline_hint: Option<String>,
    /// Live system diagnostics (GPU/HIP/kernel-cache/loaded-model). Probed off
    /// the UI thread by the worker. `None` until the first probe completes.
    pub system: Option<SystemInfo>,
}

impl Dashboard {
    /// Offline dashboard: serve unreachable. Carries a start hint and whatever
    /// VRAM we could still read (the stats CLI works without serve).
    pub fn offline(endpoint: String, vram: VramState, gpu_load: LoadState) -> Self {
        Self {
            serve_up: false,
            endpoint,
            health_text: String::new(),
            model: None,
            stats: None,
            model_ids: Vec::new(),
            vram,
            gpu_load,
            offline_hint: Some("serve offline — start it with `hipfire serve -d`".into()),
            system: None,
        }
    }

    /// Record why the serve cannot be probed (key-file or endpoint error,
    /// rejected bearer). Replaces the generic start hint and is mirrored into
    /// `health_text` so the System tab shows the real cause.
    fn with_failure(mut self, reason: String) -> Self {
        self.offline_hint = Some(reason.clone());
        self.health_text = reason;
        self
    }

    /// Demote a stale "up" snapshot when the always-on `/health` poller has
    /// since seen the serve go down, so no tab shows live stats for a dead
    /// serve. VRAM / GPU load (independent of the serve) are kept.
    pub fn mark_offline(&mut self, health: &HealthSnapshot) {
        self.serve_up = false;
        self.model = None;
        self.stats = None;
        self.model_ids.clear();
        self.health_text = health.text.clone();
        self.offline_hint = Some(
            health
                .failure
                .clone()
                .unwrap_or_else(|| "serve offline — start it with `hipfire serve -d`".into()),
        );
    }
}

/// Result of the lightweight `/health` poller. Published independently of the
/// full [`Dashboard`] snapshot so serve status is known on every tab.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HealthSnapshot {
    /// At least one probe has completed. Until then `serve_up == false` means
    /// "unknown", not "down".
    pub checked: bool,
    /// `/health` answered with a success status.
    pub serve_up: bool,
    /// Loaded model from the `/health` body (None when idle or serve down).
    pub model: Option<String>,
    /// Raw `/health` body on success; the failure reason for a config error or
    /// a rejected bearer; empty when the serve is simply unreachable.
    pub text: String,
    /// Set when the serve cannot be probed for a reason a restart would not fix
    /// (unloadable key file, malformed endpoint, bearer rejected). A serve may
    /// well be answering in that case, so callers must not start another one.
    pub failure: Option<String>,
}

/// One cheap, authenticated `GET /health`. Never panics, never blocks past the
/// agent's global timeout.
fn probe_health(agent: &ureq::Agent, config: &ConfigState) -> HealthSnapshot {
    let endpoint = match config.service_endpoint() {
        Ok(endpoint) => endpoint,
        Err(reason) => {
            return HealthSnapshot {
                checked: true,
                text: reason.clone(),
                failure: Some(reason),
                ..HealthSnapshot::default()
            };
        }
    };
    match get_text(agent, &endpoint, "/health") {
        TextFetch::Body(body) => HealthSnapshot {
            checked: true,
            serve_up: true,
            model: parse_health_model(&body),
            text: body,
            failure: None,
        },
        TextFetch::Rejected(status) => {
            let reason = format!(
                "serve rejected the bearer token (HTTP {status}); check serve.api_key_file"
            );
            HealthSnapshot {
                checked: true,
                text: reason.clone(),
                failure: Some(reason),
                ..HealthSnapshot::default()
            }
        }
        TextFetch::Unavailable => HealthSnapshot {
            checked: true,
            ..HealthSnapshot::default()
        },
    }
}

/// A single live system-diagnostic field. Each is either a real probed value
/// or an explicit honest `Unavailable(reason)` — we never fabricate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Probe {
    Value(String),
    Unavailable(String),
}

impl Probe {
    pub fn display(&self) -> &str {
        match self {
            Probe::Value(s) => s,
            Probe::Unavailable(s) => s,
        }
    }
    pub fn is_available(&self) -> bool {
        matches!(self, Probe::Value(_))
    }
}

/// Live system diagnostics for the System tab. Probed OFF the UI thread (by the
/// DashboardWorker) so a slow `amd-smi` / `hipcc` invocation never blocks
/// render or input. Every field is an honest [`Probe`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemInfo {
    /// GPU product name(s) from `amd-smi static --asic` (or `rocm-smi --showproductname`).
    pub gpu_name: Probe,
    /// GPU arch / gfx target (e.g. `gfx1100`) from `amd-smi static --asic` or `rocminfo`.
    pub gpu_arch: Probe,
    /// HIP / ROCm runtime version (`hipconfig --version` or /opt/rocm/.info).
    pub hip_version: Probe,
    /// Kernel-cache directory presence + path.
    pub kernel_cache: Probe,
    /// Currently loaded serve model (from the live snapshot, None when idle).
    pub loaded_model: Probe,
    /// Checksum of the loaded model file when readable (size-based fast hash).
    pub model_checksum: Probe,
}

impl SystemInfo {
    /// Probe everything that is platform-static (GPU, HIP, kernel cache). The
    /// loaded-model + checksum are filled in separately from the live serve
    /// snapshot via [`SystemInfo::with_loaded_model`].
    pub fn probe() -> Self {
        if !cfg!(target_os = "linux") {
            let na = || Probe::Unavailable("unavailable: Linux-only probe".into());
            return Self {
                gpu_name: na(),
                gpu_arch: na(),
                hip_version: na(),
                kernel_cache: probe_kernel_cache(),
                loaded_model: Probe::Unavailable("no model loaded".into()),
                model_checksum: Probe::Unavailable("no model loaded".into()),
            };
        }
        let (gpu_name, gpu_arch) = probe_gpu();
        Self {
            gpu_name,
            gpu_arch,
            hip_version: probe_hip_version(),
            kernel_cache: probe_kernel_cache(),
            loaded_model: Probe::Unavailable("no model loaded".into()),
            model_checksum: Probe::Unavailable("no model loaded".into()),
        }
    }

    /// Fold the live loaded-model (from the serve snapshot) into the static
    /// probe, computing a cheap file checksum when the path is readable.
    pub fn with_loaded_model(mut self, model: Option<&str>) -> Self {
        match model {
            Some(m) if !m.is_empty() => {
                self.loaded_model = Probe::Value(m.to_string());
                self.model_checksum = probe_model_checksum(m);
            }
            _ => {
                self.loaded_model = Probe::Unavailable("no model loaded".into());
                self.model_checksum = Probe::Unavailable("no model loaded".into());
            }
        }
        self
    }
}

/// Probe GPU product name + arch. Product name and arch come from `amd-smi`
/// when present, else `rocm-smi` (name) / `rocminfo` (arch); every tool is
/// resolved through the ROCm root and bounded so a hung tool cannot wedge the
/// worker. Returns honest unavailable reasons on failure.
fn probe_gpu() -> (Probe, Probe) {
    let (name, smi_arch) = match resolve_smi_tool() {
        Some(tool @ SmiTool::AmdSmi(_)) => match smi_text(&tool, &AMD_SMI_ASIC_ARGS) {
            Ok(text) => parse_amd_smi_asic(&text),
            Err(e) => {
                let reason = format!("unavailable: {e}");
                (
                    Probe::Unavailable(reason.clone()),
                    Probe::Unavailable(reason),
                )
            }
        },
        Some(tool @ SmiTool::RocmSmi(_)) => {
            let name = match smi_text(&tool, &["--showproductname", "--json"]) {
                Ok(text) => parse_rocm_product_name(&text),
                Err(e) => Probe::Unavailable(format!("unavailable: {e}")),
            };
            (
                name,
                Probe::Unavailable("unavailable: rocm-smi reports no gfx target".into()),
            )
        }
        None => {
            let reason = gpu_tool_missing_reason(hipfire_config::rocm::root().as_deref());
            (
                Probe::Unavailable(reason.clone()),
                Probe::Unavailable(reason),
            )
        }
    };
    if smi_arch.is_available() {
        return (name, smi_arch);
    }

    let arch = match resolve_tool("rocminfo") {
        None => Probe::Unavailable("unavailable: rocminfo not found".into()),
        Some(rocminfo) => match run_bounded(Command::new(rocminfo), ROCM_SMI_TIMEOUT) {
            RocmSmiRun::Output(text) => parse_rocminfo_arch(&text),
            RocmSmiRun::NotFound => Probe::Unavailable("unavailable: rocminfo not found".into()),
            RocmSmiRun::TimedOut => Probe::Unavailable("unavailable: rocminfo timed out".into()),
            RocmSmiRun::SpawnError(e) => {
                Probe::Unavailable(format!("unavailable: rocminfo error: {e}"))
            }
        },
    };
    (name, arch)
}

/// Parse `rocm-smi --showproductname --json`: ROCm emits per-card objects with
/// a `"Card Series"` / `"Card Model"` / `"Card SKU"` string field (key varies
/// by ROCm version). Returns the first non-empty product string found.
pub fn parse_rocm_product_name(body: &str) -> Probe {
    let Ok(v) = serde_json::from_str::<Value>(body) else {
        return Probe::Unavailable("unavailable: rocm-smi product JSON unparseable".into());
    };
    let Some(obj) = v.as_object() else {
        return Probe::Unavailable("unavailable: rocm-smi product JSON not an object".into());
    };
    let keys = [
        "Card Series",
        "Card Model",
        "Card SKU",
        "Device Name",
        "GPU ID",
    ];
    let mut names: Vec<String> = Vec::new();
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    for (_, card) in cards {
        for key in keys {
            if let Some(Value::String(s)) = card.get(key) {
                let s = s.trim();
                if !s.is_empty() {
                    names.push(s.to_string());
                    break;
                }
            }
        }
    }
    if names.is_empty() {
        Probe::Unavailable("unavailable: no product name in rocm-smi output".into())
    } else {
        names.dedup();
        Probe::Value(names.join(", "))
    }
}

/// Parse `rocminfo` text output for the first GPU agent's `gfx` name (the
/// `Name:` line under an Agent of Device Type GPU). Returns the gfx target.
pub fn parse_rocminfo_arch(body: &str) -> Probe {
    // rocminfo lists agents; the GPU agent has a `Name:  gfxNNNN` line.
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Name:") {
            let name = rest.trim();
            if name.starts_with("gfx") {
                return Probe::Value(name.to_string());
            }
        }
        // Some builds use "gfx Name:" / "Marketing Name:" — catch the gfx token.
        if let Some(idx) = t.find("gfx") {
            let tok: String = t[idx..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if tok.len() > 3 && tok[3..].chars().any(|c| c.is_ascii_digit()) {
                return Probe::Value(tok);
            }
        }
    }
    Probe::Unavailable("unavailable: no gfx target in rocminfo output".into())
}

/// Probe the HIP/ROCm version. Prefers `hipconfig --version` (resolved through
/// the ROCm root), falls back to the selected root's `.info/version`.
fn probe_hip_version() -> Probe {
    if let Some(hipconfig) = resolve_tool("hipconfig") {
        let mut cmd = Command::new(hipconfig);
        cmd.arg("--version");
        if let RocmSmiRun::Output(text) = run_bounded(cmd, ROCM_SMI_TIMEOUT) {
            let v = text.trim();
            if !v.is_empty() {
                return Probe::Value(format!("HIP {v}"));
            }
        }
    }
    if let Some(v) = hipfire_config::rocm::version() {
        return Probe::Value(format!("ROCm {v}"));
    }
    Probe::Unavailable("unavailable: hipconfig not found and no ROCm .info/version".into())
}

/// Resolve the kernel-cache directory the engine uses and report its presence.
/// Honors `HIPFIRE_KERNEL_CACHE` then falls back to `~/.hipfire_kernels` and the
/// HIP default `~/.cache/hip`.
fn kernel_cache_dir() -> Option<PathBuf> {
    if let Some(v) = env::var_os("HIPFIRE_KERNEL_CACHE") {
        return Some(PathBuf::from(v));
    }
    let home = env::var_os("HOME").map(PathBuf::from)?;
    let primary = home.join(".hipfire_kernels");
    if primary.exists() {
        return Some(primary);
    }
    Some(home.join(".cache").join("hip"))
}

fn probe_kernel_cache() -> Probe {
    match kernel_cache_dir() {
        Some(dir) => {
            let exists = dir.exists();
            let count = if exists {
                fs::read_dir(&dir).map(|rd| rd.count()).unwrap_or(0)
            } else {
                0
            };
            let disp = dir.display();
            if exists {
                Probe::Value(format!("{disp} (present, {count} entries)"))
            } else {
                Probe::Unavailable(format!("{disp} (absent — populated on first run)"))
            }
        }
        None => Probe::Unavailable("unavailable: cannot resolve cache dir (no HOME)".into()),
    }
}

/// Compute a cheap, stable checksum for the loaded model file. The model field
/// is often an absolute path; when it is and the file is readable we emit a
/// size + FNV-1a hash of the first/last 64 KiB (full hashing a multi-GB weight
/// file on the worker thread would be wasteful). Honest unavailable otherwise.
fn probe_model_checksum(model: &str) -> Probe {
    let path = PathBuf::from(model);
    if !path.is_absolute() || !path.exists() {
        return Probe::Unavailable("unavailable: model path not a readable file".into());
    }
    let meta = match fs::metadata(&path) {
        Ok(m) => m,
        Err(e) => return Probe::Unavailable(format!("unavailable: stat failed: {e}")),
    };
    if !meta.is_file() {
        return Probe::Unavailable("unavailable: model path is not a file".into());
    }
    let size = meta.len();
    match fnv_head_tail(&path, size) {
        Some(hash) => Probe::Value(format!("{hash:016x} ({:.2} GB)", size as f64 / 1e9)),
        None => Probe::Unavailable("unavailable: model file unreadable".into()),
    }
}

/// FNV-1a over (size ++ first 64 KiB ++ last 64 KiB) — fast, deterministic,
/// good enough to distinguish loaded weights without reading gigabytes.
fn fnv_head_tail(path: &std::path::Path, size: u64) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};
    const CHUNK: u64 = 64 * 1024;
    let mut f = fs::File::open(path).ok()?;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fold = |bytes: &[u8], hash: &mut u64| {
        for b in bytes {
            *hash ^= *b as u64;
            *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    fold(&size.to_le_bytes(), &mut hash);
    let mut head = vec![0u8; CHUNK.min(size) as usize];
    f.read_exact(&mut head).ok()?;
    fold(&head, &mut hash);
    if size > CHUNK {
        let tail_len = CHUNK.min(size - CHUNK);
        f.seek(SeekFrom::End(-(tail_len as i64))).ok()?;
        let mut tail = vec![0u8; tail_len as usize];
        f.read_exact(&mut tail).ok()?;
        fold(&tail, &mut hash);
    }
    Some(hash)
}

// ── Pure parse helpers (no I/O) ─────────────────────────────────────────────

/// Parse the GET /health JSON body. Returns the loaded-model field (which may
/// be JSON null → None). Returns None for garbage / non-object input rather
/// than fabricating a value.
pub fn parse_health_model(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Parse the GET /stats JSON body into [`ServeStats`]. Returns None on garbage
/// / non-object input. Missing numeric fields default to 0 (the serve always
/// emits them, but be defensive against an older build); `recent_tok_s` stays
/// None unless present and finite/positive.
pub fn parse_stats(body: &str) -> Option<ServeStats> {
    let v: Value = serde_json::from_str(body).ok()?;
    if !v.is_object() {
        return None;
    }
    let model = match v.get("model") {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    let as_u64 = |key: &str| v.get(key).and_then(|x| x.as_u64()).unwrap_or(0);
    let recent_tok_s = v
        .get("recent_tok_s")
        .and_then(|x| x.as_f64())
        .filter(|f| f.is_finite() && *f > 0.0);
    Some(ServeStats {
        model,
        // serve emits `uptime_sec`; `uptime_s` is the older test-fixture spelling.
        uptime_s: v
            .get("uptime_sec")
            .or_else(|| v.get("uptime_s"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0),
        queue_depth: as_u64("queue_depth"),
        requests_served: as_u64("requests_served"),
        recent_tok_s,
        loaded_config: parse_loaded_config(v.get("loaded_config")),
    })
}

/// Parse the `/stats` `loaded_config` object. A missing field, a non-object
/// (`null` while no model is resident) or an object without a model name is
/// `None`, so an older serve that lacks the field still parses; the inner
/// KV/context fields stay `None` individually when null or mistyped.
fn parse_loaded_config(value: Option<&Value>) -> Option<LoadedConfig> {
    let obj = value?.as_object()?;
    let model = match obj.get("model") {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        _ => return None,
    };
    let text = |key: &str| match obj.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    };
    Some(LoadedConfig {
        model,
        kv_mode: text("kv_mode"),
        kv_backend: text("kv_backend"),
        max_seq: obj.get("max_seq").and_then(|x| x.as_u64()),
    })
}

/// Parse the GET /v1/models JSON body into the list of model ids. Empty vec on
/// garbage input.
pub fn parse_model_ids(body: &str) -> Vec<String> {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Parse `rocm-smi --showmeminfo vram --json` output. ROCm emits a JSON object
/// keyed by `card0`, `card1`, … with `"VRAM Total Memory (B)"` and
/// `"VRAM Total Used Memory (B)"` string fields. Returns an honest
/// [`VramState::Unavailable`] when nothing parses.
pub fn parse_rocm_smi_json(body: &str) -> VramState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return VramState::Unavailable(format!("rocm-smi JSON parse failed: {e}")),
    };
    let Some(obj) = v.as_object() else {
        return VramState::Unavailable("rocm-smi JSON not an object".into());
    };
    let mut gpus: Vec<GpuMem> = Vec::new();
    // Iterate cardN keys in numeric order.
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    for (idx, card) in cards {
        let total = field_u64(card, "VRAM Total Memory (B)");
        let used = field_u64(card, "VRAM Total Used Memory (B)");
        if let (Some(total), Some(used)) = (total, used) {
            if total > 0 {
                gpus.push(GpuMem {
                    index: idx,
                    used_bytes: used,
                    total_bytes: total,
                });
            }
        }
    }
    if gpus.is_empty() {
        VramState::Unavailable("rocm-smi returned no VRAM fields".into())
    } else {
        VramState::Available(gpus)
    }
}

/// rocm-smi reports numbers as JSON strings; read either a string-encoded or a
/// raw number.
fn field_u64(card: &Value, key: &str) -> Option<u64> {
    match card.get(key)? {
        Value::String(s) => s.trim().parse::<u64>().ok(),
        Value::Number(n) => n.as_u64(),
        _ => None,
    }
}

// ── amd-smi parsers (pure) ──────────────────────────────────────────────────

/// The per-GPU entries of an `amd-smi … --json` document as `(gpu index,
/// entry)`. amd-smi versions emit either `{"gpu_data": [...]}` or a bare
/// top-level array; both are accepted. The index comes from the entry's `gpu`
/// field, falling back to its position.
fn amd_smi_entries(v: &Value) -> Option<Vec<(u32, &Value)>> {
    let arr = match v {
        Value::Array(a) => a,
        Value::Object(o) => o.get("gpu_data")?.as_array()?,
        _ => return None,
    };
    Some(
        arr.iter()
            .enumerate()
            .map(|(pos, entry)| {
                let idx = entry
                    .get("gpu")
                    .and_then(Value::as_u64)
                    .map_or(pos as u32, |g| g as u32);
                (idx, entry)
            })
            .collect(),
    )
}

/// A finite number from a JSON number or numeric string. amd-smi's `"N/A"`
/// (and anything else non-numeric) is `None`.
fn finite_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
    .filter(|n| n.is_finite())
}

/// An amd-smi reading: `{"value": N, "unit": "MB"}` (or a bare number), with
/// the unit when present. `"N/A"` / missing → `None`.
fn amd_reading(v: Option<&Value>) -> Option<(f64, Option<&str>)> {
    match v? {
        Value::Object(o) => {
            let n = finite_number(o.get("value")?)?;
            Some((n, o.get("unit").and_then(Value::as_str)))
        }
        other => finite_number(other).map(|n| (n, None)),
    }
}

/// A memory reading in bytes. amd-smi's "MB" is MiB (a 32 GB card reports
/// 32624 MB). A reading without a recognised unit is `None` rather than a guess.
fn amd_bytes(v: Option<&Value>) -> Option<u64> {
    let (n, unit) = amd_reading(v)?;
    let unit = unit?.trim().to_ascii_lowercase();
    let mult = match unit.as_str() {
        "b" => 1.0,
        "kb" | "kib" => 1024.0,
        "mb" | "mib" => 1024.0 * 1024.0,
        "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    (n >= 0.0).then(|| (n * mult).round() as u64)
}

/// Parse the JSON of `amd-smi metric --mem-usage --json`
/// (`gpu_data[].mem_usage.{total_vram,used_vram}`) into per-GPU VRAM.
pub fn parse_amd_smi_vram(body: &str) -> VramState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return VramState::Unavailable(format!("amd-smi JSON parse failed: {e}")),
    };
    let Some(entries) = amd_smi_entries(&v) else {
        return VramState::Unavailable("amd-smi JSON has no gpu_data".into());
    };
    let mut gpus = Vec::new();
    for (idx, entry) in entries {
        let mem = entry.get("mem_usage");
        let total = amd_bytes(mem.and_then(|m| m.get("total_vram")));
        let used = amd_bytes(mem.and_then(|m| m.get("used_vram")));
        if let (Some(total), Some(used)) = (total, used) {
            if total > 0 {
                gpus.push(GpuMem {
                    index: idx,
                    used_bytes: used,
                    total_bytes: total,
                });
            }
        }
    }
    if gpus.is_empty() {
        VramState::Unavailable("amd-smi returned no VRAM fields".into())
    } else {
        VramState::Available(gpus)
    }
}

/// Parse the JSON of `amd-smi metric --usage --power --temperature --json`
/// (`usage.gfx_activity` %, `power.socket_power` W, `temperature.edge` /
/// `hotspot` °C) into per-GPU load. Each reading is range-validated; `"N/A"`
/// is an honest `None`.
pub fn parse_amd_smi_load(body: &str) -> LoadState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return LoadState::Unavailable(format!("amd-smi load JSON parse failed: {e}")),
    };
    let Some(entries) = amd_smi_entries(&v) else {
        return LoadState::Unavailable("amd-smi JSON has no gpu_data".into());
    };
    let first_reading =
        |section: Option<&Value>, keys: &[&str], range: std::ops::RangeInclusive<f64>| {
            keys.iter()
                .find_map(|k| amd_reading(section.and_then(|s| s.get(*k))))
                .map(|(n, _)| n)
                .filter(|n| range.contains(n))
        };
    let mut loads = Vec::new();
    for (idx, entry) in entries {
        let util = first_reading(entry.get("usage"), &["gfx_activity"], 0.0..=100.0);
        let temp = first_reading(entry.get("temperature"), &["edge", "hotspot"], 0.0..=200.0);
        let power = first_reading(
            entry.get("power"),
            &["socket_power", "average_socket_power"],
            0.0..=2000.0,
        );
        if util.is_some() || temp.is_some() || power.is_some() {
            loads.push(GpuLoad {
                index: idx,
                util_pct: util,
                temp_c: temp,
                power_w: power,
            });
        }
    }
    if loads.is_empty() {
        LoadState::Unavailable("amd-smi returned no load fields".into())
    } else {
        LoadState::Available(loads)
    }
}

/// Parse `amd-smi static --asic --json` (`gpu_data[].asic.market_name`,
/// `target_graphics_version`) into `(product name, gfx arch)` probes. Repeated
/// identical values collapse; differing ones are comma-joined in GPU order.
pub fn parse_amd_smi_asic(body: &str) -> (Probe, Probe) {
    let fail = |reason: String| {
        (
            Probe::Unavailable(reason.clone()),
            Probe::Unavailable(reason),
        )
    };
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return fail("unavailable: amd-smi asic JSON unparseable".into()),
    };
    let Some(entries) = amd_smi_entries(&v) else {
        return fail("unavailable: amd-smi asic JSON has no gpu_data".into());
    };
    let text_field = |entry: &Value, key: &str| -> Option<String> {
        let s = entry.get("asic")?.get(key)?.as_str()?.trim();
        (!s.is_empty() && !s.eq_ignore_ascii_case("n/a")).then(|| s.to_string())
    };
    let mut names: Vec<String> = Vec::new();
    let mut arches: Vec<String> = Vec::new();
    for (_, entry) in entries {
        names.extend(text_field(entry, "market_name"));
        arches.extend(text_field(entry, "target_graphics_version"));
    }
    names.dedup();
    arches.dedup();
    let name = if names.is_empty() {
        Probe::Unavailable("unavailable: no product name in amd-smi output".into())
    } else {
        Probe::Value(names.join(", "))
    };
    let arch = if arches.is_empty() {
        Probe::Unavailable("unavailable: no gfx target in amd-smi output".into())
    } else {
        Probe::Value(arches.join(", "))
    };
    (name, arch)
}

// ── Fetchers (I/O; thin wrappers around the pure parsers) ───────────────────

/// Result of one authenticated text GET against the serve.
enum TextFetch {
    Body(String),
    /// 401/403: serve is reachable but refused (or required) the bearer token.
    Rejected(u16),
    /// Transport failure, other non-success status, or unreadable body.
    Unavailable,
}

fn get_text(agent: &ureq::Agent, endpoint: &ServiceEndpoint, path: &str) -> TextFetch {
    match endpoint
        .authorize(agent.get(&service_url(endpoint, path)))
        .call()
    {
        Ok(mut resp) => {
            let status = resp.status().as_u16();
            if matches!(status, 401 | 403) {
                TextFetch::Rejected(status)
            } else if status >= 400 {
                TextFetch::Unavailable
            } else {
                resp.body_mut()
                    .read_to_string()
                    .map_or(TextFetch::Unavailable, TextFetch::Body)
            }
        }
        Err(_) => TextFetch::Unavailable,
    }
}

/// Body of a successful GET; `None` for any failure.
fn get_body(agent: &ureq::Agent, endpoint: &ServiceEndpoint, path: &str) -> Option<String> {
    match get_text(agent, endpoint, path) {
        TextFetch::Body(body) => Some(body),
        TextFetch::Rejected(_) | TextFetch::Unavailable => None,
    }
}

/// Outcome of a bounded subprocess run. Separated from parsing so the timeout /
/// kill path stays testable in isolation. Shared with the doctor module.
pub(crate) enum RocmSmiRun {
    /// Process exited within the deadline; carries captured stdout.
    Output(String),
    /// Binary not present on PATH.
    NotFound,
    /// Spawn or wait failed for some other reason.
    SpawnError(String),
    /// Process exceeded [`ROCM_SMI_TIMEOUT`] and was killed.
    TimedOut,
}

/// `amd-smi` arguments for the one call that feeds both the VRAM and the load
/// panels (`gpu_data[].{mem_usage,usage,power,temperature}`).
const AMD_SMI_METRIC_ARGS: [&str; 6] = [
    "metric",
    "--usage",
    "--power",
    "--temperature",
    "--mem-usage",
    "--json",
];
/// `amd-smi` arguments for product name + gfx target (`gpu_data[].asic`).
const AMD_SMI_ASIC_ARGS: [&str; 3] = ["static", "--asic", "--json"];

/// The GPU-stats CLI the dashboard talks to, with its resolved path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SmiTool {
    AmdSmi(PathBuf),
    RocmSmi(PathBuf),
}

impl SmiTool {
    fn name(&self) -> &'static str {
        match self {
            SmiTool::AmdSmi(_) => "amd-smi",
            SmiTool::RocmSmi(_) => "rocm-smi",
        }
    }

    fn path(&self) -> &Path {
        match self {
            SmiTool::AmdSmi(p) | SmiTool::RocmSmi(p) => p,
        }
    }
}

/// Pure selection over already-resolved candidate paths: `amd-smi` is preferred
/// (newer ROCm ships only it); `rocm-smi` is the fallback.
pub(crate) fn select_smi_tool(
    amd_smi: Option<PathBuf>,
    rocm_smi: Option<PathBuf>,
) -> Option<SmiTool> {
    amd_smi
        .map(SmiTool::AmdSmi)
        .or_else(|| rocm_smi.map(SmiTool::RocmSmi))
}

/// First executable regular file named `name` in `dirs`.
fn find_executable_in<I: IntoIterator<Item = PathBuf>>(dirs: I, name: &str) -> Option<PathBuf> {
    dirs.into_iter()
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable_file(candidate))
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Resolve a ROCm tool the way the CLI does: through the selected ROCm root
/// (`hipfire_config::rocm::tool`, which only consults PATH when no root is
/// selectable), then a plain PATH search for a tool that root lacks.
fn resolve_tool(name: &str) -> Option<PathBuf> {
    hipfire_config::rocm::tool(name).or_else(|| {
        let path = env::var_os("PATH")?;
        find_executable_in(env::split_paths(&path), name)
    })
}

fn resolve_smi_tool() -> Option<SmiTool> {
    select_smi_tool(resolve_tool("amd-smi"), resolve_tool("rocm-smi"))
}

/// Parse `rocm-smi --showuse --showtemp --showpower --json`. Field key names
/// vary by ROCm version, so each metric is matched by substring (in preference
/// order) and a missing field is honestly `None`.
pub fn parse_rocm_smi_load_json(body: &str) -> LoadState {
    let v: Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => return LoadState::Unavailable(format!("rocm-smi load JSON parse failed: {e}")),
    };
    let Some(obj) = v.as_object() else {
        return LoadState::Unavailable("rocm-smi load JSON not an object".into());
    };
    let mut cards: Vec<(u32, &Value)> = obj
        .iter()
        .filter_map(|(k, val)| {
            k.strip_prefix("card")
                .and_then(|n| n.parse::<u32>().ok())
                .map(|idx| (idx, val))
        })
        .collect();
    cards.sort_by_key(|(idx, _)| *idx);
    let mut loads = Vec::new();
    for (idx, card) in cards {
        // Specific needles only (no bare "Power"/"Temperature" fallback, which
        // could latch onto a cap/sensor key); each reading is range-validated so
        // an implausible value becomes an honest None.
        let util = field_f64_containing(card, &["GPU use (%)", "GPU use"])
            .filter(|u| (0.0..=100.0).contains(u));
        let temp = field_f64_containing(
            card,
            &["Temperature (Sensor edge)", "Temperature (Sensor junction)"],
        )
        .filter(|t| (0.0..=200.0).contains(t));
        let power = field_f64_containing(
            card,
            &[
                "Average Graphics Package Power",
                "Current Socket Graphics Package Power",
                "Current Graphics Package Power",
            ],
        )
        .filter(|p| (0.0..=2000.0).contains(p));
        if util.is_some() || temp.is_some() || power.is_some() {
            loads.push(GpuLoad {
                index: idx,
                util_pct: util,
                temp_c: temp,
                power_w: power,
            });
        }
    }
    if loads.is_empty() {
        LoadState::Unavailable("rocm-smi returned no load fields".into())
    } else {
        LoadState::Available(loads)
    }
}

/// First card field whose key CONTAINS one of `needles` (tried in order),
/// parsed as a FINITE f64. rocm-smi emits numbers as strings, and
/// `f64::from_str` accepts "NaN"/"inf" (which rocm-smi reports for unsupported
/// sensors on some SKUs) — we reject those so a non-finite reading falls through
/// to an honest `None` rather than rendering "NaN% util". Cap / Max / Min
/// sentinel keys are skipped so a "Power Cap" can't be mistaken for live draw.
fn field_f64_containing(card: &Value, needles: &[&str]) -> Option<f64> {
    let obj = card.as_object()?;
    for needle in needles {
        for (k, val) in obj {
            if k.contains("Cap") || k.contains("Max") || k.contains("Min") {
                continue;
            }
            if k.contains(needle) {
                let parsed = val
                    .as_str()
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .or_else(|| val.as_f64());
                if let Some(n) = parsed {
                    if n.is_finite() {
                        return Some(n);
                    }
                }
            }
        }
    }
    None
}

/// Spawn `cmd`, capturing stdout, and enforce a hard `timeout`. Polls
/// `try_wait`; on deadline it `kill()`s and reaps the child so no zombie leaks.
/// Generic over the command so the timeout/kill path is unit-testable with a
/// stand-in (`sleep`) without depending on rocm-smi.
pub(crate) fn run_bounded(mut cmd: Command, timeout: Duration) -> RocmSmiRun {
    let mut child = match cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RocmSmiRun::NotFound,
        Err(e) => return RocmSmiRun::SpawnError(e.to_string()),
    };

    // Drain stdout CONCURRENTLY on a reader thread. A verbose command (rocminfo)
    // can emit more than one OS pipe buffer (~64 KiB) of output BEFORE it exits;
    // if we only read after try_wait() reports exit, the child blocks on its
    // write() — full pipe, no reader — never exits, and we kill it as a FALSE
    // timeout. With a dedicated reader thread the pipe never backs up, so the
    // child can run to completion while the watchdog below still bounds the
    // wall-clock and kills a genuinely hung child.
    let stdout = child.stdout.take();
    let reader = stdout.map(|mut out| {
        thread::spawn(move || {
            let mut buf = String::new();
            let _ = out.read_to_string(&mut buf);
            buf
        })
    });

    // Watchdog: poll for exit, enforce the hard deadline (kill + reap on timeout).
    let deadline = Instant::now() + timeout;
    let outcome = loop {
        match child.try_wait() {
            Ok(Some(_status)) => break None, // exited cleanly within deadline
            Ok(None) => {
                if Instant::now() >= deadline {
                    // Hung: kill and reap so we don't leak a zombie. Killing the
                    // child also closes the pipe, so the reader thread's
                    // read_to_string returns and we can join it without hanging.
                    let _ = child.kill();
                    let _ = child.wait();
                    break Some(RocmSmiRun::TimedOut);
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                // Reap best-effort so the reader thread unblocks, then report.
                let _ = child.kill();
                let _ = child.wait();
                break Some(RocmSmiRun::SpawnError(e.to_string()));
            }
        }
    };

    // Join the reader (it has finished once the pipe closed: clean exit OR kill).
    let captured = match reader {
        Some(h) => h.join().unwrap_or_default(),
        None => String::new(),
    };

    match outcome {
        Some(run) => run, // TimedOut / SpawnError — discard partial output
        None => RocmSmiRun::Output(captured),
    }
}

/// Why neither GPU stats CLI could be used. Names the ROCm root that was
/// searched so the message is checkable by hand; deliberately does NOT claim
/// "no GPU" — a missing stats tool says nothing about the PCI devices.
pub(crate) fn gpu_tool_missing_reason(root: Option<&Path>) -> String {
    let root = root.map_or_else(|| "none".to_string(), |r| r.display().to_string());
    format!("GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: {root})")
}

/// Run `tool` with `args` under the hard deadline; `Err` carries a short,
/// tool-named reason ("amd-smi timed out after 2000ms (killed)").
fn smi_text(tool: &SmiTool, args: &[&str]) -> Result<String, String> {
    let mut cmd = Command::new(tool.path());
    cmd.args(args);
    match run_bounded(cmd, ROCM_SMI_TIMEOUT) {
        RocmSmiRun::Output(text) => Ok(text),
        RocmSmiRun::NotFound => Err(format!("{} not found", tool.name())),
        RocmSmiRun::TimedOut => Err(format!(
            "{} timed out after {}ms (killed)",
            tool.name(),
            ROCM_SMI_TIMEOUT.as_millis()
        )),
        RocmSmiRun::SpawnError(e) => Err(format!("{} error: {e}", tool.name())),
    }
}

/// Surface "the tool ran but printed nothing" as its own clean reason instead
/// of the JSON parser's error.
fn vram_or_no_output(parsed: VramState, text: &str, tool: &str) -> VramState {
    match parsed {
        VramState::Unavailable(_) if text.trim().is_empty() => {
            VramState::Unavailable(format!("VRAM unavailable: {tool} produced no output"))
        }
        other => other,
    }
}

/// Probe per-GPU VRAM and live load with whichever stats CLI the ROCm root
/// provides (`amd-smi` preferred, `rocm-smi` fallback). `amd-smi` serves both
/// readings from ONE bounded call; `rocm-smi` needs two. Honest
/// `Unavailable` states on non-Linux, missing tools, timeouts (hard-killed) and
/// unparseable output. Never blocks indefinitely.
pub fn fetch_gpu_stats() -> (VramState, LoadState) {
    if !cfg!(target_os = "linux") {
        return (
            VramState::Unavailable("VRAM unavailable: GPU stats are Linux-only".into()),
            LoadState::Unavailable("GPU load unavailable: GPU stats are Linux-only".into()),
        );
    }
    let Some(tool) = resolve_smi_tool() else {
        let reason = gpu_tool_missing_reason(hipfire_config::rocm::root().as_deref());
        return (
            VramState::Unavailable(reason),
            LoadState::Unavailable("GPU load unavailable: no GPU stats tool".into()),
        );
    };
    match &tool {
        SmiTool::AmdSmi(_) => match smi_text(&tool, &AMD_SMI_METRIC_ARGS) {
            Ok(text) => (
                vram_or_no_output(parse_amd_smi_vram(&text), &text, tool.name()),
                parse_amd_smi_load(&text),
            ),
            Err(e) => (
                VramState::Unavailable(format!("VRAM unavailable: {e}")),
                LoadState::Unavailable(format!("GPU load unavailable: {e}")),
            ),
        },
        SmiTool::RocmSmi(_) => {
            let vram = match smi_text(&tool, &["--showmeminfo", "vram", "--json"]) {
                Ok(text) => vram_or_no_output(parse_rocm_smi_json(&text), &text, tool.name()),
                Err(e) => VramState::Unavailable(format!("VRAM unavailable: {e}")),
            };
            let load = match smi_text(&tool, &["--showuse", "--showtemp", "--showpower", "--json"])
            {
                Ok(text) => parse_rocm_smi_load_json(&text),
                Err(e) => LoadState::Unavailable(format!("GPU load unavailable: {e}")),
            };
            (vram, load)
        }
    }
}

/// Build a full live Dashboard snapshot: probes /health, /stats, /v1/models,
/// and the GPU stats CLI. Tolerant of serve being down (returns
/// [`Dashboard::offline`]) and of the stats tool being absent
/// (VramState::Unavailable). All timeouts are
/// short so the refresh tick never wedges the UI.
pub fn fetch_dashboard(config: &ConfigState) -> Dashboard {
    let display_endpoint = format!("{}:{}", config.probe_host(), config.port);
    let (vram, gpu_load) = fetch_gpu_stats();

    // An unloadable key file or malformed endpoint is surfaced as-is — never
    // downgraded to an unauthenticated probe.
    let endpoint = match config.service_endpoint() {
        Ok(endpoint) => endpoint,
        Err(reason) => {
            return Dashboard::offline(display_endpoint, vram, gpu_load).with_failure(reason);
        }
    };

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(450)))
        .http_status_as_error(false)
        // Never forward the bearer to another origin.
        .max_redirects(0)
        .build()
        .into();

    let health = match get_text(&agent, &endpoint, "/health") {
        TextFetch::Body(body) => body,
        TextFetch::Rejected(status) => {
            return Dashboard::offline(display_endpoint, vram, gpu_load).with_failure(format!(
                "serve rejected the bearer token (HTTP {status}); check serve.api_key_file"
            ));
        }
        // Serve unreachable — honest offline state (VRAM/load may still read).
        TextFetch::Unavailable => return Dashboard::offline(display_endpoint, vram, gpu_load),
    };
    let health_model = parse_health_model(&health);

    let stats = get_body(&agent, &endpoint, "/stats").and_then(|b| parse_stats(&b));
    let model_ids = get_body(&agent, &endpoint, "/v1/models")
        .map(|b| parse_model_ids(&b))
        .unwrap_or_default();

    // Prefer the model from /stats (authoritative), fall back to /health.
    let model = stats
        .as_ref()
        .and_then(|s| s.model.clone())
        .or(health_model);

    Dashboard {
        serve_up: true,
        endpoint: display_endpoint,
        health_text: health,
        model,
        stats,
        model_ids,
        vram,
        gpu_load,
        offline_hint: None,
        system: None,
    }
}

// ── Background fetch worker ─────────────────────────────────────────────────

/// How often the background thread re-fetches a snapshot while the Dashboard
/// tab is active.
const WORKER_REFRESH: Duration = Duration::from_millis(1500);
/// Poll cadence for the worker's "is the tab active?" / "should I exit?" flags
/// while it is idle (Dashboard tab not focused).
const WORKER_IDLE_POLL: Duration = Duration::from_millis(150);
/// Cadence of the always-on `/health` poller (every tab, whole TUI lifetime).
const HEALTH_REFRESH: Duration = Duration::from_secs(2);

/// Owns the background fetch threads. The UI thread NEVER calls
/// [`fetch_dashboard`] / rocm-smi / HTTP directly; it only reads the latest
/// snapshot via [`DashboardWorker::snapshot`]. The worker fetches on its own
/// thread every [`WORKER_REFRESH`] while the Dashboard tab is active, writing
/// the result into a shared `Arc<Mutex<Option<Dashboard>>>`, so a hung
/// rocm-smi or slow HTTP probe can stall only the worker — never render/input.
/// A second thread scrapes `/metrics` at the same cadence for the TUI's whole
/// lifetime, so the Live metrics card has history the moment it opens.
pub struct DashboardWorker {
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    /// Latest Live metrics view, published by the `/metrics` thread.
    metrics: Arc<Mutex<MetricsView>>,
    /// Latest `/health` result, published by the always-on health thread.
    health: Arc<Mutex<HealthSnapshot>>,
    /// Whether the Dashboard tab is currently focused. The worker only fetches
    /// when this is true (plus an immediate fetch when it flips on).
    active: Arc<AtomicBool>,
    /// Set by the UI to force an immediate re-fetch (manual `r` refresh).
    force: Arc<AtomicBool>,
    /// Live config snapshot the worker reads each cycle (host/port). Updated by
    /// the UI on reload so a port change is picked up without restarting.
    config: Arc<Mutex<ConfigState>>,
    /// Signals both threads to exit; also wakes them via `wake`/`metrics_wake`.
    shutdown: Arc<AtomicBool>,
    wake: mpsc::Sender<()>,
    metrics_wake: mpsc::Sender<()>,
    health_wake: mpsc::Sender<()>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl DashboardWorker {
    /// Spawn the background fetch thread. It starts idle (no fetch) until the
    /// Dashboard tab is marked active via [`DashboardWorker::set_active`].
    pub fn spawn(config: ConfigState) -> Self {
        let agent = health_agent();
        Self::spawn_with_health_probe(config, move |cfg| probe_health(&agent, cfg))
    }

    /// [`DashboardWorker::spawn`] with the `/health` probe supplied by the
    /// caller. Tests drive the health loop through a probe they control, so
    /// what they check (which config a probe reads, what gets published)
    /// never depends on network or scheduling timing.
    fn spawn_with_health_probe(
        config: ConfigState,
        probe: impl Fn(&ConfigState) -> HealthSnapshot + Send + 'static,
    ) -> Self {
        let snapshot = Arc::new(Mutex::new(None));
        let metrics = Arc::new(Mutex::new(MetricsView::Collecting));
        let active = Arc::new(AtomicBool::new(false));
        let force = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let config = Arc::new(Mutex::new(config));
        let (wake, wake_rx) = mpsc::channel::<()>();
        let (metrics_wake, metrics_wake_rx) = mpsc::channel::<()>();
        let health = Arc::new(Mutex::new(HealthSnapshot::default()));
        let (health_wake, health_wake_rx) = mpsc::channel::<()>();

        let handle = {
            let snapshot = Arc::clone(&snapshot);
            let active = Arc::clone(&active);
            let force = Arc::clone(&force);
            let shutdown = Arc::clone(&shutdown);
            let config = Arc::clone(&config);
            thread::spawn(move || {
                worker_loop(snapshot, active, force, shutdown, config, wake_rx);
            })
        };
        let metrics_handle = {
            let metrics = Arc::clone(&metrics);
            let shutdown = Arc::clone(&shutdown);
            let config = Arc::clone(&config);
            thread::spawn(move || metrics_loop(metrics, shutdown, config, metrics_wake_rx))
        };
        let health_handle = {
            let health = Arc::clone(&health);
            let shutdown = Arc::clone(&shutdown);
            let config = Arc::clone(&config);
            thread::spawn(move || health_loop(health, shutdown, config, health_wake_rx, probe))
        };

        Self {
            snapshot,
            metrics,
            health,
            active,
            force,
            config,
            shutdown,
            wake,
            metrics_wake,
            health_wake,
            handles: vec![handle, metrics_handle, health_handle],
        }
    }

    /// Read the latest snapshot (cheap clone). `None` until the first fetch
    /// completes. This is the ONLY path the UI thread uses for dashboard data.
    pub fn snapshot(&self) -> Option<Dashboard> {
        // Poison-tolerant: this runs on the UI thread every frame, so a worker
        // panic while holding the lock must NOT crash render — recover the guard.
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Latest Live metrics view (cheap clone; ~120-point series at most).
    pub fn metrics(&self) -> MetricsView {
        self.metrics
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Latest `/health` result (cheap clone). `checked == false` until the
    /// first probe completes. Runs on every tab, independent of the full
    /// snapshot's focus gating.
    pub fn health(&self) -> HealthSnapshot {
        self.health
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Mark the Dashboard tab focused / unfocused. Flipping to active wakes the
    /// worker so the first fetch starts immediately instead of after an idle
    /// poll.
    pub fn set_active(&self, active: bool) {
        let was = self.active.swap(active, Ordering::SeqCst);
        if active && !was {
            let _ = self.wake.send(());
        }
    }

    /// Request an immediate re-fetch (manual refresh). No-op cost if the worker
    /// is mid-cycle; it picks the flag up at the next loop turn.
    pub fn force_refresh(&self) {
        self.force.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        let _ = self.health_wake.send(());
    }

    /// Update the config the worker fetches against (e.g. after a port change
    /// on reload).
    pub fn update_config(&self, config: ConfigState) {
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = config;
        let _ = self.health_wake.send(());
    }
}

impl Drop for DashboardWorker {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.wake.send(());
        let _ = self.metrics_wake.send(());
        let _ = self.health_wake.send(());
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(
    snapshot: Arc<Mutex<Option<Dashboard>>>,
    active: Arc<AtomicBool>,
    force: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    config: Arc<Mutex<ConfigState>>,
    wake_rx: mpsc::Receiver<()>,
) {
    let mut last_fetch: Option<Instant> = None;
    // Static system diagnostics (GPU/HIP/kernel-cache) are probed once and
    // cached — they don't change within a session and re-running rocm-smi /
    // rocminfo every 1.5s would be wasteful. The loaded-model + checksum
    // overlay IS refreshed each cycle from the live serve snapshot.
    let mut system_base: Option<SystemInfo> = None;
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        let is_active = active.load(Ordering::SeqCst);
        let forced = force.swap(false, Ordering::SeqCst);
        // A forced refresh (manual `r`) ALWAYS fetches, even when the live tab
        // is not focused — otherwise the force flag is swapped-then-dropped and
        // the snapshot goes stale until Dashboard/System is next opened. The
        // time-based cadence still only fires while active.
        let due = forced
            || (is_active
                && last_fetch
                    .map(|t| t.elapsed() >= WORKER_REFRESH)
                    .unwrap_or(true));

        if due {
            // Clone the config out of the lock so the (potentially slow) fetch
            // never holds it.
            let cfg = config.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let mut dash = fetch_dashboard(&cfg);
            // Probe static system info on first need; reuse thereafter.
            let base = system_base.get_or_insert_with(SystemInfo::probe).clone();
            dash.system = Some(base.with_loaded_model(dash.model.as_deref()));
            *snapshot.lock().unwrap_or_else(|e| e.into_inner()) = Some(dash);
            last_fetch = Some(Instant::now());
        }

        // Sleep until the next refresh is due (when active) or a short idle
        // poll (when inactive), but wake early on force/active/shutdown signals.
        let wait = if is_active {
            last_fetch
                .map(|t| WORKER_REFRESH.saturating_sub(t.elapsed()))
                .unwrap_or(WORKER_REFRESH)
                .max(Duration::from_millis(20))
        } else {
            WORKER_IDLE_POLL
        };
        // Drain a single wake (coalesces bursts) or time out.
        let _ = wake_rx.recv_timeout(wait);
    }
}

/// Scrape `/metrics` every [`WORKER_REFRESH`] regardless of tab focus. Only
/// this one short-timeout HTTP GET runs in the background; rocm-smi and the
/// other probes stay focus-gated in [`worker_loop`].
fn metrics_loop(
    view: Arc<Mutex<MetricsView>>,
    shutdown: Arc<AtomicBool>,
    config: Arc<Mutex<ConfigState>>,
    wake_rx: mpsc::Receiver<()>,
) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(450)))
        .http_status_as_error(false)
        // Never forward the bearer to another origin.
        .max_redirects(0)
        .build()
        .into();
    let mut ring = MetricsRing::default();
    while !shutdown.load(Ordering::SeqCst) {
        let endpoint = {
            let cfg = config.lock().unwrap_or_else(|e| e.into_inner());
            cfg.service_endpoint()
        };
        // An unloadable bearer/endpoint is never retried unauthenticated.
        let scrape = match endpoint {
            Err(_) => Err(MetricsView::Unavailable),
            Ok(endpoint) => match endpoint
                .authorize(agent.get(&service_url(&endpoint, "/metrics")))
                .call()
            {
                Ok(resp) if resp.status().as_u16() >= 400 => Err(MetricsView::Unavailable),
                Ok(mut resp) => resp
                    .body_mut()
                    .read_to_string()
                    .map(|body| metrics::parse(&body))
                    .map_err(|_| MetricsView::Offline),
                Err(_) => Err(MetricsView::Offline),
            },
        };
        let next = ring.record(Instant::now(), scrape);
        *view.lock().unwrap_or_else(|e| e.into_inner()) = next;
        let _ = wake_rx.recv_timeout(WORKER_REFRESH);
    }
}

/// Probe `/health` immediately, then every [`HEALTH_REFRESH`], on every tab.
/// The endpoint is re-read from the shared config each cycle so a port change
/// is picked up; `force_refresh` / `update_config` wake it early.
fn health_loop(
    health: Arc<Mutex<HealthSnapshot>>,
    shutdown: Arc<AtomicBool>,
    config: Arc<Mutex<ConfigState>>,
    wake_rx: mpsc::Receiver<()>,
    probe: impl Fn(&ConfigState) -> HealthSnapshot,
) {
    while !shutdown.load(Ordering::SeqCst) {
        // Clone out of the lock so the probe never holds it.
        let cfg = config.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let next = probe(&cfg);
        *health.lock().unwrap_or_else(|e| e.into_inner()) = next;
        let _ = wake_rx.recv_timeout(HEALTH_REFRESH);
        // Coalesce a burst of wakes into a single re-probe.
        while wake_rx.try_recv().is_ok() {}
    }
}

/// The `/health` client: 450 ms global timeout, statuses as values, no
/// redirects (the bearer is never forwarded to another origin).
fn health_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_millis(450)))
        .http_status_as_error(false)
        .max_redirects(0)
        .build()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_load_json_extracts_util_temp_power() {
        let body = r#"{
            "card0": {
                "GPU use (%)": "37",
                "Temperature (Sensor edge) (C)": "45.0",
                "Average Graphics Package Power (W)": "120.0"
            },
            "card1": {
                "GPU use (%)": "0",
                "Temperature (Sensor junction) (C)": "33.0",
                "Current Socket Graphics Package Power (W)": "18.0"
            }
        }"#;
        match parse_rocm_smi_load_json(body) {
            LoadState::Available(loads) => {
                assert_eq!(loads.len(), 2);
                assert_eq!(loads[0].util_pct, Some(37.0));
                assert_eq!(loads[0].temp_c, Some(45.0));
                assert_eq!(loads[0].power_w, Some(120.0));
                // Version-variant key names still resolve via substring match.
                assert_eq!(loads[1].temp_c, Some(33.0));
                assert_eq!(loads[1].power_w, Some(18.0));
            }
            other => panic!("expected Available, got {other:?}"),
        }
    }

    #[test]
    fn parse_load_json_tolerates_missing_fields_and_bad_input() {
        // Only util present -> Available with temp/power honestly None.
        match parse_rocm_smi_load_json(r#"{"card0": {"GPU use (%)": "12"}}"#) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].util_pct, Some(12.0));
                assert_eq!(loads[0].temp_c, None);
                assert_eq!(loads[0].power_w, None);
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // No load fields -> Unavailable, not a fabricated zero.
        assert!(matches!(
            parse_rocm_smi_load_json(r#"{"card0": {}}"#),
            LoadState::Unavailable(_)
        ));
        // Bad JSON -> Unavailable.
        assert!(matches!(
            parse_rocm_smi_load_json("not json"),
            LoadState::Unavailable(_)
        ));
    }

    #[test]
    fn parse_load_json_rejects_nonfinite_and_cap_keys() {
        // rocm-smi reports "NaN"/"inf" for unsupported sensors on some SKUs;
        // f64::from_str accepts them, so they must be rejected to honest None.
        let body = r#"{"card0": {
            "GPU use (%)": "NaN",
            "Average Graphics Package Power (W)": "inf",
            "Temperature (Sensor edge) (C)": "45.0"
        }}"#;
        match parse_rocm_smi_load_json(body) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].util_pct, None, "NaN util rejected");
                assert_eq!(loads[0].power_w, None, "inf power rejected");
                assert_eq!(loads[0].temp_c, Some(45.0), "finite temp kept");
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // A power CAP emitted before the draw key must not be read as live power.
        let body2 = r#"{"card0": {
            "Max Graphics Package Power (W)": "327",
            "Average Graphics Package Power (W)": "120"
        }}"#;
        match parse_rocm_smi_load_json(body2) {
            LoadState::Available(loads) => {
                assert_eq!(loads[0].power_w, Some(120.0), "live draw, not the cap");
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // Out-of-range util (sensor sentinel) -> None.
        match parse_rocm_smi_load_json(r#"{"card0": {"GPU use (%)": "65535"}}"#) {
            LoadState::Unavailable(_) => {}
            LoadState::Available(loads) => assert_eq!(loads[0].util_pct, None),
        }
    }

    #[test]
    fn health_model_present() {
        let b = r#"{"status":"ok","model":"/home/u/.hipfire/models/q.mq4","pid":42}"#;
        assert_eq!(
            parse_health_model(b),
            Some("/home/u/.hipfire/models/q.mq4".to_string())
        );
    }

    #[test]
    fn health_model_null_or_idle() {
        assert_eq!(parse_health_model(r#"{"status":"ok","model":null}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok","model":""}"#), None);
        assert_eq!(parse_health_model(r#"{"status":"ok"}"#), None);
    }

    #[test]
    fn health_garbage_is_none_not_panic() {
        assert_eq!(parse_health_model("not json"), None);
        assert_eq!(parse_health_model(""), None);
        assert_eq!(parse_health_model("[1,2,3]"), None);
    }

    #[test]
    fn stats_loaded_config_present() {
        let b = r#"{"model":"q","uptime_s":1,"queue_depth":0,"requests_served":0,"recent_tok_s":null,"loaded_config":{"model":"qwen3.5:4b","kv_mode":"asym3","kv_backend":"paged","max_seq":32768}}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(
            s.loaded_config,
            Some(LoadedConfig {
                model: "qwen3.5:4b".into(),
                kv_mode: Some("asym3".into()),
                kv_backend: Some("paged".into()),
                max_seq: Some(32768),
            })
        );
    }

    #[test]
    fn stats_loaded_config_absent_or_null_is_none() {
        // Older serve: no field at all.
        let old = r#"{"model":"q","uptime_s":1,"queue_depth":0,"requests_served":0}"#;
        assert_eq!(parse_stats(old).unwrap().loaded_config, None);
        // No model resident: serve sends null.
        let none = r#"{"model":null,"uptime_s":1,"queue_depth":0,"requests_served":0,"loaded_config":null}"#;
        assert_eq!(parse_stats(none).unwrap().loaded_config, None);
    }

    #[test]
    fn stats_loaded_config_null_fields_stay_none() {
        let b = r#"{"model":"q","loaded_config":{"model":"q","kv_mode":null,"kv_backend":null,"max_seq":null}}"#;
        assert_eq!(
            parse_stats(b).unwrap().loaded_config,
            Some(LoadedConfig {
                model: "q".into(),
                kv_mode: None,
                kv_backend: None,
                max_seq: None,
            })
        );
    }

    #[test]
    fn stats_full_shape() {
        let b = r#"{"model":"q.mq4","uptime_s":63,"queue_depth":2,"requests_served":42,"recent_tok_s":151.24}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model.as_deref(), Some("q.mq4"));
        assert_eq!(s.uptime_s, 63);
        assert_eq!(s.queue_depth, 2);
        assert_eq!(s.requests_served, 42);
        assert_eq!(s.recent_tok_s, Some(151.24));
    }

    #[test]
    fn stats_reads_serve_uptime_sec() {
        // The real serve `/stats` spells it `uptime_sec` (serve/http.rs).
        let b = r#"{"model":"q","uptime_sec":120,"queue_depth":0,"requests_served":0}"#;
        assert_eq!(parse_stats(b).expect("parse").uptime_s, 120);
    }

    #[test]
    fn stats_idle_no_model_no_tok_s() {
        let b = r#"{"model":null,"uptime_s":5,"queue_depth":0,"requests_served":0}"#;
        let s = parse_stats(b).expect("parse");
        assert_eq!(s.model, None);
        assert_eq!(s.queue_depth, 0);
        assert_eq!(s.requests_served, 0);
        assert_eq!(s.recent_tok_s, None);
    }

    #[test]
    fn stats_garbage_is_none() {
        assert_eq!(parse_stats("not json"), None);
        assert_eq!(parse_stats("[1,2]"), None);
        assert_eq!(parse_stats(""), None);
    }

    #[test]
    fn stats_rejects_nonfinite_tok_s() {
        // serde won't parse NaN, but a zero/negative must be dropped.
        let b =
            r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":0}"#;
        assert_eq!(parse_stats(b).unwrap().recent_tok_s, None);
        let b2 =
            r#"{"model":"m","uptime_s":1,"queue_depth":0,"requests_served":1,"recent_tok_s":-3}"#;
        assert_eq!(parse_stats(b2).unwrap().recent_tok_s, None);
    }

    #[test]
    fn model_ids_parsed() {
        let b = r#"{"data":[{"id":"qwen3.5:9b"},{"id":"qwen3.5:27b"}]}"#;
        assert_eq!(parse_model_ids(b), vec!["qwen3.5:9b", "qwen3.5:27b"]);
        assert!(parse_model_ids("garbage").is_empty());
        assert!(parse_model_ids(r#"{"data":[]}"#).is_empty());
    }

    #[test]
    fn rocm_smi_two_gpus() {
        let b = r#"{
            "card1": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "10000000000"},
            "card0": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "2000000000"}
        }"#;
        let vram = parse_rocm_smi_json(b);
        match vram {
            VramState::Available(gpus) => {
                assert_eq!(gpus.len(), 2);
                // Sorted by card index.
                assert_eq!(gpus[0].index, 0);
                assert_eq!(gpus[0].total_bytes, 34208743424);
                assert_eq!(gpus[0].used_bytes, 2000000000);
                assert_eq!(gpus[0].free_bytes(), 32208743424);
                assert_eq!(gpus[1].index, 1);
            }
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    fn rocm_smi_numeric_fields_also_work() {
        let b = r#"{"card0":{"VRAM Total Memory (B)": 8000000000, "VRAM Total Used Memory (B)": 1000000000}}"#;
        match parse_rocm_smi_json(b) {
            VramState::Available(g) => assert_eq!(g[0].total_bytes, 8000000000),
            VramState::Unavailable(r) => panic!("expected Available, got {r}"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_kills_hung_process() {
        // `tail -f /dev/null` never exits on its own, so `run_bounded` can only
        // return by killing it at the deadline: returning `TimedOut` at all is
        // the proof that a hung child is killed rather than waited on. No
        // wall-clock bound on how long that took (host load must not decide
        // the result).
        let mut cmd = Command::new("tail");
        cmd.args(["-f", "/dev/null"]);
        let run = run_bounded(cmd, Duration::from_millis(200));
        assert!(matches!(run, RocmSmiRun::TimedOut), "expected TimedOut");
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_captures_fast_output() {
        let mut cmd = Command::new("printf");
        cmd.arg("hello");
        match run_bounded(cmd, Duration::from_secs(5)) {
            RocmSmiRun::Output(s) => assert_eq!(s, "hello"),
            other => panic!("expected Output, got {:?}", run_kind(&other)),
        }
    }

    #[test]
    #[cfg(unix)]
    fn bounded_run_stdout_fill_is_not_falsely_timed_out() {
        // F2: a command that emits MORE than one OS pipe buffer (~64 KiB) of
        // stdout BEFORE exiting must run to completion, not be killed as a false
        // timeout. Before the concurrent-drain fix, the child would block on
        // write() against a full pipe (no reader until after exit) and get
        // killed. We emit ~1 MiB and assert we capture all of it.
        let mut cmd = Command::new("sh");
        // `yes` would never exit; use a bounded large emission via head.
        cmd.args(["-c", "yes ABCDEFGHIJ | head -c 1048576"]);
        let run = run_bounded(cmd, Duration::from_secs(10));
        match run {
            RocmSmiRun::Output(s) => {
                assert_eq!(
                    s.len(),
                    1_048_576,
                    "must capture the full 1 MiB without deadlocking the pipe"
                );
            }
            other => panic!(
                "expected Output (not a false timeout), got {:?}",
                run_kind(&other)
            ),
        }
    }

    #[test]
    fn bounded_run_missing_binary_is_notfound() {
        let cmd = Command::new("definitely-not-a-real-binary-xyz-123");
        assert!(matches!(
            run_bounded(cmd, Duration::from_secs(1)),
            RocmSmiRun::NotFound
        ));
    }

    #[cfg(unix)]
    fn run_kind(r: &RocmSmiRun) -> &'static str {
        match r {
            RocmSmiRun::Output(_) => "Output",
            RocmSmiRun::NotFound => "NotFound",
            RocmSmiRun::SpawnError(_) => "SpawnError",
            RocmSmiRun::TimedOut => "TimedOut",
        }
    }

    #[test]
    fn worker_spawns_and_drops_cleanly() {
        // Spawn the worker, leave it idle (Dashboard tab not active), and drop
        // it — Drop must signal shutdown and join without hanging. No GPU /
        // serve needed; idle worker performs no fetch.
        let cfg = ConfigState::load_hermetic(&crate::hipfire::HipfirePaths::under(
            std::env::temp_dir().join(format!(
                "hipfire-worker-spawn-absent-{}",
                std::process::id()
            )),
        ));
        let worker = DashboardWorker::spawn(cfg);
        // No snapshot before any fetch.
        assert!(worker.snapshot().is_none());
        // set_active toggling must not panic; we don't assert a fetch result
        // (no serve in CI) — just that the worker stays responsive and Drop
        // joins promptly.
        worker.set_active(true);
        worker.set_active(false);
        drop(worker); // joins; would hang the test if shutdown were broken.
    }

    #[test]
    fn rocm_smi_garbage_is_unavailable_not_panic() {
        assert!(matches!(
            parse_rocm_smi_json("not json"),
            VramState::Unavailable(_)
        ));
        assert!(matches!(
            parse_rocm_smi_json("{}"),
            VramState::Unavailable(_)
        ));
        // Object without the VRAM fields → unavailable, not a fake 0.
        assert!(matches!(
            parse_rocm_smi_json(r#"{"card0":{"GPU use (%)":"50"}}"#),
            VramState::Unavailable(_)
        ));
    }

    /// Minimal `/health` server on an ephemeral loopback port. Dropping the
    /// returned guard stops it and closes the listener.
    struct HealthServer {
        port: u16,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl HealthServer {
        fn start(body: &'static str) -> Self {
            use std::io::Write;
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let port = listener.local_addr().unwrap().port();
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                while !flag.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
                            let mut request = Vec::new();
                            let mut buf = [0u8; 512];
                            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                                match stream.read(&mut buf) {
                                    Ok(0) | Err(_) => break,
                                    Ok(n) => request.extend_from_slice(&buf[..n]),
                                }
                            }
                            let response = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                                body.len()
                            );
                            let _ = stream.write_all(response.as_bytes());
                        }
                        Err(_) => thread::sleep(Duration::from_millis(10)),
                    }
                }
                // `listener` drops here: the port stops accepting.
            });
            Self {
                port,
                stop,
                handle: Some(handle),
            }
        }

        fn shutdown(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    impl Drop for HealthServer {
        fn drop(&mut self) {
            self.shutdown();
        }
    }

    fn config_for_port(port: u16) -> ConfigState {
        ConfigState {
            host: "127.0.0.1".into(),
            port,
            default_model: String::new(),
            values: Default::default(),
            per_model_count: 0,
            loaded_from_disk: false,
            warning: None,
        }
    }

    /// A `/health` client with no timeout: the local test server always
    /// answers, so the result never depends on how loaded the host is.
    fn untimed_agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            .http_status_as_error(false)
            .max_redirects(0)
            .build()
            .into()
    }

    #[test]
    fn health_probe_reports_up_then_down_when_the_listener_closes() {
        let mut server = HealthServer::start(r#"{"status":"ok","model":"m"}"#);
        let agent = untimed_agent();
        let up = probe_health(&agent, &config_for_port(server.port));
        assert!(up.checked && up.serve_up);
        assert_eq!(up.model.as_deref(), Some("m"));
        assert!(up.failure.is_none());
        assert!(up.text.contains(r#""status":"ok""#));

        let port = server.port;
        server.shutdown();
        let down = probe_health(&agent, &config_for_port(port));
        assert!(down.checked && !down.serve_up);
        assert!(down.model.is_none());
        assert!(down.failure.is_none(), "unreachable is down, not a failure");
    }

    /// A health probe the test steps by hand: each call reports the port it
    /// was given on `started`, then blocks until the test sends on `release`
    /// (or drops it), then reports serve up only for `up_port`.
    fn stepped_probe(
        up_port: u16,
    ) -> (
        impl Fn(&ConfigState) -> HealthSnapshot + Send + 'static,
        mpsc::Receiver<u16>,
        mpsc::Sender<()>,
    ) {
        let (started_tx, started) = mpsc::channel::<u16>();
        let (release, release_rx) = mpsc::channel::<()>();
        let probe = move |cfg: &ConfigState| {
            let _ = started_tx.send(cfg.port);
            let _ = release_rx.recv();
            HealthSnapshot {
                checked: true,
                serve_up: cfg.port == up_port,
                ..HealthSnapshot::default()
            }
        };
        (probe, started, release)
    }

    #[test]
    fn health_poller_picks_up_a_port_change_from_update_config() {
        // The loop runs on every tab (the worker starts inactive) and re-reads
        // the shared config for every probe. Each probe is stepped by hand, so
        // the order of events is fixed and nothing depends on timing.
        let (dead, live) = (1, 2);
        let (probe, started, release) = stepped_probe(live);
        let worker = DashboardWorker::spawn_with_health_probe(config_for_port(dead), probe);
        // Declared after `worker`, so it drops first (also when an assert
        // unwinds): the blocked probe returns and the worker can join.
        let release = release;

        assert_eq!(
            started.recv().unwrap(),
            dead,
            "first probe uses the initial config"
        );
        release.send(()).unwrap();
        worker.update_config(config_for_port(live));
        // A probe that read the config before `update_config` may still be
        // reported (it blocks until released); every probe after it must
        // read the new port.
        let mut port = started.recv().unwrap();
        if port == dead {
            release.send(()).unwrap();
            port = started.recv().unwrap();
        }
        assert_eq!(
            port, live,
            "new port picked up without restarting the worker"
        );
        release.send(()).unwrap();
        // The live probe's snapshot is published before the loop starts its
        // next probe; wake only the health loop and wait for that start.
        worker.update_config(config_for_port(live));
        assert_eq!(started.recv().unwrap(), live);
        let health = worker.health();
        assert!(health.checked && health.serve_up, "{health:?}");
    }

    // ── amd-smi parsers + tool selection ────────────────────────────────────

    const AMD_METRIC: &str = include_str!("testdata/amd_smi_metric.json");
    const AMD_ASIC: &str = include_str!("testdata/amd_smi_static_asic.json");

    /// The fixture's `gpu_data` array re-serialised as a bare top-level array
    /// (the other JSON root amd-smi versions emit).
    fn bare_array(fixture: &str) -> String {
        let v: Value = serde_json::from_str(fixture).unwrap();
        v["gpu_data"].to_string()
    }

    #[test]
    fn amd_smi_asic_fixture_yields_name_and_arch() {
        let (name, arch) = parse_amd_smi_asic(AMD_ASIC);
        // Two identical GPUs collapse to one value.
        assert_eq!(name, Probe::Value("AMD Radeon AI PRO R9700".into()));
        assert_eq!(arch, Probe::Value("gfx1201".into()));
        let (name, arch) = parse_amd_smi_asic(&bare_array(AMD_ASIC));
        assert_eq!(name, Probe::Value("AMD Radeon AI PRO R9700".into()));
        assert_eq!(arch, Probe::Value("gfx1201".into()));
    }

    #[test]
    fn amd_smi_asic_mixed_gpus_join_and_na_is_skipped() {
        let body = r#"{"gpu_data": [
            {"gpu": 0, "asic": {"market_name": "A", "target_graphics_version": "gfx1100"}},
            {"gpu": 1, "asic": {"market_name": "N/A", "target_graphics_version": "gfx1201"}}
        ]}"#;
        let (name, arch) = parse_amd_smi_asic(body);
        assert_eq!(name, Probe::Value("A".into()));
        assert_eq!(arch, Probe::Value("gfx1100, gfx1201".into()));
    }

    #[test]
    fn amd_smi_vram_fixture_converts_mb_to_bytes() {
        for body in [AMD_METRIC.to_string(), bare_array(AMD_METRIC)] {
            match parse_amd_smi_vram(&body) {
                VramState::Available(gpus) => {
                    assert_eq!(gpus.len(), 2);
                    assert_eq!(gpus[0].index, 0);
                    assert_eq!(gpus[1].index, 1);
                    assert_eq!(gpus[0].total_bytes, 32624 * 1024 * 1024);
                    assert_eq!(gpus[0].used_bytes, 57 * 1024 * 1024);
                    assert_eq!(gpus[0].free_bytes(), (32624 - 57) * 1024 * 1024);
                }
                other => panic!("expected Available, got {other:?}"),
            }
        }
    }

    #[test]
    fn amd_smi_load_fixture_extracts_util_power_temp() {
        for body in [AMD_METRIC.to_string(), bare_array(AMD_METRIC)] {
            match parse_amd_smi_load(&body) {
                LoadState::Available(loads) => {
                    assert_eq!(loads.len(), 2);
                    assert_eq!(loads[0].index, 0);
                    assert_eq!(loads[0].util_pct, Some(0.0));
                    assert_eq!(loads[0].power_w, Some(23.0));
                    // Edge temperature is preferred over hotspot.
                    assert_eq!(loads[0].temp_c, Some(47.0));
                    assert_eq!(loads[1].index, 1);
                }
                other => panic!("expected Available, got {other:?}"),
            }
        }
    }

    #[test]
    fn amd_smi_na_fields_become_none() {
        let body = r#"{"gpu_data": [{"gpu": 3,
            "usage": {"gfx_activity": "N/A"},
            "power": {"socket_power": "N/A"},
            "temperature": {"edge": "N/A", "hotspot": {"value": 51, "unit": "C"}},
            "mem_usage": {"total_vram": "N/A", "used_vram": {"value": 1, "unit": "MB"}}
        }]}"#;
        match parse_amd_smi_load(body) {
            LoadState::Available(loads) => {
                assert_eq!(loads.len(), 1);
                assert_eq!(loads[0].index, 3);
                assert_eq!(loads[0].util_pct, None);
                assert_eq!(loads[0].power_w, None);
                // edge is N/A, so the hotspot reading is used.
                assert_eq!(loads[0].temp_c, Some(51.0));
            }
            other => panic!("expected Available, got {other:?}"),
        }
        // N/A total → no VRAM row, honestly unavailable.
        assert!(matches!(
            parse_amd_smi_vram(body),
            VramState::Unavailable(_)
        ));
        let all_na = r#"{"gpu_data": [{"gpu": 0, "usage": {"gfx_activity": "N/A"}}]}"#;
        assert!(matches!(
            parse_amd_smi_load(all_na),
            LoadState::Unavailable(_)
        ));
    }

    #[test]
    fn amd_smi_unit_less_or_unknown_unit_memory_is_not_guessed() {
        let body = r#"[{"gpu": 0, "mem_usage": {
            "total_vram": {"value": 100}, "used_vram": {"value": 1, "unit": "MB"}}}]"#;
        assert!(matches!(
            parse_amd_smi_vram(body),
            VramState::Unavailable(_)
        ));
    }

    #[test]
    fn amd_smi_garbage_is_unavailable_not_panic() {
        for body in ["not json", "{}", "[]", "42", r#"{"gpu_data": "x"}"#, ""] {
            assert!(
                matches!(parse_amd_smi_vram(body), VramState::Unavailable(_)),
                "vram: {body:?}"
            );
            assert!(
                matches!(parse_amd_smi_load(body), LoadState::Unavailable(_)),
                "load: {body:?}"
            );
            let (name, arch) = parse_amd_smi_asic(body);
            assert!(!name.is_available(), "name: {body:?}");
            assert!(!arch.is_available(), "arch: {body:?}");
        }
    }

    #[cfg(unix)]
    fn stub_dir(tag: &str, tools: &[&str]) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir =
            std::env::temp_dir().join(format!("hipfire-smistub-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for tool in tools {
            let p = dir.join(tool);
            fs::write(&p, "#!/bin/sh\nexit 0\n").unwrap();
            fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        }
        dir
    }

    #[test]
    #[cfg(unix)]
    fn lookup_finds_amd_smi_stub_and_selects_it() {
        let dir = stub_dir("amd-only", &["amd-smi"]);
        let amd = find_executable_in([dir.clone()], "amd-smi");
        let rocm = find_executable_in([dir.clone()], "rocm-smi");
        assert_eq!(amd, Some(dir.join("amd-smi")));
        assert_eq!(rocm, None);
        assert_eq!(
            select_smi_tool(amd, rocm),
            Some(SmiTool::AmdSmi(dir.join("amd-smi")))
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(unix)]
    fn lookup_ignores_non_executable_and_prefers_amd_smi_over_rocm_smi() {
        use std::os::unix::fs::PermissionsExt;
        let dir = stub_dir("both", &["amd-smi", "rocm-smi"]);
        let amd = find_executable_in([dir.clone()], "amd-smi");
        let rocm = find_executable_in([dir.clone()], "rocm-smi");
        assert_eq!(
            select_smi_tool(amd, rocm.clone()),
            Some(SmiTool::AmdSmi(dir.join("amd-smi")))
        );
        // Strip the exec bit: amd-smi is no longer a candidate → rocm-smi.
        fs::set_permissions(dir.join("amd-smi"), fs::Permissions::from_mode(0o644)).unwrap();
        let amd = find_executable_in([dir.clone()], "amd-smi");
        assert_eq!(amd, None);
        assert_eq!(
            select_smi_tool(amd, rocm),
            Some(SmiTool::RocmSmi(dir.join("rocm-smi")))
        );
        assert_eq!(select_smi_tool(None, None), None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_tool_reason_names_root_and_never_claims_no_gpu() {
        let with_root = gpu_tool_missing_reason(Some(Path::new("/opt/rocm/core")));
        assert_eq!(
            with_root,
            "GPU stats unavailable: neither amd-smi nor rocm-smi found (ROCm root: /opt/rocm/core)"
        );
        let none = gpu_tool_missing_reason(None);
        assert!(none.ends_with("(ROCm root: none)"), "{none}");
        assert!(!none.contains("No GPU"));
    }
}
