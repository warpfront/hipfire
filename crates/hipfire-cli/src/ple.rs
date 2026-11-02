// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next PLE row sidecars: `hipfire ple derive|split`, the registry pull
//! plan (`hipfire pull --ple`), and the on-demand download of a missing
//! sidecar when `run`/`serve`/`bench` load a registry split model.
//!
//! A split base (`<name>-base.<ext>`) carries everything but the 128 PLE n-gram
//! shards; the rows live in `<base>.ple-q8` (exact Q8F16) and `<base>.ple-q4`
//! (4-bit Lloyd-Max). The decisions here are pure functions over the registry
//! entry and which files exist, so they are testable without a GPU or network.

use crate::Paths;
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use hipfire_config::memory::{ple_sidecar_path, PleFormat, PleFormatRequest};
use hipfire_registry::{
    GrandfatheredMonolithic, ModelEntry, Sidecar, FLASH_NEXT_XTS_CANONICAL_FILE,
    FLASH_NEXT_XTS_LEGACY_FILE, PLE_EXACT_FORMAT,
};
use hipfire_runtime::ple_convert::{self, DeriveReport, PleProgress, SplitReport};
use hipfire_runtime::verified_cache::VerifiedCache;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Minimum spacing of two progress lines inside one conversion phase.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);

/// `hipfire pull --ple` selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum PleSelection {
    /// The formats `memory.ple_format` and every detected GPU arch prefer (q4
    /// on gfx1151, q8 elsewhere; the union on a mixed host); no GPU or an
    /// unknown arch means q8.
    Auto,
    /// The 4-bit Lloyd-Max rows.
    Q4,
    /// The exact Q8F16 rows.
    Q8,
    /// Both sidecars.
    All,
    /// The base only; the sidecar is fetched on first load.
    #[value(name = "none")]
    Skip,
}

#[derive(Args, Debug)]
pub(crate) struct PleArgs {
    #[command(subcommand)]
    action: PleAction,
}

#[derive(Subcommand, Debug)]
enum PleAction {
    /// Derive the 4-bit PLE rows (`<model>.ple-q4`) from the model's Q8 rows.
    Derive(DeriveArgs),
    /// Split a monolithic model into a PLE-less base plus `<base>.ple-q8`.
    Split(SplitArgs),
    /// Copy a Q4 table, rebinding only after proving embedded PLE rows unchanged.
    Rebind {
        #[arg(long)]
        old_base: PathBuf,
        #[arg(long)]
        new_base: PathBuf,
        #[arg(long)]
        source: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DeriveFormat {
    Q4,
    Q8,
}

#[derive(Args, Debug)]
struct DeriveArgs {
    /// Row format to derive. Only q4 is derivable: q8 rows come from the
    /// model itself or from `hipfire ple split`.
    #[arg(long, value_enum, default_value_t = DeriveFormat::Q4)]
    format: DeriveFormat,
    /// Frequency rank (`.plehot`) that orders the derived rows. Default:
    /// `<model>.plehot` when present, else row order.
    #[arg(long, value_name = "PATH")]
    rank: Option<PathBuf>,
    /// Model path or registry tag.
    model: String,
}

#[derive(Args, Debug)]
struct SplitArgs {
    /// Base output path. Default: `-base` inserted before the model's extension
    /// (`qwen3.8-flash-next.mq4-xts` -> `qwen3.8-flash-next-base.mq4-xts`).
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,
    /// Keep the monolithic input after a verified split (default: remove it).
    #[arg(long)]
    keep_original: bool,
    /// Model path or registry tag.
    model: String,
}

pub(crate) fn ple_command(paths: &Paths, args: PleArgs) -> Result<()> {
    match args.action {
        PleAction::Derive(args) => derive_command(paths, args),
        PleAction::Split(args) => split_command(paths, args),
        PleAction::Rebind {
            old_base,
            new_base,
            source,
        } => {
            let output = ple_convert::rebind_q4(&old_base, &new_base, &source, &mut |_| {})
                .map_err(anyhow::Error::msg)?;
            println!("rebound q4 rows: {}", output.display());
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Pure decisions
// ---------------------------------------------------------------------------

/// The PLE formats this client can load (`hipfire_config` [`PleFormat`]), in
/// download order. A registry entry may publish other keys (a future q2); they
/// are ignored with a note ([`unloadable_formats`]).
pub(crate) const LOADABLE_FORMATS: [PleFormat; 2] = [PleFormat::Q4, PleFormat::Q8];

/// The registry sidecar of `format`.
pub(crate) fn ple_slot(entry: &ModelEntry, format: PleFormat) -> Option<&Sidecar> {
    entry.ple_sidecar(format.name())
}

/// True for the exact reference format (`q8`): never derived, never lossy.
fn is_exact(format: PleFormat) -> bool {
    format.name() == PLE_EXACT_FORMAT
}

/// Keys of `entry.ple` this client cannot load, in key order.
pub(crate) fn unloadable_formats(entry: &ModelEntry) -> Vec<&str> {
    entry
        .ple
        .keys()
        .map(String::as_str)
        .filter(|key| PleFormat::from_name(key).is_none())
        .collect()
}

/// The published format a wanted-but-unpublished `wanted` falls back to when
/// the selection allows it. The exact format (q8) is the reference every lossy
/// format derives from, so an unpublished lossy format falls back to it; an
/// unpublished exact format falls back to the first published lossy format
/// this client can load (a request that resolved to q8 on its own, never an
/// explicit `--ple q8`, which does not fall back at all).
fn fallback_format(entry: &ModelEntry, wanted: PleFormat) -> Option<PleFormat> {
    LOADABLE_FORMATS.into_iter().find(|candidate| {
        *candidate != wanted
            && is_exact(*candidate) != is_exact(wanted)
            && ple_slot(entry, *candidate).is_some()
    })
}

/// Sidecars a pull downloads next to the base, in download order.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct PlePullPlan<'a> {
    pub sidecars: Vec<(PleFormat, &'a Sidecar)>,
    /// Why the plan differs from the request (missing registry slot, or a
    /// published format this client cannot load).
    pub notes: Vec<String>,
}

/// The formats `request` defaults to on every arch in `gpu_arches` (the union,
/// in download order). No GPU resolves with an empty arch: q8 under `auto`.
pub(crate) fn formats_for_arches(
    request: PleFormatRequest,
    gpu_arches: &[String],
) -> Vec<PleFormat> {
    let resolved: Vec<PleFormat> = if gpu_arches.is_empty() {
        vec![request.resolve("")]
    } else {
        gpu_arches
            .iter()
            .map(|arch| request.resolve(arch))
            .collect()
    };
    LOADABLE_FORMATS
        .into_iter()
        .filter(|format| resolved.contains(format))
        .collect()
}

/// Choose the sidecars `hipfire pull --ple <selection>` downloads.
///
/// `Auto` fetches the union of the formats `request` (the `memory.ple_format`
/// config) resolves to on every detected GPU arch (a mixed gfx1100 + gfx1151
/// host plans q4 and q8; no GPU plans q8). A missing wanted slot falls back
/// ([`fallback_format`]) for `Auto` and `Q4`; an explicit `Q8` never falls
/// back, since lossy rows cannot produce exact ones. `All` plans every
/// published format this client can load. Published formats it cannot load are
/// ignored with a note.
pub(crate) fn plan_pull<'a>(
    entry: &'a ModelEntry,
    selection: PleSelection,
    request: PleFormatRequest,
    gpu_arches: &[String],
) -> PlePullPlan<'a> {
    let (wanted, fallback): (Vec<PleFormat>, bool) = match selection {
        PleSelection::Skip => (Vec::new(), false),
        PleSelection::Q4 => (vec![PleFormat::Q4], true),
        PleSelection::Q8 => (vec![PleFormat::Q8], false),
        PleSelection::All => (LOADABLE_FORMATS.to_vec(), false),
        PleSelection::Auto => (formats_for_arches(request, gpu_arches), true),
    };
    let mut plan = PlePullPlan::default();
    for format in wanted {
        let chosen = match ple_slot(entry, format) {
            Some(sidecar) => Some((format, sidecar)),
            None => match fallback_format(entry, format).filter(|_| fallback) {
                Some(other) => {
                    let derive = if is_exact(other) {
                        " (derive q4 from it with `hipfire ple derive --format q4`)"
                    } else {
                        ""
                    };
                    plan.notes.push(format!(
                        "registry publishes no ple-{} sidecar; fetching ple-{} instead{derive}",
                        format.name(),
                        other.name()
                    ));
                    ple_slot(entry, other).map(|sidecar| (other, sidecar))
                }
                None => {
                    plan.notes.push(format!(
                        "registry publishes no ple-{} sidecar for this model",
                        format.name()
                    ));
                    None
                }
            },
        };
        if let Some((format, sidecar)) = chosen {
            if !plan.sidecars.iter().any(|(planned, _)| *planned == format) {
                plan.sidecars.push((format, sidecar));
            }
        }
    }
    if selection != PleSelection::Skip {
        for key in unloadable_formats(entry) {
            plan.notes.push(format!(
                "registry publishes ple-{key}, which this client cannot load; ignoring it"
            ));
        }
    }
    plan
}

/// Which PLE sidecar files sit next to a split base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlePresence {
    pub q4: bool,
    pub q8: bool,
}

impl PlePresence {
    fn has(self, format: PleFormat) -> bool {
        match format {
            PleFormat::Q4 => self.q4,
            PleFormat::Q8 => self.q8,
        }
    }

    fn any(self) -> bool {
        LOADABLE_FORMATS.into_iter().any(|format| self.has(format))
    }
}

/// The sidecar a load must download first, if any. `gpu_arch` is the arch of
/// the card the load targets ([`load_target_arch`]).
///
/// * `q8`: download ple-q8 when absent.
/// * `q4`: nothing when ple-q4 *or* ple-q8 is present (with only q8 the load
///   refuses until `hipfire ple derive --format q4` is run: a load never
///   derives); otherwise download ple-q4, or ple-q8 when the registry
///   publishes no q4.
/// * `auto`: nothing when any loadable PLE file is present (auto never
///   downloads over an available file); otherwise the arch's preferred format
///   (or its [`fallback_format`] when unpublished).
///
/// The result always names a registry slot; `None` also covers "the registry
/// has nothing to offer", where the loader's own refusal explains it.
pub(crate) fn plan_on_demand(
    entry: &ModelEntry,
    request: PleFormatRequest,
    gpu_arch: &str,
    present: PlePresence,
) -> Option<PleFormat> {
    let (target, fallback) = match request {
        PleFormatRequest::Q8 => {
            if present.q8 {
                return None;
            }
            (PleFormat::Q8, false)
        }
        PleFormatRequest::Q4 => {
            if present.any() {
                return None;
            }
            (PleFormat::Q4, true)
        }
        PleFormatRequest::Auto => {
            if present.any() {
                return None;
            }
            (request.resolve(gpu_arch), true)
        }
    };
    if ple_slot(entry, target).is_some() {
        return Some(target);
    }
    fallback_format(entry, target).filter(|_| fallback)
}

/// How a resolved model file relates to a registry split entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelLayout {
    /// The split base itself (`entry.file`).
    SplitBase,
    /// The grandfathered monolithic (either pinned spelling).
    Monolithic,
    /// Anything else (a user path, a different artifact).
    Other,
}

pub(crate) fn classify_layout(entry: &ModelEntry, file_name: &str) -> ModelLayout {
    if file_name == entry.file {
        return ModelLayout::SplitBase;
    }
    match crate::grandfather_pin(entry) {
        Some(pin) if file_name == pin.canonical_file || file_name == pin.legacy_file => {
            ModelLayout::Monolithic
        }
        _ => ModelLayout::Other,
    }
}

/// Path of the PLE sidecar of format `key` beside `model`
/// (`<model file name>.ple-<key>`): [`ple_sidecar_path`] for any key, including
/// ones this client cannot load.
pub(crate) fn ple_sidecar_path_for_key(model: &Path, key: &str) -> PathBuf {
    let mut path = model.as_os_str().to_owned();
    path.push(hipfire_registry::ple_sidecar_suffix(key));
    PathBuf::from(path)
}

/// The PLE row sidecars of `model` and their hot-row packs; `rm` removes the
/// ones that exist. Covers every format this client can load plus every key the
/// registry `entry` (when given) publishes.
pub(crate) fn ple_sidecar_paths(model: &Path, entry: Option<&ModelEntry>) -> Vec<PathBuf> {
    let mut keys: Vec<&str> = LOADABLE_FORMATS
        .iter()
        .map(|format| format.name())
        .collect();
    for key in entry.iter().flat_map(|entry| entry.ple.keys()) {
        if !keys.contains(&key.as_str()) {
            keys.push(key);
        }
    }
    let mut paths = Vec::new();
    for key in keys {
        let sidecar = ple_sidecar_path_for_key(model, key);
        let mut hot = sidecar.as_os_str().to_owned();
        hot.push(".plehotrows");
        paths.push(sidecar);
        paths.push(PathBuf::from(hot));
    }
    // The frequency rank (`<model>.plehot`) belongs to the model it ranks.
    let mut rank = model.as_os_str().to_owned();
    rank.push(".plehot");
    paths.push(PathBuf::from(rank));
    paths
}

fn format_gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn format_size(bytes: Option<u64>) -> String {
    bytes.map_or_else(|| "size unknown".to_owned(), format_gb)
}

/// One line of the pull plan.
pub(crate) fn plan_line(label: &str, file: &str, size: Option<u64>, state: &str) -> String {
    format!("  {label:<8} {file}  ({})  {state}", format_size(size))
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h{:02}m{:02}s", secs / 3600, secs % 3600 / 60, secs % 60)
    } else if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

/// Throttles conversion progress to one line per phase every
/// [`PROGRESS_INTERVAL`], plus a line at each phase start and completion.
pub(crate) struct ProgressPrinter {
    phase: Option<&'static str>,
    phase_start: Instant,
    last_print: Instant,
    completed: bool,
}

impl ProgressPrinter {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            phase: None,
            phase_start: now,
            last_print: now,
            completed: false,
        }
    }

    /// The line to print for `progress` at `now`, if one is due.
    pub(crate) fn update(&mut self, progress: &PleProgress, now: Instant) -> Option<String> {
        let new_phase = self.phase != Some(progress.phase);
        if new_phase {
            self.phase = Some(progress.phase);
            self.phase_start = now;
            self.completed = false;
        }
        let finished = progress.total_bytes > 0 && progress.done_bytes >= progress.total_bytes;
        let due = new_phase
            || (finished && !self.completed)
            || now.saturating_duration_since(self.last_print) >= PROGRESS_INTERVAL;
        if !due {
            return None;
        }
        self.completed |= finished;
        self.last_print = now;
        let elapsed = now
            .saturating_duration_since(self.phase_start)
            .as_secs_f64();
        let rate = progress.done_bytes as f64 / elapsed.max(0.001);
        let percent = if progress.total_bytes > 0 {
            format!(
                "{:>5.1}%  ",
                progress.done_bytes as f64 / progress.total_bytes as f64 * 100.0
            )
        } else {
            String::new()
        };
        Some(format!(
            "ple: {}: {percent}{:.2}/{:.2} GB  {:.0} MB/s",
            progress.phase,
            progress.done_bytes as f64 / 1e9,
            progress.total_bytes as f64 / 1e9,
            rate / 1e6
        ))
    }
}

fn derive_summary(report: &DeriveReport) -> Vec<String> {
    vec![
        format!("derived q4 rows: {}", report.output.display()),
        format!(
            "  size            {} B ({})",
            report.bytes,
            format_gb(report.bytes)
        ),
        format!("  rows source     {}", report.source.display()),
        format!("  payload_sha256  {}", report.payload_sha256),
        format!("  q8_rows_sha256  {}", report.source_rows_sha256),
        format!(
            "  elapsed         {} ({} threads)",
            format_duration(report.elapsed),
            report.threads
        ),
        format!("  order           {}", report.order),
    ]
}

fn split_summary(report: &SplitReport) -> Vec<String> {
    vec![
        format!("split complete: base {}", report.base.display()),
        format!(
            "  base size        {} B ({})",
            report.base_bytes,
            format_gb(report.base_bytes)
        ),
        format!("  base_index_sha256 {}", report.base_index_sha256),
        format!("  ple-q8           {}", report.ple_q8.display()),
        format!(
            "  ple-q8 size      {} B ({})",
            report.ple_q8_bytes,
            format_gb(report.ple_q8_bytes)
        ),
        format!("  q8_rows_sha256   {}", report.q8_rows_sha256),
        format!("  elapsed          {}", format_duration(report.elapsed)),
    ]
}

/// Other hard links of the pinned monolithic (the canonical and legacy
/// spellings share one inode after a pull): removing only the named one would
/// not free the disk.
pub(crate) fn monolithic_twins(model: &Path) -> Vec<PathBuf> {
    let Some(dir) = model.parent() else {
        return Vec::new();
    };
    [FLASH_NEXT_XTS_CANONICAL_FILE, FLASH_NEXT_XTS_LEGACY_FILE]
        .into_iter()
        .map(|name| dir.join(name))
        .filter(|twin| twin != model && twin.is_file() && crate::same_installed_file(twin, model))
        .collect()
}

/// `<model>.plehot` / `.plehotrows` files next to `model` that exist.
fn plehot_siblings(model: &Path) -> Vec<PathBuf> {
    [".plehot", ".plehotrows"]
        .into_iter()
        .map(|suffix| {
            let mut path = model.as_os_str().to_owned();
            path.push(suffix);
            PathBuf::from(path)
        })
        .filter(|path| path.is_file())
        .collect()
}

// ---------------------------------------------------------------------------
// Environment-dependent helpers
// ---------------------------------------------------------------------------

#[cfg(test)]
thread_local! {
    /// Tests pin the GPU arches of the pull plan (the real ones are read from sysfs).
    static GPU_ARCH_OVERRIDE: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

/// Test guard: `local_gpu_arches()` returns `arches` on this thread until dropped.
#[cfg(test)]
pub(crate) struct GpuArchOverride;

#[cfg(test)]
impl GpuArchOverride {
    pub(crate) fn set(arches: &[&str]) -> Self {
        GPU_ARCH_OVERRIDE.with(|cell| {
            *cell.borrow_mut() = Some(arches.iter().map(|arch| (*arch).to_owned()).collect());
        });
        Self
    }
}

#[cfg(test)]
impl Drop for GpuArchOverride {
    fn drop(&mut self) {
        GPU_ARCH_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Every GPU arch detected on this host, sorted and deduplicated (the order is
/// stable, unlike the KFD node order); empty when none is detected (an unknown
/// arch resolves to q8).
pub(crate) fn local_gpu_arches() -> Vec<String> {
    #[cfg(test)]
    if let Some(arches) = GPU_ARCH_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return arches;
    }
    let mut arches = crate::detect_gpu_arches();
    arches.sort();
    arches.dedup();
    arches
}

/// Arch of the card behind logical device 0 of a load with the `hardware.devices`
/// selector `spec` (`HIPFIRE_DEVICES`), else the inherited `ROCR_VISIBLE_DEVICES`
/// filter `rocr_visible`, else ROCr agent order (the lowest ROCr ordinal): the
/// card a single-GPU `run`/`serve`/`bench` load lands on. Reads only the KFD
/// topology under `nodes_dir`; nothing is claimed. `None` when the topology is
/// unreadable or a selector names no GPU.
pub(crate) fn device_arch_in(
    nodes_dir: &Path,
    spec: Option<&str>,
    rocr_visible: Option<&str>,
) -> Option<String> {
    use hipfire_config::devices::{enumerate_gpus, resolve_device_selectors, Claim, GpuDevice};
    let gpus = enumerate_gpus(nodes_dir).ok()?;
    if let Some(spec) = spec.map(str::trim).filter(|spec| !spec.is_empty()) {
        let mut no_claim = |_: &GpuDevice| Ok(Claim::Claimed);
        let chosen = resolve_device_selectors(spec, &gpus, &mut no_claim).ok()?;
        return chosen.into_iter().next().map(|gpu| gpu.arch);
    }
    if let Some(filter) = rocr_visible
        .map(str::trim)
        .filter(|filter| !filter.is_empty())
    {
        let token = filter.split(',').next().unwrap_or_default().trim();
        return gpus
            .iter()
            .find(|gpu| match token.parse::<usize>() {
                Ok(ordinal) => gpu.rocr_index == ordinal,
                Err(_) => gpu
                    .uuid()
                    .is_some_and(|uuid| uuid.eq_ignore_ascii_case(token)),
            })
            .map(|gpu| gpu.arch.clone());
    }
    gpus.iter()
        .min_by_key(|gpu| gpu.rocr_index)
        .map(|gpu| gpu.arch.clone())
}

/// The GPU arch the load-time on-demand PLE download resolves its format with:
/// the arch of the device the load targets ([`device_arch_in`] over the
/// resolved `hardware.devices`, else `ROCR_VISIBLE_DEVICES`, else ROCr agent
/// order), so a mixed gfx1100 + gfx1151 host fetches what the card it loads on
/// prefers.
pub(crate) fn load_target_arch(resolved: &hipfire_config::ResolvedConfig) -> String {
    let spec = hipfire_config::ProcessConfig::from_resolved(resolved)
        .ok()
        .and_then(|config| config.legacy_value("HIPFIRE_DEVICES"));
    let rocr_visible = std::env::var("ROCR_VISIBLE_DEVICES").ok();
    device_arch_in(
        Path::new("/sys/class/kfd/kfd/topology/nodes"),
        spec.as_deref(),
        rocr_visible.as_deref(),
    )
    // The CLI cannot tell which card the load targets (no readable KFD
    // topology, as on WSL2 / native Windows, or a selector it cannot resolve
    // without the daemon's claim step): keep the original rule, the first
    // detected arch; empty (q8) when there is none.
    .or_else(|| local_gpu_arches().into_iter().next())
    .unwrap_or_default()
}

/// `memory.ple_format` from a resolved config; `auto` when absent or invalid.
pub(crate) fn request_from_config(resolved: &hipfire_config::ResolvedConfig) -> PleFormatRequest {
    crate::config_string(resolved, "memory.ple_format")
        .ok()
        .and_then(|value| PleFormatRequest::parse(&value))
        .unwrap_or(PleFormatRequest::Auto)
}

/// `memory.ple_format` from the global config + environment (pull has no
/// model context).
pub(crate) fn request_from_global(paths: &Paths) -> PleFormatRequest {
    crate::resolved_global(paths, true)
        .map(|(_, resolved)| request_from_config(&resolved))
        .unwrap_or(PleFormatRequest::Auto)
}

/// How the base of a split pull is satisfied.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PullBase<'a> {
    /// The split base is downloaded. `monolithic` is the pin of a monolithic a
    /// local file may exist for (not served: absent, unverified or `--force`).
    Download {
        monolithic: Option<&'a GrandfatheredMonolithic>,
    },
    /// The split base is already installed and verified.
    Present,
    /// A verified local monolithic at `path` serves the tags; no base download.
    Served {
        pin: &'a GrandfatheredMonolithic,
        path: &'a Path,
    },
}

/// Drop a planned exact (q8) sidecar when a served monolithic embeds the exact
/// rows: nothing is fetched for it and a note says why. A planned lossy format
/// stays (it is linked beside the monolithic).
pub(crate) fn drop_embedded_exact(plan: &mut PlePullPlan<'_>, model_file: &str) {
    let planned = plan.sidecars.len();
    plan.sidecars.retain(|(format, _)| !is_exact(*format));
    if plan.sidecars.len() != planned {
        plan.notes.push(format!(
            "ple-{PLE_EXACT_FORMAT} is not fetched: the exact {PLE_EXACT_FORMAT} rows are \
             embedded in the monolithic {model_file}"
        ));
    }
}

/// The sidecars a pull of `selection` fetches next to `base`: [`plan_pull`]
/// over every GPU arch in `gpu_arches`, minus the exact rows a served
/// monolithic embeds. `model_file` is the file the loader opens.
pub(crate) fn plan_for_base<'a>(
    entry: &'a ModelEntry,
    selection: PleSelection,
    request: PleFormatRequest,
    gpu_arches: &[String],
    base: PullBase<'_>,
    model_file: &str,
) -> PlePullPlan<'a> {
    let mut plan = plan_pull(entry, selection, request, gpu_arches);
    if let PullBase::Served { .. } = base {
        drop_embedded_exact(&mut plan, model_file);
    }
    plan
}

/// Print the pull plan for a split entry and return the sidecars to fetch.
///
/// `--ple auto` plans the union of the formats every detected GPU arch
/// defaults to. For a served monolithic the exact q8 rows are embedded in it,
/// so a planned q8 is dropped with a note; a planned q4 stays (it is linked
/// beside the monolithic).
pub(crate) fn announce_pull_plan<'a>(
    paths: &Paths,
    entry: &'a ModelEntry,
    selection: PleSelection,
    base: PullBase<'_>,
) -> PlePullPlan<'a> {
    let request = if selection == PleSelection::Auto {
        request_from_global(paths)
    } else {
        PleFormatRequest::Auto
    };
    let arches = local_gpu_arches();
    // The file the loader opens: its `<file>.ple-qN` names the links.
    let model_file = match base {
        PullBase::Served { path, .. } => path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&entry.file),
        _ => &entry.file,
    };
    let plan = plan_for_base(entry, selection, request, &arches, base, model_file);
    eprintln!("Pull plan (--ple {}):", selection_name(selection));
    eprintln!(
        "{}",
        match base {
            PullBase::Served { pin, .. } => plan_line(
                "base",
                model_file,
                Some(pin.size_bytes),
                "present (already satisfied by the monolithic; no base download)"
            ),
            PullBase::Present => plan_line("base", &entry.file, entry.size_bytes, "present"),
            PullBase::Download { .. } =>
                plan_line("base", &entry.file, entry.size_bytes, "download"),
        }
    );
    for (format, sidecar) in &plan.sidecars {
        let state = if paths.models.join(&sidecar.file).is_file() {
            "present (verified)"
        } else {
            "download"
        };
        let own_name = format!("{model_file}.{}", format.sidecar_suffix());
        let state = if sidecar.file != own_name {
            format!("{state}, shared; linked as {own_name}")
        } else {
            state.to_owned()
        };
        eprintln!(
            "{}",
            plan_line(
                &format!("ple-{}", format.name()),
                &sidecar.file,
                sidecar.size_bytes,
                &state
            )
        );
    }
    for note in &plan.notes {
        eprintln!("  note: {note}");
    }
    if plan.sidecars.is_empty() && selection != PleSelection::Skip {
        eprintln!("  note: no PLE sidecar will be downloaded");
    }
    if selection == PleSelection::Skip {
        eprintln!("  (the selected sidecar is fetched on first load)");
    }
    if let PullBase::Download {
        monolithic: Some(pin),
    } = base
    {
        let local_monolithic = [pin.canonical_file, pin.legacy_file]
            .into_iter()
            .find(|file| paths.models.join(file).is_file());
        if let Some(file) = local_monolithic {
            eprintln!(
                "  note: a local monolithic {file} is present; `hipfire ple split` converts it \
                 in place without downloading the base"
            );
        }
    }
    plan
}

fn selection_name(selection: PleSelection) -> &'static str {
    match selection {
        PleSelection::Auto => "auto",
        PleSelection::Q4 => "q4",
        PleSelection::Q8 => "q8",
        PleSelection::All => "all",
        PleSelection::Skip => "none",
    }
}

/// True when `sidecar` is the family-shared PLE file of a Flash-Next split base
/// (`qwen3.8-flash-next.ple-q4`) rather than the per-base `<base file>.ple-q4`.
/// The runtime loader only ever opens `<base file>.ple-qN`, so a shared slot is
/// installed as a link under that name ([`install_shared_alias`]).
pub(crate) fn is_shared_slot(entry: &ModelEntry, key: &str, sidecar: &Sidecar) -> bool {
    sidecar.file != format!("{}.ple-{key}", entry.file)
}

/// Make `alias` (`<base file>.ple-qN`, the name the loader opens) a link to the
/// verified shared file `canonical`, never copying its bytes.
///
/// The caller has verified `canonical` against the registry pin. An alias that
/// is already the same file is left alone. A different one is replaced
/// (atomically, through a temporary link) only when it is a dangling symlink or
/// verifies against the same pin; any other file is somebody else's sidecar
/// (e.g. a private `hipfire ple split` output) and is never touched.
fn install_shared_alias(
    cache: &VerifiedCache,
    canonical: &Path,
    alias: &Path,
    sidecar: &Sidecar,
    format: PleFormat,
) -> Result<()> {
    match fs::symlink_metadata(alias) {
        Ok(link) => {
            if crate::same_installed_file(alias, canonical) {
                return Ok(());
            }
            let dangling = link.file_type().is_symlink() && fs::metadata(alias).is_err();
            if !dangling
                && !crate::existing_artifact_valid(
                    cache,
                    alias,
                    sidecar.sha256.as_deref(),
                    sidecar.size_bytes,
                )
            {
                bail!(
                    "{} exists but is not the verified shared {}; remove it to install the \
                     shared pair (nothing was changed)",
                    alias.display(),
                    sidecar.file
                );
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspecting {}", alias.display()));
        }
    }
    let mut temporary = alias.as_os_str().to_owned();
    temporary.push(format!(".link.{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("clearing {}", temporary.display()));
        }
    }
    if !crate::install_verified_link(canonical, &temporary)? {
        bail!(
            "cannot link {} to {}: the filesystem supports neither hard links nor symlinks",
            alias.display(),
            canonical.display()
        );
    }
    if let Err(error) = fs::rename(&temporary, alias) {
        let _ = fs::remove_file(&temporary);
        return Err(error).with_context(|| format!("installing {}", alias.display()));
    }
    eprintln!(
        "  ple-{} linked: {} -> {}",
        format.name(),
        alias.display(),
        canonical.display()
    );
    Ok(())
}

/// Download `destination` unless a verified copy exists (`force` re-downloads).
/// On `Ok` the file matches the registry pin.
fn fetch_verified_slot(
    cache: &VerifiedCache,
    entry: &ModelEntry,
    format: PleFormat,
    sidecar: &Sidecar,
    destination: &Path,
    force: bool,
) -> Result<()> {
    if destination.exists() && !force {
        if crate::existing_artifact_valid(
            cache,
            destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
        ) {
            eprintln!(
                "  ple-{} sidecar already present: {}",
                format.name(),
                sidecar.file
            );
            return Ok(());
        }
        eprintln!(
            "  ple-{} sidecar stale, refreshing: {}",
            format.name(),
            sidecar.file
        );
    }
    eprintln!(
        "  Fetching ple-{} sidecar: {} ({})",
        format.name(),
        sidecar.file,
        format_size(sidecar.size_bytes)
    );
    crate::download_verified(
        cache,
        &crate::artifact_url(entry, &sidecar.file),
        destination,
        sidecar.sha256.as_deref(),
        sidecar.size_bytes,
        false,
    )
    .with_context(|| format!("downloading {}", sidecar.file))
}

/// Install one PLE sidecar of `entry` into `models` unless a verified copy
/// exists. The file is fetched once under its registry name, then, when the
/// loader's name for `model_path` (`<model file>.ple-qN`) differs (a shared
/// slot, or a served grandfathered monolithic), linked there once the
/// canonical file has verified.
pub(crate) fn fetch_ple_sidecar(
    cache: &VerifiedCache,
    entry: &ModelEntry,
    format: PleFormat,
    sidecar: &Sidecar,
    models: &Path,
    model_path: &Path,
    force: bool,
) -> Result<()> {
    let canonical = models.join(&sidecar.file);
    fetch_verified_slot(cache, entry, format, sidecar, &canonical, force)?;
    let alias = ple_sidecar_path(model_path, format);
    if alias != canonical {
        install_shared_alias(cache, &canonical, &alias, sidecar, format)?;
    }
    Ok(())
}

/// Make the PLE rows of a resolved registry model available before a load:
/// downloads the selected sidecar when it (and every acceptable substitute) is
/// missing, per [`plan_on_demand`]. A no-op for models without PLE slots, for
/// a split entry served from its grandfathered monolithic (one note), and for
/// paths that are not the registry artifact.
///
/// `gpu_arch` is the arch of the device the load targets
/// ([`load_target_arch`]); it resolves `memory.ple_format = auto`.
/// `allow_download = false` (a serve request with downloads disabled) turns a
/// needed download into an error naming the pull command.
pub(crate) fn ensure_ple_sidecar(
    entry: Option<&ModelEntry>,
    tag: Option<&str>,
    model_path: &Path,
    request: PleFormatRequest,
    gpu_arch: &str,
    allow_download: bool,
) -> Result<()> {
    let Some(entry) = entry.filter(|entry| entry.has_ple_sidecars()) else {
        return Ok(());
    };
    let file_name = model_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let label = tag.unwrap_or(&entry.file);
    match classify_layout(entry, file_name) {
        ModelLayout::SplitBase => {}
        ModelLayout::Monolithic => {
            note_monolithic(label, file_name);
            return Ok(());
        }
        ModelLayout::Other => return Ok(()),
    }
    // The same lookup order as the loader: the path as selected, then its
    // canonical target (a symlinked base keeps the sidecars beside either).
    let sidecar_present = |format: PleFormat| {
        std::iter::once(model_path.to_path_buf())
            .chain(std::fs::canonicalize(model_path).ok())
            .any(|base| ple_sidecar_path(&base, format).is_file())
    };
    let present = PlePresence {
        q4: sidecar_present(PleFormat::Q4),
        q8: sidecar_present(PleFormat::Q8),
    };
    let Some(format) = plan_on_demand(entry, request, gpu_arch, present) else {
        return Ok(());
    };
    let sidecar = ple_slot(entry, format)
        .ok_or_else(|| anyhow!("registry slot ple-{} vanished", format.name()))?;
    let shared = is_shared_slot(entry, format.name(), sidecar);
    let alias = ple_sidecar_path(model_path, format);
    let destination = if shared {
        model_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(&sidecar.file)
    } else {
        alias.clone()
    };
    // A shared file another base already downloaded only needs linking.
    let local_shared = shared && destination.is_file();
    if !allow_download && !local_shared {
        bail!(
            "ple: {} is missing; run `hipfire pull {label} --ple {}` on the server first \
             ({}; downloads named by a request are off: serve.allow_request_pull)",
            sidecar.file,
            format.name(),
            format_size(sidecar.size_bytes)
        );
    }
    // The on-demand path runs inside a load with no hipfire home in scope and
    // only links or fetches a sidecar: `hipfire pull` is what records proofs.
    let cache = VerifiedCache::disabled();
    if local_shared {
        eprintln!("ple: verifying shared {}", sidecar.file);
        if crate::existing_artifact_valid(
            &cache,
            &destination,
            sidecar.sha256.as_deref(),
            sidecar.size_bytes,
        ) {
            return install_shared_alias(&cache, &destination, &alias, sidecar, format);
        }
        if !allow_download {
            bail!(
                "ple: shared {} fails verification; run `hipfire pull {label} --ple {}` on the \
                 server to refresh it (downloads named by a request are off: \
                 serve.allow_request_pull)",
                sidecar.file,
                format.name()
            );
        }
        eprintln!("ple: shared {} is stale, refreshing", sidecar.file);
    }
    eprintln!(
        "ple: downloading {} ({}) for memory.ple_format={}",
        sidecar.file,
        format_size(sidecar.size_bytes),
        request.name()
    );
    crate::download_verified(
        &cache,
        &crate::artifact_url(entry, &sidecar.file),
        &destination,
        sidecar.sha256.as_deref(),
        sidecar.size_bytes,
        false,
    )
    .with_context(|| format!("downloading {}", sidecar.file))?;
    if shared {
        install_shared_alias(&cache, &destination, &alias, sidecar, format)?;
    }
    Ok(())
}

fn note_monolithic(label: &str, file: &str) {
    static NOTED: std::sync::Once = std::sync::Once::new();
    NOTED.call_once(|| {
        eprintln!(
            "note: serving the local monolithic {file} for {label}; `hipfire ple split {label}` \
             or `hipfire pull {label} --force` gets the split PLE layout (base + swappable PLE \
             rows)"
        );
    });
}

// ---------------------------------------------------------------------------
// hipfire ple derive / split
// ---------------------------------------------------------------------------

fn resolve_local_model(paths: &Paths, model: &str) -> Result<PathBuf> {
    let registry = hipfire_registry::load(&paths.registry).registry;
    crate::resolve_model_path(paths, &registry, model)?.ok_or_else(|| {
        anyhow!("model not found locally: {model}; pass a model path or `hipfire pull` it first")
    })
}

fn derive_command(paths: &Paths, args: DeriveArgs) -> Result<()> {
    if args.format == DeriveFormat::Q8 {
        bail!(
            "q8 rows cannot be derived: they are the model's own exact rows (embedded in a \
             monolithic model) or the `<base>.ple-q8` written by `hipfire ple split`; only \
             `--format q4` derives (from the q8 rows)"
        );
    }
    let model = resolve_local_model(paths, &args.model)?;
    eprintln!("ple: deriving q4 rows for {}", model.display());
    let mut printer = ProgressPrinter::new(Instant::now());
    let mut report_progress = |progress: &PleProgress| {
        if let Some(line) = printer.update(progress, Instant::now()) {
            eprintln!("{line}");
        }
    };
    let report = ple_convert::derive_q4(&model, args.rank.as_deref(), &mut report_progress)
        .map_err(|error| anyhow!(error))?;
    for line in derive_summary(&report) {
        println!("{line}");
    }
    Ok(())
}

fn split_command(paths: &Paths, args: SplitArgs) -> Result<()> {
    let model = resolve_local_model(paths, &args.model)?;
    let base = args
        .out
        .clone()
        .unwrap_or_else(|| ple_convert::default_split_base_path(&model));
    if base == model {
        bail!(
            "split output {} is the input model; pass --out to name a different base",
            base.display()
        );
    }
    let q8 = ple_sidecar_path(&base, PleFormat::Q8);
    for output in [&base, &q8] {
        if output.exists() {
            bail!(
                "split output {} already exists; remove it or pass --out",
                output.display()
            );
        }
    }
    eprintln!(
        "ple: splitting {} into {} + {}",
        model.display(),
        base.display(),
        q8.display()
    );
    let mut printer = ProgressPrinter::new(Instant::now());
    let mut report_progress = |progress: &PleProgress| {
        if let Some(line) = printer.update(progress, Instant::now()) {
            eprintln!("{line}");
        }
    };
    let report = ple_convert::split(&model, args.out.as_deref(), &mut report_progress)
        .map_err(|error| anyhow!(error))?;
    for line in split_summary(&report) {
        println!("{line}");
    }
    let plehot = plehot_siblings(&model);
    let stale_q4 = ple_sidecar_path(&model, PleFormat::Q4);
    if args.keep_original {
        println!("kept original: {}", model.display());
    } else {
        // The twins share the monolithic's inode; list them before removal.
        let mut removed = monolithic_twins(&model);
        removed.insert(0, model.clone());
        for path in removed {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            println!("removed original: {}", path.display());
        }
    }
    if !plehot.is_empty() {
        let names: Vec<String> = plehot
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        println!(
            "note: {} belong to the monolithic layout and were left in place; they do not \
             apply to the split base",
            names.join(", ")
        );
    }
    if stale_q4.is_file() {
        println!(
            "note: {} is bound to the monolithic and does not load with the split base; \
             delete it or run `hipfire ple derive {}`",
            stale_q4.display(),
            report.base.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cli, Commands};
    use clap::Parser;
    use hipfire_registry::RegistryV1;

    fn sidecar(file: &str) -> Sidecar {
        Sidecar {
            file: file.into(),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(1_000_000_000),
        }
    }

    fn split_entry(q4: bool, q8: bool) -> ModelEntry {
        ModelEntry {
            repo: "hipfire-models/qwen3.8-flash-next".into(),
            file: "qwen3.8-flash-next-base.mq4-xts".into(),
            ple: [
                (
                    "q4",
                    q4.then(|| sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q4")),
                ),
                (
                    "q8",
                    q8.then(|| sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q8")),
                ),
            ]
            .into_iter()
            .filter_map(|(key, sidecar)| Some((key.to_owned(), sidecar?)))
            .collect(),
            ..Default::default()
        }
    }

    fn formats(plan: &PlePullPlan<'_>) -> Vec<PleFormat> {
        plan.sidecars.iter().map(|(format, _)| *format).collect()
    }

    /// The detected arches of a one-GPU host (none for an unknown arch).
    fn arches(arch: &str) -> Vec<String> {
        if arch.is_empty() {
            Vec::new()
        } else {
            vec![arch.to_owned()]
        }
    }

    const NONE: PlePresence = PlePresence {
        q4: false,
        q8: false,
    };
    const Q4: PlePresence = PlePresence {
        q4: true,
        q8: false,
    };
    const Q8: PlePresence = PlePresence {
        q4: false,
        q8: true,
    };
    const BOTH: PlePresence = PlePresence { q4: true, q8: true };

    #[test]
    fn pull_selection_picks_files_per_flag_and_arch() {
        let entry = split_entry(true, true);
        let pick = |selection, request, arch: &str| {
            formats(&plan_pull(&entry, selection, request, &arches(arch)))
        };
        let auto = PleFormatRequest::Auto;
        assert_eq!(pick(PleSelection::Auto, auto, "gfx1151"), [PleFormat::Q4]);
        assert_eq!(pick(PleSelection::Auto, auto, "gfx1100"), [PleFormat::Q8]);
        assert_eq!(
            pick(PleSelection::Auto, auto, ""),
            [PleFormat::Q8],
            "unknown arch is q8"
        );
        // The configured request wins over the arch default.
        assert_eq!(
            pick(PleSelection::Auto, PleFormatRequest::Q8, "gfx1151"),
            [PleFormat::Q8]
        );
        assert_eq!(
            pick(PleSelection::Auto, PleFormatRequest::Q4, "gfx1100"),
            [PleFormat::Q4]
        );
        // Explicit selections ignore config and arch.
        assert_eq!(pick(PleSelection::Q4, auto, "gfx1100"), [PleFormat::Q4]);
        assert_eq!(pick(PleSelection::Q8, auto, "gfx1151"), [PleFormat::Q8]);
        assert_eq!(
            pick(PleSelection::All, auto, "gfx1100"),
            [PleFormat::Q4, PleFormat::Q8]
        );
        assert!(pick(PleSelection::Skip, auto, "gfx1151").is_empty());
        let plan = plan_pull(&entry, PleSelection::All, auto, &arches("gfx1100"));
        assert_eq!(
            plan.sidecars[1].1.file,
            "qwen3.8-flash-next-base.mq4-xts.ple-q8"
        );
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn pull_selection_handles_unpublished_slots() {
        let q8_only = split_entry(false, true);
        let auto = PleFormatRequest::Auto;
        // q4 falls back to q8 (`hipfire ple derive` makes q4 from it), with a note.
        for selection in [PleSelection::Q4, PleSelection::Auto] {
            let plan = plan_pull(&q8_only, selection, auto, &arches("gfx1151"));
            assert_eq!(formats(&plan), [PleFormat::Q8], "{selection:?}");
            assert!(
                plan.notes[0].contains("hipfire ple derive --format q4"),
                "{:?}",
                plan.notes
            );
        }
        // `all` takes what exists and explains what does not.
        let plan = plan_pull(&q8_only, PleSelection::All, auto, &arches("gfx1151"));
        assert_eq!(formats(&plan), [PleFormat::Q8]);
        assert_eq!(plan.notes.len(), 1);

        let q4_only = split_entry(true, false);
        // An explicit q8 never degrades to q4.
        let plan = plan_pull(&q4_only, PleSelection::Q8, auto, &arches("gfx1151"));
        assert!(plan.sidecars.is_empty());
        assert!(plan.notes[0].contains("no ple-q8"));
        // Auto prefers q8 off gfx1151 but takes the published q4.
        let plan = plan_pull(&q4_only, PleSelection::Auto, auto, &arches("gfx1100"));
        assert_eq!(formats(&plan), [PleFormat::Q4]);
        assert_eq!(plan.notes.len(), 1);

        let bare = split_entry(false, false);
        assert!(
            plan_pull(&bare, PleSelection::Auto, auto, &arches("gfx1151"))
                .sidecars
                .is_empty()
        );
    }

    #[test]
    fn on_demand_follows_the_request_and_never_downloads_over_a_usable_file() {
        let entry = split_entry(true, true);
        let go = |request, arch: &str, present| plan_on_demand(&entry, request, arch, present);
        // q8: present -> nothing; absent -> download, even when q4 is on disk.
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", Q8), None);
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", BOTH), None);
        assert_eq!(
            go(PleFormatRequest::Q8, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(go(PleFormatRequest::Q8, "gfx1151", Q4), Some(PleFormat::Q8));
        // q4: ple-q4 or ple-q8 on disk is enough (a load never derives; `hipfire ple derive` does); else download q4.
        assert_eq!(go(PleFormatRequest::Q4, "gfx1100", Q4), None);
        assert_eq!(go(PleFormatRequest::Q4, "gfx1100", Q8), None);
        assert_eq!(
            go(PleFormatRequest::Q4, "gfx1100", NONE),
            Some(PleFormat::Q4)
        );
        // auto: preferred present, or the other present -> nothing; neither -> preferred.
        assert_eq!(go(PleFormatRequest::Auto, "gfx1151", Q4), None);
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1151", Q8),
            None,
            "falls back, no download"
        );
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1100", Q4),
            None,
            "falls back, no download"
        );
        assert_eq!(go(PleFormatRequest::Auto, "gfx1100", Q8), None);
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1151", NONE),
            Some(PleFormat::Q4)
        );
        assert_eq!(
            go(PleFormatRequest::Auto, "gfx1100", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(go(PleFormatRequest::Auto, "", NONE), Some(PleFormat::Q8));
    }

    #[test]
    fn on_demand_substitutes_the_published_slot_and_never_names_a_missing_one() {
        let q8_only = split_entry(false, true);
        // q4 requested, nothing local, registry has only q8: fetch q8 (derive q4 locally).
        assert_eq!(
            plan_on_demand(&q8_only, PleFormatRequest::Q4, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        assert_eq!(
            plan_on_demand(&q8_only, PleFormatRequest::Auto, "gfx1151", NONE),
            Some(PleFormat::Q8)
        );
        let q4_only = split_entry(true, false);
        assert_eq!(
            plan_on_demand(&q4_only, PleFormatRequest::Auto, "gfx1100", NONE),
            Some(PleFormat::Q4)
        );
        // Exact q8 rows cannot come from a q4-only registry: let the loader refuse.
        assert_eq!(
            plan_on_demand(&q4_only, PleFormatRequest::Q8, "gfx1100", NONE),
            None
        );
        let bare = split_entry(false, false);
        assert_eq!(
            plan_on_demand(&bare, PleFormatRequest::Auto, "gfx1151", NONE),
            None
        );
    }

    fn many(arches: &[&str]) -> Vec<String> {
        arches.iter().map(|arch| (*arch).to_owned()).collect()
    }

    #[test]
    fn auto_plans_the_union_of_every_gpu_archs_default_format() {
        let entry = split_entry(true, true);
        let auto = PleFormatRequest::Auto;
        let plan = |selection, request, arches: &[&str]| {
            formats(&plan_pull(&entry, selection, request, &many(arches)))
        };
        // hipx: a gfx1100 (q8) next to a gfx1151 (q4) plans both, q4 first.
        assert_eq!(
            plan(PleSelection::Auto, auto, &["gfx1100", "gfx1151"]),
            [PleFormat::Q4, PleFormat::Q8]
        );
        assert_eq!(
            plan(PleSelection::Auto, auto, &["gfx1151", "gfx1100"]),
            [PleFormat::Q4, PleFormat::Q8],
            "arch order does not matter"
        );
        assert_eq!(
            plan(PleSelection::Auto, auto, &["gfx1151", "gfx1151"]),
            [PleFormat::Q4]
        );
        assert_eq!(
            plan(PleSelection::Auto, auto, &["gfx1100", "gfx1201"]),
            [PleFormat::Q8]
        );
        assert_eq!(plan(PleSelection::Auto, auto, &[]), [PleFormat::Q8]);
        assert_eq!(
            plan(PleSelection::Auto, auto, &["gfx9999"]),
            [PleFormat::Q8],
            "unknown arch is q8"
        );
        // A configured request resolves identically on every arch.
        assert_eq!(
            plan(
                PleSelection::Auto,
                PleFormatRequest::Q8,
                &["gfx1100", "gfx1151"]
            ),
            [PleFormat::Q8]
        );
        assert_eq!(
            plan(
                PleSelection::Auto,
                PleFormatRequest::Q4,
                &["gfx1100", "gfx1151"]
            ),
            [PleFormat::Q4]
        );
        // Explicit selections ignore the arches.
        assert_eq!(
            plan(PleSelection::Q8, auto, &["gfx1100", "gfx1151"]),
            [PleFormat::Q8]
        );
        assert_eq!(
            plan(PleSelection::Q4, auto, &["gfx1100", "gfx1201"]),
            [PleFormat::Q4]
        );
        assert_eq!(
            formats_for_arches(auto, &many(&["gfx1100", "gfx1151"])),
            [PleFormat::Q4, PleFormat::Q8]
        );

        // One published format: the union collapses onto it, with a note.
        let q8_only = split_entry(false, true);
        let plan = plan_pull(
            &q8_only,
            PleSelection::Auto,
            auto,
            &many(&["gfx1100", "gfx1151"]),
        );
        assert_eq!(formats(&plan), [PleFormat::Q8]);
        assert_eq!(plan.notes.len(), 1, "{:?}", plan.notes);
        let q4_only = split_entry(true, false);
        let plan = plan_pull(
            &q4_only,
            PleSelection::Auto,
            auto,
            &many(&["gfx1100", "gfx1151"]),
        );
        assert_eq!(formats(&plan), [PleFormat::Q4]);
    }

    #[test]
    fn q8_is_the_exact_reference_and_the_lossy_fallback() {
        let q8_only = split_entry(false, true);
        let plan = plan_pull(
            &q8_only,
            PleSelection::Q4,
            PleFormatRequest::Auto,
            &many(&["gfx1151"]),
        );
        assert_eq!(formats(&plan), [PleFormat::Q8]);
        assert!(
            plan.notes[0].contains("fetching ple-q8 instead"),
            "{:?}",
            plan.notes
        );
        assert_eq!(
            fallback_format(&q8_only, PleFormat::Q4),
            Some(PleFormat::Q8)
        );
        assert_eq!(fallback_format(&q8_only, PleFormat::Q8), None);
        let q4_only = split_entry(true, false);
        assert_eq!(fallback_format(&q4_only, PleFormat::Q4), None);
        assert_eq!(
            fallback_format(&q4_only, PleFormat::Q8),
            Some(PleFormat::Q4)
        );
        // An explicit q8 never falls back to lossy rows.
        let plan = plan_pull(
            &q4_only,
            PleSelection::Q8,
            PleFormatRequest::Auto,
            &many(&["gfx1151"]),
        );
        assert!(plan.sidecars.is_empty());
    }

    #[test]
    fn formats_this_client_cannot_load_are_ignored_with_a_note() {
        let mut entry = split_entry(true, true);
        entry.ple.insert(
            "q2".into(),
            sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q2"),
        );
        assert_eq!(unloadable_formats(&entry), ["q2"]);
        assert_eq!(
            unloadable_formats(&split_entry(true, true)),
            Vec::<&str>::new()
        );
        for (selection, expected) in [
            (PleSelection::All, vec![PleFormat::Q4, PleFormat::Q8]),
            (PleSelection::Auto, vec![PleFormat::Q8]),
            (PleSelection::Q4, vec![PleFormat::Q4]),
            (PleSelection::Q8, vec![PleFormat::Q8]),
        ] {
            let plan = plan_pull(
                &entry,
                selection,
                PleFormatRequest::Auto,
                &many(&["gfx1100"]),
            );
            assert_eq!(formats(&plan), expected, "{selection:?}");
            assert!(
                plan.notes
                    .iter()
                    .any(|note| note.contains("ple-q2") && note.contains("cannot load")),
                "{selection:?}: {:?}",
                plan.notes
            );
        }
        // `--ple none` fetches and says nothing.
        let skip = plan_pull(&entry, PleSelection::Skip, PleFormatRequest::Auto, &[]);
        assert!(skip.sidecars.is_empty() && skip.notes.is_empty());
        // Only an unloadable format published: nothing to fetch, loader refuses.
        let mut only_q2 = split_entry(false, false);
        only_q2.ple.insert(
            "q2".into(),
            sidecar("qwen3.8-flash-next-base.mq4-xts.ple-q2"),
        );
        let plan = plan_pull(&only_q2, PleSelection::Auto, PleFormatRequest::Auto, &[]);
        assert!(plan.sidecars.is_empty());
        assert_eq!(
            plan_on_demand(&only_q2, PleFormatRequest::Auto, "gfx1100", NONE),
            None
        );
    }

    #[test]
    fn a_served_monolithic_embeds_q8_so_only_lossy_rows_are_planned() {
        let entry = split_entry(true, true);
        let pin = GrandfatheredMonolithic {
            repo: "hipfire-models/qwen3.8-flash-next",
            split_file: "qwen3.8-flash-next-base.mq4-xts",
            split_sha256: "e",
            split_size_bytes: 1,
            canonical_file: "qwen3.8-flash-next.mq4-xts",
            legacy_file: FLASH_NEXT_XTS_LEGACY_FILE,
            sha256: "f",
            size_bytes: 2,
        };
        let served = PullBase::Served {
            pin: &pin,
            path: Path::new("/m/qwen3.8-flash-next.mq4-xts"),
        };
        let auto = PleFormatRequest::Auto;
        let plan_for = |selection, arches: &[&str], base| {
            plan_for_base(
                &entry,
                selection,
                auto,
                &many(arches),
                base,
                "qwen3.8-flash-next.mq4-xts",
            )
        };
        // hipx with the monolithic: only q4 (gfx1151); q8 is embedded.
        let plan = plan_for(PleSelection::Auto, &["gfx1100", "gfx1151"], served);
        assert_eq!(formats(&plan), [PleFormat::Q4]);
        assert!(plan
            .notes
            .iter()
            .any(|note| note.contains("embedded in the monolithic")));
        // gfx1100 only (q8 by default) with the monolithic: nothing to fetch.
        assert!(plan_for(PleSelection::Auto, &["gfx1100"], served)
            .sidecars
            .is_empty());
        // An explicit q8 on Halo is satisfied by the monolithic as well.
        assert!(plan_for(PleSelection::Q8, &["gfx1151"], served)
            .sidecars
            .is_empty());
        // `--ple all` still links q4 beside the monolithic.
        assert_eq!(
            formats(&plan_for(PleSelection::All, &["gfx1151"], served)),
            [PleFormat::Q4]
        );
        // Without the monolithic the same hipx host plans both.
        let download = PullBase::Download {
            monolithic: Some(&pin),
        };
        assert_eq!(
            formats(&plan_for(
                PleSelection::Auto,
                &["gfx1100", "gfx1151"],
                download
            )),
            [PleFormat::Q4, PleFormat::Q8]
        );
    }

    /// A fixture KFD topology: (node, gfx_target_version, location_id, unique_id).
    fn kfd_fixture(label: &str, nodes: &[(u32, u32, u32, u64)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "hipfire-kfd-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for (node, version, location, unique) in nodes {
            let dir = root.join(node.to_string());
            fs::create_dir_all(&dir).unwrap();
            fs::write(
                dir.join("properties"),
                format!(
                    "simd_count 96\nlocation_id {location}\ndomain 0\n\
                     gfx_target_version {version}\nunique_id {unique}\n"
                ),
            )
            .unwrap();
        }
        root
    }

    #[test]
    fn the_load_target_arch_follows_devices_then_rocr_then_agent_order() {
        // Node 0 = gfx1151 at bus 7, node 1 = gfx1100 at bus 3 (BDF order is
        // the reverse of ROCr order).
        let root = kfd_fixture(
            "load-arch",
            &[(0, 110501, 0x0700, 0x1111), (1, 110000, 0x0300, 0x2222)],
        );
        let arch = |spec, rocr| device_arch_in(&root, spec, rocr);
        // No filter: ROCr agent 0 (the lowest ordinal) is HIP device 0.
        assert_eq!(arch(None, None).as_deref(), Some("gfx1151"));
        assert_eq!(arch(Some(""), Some("")).as_deref(), Some("gfx1151"));
        // `hardware.devices`: an arch, a BDF-order index or a UUID.
        assert_eq!(arch(Some("gfx1100"), None).as_deref(), Some("gfx1100"));
        assert_eq!(arch(Some("0"), None).as_deref(), Some("gfx1100"));
        assert_eq!(arch(Some("1"), None).as_deref(), Some("gfx1151"));
        assert_eq!(arch(Some("1,0"), None).as_deref(), Some("gfx1151"));
        assert_eq!(
            arch(Some("GPU-0000000000002222"), None).as_deref(),
            Some("gfx1100")
        );
        // It wins over an inherited ROCR_VISIBLE_DEVICES.
        assert_eq!(arch(Some("gfx1100"), Some("0")).as_deref(), Some("gfx1100"));
        // ROCR_VISIBLE_DEVICES: ROCr ordinal or UUID, first entry.
        assert_eq!(arch(None, Some("1")).as_deref(), Some("gfx1100"));
        assert_eq!(arch(None, Some("1,0")).as_deref(), Some("gfx1100"));
        assert_eq!(arch(None, Some("0")).as_deref(), Some("gfx1151"));
        assert_eq!(
            arch(None, Some("GPU-0000000000002222")).as_deref(),
            Some("gfx1100")
        );
        // Unresolvable selectors and an unreadable topology are unknown.
        assert_eq!(arch(Some("gfx9999"), None), None);
        assert_eq!(arch(Some("7"), None), None);
        assert_eq!(arch(None, Some("5")), None);
        assert_eq!(arch(None, Some("-1")), None);
        assert_eq!(device_arch_in(&root.join("missing"), None, None), None);
        fs::remove_dir_all(&root).unwrap();
    }

    fn registry_with_split() -> RegistryV1 {
        // The published MQ4 XTS split base: its identity is the compiled-in
        // grandfather pin's, so the pinned monolithic serves it.
        let value = serde_json::json!({
            "schema_version": 1,
            "generated_at": "2099-01-01T00:00:00Z",
            "models": {
                "qwen3.8:flash-next": {
                    "repo": "hipfire-models/qwen3.8-flash-next",
                    "file": "qwen3.8-flash-next-base.mq4-xts",
                    "size_gb": 70.9, "min_vram_gb": 8.0, "desc": "split",
                    "sha256": "eafaff94fb9c6bb51282942e0ea6441cbdc266b634bb003b5913cfe2ccb90f45",
                    "size_bytes": 70888270360_u64,
                    "ple": {"q8": {"file": "qwen3.8-flash-next-base.mq4-xts.ple-q8"}}
                },
                "unrelated": {"repo": "x/y", "file": "plain.mq4", "size_gb": 1, "min_vram_gb": 1, "desc": "d"}
            },
            "aliases": {}
        });
        RegistryV1::parse(&value.to_string(), "ple test").unwrap()
    }

    #[test]
    fn layout_classification_and_tag_fallback_files() {
        let registry = registry_with_split();
        let split = &registry.models["qwen3.8:flash-next"];
        assert_eq!(
            classify_layout(split, "qwen3.8-flash-next-base.mq4-xts"),
            ModelLayout::SplitBase
        );
        for name in ["qwen3.8-flash-next.mq4-xts", FLASH_NEXT_XTS_LEGACY_FILE] {
            assert_eq!(
                classify_layout(split, name),
                ModelLayout::Monolithic,
                "{name}"
            );
        }
        assert_eq!(classify_layout(split, "plain.mq4"), ModelLayout::Other);
        assert_eq!(
            hipfire_registry::split_model_files(split),
            [
                "qwen3.8-flash-next-base.mq4-xts",
                "qwen3.8-flash-next.mq4-xts",
                FLASH_NEXT_XTS_LEGACY_FILE
            ]
        );
        // An entry without a pin serves only its own file.
        let plain = &registry.models["unrelated"];
        assert_eq!(hipfire_registry::split_model_files(plain), ["plain.mq4"]);
    }

    #[test]
    fn ensure_is_a_noop_without_a_split_entry_or_for_other_layouts() {
        let registry = registry_with_split();
        let request = PleFormatRequest::Q8;
        // No entry, a plain entry, and the monolithic fallback never download.
        ensure_ple_sidecar(
            None,
            None,
            Path::new("/nonexistent/x.mq4"),
            request,
            "gfx1151",
            true,
        )
        .unwrap();
        let plain = &registry.models["unrelated"];
        ensure_ple_sidecar(
            Some(plain),
            None,
            Path::new("/nonexistent/plain.mq4"),
            request,
            "gfx1151",
            true,
        )
        .unwrap();
        let split = &registry.models["qwen3.8:flash-next"];
        ensure_ple_sidecar(
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next.mq4-xts"),
            request,
            "gfx1151",
            false,
        )
        .unwrap();
        ensure_ple_sidecar(
            Some(split),
            None,
            Path::new("/nonexistent/other.mq4"),
            request,
            "gfx1151",
            false,
        )
        .unwrap();
    }

    #[test]
    fn ensure_refuses_a_needed_download_when_downloads_are_off() {
        let registry = registry_with_split();
        let split = &registry.models["qwen3.8:flash-next"];
        let error = ensure_ple_sidecar(
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next-base.mq4-xts"),
            PleFormatRequest::Q8,
            "gfx1151",
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("hipfire pull qwen3.8:flash-next --ple q8"),
            "{error}"
        );
        assert!(error.contains("serve.allow_request_pull"), "{error}");
        // q4 with nothing local and a q8-only registry points at the q8 file.
        let error = ensure_ple_sidecar(
            Some(split),
            Some("qwen3.8:flash-next"),
            Path::new("/nonexistent/qwen3.8-flash-next-base.mq4-xts"),
            PleFormatRequest::Q4,
            "gfx1151",
            false,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("--ple q8"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn ensure_sees_a_sidecar_beside_the_canonical_target_of_a_symlinked_base() {
        let registry = registry_with_split();
        let split = &registry.models["qwen3.8:flash-next"];
        let root =
            std::env::temp_dir().join(format!("hipfire-ple-ensure-symlink-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = root.join("store");
        let models = root.join("models");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let target = store.join("qwen3.8-flash-next-base.mq4-xts");
        std::fs::write(&target, b"base").unwrap();
        let link = models.join("qwen3.8-flash-next-base.mq4-xts");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let ensure = || {
            ensure_ple_sidecar(
                Some(split),
                Some("qwen3.8:flash-next"),
                &link,
                PleFormatRequest::Q8,
                "gfx1151",
                false,
            )
        };

        // Nothing anywhere: a needed download is refused (downloads off).
        assert!(ensure().is_err());
        // The q8 rows beside the canonical target satisfy the symlinked base.
        std::fs::write(ple_sidecar_path(&target, PleFormat::Q8), b"rows").unwrap();
        ensure().unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sidecar_paths_name_rows_and_their_hot_packs() {
        let paths = ple_sidecar_paths(Path::new("/m/x-base.mq4-xts"), None);
        let names: Vec<_> = paths
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        assert_eq!(
            names,
            [
                "/m/x-base.mq4-xts.ple-q4",
                "/m/x-base.mq4-xts.ple-q4.plehotrows",
                "/m/x-base.mq4-xts.ple-q8",
                "/m/x-base.mq4-xts.ple-q8.plehotrows",
                "/m/x-base.mq4-xts.plehot",
            ]
        );
        // A key this client cannot load is still removed with its model.
        let mut entry = split_entry(true, true);
        entry
            .ple
            .insert("q2".into(), sidecar("qwen3.8-flash-next.ple-q2"));
        let names: Vec<_> = ple_sidecar_paths(Path::new("/m/x-base.mq4-xts"), Some(&entry))
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        assert_eq!(names.len(), 7);
        assert!(names.contains(&"/m/x-base.mq4-xts.ple-q2".to_owned()));
        assert!(names.contains(&"/m/x-base.mq4-xts.ple-q2.plehotrows".to_owned()));
        assert_eq!(
            ple_sidecar_path_for_key(Path::new("/m/x-base.mq4-xts"), "q2"),
            Path::new("/m/x-base.mq4-xts.ple-q2")
        );
    }

    #[test]
    fn progress_is_throttled_per_phase_and_reports_completion() {
        let t0 = Instant::now();
        let mut printer = ProgressPrinter::new(t0);
        let at = |secs: f64| t0 + Duration::from_secs_f64(secs);
        let progress = |phase, done, total| PleProgress {
            phase,
            done_bytes: done,
            total_bytes: total,
        };
        let first = printer
            .update(&progress("read", 0, 10_000_000_000), at(0.0))
            .unwrap();
        assert!(first.starts_with("ple: read:"), "{first}");
        assert!(printer
            .update(&progress("read", 100, 10_000_000_000), at(0.5))
            .is_none());
        assert!(printer
            .update(&progress("read", 200, 10_000_000_000), at(1.9))
            .is_none());
        let line = printer
            .update(&progress("read", 4_000_000_000, 10_000_000_000), at(2.0))
            .unwrap();
        assert!(line.contains("40.0%"), "{line}");
        assert!(line.contains("4.00/10.00 GB"), "{line}");
        assert!(line.contains("2000 MB/s"), "{line}");
        // Completion prints once even inside the interval.
        assert!(printer
            .update(&progress("read", 10_000_000_000, 10_000_000_000), at(2.1))
            .is_some());
        assert!(printer
            .update(&progress("read", 10_000_000_000, 10_000_000_000), at(2.2))
            .is_none());
        // A new phase prints immediately and restarts the rate.
        let verify = printer
            .update(&progress("verify", 0, 5_000_000_000), at(2.3))
            .unwrap();
        assert!(verify.starts_with("ple: verify:"), "{verify}");
        // Unknown total: no percentage.
        let mut printer = ProgressPrinter::new(t0);
        let line = printer
            .update(&progress("scan", 1_000_000_000, 0), at(1.0))
            .unwrap();
        assert!(!line.contains('%'), "{line}");
    }

    #[test]
    fn durations_and_plan_lines_are_human_readable() {
        assert_eq!(format_duration(Duration::from_secs_f64(12.34)), "12.3s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m05s");
        assert_eq!(format_duration(Duration::from_secs(3723)), "1h02m03s");
        let line = plan_line("ple-q8", "m.ple-q8", Some(54_400_000_000), "download");
        assert!(
            line.contains("54.40 GB") && line.contains("m.ple-q8") && line.ends_with("download")
        );
        assert!(plan_line("base", "m", None, "present").contains("size unknown"));
    }

    #[test]
    fn summaries_print_every_digest_the_registry_needs() {
        let derive = DeriveReport {
            output: PathBuf::from("/m/x.ple-q4"),
            source: PathBuf::from("/m/x.ple-q8"),
            bytes: 28_800_000_000,
            payload_sha256: "p".repeat(64),
            source_rows_sha256: "s".repeat(64),
            elapsed: Duration::from_secs(90),
            threads: 16,
            order: "rank",
        };
        let text = derive_summary(&derive).join("\n");
        for needle in [
            "/m/x.ple-q4",
            "/m/x.ple-q8",
            "28800000000 B",
            "28.80 GB",
            "payload_sha256",
            "q8_rows_sha256",
            "1m30s",
            "16 threads",
            "order           rank",
        ] {
            assert!(text.contains(needle), "{needle}: {text}");
        }
        let split = SplitReport {
            base: PathBuf::from("/m/x-base.mq4"),
            base_bytes: 70_900_000_000,
            base_index_sha256: "b".repeat(64),
            ple_q8: PathBuf::from("/m/x-base.mq4.ple-q8"),
            ple_q8_bytes: 54_400_000_000,
            q8_rows_sha256: "q".repeat(64),
            elapsed: Duration::from_secs(3700),
        };
        let text = split_summary(&split).join("\n");
        for needle in [
            "/m/x-base.mq4",
            "70.90 GB",
            "base_index_sha256",
            "x-base.mq4.ple-q8",
            "54.40 GB",
            "q8_rows_sha256",
            "1h01m40s",
        ] {
            assert!(text.contains(needle), "{needle}: {text}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn split_removal_covers_hardlinked_spellings_and_reports_hot_files() {
        let dir = std::env::temp_dir().join(format!("hipfire-ple-twins-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let canonical = dir.join(FLASH_NEXT_XTS_CANONICAL_FILE);
        let legacy = dir.join(FLASH_NEXT_XTS_LEGACY_FILE);
        fs::write(&canonical, b"monolithic").unwrap();
        fs::hard_link(&canonical, &legacy).unwrap();
        assert_eq!(monolithic_twins(&canonical), [legacy.clone()]);
        assert_eq!(monolithic_twins(&legacy), [canonical.clone()]);
        // A different file under the other spelling is not a twin.
        fs::remove_file(&legacy).unwrap();
        fs::write(&legacy, b"different").unwrap();
        assert!(monolithic_twins(&canonical).is_empty());
        // Any other model name has no twins.
        let other = dir.join("other.mq4");
        fs::write(&other, b"x").unwrap();
        assert!(monolithic_twins(&other).is_empty());
        assert!(plehot_siblings(&canonical).is_empty());
        let rank = dir.join(format!("{FLASH_NEXT_XTS_CANONICAL_FILE}.plehot"));
        let pack = dir.join(format!("{FLASH_NEXT_XTS_CANONICAL_FILE}.plehotrows"));
        fs::write(&rank, b"r").unwrap();
        fs::write(&pack, b"p").unwrap();
        assert_eq!(plehot_siblings(&canonical), [rank, pack]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn clap_parses_ple_subcommands() {
        let parse = |args: &[&str]| {
            let mut argv = vec!["hipfire"];
            argv.extend_from_slice(args);
            Cli::try_parse_from(argv)
        };
        let Commands::Ple(args) = parse(&["ple", "derive", "qwen3.8:flash-next"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(derive.format, DeriveFormat::Q4);
        assert_eq!(derive.model, "qwen3.8:flash-next");
        assert!(derive.rank.is_none());

        let Commands::Ple(args) = parse(&["ple", "derive", "--rank", "/m/x.plehot", "m"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(derive.rank.as_deref(), Some(Path::new("/m/x.plehot")));

        let Commands::Ple(args) = parse(&["ple", "derive", "--format", "q8", "/m/x.mq4"])
            .unwrap()
            .command
            .unwrap()
        else {
            panic!("expected ple");
        };
        let PleAction::Derive(derive) = args.action else {
            panic!("expected derive");
        };
        assert_eq!(
            derive.format,
            DeriveFormat::Q8,
            "rejected at run time with an explanation"
        );

        let Commands::Ple(args) = parse(&[
            "ple",
            "split",
            "--out",
            "/o/x-base.mq4",
            "--keep-original",
            "m",
        ])
        .unwrap()
        .command
        .unwrap() else {
            panic!("expected ple");
        };
        let PleAction::Split(split) = args.action else {
            panic!("expected split");
        };
        assert_eq!(split.out.as_deref(), Some(Path::new("/o/x-base.mq4")));
        assert!(split.keep_original);
        assert_eq!(split.model, "m");

        let Commands::Ple(args) = parse(&["ple", "split", "m"]).unwrap().command.unwrap() else {
            panic!("expected ple");
        };
        let PleAction::Split(split) = args.action else {
            panic!("expected split");
        };
        assert!(split.out.is_none() && !split.keep_original);

        assert!(parse(&["ple"]).is_err());
        assert!(parse(&["ple", "derive"]).is_err());
        assert!(parse(&["ple", "derive", "--format", "q5", "m"]).is_err());
    }

    #[test]
    fn clap_parses_pull_ple_selection() {
        let selection = |args: &[&str]| {
            let mut argv = vec!["hipfire", "pull", "qwen3.8:flash-next"];
            argv.extend_from_slice(args);
            match Cli::try_parse_from(argv).unwrap().command.unwrap() {
                Commands::Pull(pull) => pull.ple,
                other => panic!("expected pull, got {other:?}"),
            }
        };
        assert_eq!(selection(&[]), None);
        assert_eq!(selection(&["--ple", "auto"]), Some(PleSelection::Auto));
        assert_eq!(selection(&["--ple", "q4"]), Some(PleSelection::Q4));
        assert_eq!(selection(&["--ple", "q8"]), Some(PleSelection::Q8));
        assert_eq!(selection(&["--ple", "all"]), Some(PleSelection::All));
        assert_eq!(selection(&["--ple", "none"]), Some(PleSelection::Skip));
        assert!(Cli::try_parse_from(["hipfire", "pull", "m", "--ple", "q5"]).is_err());
    }
}
