// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host calibration record: what `hipfire calibrate` measured on this host,
//! the fingerprint that says whether it still describes the host, and the
//! lookup the engine's measured-rate selectors call at model load.
//!
//! The record lives in the top-level `[calibration]` table of `config.toml`.
//! The user-config loader ignores that table and every `config.toml` rewrite
//! carries it over (see `write_global_toml`), so `hipfire config set` never
//! drops a measurement. Nothing here talks to a network or collects anything:
//! the record is a local file the user can read, edit or delete.
//!
//! A record is trusted only while its fingerprint matches the live host: the
//! probe version, CPU model and logical count, installed RAM (within
//! [`RAM_TOLERANCE_BYTES`]), and per device the architecture, VRAM size and
//! the PCIe link (speed and width) the card actually negotiated. Any mismatch
//! degrades to [`Lookup::Stale`] / [`Lookup::Missing`] and the caller falls
//! back to its own load-time probe.

use crate::devices::{enumerate_gpus, PciBdf, KFD_TOPOLOGY_NODES};
use crate::{host_memory, ConfigError, ConfigPaths, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Bump when probe semantics change: every stored record becomes stale.
pub const PROBE_VERSION: u32 = 1;

/// Top-level table name in `config.toml`.
pub const SECTION: &str = "calibration";

/// Installed-RAM drift tolerated before a record is stale: a BIOS carve-out or
/// kernel reservation change moves `MemTotal` by far less than a DIMM does.
pub const RAM_TOLERANCE_BYTES: u64 = 1 << 30;

/// One `hipfire calibrate` run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub probe_version: u32,
    /// RFC 3339 UTC, `2026-10-18T14:03:22Z`.
    pub measured_at: String,
    pub hipfire_version: String,
    /// Wall time of the whole `hipfire calibrate` run.
    pub duration_s: f64,
    pub host: HostFacts,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_read: Option<CpuReadRate>,
    #[serde(default)]
    pub devices: Vec<DeviceCalibration>,
    /// GPUs the kernel listed that this run did not measure (not visible to
    /// HIP, or they failed to open). They keep `--check` from asking for a
    /// re-run that could not measure them either; the engine still probes them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unmeasured: Vec<UnmeasuredDevice>,
}

/// A GPU present at calibration time but not measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnmeasuredDevice {
    /// Normalized lowercase `0000:c0:00.0`.
    pub pci_bdf: String,
    /// KFD architecture at calibration time.
    pub arch: String,
    pub reason: String,
}

impl Calibration {
    /// Whether `device` is the GPU this record lists as unmeasured (same PCI
    /// address and architecture).
    pub fn lists_unmeasured(&self, device: &DeviceIdentity) -> bool {
        let wanted = normalize_bdf(&device.pci_bdf).unwrap_or_else(|| device.pci_bdf.clone());
        self.unmeasured
            .iter()
            .any(|gpu| gpu.pci_bdf == wanted && gpu.arch == device.arch)
    }
}

/// CPU and memory identity of the host.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostFacts {
    /// Fingerprint.
    pub cpu_model: String,
    /// Fingerprint.
    pub logical_cpus: u32,
    /// Informational.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_cores: Option<u32>,
    /// Informational, e.g. `["avx2", "fma", "f16c", "avx512f"]`.
    #[serde(default)]
    pub isa: Vec<String>,
    /// Fingerprint, compared within [`RAM_TOLERANCE_BYTES`].
    pub ram_total_bytes: u64,
}

/// Aggregate host DRAM read rate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpuReadRate {
    pub gbps: f64,
    pub threads: u32,
    pub bytes: u64,
    pub reps: u32,
}

/// Negotiated PCIe path from the root port to a GPU.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcieLink {
    /// Fingerprint: minimum `max_link_speed` (GT/s) over every PCI device on
    /// the sysfs path root port → GPU.
    pub gts: f64,
    /// Fingerprint: minimum `current_link_width` over that path.
    pub width: u32,
    /// Informational: minimum `current_link_speed`; power management can
    /// lower it below `gts`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_gts: Option<f64>,
    /// Informational: the path device that identifies the narrowest link.
    pub bottleneck_bdf: String,
}

/// One two-engine probe: CPU GEMV against GPU GEMV over host-mapped bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineRate {
    /// Lowercase DType name as offload-bench spells it, e.g. `mq4g256v2`.
    pub format: String,
    pub k: u64,
    pub m: u64,
    /// `global` (rayon global pool) or `splice` (pinned physical-core pool).
    pub pool: String,
    pub reps: u32,
    pub cpu_gbps: f64,
    pub gpu_gbps: f64,
}

/// Measurements of one GPU, keyed by its PCI address.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceCalibration {
    /// Normalized lowercase `0000:67:00.0`.
    pub pci_bdf: String,
    /// Fingerprint.
    pub arch: String,
    /// Fingerprint: sysfs `mem_info_vram_total` when readable, else the HIP
    /// total.
    pub vram_bytes: u64,
    pub unified: bool,
    /// Fingerprint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<PcieLink>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_h2d_gbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zero_copy_gbps: Option<f64>,
    #[serde(default)]
    pub engine: Vec<EngineRate>,
}

impl DeviceCalibration {
    /// The engine probe recorded for `format`, depth `k` and CPU `pool`.
    pub fn engine_rate(&self, format: &str, k: u64, pool: &str) -> Option<&EngineRate> {
        self.engine
            .iter()
            .find(|rate| rate.format == format && rate.k == k && rate.pool == pool)
    }
}

/// Live identity of one device, built identically by the CLI and the engine so
/// the fingerprints compare like with like.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceIdentity {
    pub pci_bdf: String,
    pub arch: String,
    pub vram_bytes: u64,
    pub link: Option<PcieLink>,
}

impl DeviceIdentity {
    /// `pci_bdf` in any spelling HIP returns (normalized via [`normalize_bdf`]);
    /// VRAM is sysfs `mem_info_vram_total` when readable, else `hip_vram_bytes`,
    /// else 0; the link is [`pcie_link_at`] on the live system.
    pub fn live(pci_bdf: &str, arch: &str, hip_vram_bytes: Option<u64>) -> Self {
        Self::live_at(Path::new("/"), pci_bdf, arch, hip_vram_bytes)
    }

    /// [`DeviceIdentity::live`] against a sysfs tree rooted at `root`.
    pub fn live_at(root: &Path, pci_bdf: &str, arch: &str, hip_vram_bytes: Option<u64>) -> Self {
        let bdf = normalize_bdf(pci_bdf).unwrap_or_else(|| pci_bdf.trim().to_ascii_lowercase());
        Self {
            vram_bytes: sysfs_vram_bytes(root, &bdf).or(hip_vram_bytes).unwrap_or(0),
            link: pcie_link_at(root, &bdf),
            arch: arch.to_owned(),
            pci_bdf: bdf,
        }
    }
}

/// `0000:67:00.0` from any spelling [`PciBdf::parse`] accepts (HIP's
/// `0000:67:00.0`, a domain-less `67:00.0`, upper case).
pub fn normalize_bdf(raw: &str) -> Option<String> {
    PciBdf::parse(raw).map(|bdf| bdf.to_string())
}

fn sysfs_vram_bytes(root: &Path, bdf: &str) -> Option<u64> {
    fs::read_to_string(
        root.join("sys/bus/pci/devices")
            .join(bdf)
            .join("mem_info_vram_total"),
    )
    .ok()?
    .trim()
    .parse()
    .ok()
}

/// Host facts of the running system.
pub fn detect_host() -> HostFacts {
    detect_host_at(Path::new("/"))
}

/// Host facts read from `root/proc/cpuinfo` and `root/proc/meminfo`; the ISA
/// list is always the running CPU's (runtime feature detection), x86_64 only.
pub fn detect_host_at(root: &Path) -> HostFacts {
    let cpuinfo = fs::read_to_string(root.join("proc/cpuinfo")).unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find_map(|line| cpuinfo_field(line, "model name").filter(|model| !model.is_empty()))
        .map(str::to_owned)
        .or_else(cpuid_brand)
        .unwrap_or_else(|| "unknown".to_owned());
    let processors = cpuinfo
        .lines()
        .filter(|line| cpuinfo_field(line, "processor").is_some())
        .count();
    let logical_cpus = if processors > 0 {
        processors as u32
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1)
    };
    let physical = cpuinfo
        .split("\n\n")
        .filter_map(|block| {
            let field = |key: &str| block.lines().find_map(|line| cpuinfo_field(line, key));
            Some((field("physical id")?, field("core id")?))
        })
        .collect::<BTreeSet<_>>();
    let ram_total_bytes = fs::read_to_string(root.join("proc/meminfo"))
        .ok()
        .and_then(|meminfo| host_memory::meminfo_total_bytes(&meminfo))
        .or_else(|| host_memory().map(|memory| memory.total_physical_bytes))
        .unwrap_or(0);
    HostFacts {
        cpu_model,
        logical_cpus,
        physical_cores: (!physical.is_empty()).then_some(physical.len() as u32),
        isa: detect_isa(),
        ram_total_bytes,
    }
}

/// Value of `key : value` when `line` is that `/proc/cpuinfo` field.
fn cpuinfo_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (name, value) = line.split_once(':')?;
    (name.trim() == key).then(|| value.trim())
}

/// Processor brand string from CPUID leaves `0x8000_0002..=0x8000_0004`, for
/// hosts without a `model name` cpuinfo field.
#[cfg(target_arch = "x86_64")]
fn cpuid_brand() -> Option<String> {
    use std::arch::x86_64::__cpuid;
    // `__cpuid` is a safe fn on current toolchains and `unsafe` on older ones;
    // CPUID itself exists on every x86_64 CPU.
    #[allow(unused_unsafe)]
    let cpuid = |leaf: u32| unsafe { __cpuid(leaf) };
    if cpuid(0x8000_0000).eax < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002..=0x8000_0004 {
        let regs = cpuid(leaf);
        for word in [regs.eax, regs.ebx, regs.ecx, regs.edx] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    let brand = String::from_utf8_lossy(&bytes);
    let brand = brand.trim_matches(|c: char| c == '\0' || c.is_whitespace());
    (!brand.is_empty()).then(|| brand.to_owned())
}

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_brand() -> Option<String> {
    None
}

#[cfg(target_arch = "x86_64")]
fn detect_isa() -> Vec<String> {
    let mut isa = Vec::new();
    macro_rules! probe {
        ($($feature:tt),+) => {
            $(if std::arch::is_x86_feature_detected!($feature) {
                isa.push($feature.to_owned());
            })+
        };
    }
    probe!(
        "sse4.2",
        "avx",
        "avx2",
        "fma",
        "f16c",
        "avx512f",
        "avx512bw",
        "avx512vl",
        "avx512vnni",
        "avx512bf16",
        "avxvnni"
    );
    isa
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_isa() -> Vec<String> {
    Vec::new()
}

/// One PCI device on the path from the root port to a GPU.
struct PathLink {
    bdf: String,
    width: u32,
    max_gts: Option<f64>,
    current_gts: Option<f64>,
}

/// GT/s from a sysfs speed string (`16.0 GT/s PCIe`); `Unknown` is `None`.
fn parse_gts(text: &str) -> Option<f64> {
    text.split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
        .filter(|gts| gts.is_finite() && *gts > 0.0)
}

/// The PCIe path from the root port down to `pci_bdf`, read from the sysfs tree
/// at `root`: `root/sys/bus/pci/devices/<bdf>` is a symlink into
/// `root/sys/devices/pci…/<root port>/…/<bdf>`, and every BDF-named directory
/// on that path that reports `current_link_width` is one link of the chain.
///
/// - `gts`: the minimum `max_link_speed` over the chain,
/// - `width`: the minimum `current_link_width`,
/// - `current_gts`: the minimum `current_link_speed`,
/// - `bottleneck_bdf`: the device that identifies the narrowest link. A link
///   is reported by both of its ends (a root port mirrors the width of the
///   device below it), so the downstream end is named; a device that is the
///   top of the chain is named only when nothing below it is as narrow.
///
/// `None` off Linux, when the device or its sysfs tree is unreadable, or when
/// no device on the path reports a link.
pub fn pcie_link_at(root: &Path, pci_bdf: &str) -> Option<PcieLink> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let bdf = normalize_bdf(pci_bdf)?;
    let devices_root = fs::canonicalize(root.join("sys/devices")).ok()?;
    let device = fs::canonicalize(root.join("sys/bus/pci/devices").join(&bdf)).ok()?;
    let mut chain = Vec::new();
    for dir in device.ancestors() {
        if !dir.starts_with(&devices_root) || dir == devices_root {
            break;
        }
        let Some(name) = dir
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(normalize_bdf)
        else {
            continue;
        };
        let read = |file: &str| fs::read_to_string(dir.join(file)).ok();
        let Some(width) = read("current_link_width").and_then(|text| text.trim().parse().ok())
        else {
            continue;
        };
        chain.push(PathLink {
            bdf: name,
            width,
            max_gts: read("max_link_speed").and_then(|text| parse_gts(&text)),
            current_gts: read("current_link_speed").and_then(|text| parse_gts(&text)),
        });
    }
    // `ancestors` walks leaf → root; the chain reads root first.
    chain.reverse();
    let width = chain.iter().map(|link| link.width).min()?;
    let min_speed =
        |speed: fn(&PathLink) -> Option<f64>| chain.iter().filter_map(speed).min_by(f64::total_cmp);
    let current_gts = min_speed(|link| link.current_gts);
    let gts = min_speed(|link| link.max_gts).or(current_gts)?;
    let bottleneck = chain
        .iter()
        .enumerate()
        .filter(|(_, link)| link.width == width)
        .map(|(position, link)| (position, &link.bdf))
        .reduce(|first, next| if first.0 == 0 { next } else { first })?
        .1
        .clone();
    Some(PcieLink {
        gts,
        width,
        current_gts,
        bottleneck_bdf: bottleneck,
    })
}

/// A GPU found through the KFD topology, with its sysfs-derived identity.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectedGpu {
    pub pci_bdf: String,
    pub arch: String,
    pub vram_bytes: Option<u64>,
    pub link: Option<PcieLink>,
}

/// GPUs of the running system, in PCI order; empty when the KFD topology is
/// unavailable (Windows, no `amdgpu`).
pub fn detect_gpus() -> Vec<DetectedGpu> {
    detect_gpus_at(Path::new("/"))
}

/// [`detect_gpus`] against a sysfs tree rooted at `root`.
pub fn detect_gpus_at(root: &Path) -> Vec<DetectedGpu> {
    let nodes = root.join(KFD_TOPOLOGY_NODES.trim_start_matches('/'));
    enumerate_gpus(&nodes)
        .unwrap_or_default()
        .into_iter()
        .map(|gpu| {
            let pci_bdf = gpu.bdf.to_string();
            DetectedGpu {
                vram_bytes: sysfs_vram_bytes(root, &pci_bdf),
                link: pcie_link_at(root, &pci_bdf),
                arch: gpu.arch,
                pci_bdf,
            }
        })
        .collect()
}

/// Why a stored record (or one device entry) no longer describes this host.
/// `Display` is one human line.
#[derive(Clone, Debug, PartialEq)]
pub enum Stale {
    ProbeVersion {
        stored: u32,
        current: u32,
    },
    /// Model or logical count changed; both are shown.
    Cpu {
        stored: String,
        live: String,
    },
    /// `|stored - live|` exceeds [`RAM_TOLERANCE_BYTES`].
    Ram {
        stored: u64,
        live: u64,
    },
    Arch {
        stored: String,
        live: String,
    },
    /// Exact compare.
    Vram {
        stored: u64,
        live: u64,
    },
    /// Speed (`|d| < 0.01` GT/s) or width changed.
    Link {
        stored: Option<PcieLink>,
        live: Option<PcieLink>,
    },
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn link_text(link: &Option<PcieLink>) -> String {
    match link {
        Some(link) => format!("x{} @ {:.1} GT/s", link.width, link.gts),
        None => "unknown".to_owned(),
    }
}

impl fmt::Display for Stale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProbeVersion { stored, current } => write!(
                f,
                "calibration was measured by probe version {stored}, this hipfire is version {current}"
            ),
            Self::Cpu { stored, live } => {
                write!(f, "CPU changed: calibrated on {stored}, now {live}")
            }
            Self::Ram { stored, live } => write!(
                f,
                "RAM changed: calibrated with {:.1} GiB, now {:.1} GiB",
                gib(*stored),
                gib(*live)
            ),
            Self::Arch { stored, live } => {
                write!(f, "GPU architecture changed: calibrated on {stored}, now {live}")
            }
            Self::Vram { stored, live } => write!(
                f,
                "VRAM size changed: calibrated with {} bytes, now {live} bytes",
                stored
            ),
            Self::Link { stored, live } => write!(
                f,
                "PCIe link changed: calibrated at {}, now {}",
                link_text(stored),
                link_text(live)
            ),
        }
    }
}

fn cpu_text(host: &HostFacts) -> String {
    format!("{} ({} logical CPUs)", host.cpu_model, host.logical_cpus)
}

/// Host-level staleness: probe version first, then CPU, then RAM.
pub fn host_staleness(stored: &Calibration, live: &HostFacts) -> Option<Stale> {
    if stored.probe_version != PROBE_VERSION {
        return Some(Stale::ProbeVersion {
            stored: stored.probe_version,
            current: PROBE_VERSION,
        });
    }
    if stored.host.cpu_model != live.cpu_model || stored.host.logical_cpus != live.logical_cpus {
        return Some(Stale::Cpu {
            stored: cpu_text(&stored.host),
            live: cpu_text(live),
        });
    }
    if stored.host.ram_total_bytes.abs_diff(live.ram_total_bytes) > RAM_TOLERANCE_BYTES {
        return Some(Stale::Ram {
            stored: stored.host.ram_total_bytes,
            live: live.ram_total_bytes,
        });
    }
    None
}

fn links_match(stored: &Option<PcieLink>, live: &Option<PcieLink>) -> bool {
    match (stored, live) {
        (None, None) => true,
        (Some(stored), Some(live)) => {
            (stored.gts - live.gts).abs() < 0.01 && stored.width == live.width
        }
        _ => false,
    }
}

/// Device-level staleness: architecture, VRAM, then link.
pub fn device_staleness(stored: &DeviceCalibration, live: &DeviceIdentity) -> Option<Stale> {
    if stored.arch != live.arch {
        return Some(Stale::Arch {
            stored: stored.arch.clone(),
            live: live.arch.clone(),
        });
    }
    if stored.vram_bytes != live.vram_bytes {
        return Some(Stale::Vram {
            stored: stored.vram_bytes,
            live: live.vram_bytes,
        });
    }
    if !links_match(&stored.link, &live.link) {
        return Some(Stale::Link {
            stored: stored.link.clone(),
            live: live.link.clone(),
        });
    }
    None
}

/// Result of asking whether a stored measurement applies to a device.
#[derive(Clone, Debug, PartialEq)]
pub enum Lookup {
    Fresh {
        device: DeviceCalibration,
        measured_at: String,
    },
    /// `no calibration in <path>`, `device 0000:.. was not calibrated` or
    /// `unreadable calibration: <err>`.
    Missing(String),
    /// [`Stale`]'s `Display`.
    Stale(String),
}

/// Pure lookup of `device` in `record` against the live host.
pub fn lookup_in(
    record: Option<&Calibration>,
    live_host: &HostFacts,
    device: &DeviceIdentity,
) -> Lookup {
    let Some(record) = record else {
        return Lookup::Missing("no calibration recorded".to_owned());
    };
    if let Some(stale) = host_staleness(record, live_host) {
        return Lookup::Stale(stale.to_string());
    }
    let wanted = normalize_bdf(&device.pci_bdf).unwrap_or_else(|| device.pci_bdf.clone());
    let Some(stored) = record.devices.iter().find(|stored| {
        normalize_bdf(&stored.pci_bdf).unwrap_or_else(|| stored.pci_bdf.clone()) == wanted
    }) else {
        return Lookup::Missing(format!("device {wanted} was not calibrated"));
    };
    if let Some(stale) = device_staleness(stored, device) {
        return Lookup::Stale(stale.to_string());
    }
    Lookup::Fresh {
        device: stored.clone(),
        measured_at: record.measured_at.clone(),
    }
}

/// Engine entry point: the record is loaded once per process from
/// [`ConfigPaths::try_discover`] (`config.toml`'s `[calibration]`), the live
/// host facts are detected once per process. Never panics; any error is
/// [`Lookup::Missing`] with the reason.
pub fn lookup(device: &DeviceIdentity) -> Lookup {
    type Loaded = (PathBuf, std::result::Result<Option<Calibration>, String>);
    static RECORD: LazyLock<Loaded> = LazyLock::new(|| match ConfigPaths::try_discover() {
        Ok(paths) => (
            paths.config_toml.clone(),
            load(&paths).map_err(|error| error.to_string()),
        ),
        Err(error) => (PathBuf::new(), Err(error.to_string())),
    });
    static HOST: LazyLock<HostFacts> = LazyLock::new(detect_host);
    let (path, record) = &*RECORD;
    match record {
        Err(error) => Lookup::Missing(format!("unreadable calibration: {error}")),
        Ok(None) => Lookup::Missing(format!("no calibration in {}", path.display())),
        Ok(Some(record)) => lookup_in(Some(record), &HOST, device),
    }
}

/// The record in `config.toml`; `None` when the file or the table is absent.
pub fn load(paths: &ConfigPaths) -> Result<Option<Calibration>> {
    let Some(table) = read_table(paths)? else {
        return Ok(None);
    };
    table
        .try_into()
        .map(Some)
        .map_err(|source: toml::de::Error| ConfigError::Parse {
            path: paths.config_toml.clone(),
            message: format!("[{SECTION}]: {source}"),
        })
}

/// Replace the record in `config.toml`. The rest of the file is the user layer
/// as `hipfire config set` would rewrite it: legacy `config.json` migrates the
/// same way.
pub fn store(paths: &ConfigPaths, record: &Calibration) -> Result<()> {
    let layer = crate::load_global(paths)?.layer;
    let table = toml::Value::try_from(record).map_err(|source| ConfigError::Parse {
        path: paths.config_toml.clone(),
        message: format!("[{SECTION}]: {source}"),
    })?;
    crate::write_global_with_calibration(paths, &layer, Some(table))
}

/// Remove the record from `config.toml`, keeping the user layer; `true` when a
/// table was removed.
pub fn clear(paths: &ConfigPaths) -> Result<bool> {
    if read_table(paths)?.is_none() {
        return Ok(false);
    }
    let layer = crate::load_global(paths)?.layer;
    crate::write_global_with_calibration(paths, &layer, None)?;
    Ok(true)
}

/// The raw `[calibration]` table of `config.toml`.
fn read_table(paths: &ConfigPaths) -> Result<Option<toml::Value>> {
    let raw = match fs::read_to_string(&paths.config_toml) {
        Ok(raw) => raw,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(ConfigError::Read {
                path: paths.config_toml.clone(),
                source,
            })
        }
    };
    let mut root = toml::from_str::<toml::Table>(&raw).map_err(|source| ConfigError::Parse {
        path: paths.config_toml.clone(),
        message: source.to_string(),
    })?;
    Ok(root.remove(SECTION))
}

/// `2026-10-18T14:03:22Z` for `unix_secs` since the epoch.
pub fn utc_rfc3339(unix_secs: u64) -> String {
    let (year, month, day) = civil_from_days((unix_secs / 86_400) as i64);
    let secs = unix_secs % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

/// Days since the Unix epoch to a proleptic-Gregorian `(year, month, day)`:
/// Howard Hinnant's `civil_from_days`, so no `chrono` dependency.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = yoe + era * 400;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_global, write_global_toml, ConfigValue};

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hipfire-calibration-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn link(gts: f64, width: u32) -> PcieLink {
        PcieLink {
            gts,
            width,
            current_gts: Some(gts),
            bottleneck_bdf: "0000:65:00.0".to_owned(),
        }
    }

    fn host() -> HostFacts {
        HostFacts {
            cpu_model: "AMD Test CPU".to_owned(),
            logical_cpus: 32,
            physical_cores: Some(16),
            isa: vec!["avx2".to_owned(), "fma".to_owned()],
            ram_total_bytes: 128 << 30,
        }
    }

    fn device() -> DeviceCalibration {
        DeviceCalibration {
            pci_bdf: "0000:67:00.0".to_owned(),
            arch: "gfx1100".to_owned(),
            vram_bytes: 25_753_026_560,
            unified: false,
            link: Some(link(16.0, 4)),
            pinned_h2d_gbps: Some(7.09),
            zero_copy_gbps: Some(7.1),
            engine: vec![EngineRate {
                format: "mq4g256v2".to_owned(),
                k: 2560,
                m: 123,
                pool: "global".to_owned(),
                reps: 5,
                cpu_gbps: 40.1,
                gpu_gbps: 6.1,
            }],
        }
    }

    fn record() -> Calibration {
        Calibration {
            probe_version: PROBE_VERSION,
            measured_at: "2026-10-18T14:03:22Z".to_owned(),
            hipfire_version: "0.4.3".to_owned(),
            duration_s: 21.4,
            host: host(),
            cpu_read: Some(CpuReadRate {
                gbps: 113.5,
                threads: 16,
                bytes: 4 << 30,
                reps: 3,
            }),
            devices: vec![device()],
            unmeasured: Vec::new(),
        }
    }

    fn identity() -> DeviceIdentity {
        let device = device();
        DeviceIdentity {
            pci_bdf: device.pci_bdf,
            arch: device.arch,
            vram_bytes: device.vram_bytes,
            link: device.link,
        }
    }

    /// One PCI directory of the fixture: the link files a kernel exposes.
    #[cfg(unix)]
    fn pci_dir(dir: &Path, width: u32, max_speed: &str, current_speed: &str) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("current_link_width"), format!("{width}\n")).unwrap();
        fs::write(dir.join("max_link_speed"), format!("{max_speed}\n")).unwrap();
        fs::write(dir.join("current_link_speed"), format!("{current_speed}\n")).unwrap();
    }

    /// hipx: GPU 0000:67:00.0 behind a switch (66 → 65) on root port 00:03.1.
    /// The switch upstream port 65:00.0 trained to x4 though it supports x16;
    /// the root port is x4 by design.
    #[cfg(unix)]
    fn hipx_sysfs(root: &Path) {
        let bus = root.join("sys/devices/pci0000:00");
        let port = bus.join("0000:00:03.1");
        let up = port.join("0000:65:00.0");
        let down = up.join("0000:66:00.0");
        let gpu = down.join("0000:67:00.0");
        fs::create_dir_all(&bus).unwrap();
        pci_dir(&port, 4, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        pci_dir(&up, 4, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        pci_dir(&down, 16, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        pci_dir(&gpu, 16, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        fs::write(gpu.join("mem_info_vram_total"), "25753026560\n").unwrap();
        let devices = root.join("sys/bus/pci/devices");
        fs::create_dir_all(&devices).unwrap();
        std::os::unix::fs::symlink(
            "../../../devices/pci0000:00/0000:00:03.1/0000:65:00.0/0000:66:00.0/0000:67:00.0",
            devices.join("0000:67:00.0"),
        )
        .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn pcie_link_walks_the_sysfs_path_to_the_narrowest_link() {
        let root = temp_dir("pcie");
        hipx_sysfs(&root);
        let link = pcie_link_at(&root, "0000:67:00.0").unwrap();
        assert_eq!(link.width, 4);
        assert_eq!(link.gts, 16.0);
        assert_eq!(link.current_gts, Some(16.0));
        assert_eq!(link.bottleneck_bdf, "0000:65:00.0");
        // Callers may drop the domain; an unknown device has no link.
        assert_eq!(
            pcie_link_at(&root, "0000:67:00.0"),
            pcie_link_at(&root, "67:00.0")
        );
        assert!(pcie_link_at(&root, "0000:99:00.0").is_none());
        let identity = DeviceIdentity::live_at(&root, "0000:67:00.0", "gfx1100", Some(1));
        assert_eq!(identity.vram_bytes, 25_753_026_560, "sysfs beats HIP");
        assert_eq!(identity.link, Some(link));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn pcie_link_takes_the_lowest_speed_and_ignores_unknown() {
        let root = temp_dir("pcie-speed");
        hipx_sysfs(&root);
        let gpu =
            root.join("sys/devices/pci0000:00/0000:00:03.1/0000:65:00.0/0000:66:00.0/0000:67:00.0");
        fs::write(gpu.join("current_link_speed"), "2.5 GT/s PCIe\n").unwrap();
        let down = gpu.parent().unwrap();
        fs::write(down.join("max_link_speed"), "8.0 GT/s PCIe\n").unwrap();
        fs::write(down.join("current_link_speed"), "Unknown\n").unwrap();
        let link = pcie_link_at(&root, "0000:67:00.0").unwrap();
        assert_eq!(link.gts, 8.0);
        assert_eq!(link.current_gts, Some(2.5));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn gpu_directly_on_a_root_port_names_itself_when_equally_narrow() {
        let root = temp_dir("pcie-direct");
        let port = root.join("sys/devices/pci0000:00/0000:00:01.1");
        let gpu = port.join("0000:03:00.0");
        pci_dir(&port, 16, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        pci_dir(&gpu, 16, "16.0 GT/s PCIe", "16.0 GT/s PCIe");
        let devices = root.join("sys/bus/pci/devices");
        fs::create_dir_all(&devices).unwrap();
        std::os::unix::fs::symlink(
            "../../../devices/pci0000:00/0000:00:01.1/0000:03:00.0",
            devices.join("0000:03:00.0"),
        )
        .unwrap();
        let link = pcie_link_at(&root, "0000:03:00.0").unwrap();
        assert_eq!(
            (link.width, link.bottleneck_bdf.as_str()),
            (16, "0000:03:00.0")
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn detect_host_parses_cpuinfo_and_meminfo() {
        let root = temp_dir("host");
        fs::create_dir_all(root.join("proc")).unwrap();
        let mut cpuinfo = String::new();
        for processor in 0..4 {
            // Two sockets x one core each, two hardware threads per core.
            cpuinfo.push_str(&format!(
                "processor\t: {processor}\nvendor_id\t: AuthenticAMD\n\
                 model name\t: AMD Fixture CPU 9000\nphysical id\t: {}\ncore id\t\t: 0\n\n",
                processor % 2
            ));
        }
        fs::write(root.join("proc/cpuinfo"), cpuinfo).unwrap();
        fs::write(
            root.join("proc/meminfo"),
            "MemTotal:       131072000 kB\nMemFree:         1000 kB\n",
        )
        .unwrap();
        let facts = detect_host_at(&root);
        assert_eq!(facts.cpu_model, "AMD Fixture CPU 9000");
        assert_eq!(facts.logical_cpus, 4);
        assert_eq!(facts.physical_cores, Some(2));
        assert_eq!(facts.ram_total_bytes, 131_072_000 * 1024);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn detect_host_falls_back_without_proc() {
        let root = temp_dir("host-bare");
        let facts = detect_host_at(&root);
        assert!(!facts.cpu_model.is_empty());
        assert!(facts.logical_cpus >= 1);
        assert_eq!(facts.physical_cores, None);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn staleness_matrix() {
        let stored = record();
        let live_host = host();
        let live = identity();
        assert_eq!(host_staleness(&stored, &live_host), None);
        assert_eq!(device_staleness(&stored.devices[0], &live), None);
        assert!(matches!(
            lookup_in(Some(&stored), &live_host, &live),
            Lookup::Fresh { ref measured_at, .. } if measured_at == "2026-10-18T14:03:22Z"
        ));

        let mut other = identity();
        other.arch = "gfx1201".to_owned();
        assert!(matches!(
            device_staleness(&stored.devices[0], &other),
            Some(Stale::Arch { .. })
        ));
        let mut other = identity();
        other.vram_bytes += 1;
        assert!(matches!(
            device_staleness(&stored.devices[0], &other),
            Some(Stale::Vram { .. })
        ));
        let mut other = identity();
        other.link = Some(link(16.0, 16));
        assert!(matches!(
            device_staleness(&stored.devices[0], &other),
            Some(Stale::Link { .. })
        ));
        other.link = None;
        assert!(matches!(
            device_staleness(&stored.devices[0], &other),
            Some(Stale::Link { .. })
        ));
        let mut other = identity();
        other.link = Some(link(16.004, 4));
        assert_eq!(device_staleness(&stored.devices[0], &other), None);

        let mut other = host();
        other.ram_total_bytes += 16 << 30;
        assert!(matches!(
            host_staleness(&stored, &other),
            Some(Stale::Ram { .. })
        ));
        let mut other = host();
        other.ram_total_bytes += 100 << 20;
        assert_eq!(host_staleness(&stored, &other), None);
        let mut other = host();
        other.cpu_model = "Intel Other".to_owned();
        assert!(matches!(
            host_staleness(&stored, &other),
            Some(Stale::Cpu { .. })
        ));
        let mut other = host();
        other.logical_cpus = 16;
        assert!(matches!(
            host_staleness(&stored, &other),
            Some(Stale::Cpu { .. })
        ));
        let mut old = record();
        old.probe_version = PROBE_VERSION - 1;
        assert_eq!(
            host_staleness(&old, &live_host),
            Some(Stale::ProbeVersion {
                stored: PROBE_VERSION - 1,
                current: PROBE_VERSION
            })
        );

        assert!(matches!(
            lookup_in(None, &live_host, &live),
            Lookup::Missing(_)
        ));
        let mut elsewhere = identity();
        elsewhere.pci_bdf = "0000:03:00.0".to_owned();
        assert!(matches!(
            lookup_in(Some(&stored), &live_host, &elsewhere),
            Lookup::Missing(reason) if reason.contains("0000:03:00.0")
        ));
        let mut shifted = identity();
        shifted.vram_bytes += 1;
        assert!(matches!(
            lookup_in(Some(&stored), &live_host, &shifted),
            Lookup::Stale(reason) if reason.contains("VRAM")
        ));
        assert!(matches!(
            lookup_in(Some(&old), &live_host, &live),
            Lookup::Stale(reason) if reason.contains("probe version")
        ));
    }

    #[test]
    fn engine_rate_selects_format_depth_and_pool() {
        let device = device();
        assert_eq!(
            device.engine_rate("mq4g256v2", 2560, "global").map(|r| r.m),
            Some(123)
        );
        assert!(device.engine_rate("mq4g256v2", 2560, "splice").is_none());
        assert!(device.engine_rate("mq4g256v2", 4096, "global").is_none());
        assert!(device.engine_rate("q8", 2560, "global").is_none());
    }

    #[test]
    fn store_load_round_trip() {
        let paths = ConfigPaths::under(temp_dir("roundtrip"));
        assert_eq!(load(&paths).unwrap(), None);
        let stored = record();
        store(&paths, &stored).unwrap();
        assert_eq!(load(&paths).unwrap(), Some(stored.clone()));
        // Optional fields stay absent rather than serializing as placeholders.
        let mut sparse = stored;
        sparse.cpu_read = None;
        sparse.host.physical_cores = None;
        sparse.devices[0].link = None;
        sparse.devices[0].engine.clear();
        store(&paths, &sparse).unwrap();
        assert_eq!(load(&paths).unwrap(), Some(sparse));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn unmeasured_devices_round_trip_and_match_by_address_and_arch() {
        let paths = ConfigPaths::under(temp_dir("unmeasured"));
        let mut stored = record();
        stored.unmeasured.push(UnmeasuredDevice {
            pci_bdf: "0000:c0:00.0".to_owned(),
            arch: "gfx1151".to_owned(),
            reason: "not opened by this run".to_owned(),
        });
        store(&paths, &stored).unwrap();
        let loaded = load(&paths).unwrap().unwrap();
        assert_eq!(loaded, stored);
        let mut igpu = identity();
        igpu.pci_bdf = "C0:00.0".to_owned();
        igpu.arch = "gfx1151".to_owned();
        assert!(loaded.lists_unmeasured(&igpu));
        // Never a measurement: the engine still probes it.
        assert!(matches!(
            lookup_in(Some(&loaded), &host(), &igpu),
            Lookup::Missing(_)
        ));
        igpu.arch = "gfx1150".to_owned();
        assert!(!loaded.lists_unmeasured(&igpu));
        assert!(!loaded.lists_unmeasured(&identity()));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn user_loader_ignores_the_table_and_config_set_keeps_it() {
        let paths = ConfigPaths::under(temp_dir("coexist"));
        let mut layer = load_global(&paths).unwrap().layer;
        layer
            .set("memory.max_seq", ConfigValue::Integer(8192))
            .unwrap();
        write_global_toml(&paths, &layer).unwrap();
        store(&paths, &record()).unwrap();

        let loaded = load_global(&paths).unwrap();
        assert_eq!(
            loaded.layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(8192))
        );
        assert!(loaded.warnings.is_empty());

        let table_before = read_table(&paths).unwrap().unwrap();
        // `hipfire config set`: load the layer, change one key, write it back.
        let mut layer = loaded.layer;
        layer
            .set("memory.max_seq", ConfigValue::Integer(4096))
            .unwrap();
        write_global_toml(&paths, &layer).unwrap();
        assert_eq!(read_table(&paths).unwrap().unwrap(), table_before);
        assert_eq!(load(&paths).unwrap(), Some(record()));
        assert_eq!(
            load_global(&paths).unwrap().layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(4096))
        );

        assert!(clear(&paths).unwrap());
        assert_eq!(load(&paths).unwrap(), None);
        assert!(!fs::read_to_string(&paths.config_toml)
            .unwrap()
            .contains(SECTION));
        assert_eq!(
            load_global(&paths).unwrap().layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(4096))
        );
        assert!(!clear(&paths).unwrap());
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn clear_without_config_is_a_no_op() {
        let paths = ConfigPaths::under(temp_dir("clear-empty"));
        assert!(!clear(&paths).unwrap());
        assert!(!paths.config_toml.exists());
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn store_migrates_a_legacy_json_config() {
        let paths = ConfigPaths::under(temp_dir("legacy"));
        fs::write(&paths.config_json, r#"{"max_seq":4096}"#).unwrap();
        assert!(!paths.config_toml.exists());
        store(&paths, &record()).unwrap();
        assert!(paths.config_json.exists(), "rollback file stays");
        let loaded = load_global(&paths).unwrap();
        assert_eq!(loaded.path, paths.config_toml);
        assert_eq!(
            loaded.layer.get("memory.max_seq"),
            Some(&ConfigValue::Integer(4096))
        );
        assert_eq!(load(&paths).unwrap(), Some(record()));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn unreadable_table_is_an_error_not_a_panic() {
        let paths = ConfigPaths::under(temp_dir("malformed"));
        fs::write(
            &paths.config_toml,
            "schema_version = 1\n[calibration]\nprobe_version = \"one\"\n",
        )
        .unwrap();
        assert!(load(&paths).is_err());
        assert!(load_global(&paths).is_ok(), "user layer ignores the table");
        // A rewrite replaces the broken table.
        store(&paths, &record()).unwrap();
        assert_eq!(load(&paths).unwrap(), Some(record()));
        let _ = fs::remove_dir_all(&paths.root);
    }

    #[test]
    fn utc_rfc3339_known_dates() {
        assert_eq!(utc_rfc3339(0), "1970-01-01T00:00:00Z");
        // `date -u -d @1760796202`.
        assert_eq!(utc_rfc3339(1_760_796_202), "2025-10-18T14:03:22Z");
        assert_eq!(utc_rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc_rfc3339(1_709_164_799), "2024-02-28T23:59:59Z");
        assert_eq!(utc_rfc3339(4_102_444_800), "2100-01-01T00:00:00Z");
    }
}
