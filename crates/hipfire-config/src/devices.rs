// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Physical GPU identity and `hardware.devices` resolution.
//!
//! Three numberings exist on a multi-GPU ROCm host and none of them agree:
//!
//! * the **ROCr agent ordinal** consumed by numeric `ROCR_VISIBLE_DEVICES`
//!   entries — the order of GPU nodes in the KFD topology;
//! * the **HIP ordinal** — position inside whatever ROCr exposed, further
//!   filtered by `HIP_VISIBLE_DEVICES`;
//! * the **rocm-smi index** — PCI bus order.
//!
//! `hardware.devices` (`HIPFIRE_DEVICES`) names cards by one objective
//! identity instead: an index into the PCI-BDF-sorted GPU list, an
//! architecture (`gfx1201`: first free card of that arch), an exact
//! `GPU-<uuid>`, or an exact PCI address. Resolution reads only the KFD
//! topology in sysfs — it never touches a GPU — and lowers the chosen cards to
//! `ROCR_VISIBLE_DEVICES` selectors (UUID when the card has one, else its ROCr
//! agent ordinal) plus HIP logical `0..N-1`. The list order is the logical
//! device order for TP/EP. After HIP initializes, callers compare every
//! logical device's `gcnArchName` and PCI bus ID with the resolved card
//! ([`ActiveDevices::verify_visible`]) and abort on any disagreement.

use crate::{ConfigError, ProcessConfig, Result};
use std::{
    fmt, fs,
    path::Path,
    sync::OnceLock,
};

pub const HIP_VISIBLE_DEVICES: &str = "HIP_VISIBLE_DEVICES";
pub const ROCR_VISIBLE_DEVICES: &str = "ROCR_VISIBLE_DEVICES";
/// ROCr's global ISA override; HIP then reports this arch for every agent.
pub const HSA_OVERRIDE_GFX_VERSION: &str = "HSA_OVERRIDE_GFX_VERSION";
/// KFD topology root. Its GPU nodes, in node order, are ROCr's agent order.
pub const KFD_TOPOLOGY_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

const KEY: &str = "hardware.devices";

/// PCI address `domain:bus:device.function`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PciBdf {
    pub domain: u32,
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciBdf {
    /// KFD `location_id` is the PCI routing id `bus << 8 | device << 3 | function`.
    pub fn from_kfd(domain: u32, location_id: u32) -> Self {
        Self {
            domain,
            bus: (location_id >> 8) as u8,
            device: ((location_id >> 3) & 0x1f) as u8,
            function: (location_id & 0x7) as u8,
        }
    }

    /// Parse `DDDD:BB:DD.F` (including HIP's `hipDeviceGetPCIBusId` spelling)
    /// or `BB:DD.F`; hex, any case. The flag reports whether a domain was
    /// given.
    fn parse_parts(value: &str) -> Option<(Self, bool)> {
        fn hex(field: &str, max_digits: usize) -> Option<u32> {
            if field.is_empty()
                || field.len() > max_digits
                || !field.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                return None;
            }
            u32::from_str_radix(field, 16).ok()
        }
        let (head, function) = value.rsplit_once('.')?;
        let fields = head.split(':').collect::<Vec<_>>();
        let (domain, bus, device, has_domain) = match fields.as_slice() {
            [domain, bus, device] => (hex(domain, 8)?, bus, device, true),
            [bus, device] => (0, bus, device, false),
            _ => return None,
        };
        let bus = hex(bus, 2)?;
        let device = hex(device, 2)?;
        let function = hex(function, 1)?;
        (device <= 0x1f && function <= 0x7).then_some((
            Self {
                domain,
                bus: bus as u8,
                device: device as u8,
                function: function as u8,
            },
            has_domain,
        ))
    }

    /// Parse a fully qualified or domain-less address; a missing domain is 0.
    pub fn parse(value: &str) -> Option<Self> {
        Self::parse_parts(value.trim()).map(|(bdf, _)| bdf)
    }
}

impl fmt::Display for PciBdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.domain, self.bus, self.device, self.function
        )
    }
}

/// One GPU node of the KFD topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDevice {
    /// Objective index: position in PCI BDF order (rocm-smi order).
    pub index: usize,
    /// ROCr agent ordinal: position among KFD GPU nodes in node order.
    pub rocr_index: usize,
    /// KFD topology node id.
    pub node: u32,
    /// `gfxNNNN` from the node's `gfx_target_version`.
    pub arch: String,
    /// KFD `unique_id`; `None` when the card reports none (0).
    pub unique_id: Option<u64>,
    pub bdf: PciBdf,
}

impl GpuDevice {
    /// ROCr's UUID spelling, `GPU-` plus 16 lowercase hex digits.
    pub fn uuid(&self) -> Option<String> {
        self.unique_id.map(|id| format!("GPU-{id:016x}"))
    }

    /// Selector for `ROCR_VISIBLE_DEVICES`: the UUID when the card has one,
    /// else its ROCr agent ordinal.
    pub fn rocr_selector(&self) -> String {
        self.uuid().unwrap_or_else(|| self.rocr_index.to_string())
    }

    /// Machine-wide lock identity: the UUID when real, else `pci-<bdf>`.
    /// No-UUID cards must not share one name.
    pub fn lock_identity(&self) -> String {
        self.uuid().unwrap_or_else(|| format!("pci-{}", self.bdf))
    }

    pub fn lock_file_name(&self) -> String {
        format!("gpu-{}.lock", self.lock_identity())
    }

    fn describe(&self) -> String {
        format!(
            "GPU index {} ({} {} PCI {})",
            self.index,
            self.arch,
            self.uuid().unwrap_or_else(|| "no-UUID".into()),
            self.bdf
        )
    }
}

/// Decode `gfx_target_version` (`major*10000 + minor*100 + stepping`).
/// Minor and stepping are single hex digits in the ISA name (`90010` → `gfx90a`).
fn gfx_name(version: u64) -> String {
    format!(
        "gfx{}{:x}{:x}",
        version / 10000,
        (version / 100) % 100,
        version % 100
    )
}

fn read_error(path: &Path, source: std::io::Error) -> ConfigError {
    ConfigError::Read {
        path: path.to_owned(),
        source,
    }
}

/// Enumerate GPU nodes of a KFD topology directory, sorted by PCI BDF.
///
/// A node is a GPU when it has SIMDs, the same test ROCr applies when it
/// builds agents; CPU nodes are skipped. Node order assigns ROCr ordinals.
pub fn enumerate_gpus(nodes_dir: &Path) -> Result<Vec<GpuDevice>> {
    let mut nodes = fs::read_dir(nodes_dir)
        .map_err(|source| read_error(nodes_dir, source))?
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<u32>().ok())
        .collect::<Vec<_>>();
    nodes.sort_unstable();
    let mut devices = Vec::new();
    for node in nodes {
        let path = nodes_dir.join(node.to_string()).join("properties");
        let text = fs::read_to_string(&path).map_err(|source| read_error(&path, source))?;
        let property = |name: &str| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(' ')?;
                (key == name).then(|| value.trim().parse::<u64>().ok())?
            })
        };
        if property("simd_count").unwrap_or(0) == 0 {
            continue;
        }
        let malformed = |field: &str| ConfigError::InvalidValue {
            key: KEY.into(),
            message: format!("KFD node {} lacks a valid {field}", path.display()),
        };
        let location_id = property("location_id").ok_or_else(|| malformed("location_id"))?;
        let domain = property("domain").unwrap_or(0);
        let version = property("gfx_target_version").ok_or_else(|| malformed("gfx_target_version"))?;
        devices.push(GpuDevice {
            index: 0,
            rocr_index: devices.len(),
            node,
            arch: gfx_name(version),
            unique_id: property("unique_id").filter(|&id| id != 0),
            bdf: PciBdf::from_kfd(domain as u32, location_id as u32),
        });
    }
    devices.sort_by_key(|device| device.bdf);
    for (index, device) in devices.iter_mut().enumerate() {
        device.index = index;
    }
    Ok(devices)
}

/// Human-readable device table carried by every resolution error.
pub fn device_table(devices: &[GpuDevice]) -> String {
    let mut table = String::from(
        "GPUs (index = PCI BDF order = rocm-smi order; rocr = ROCr agent ordinal):\n  index  rocr  node  arch      uuid                  pci           lock",
    );
    if devices.is_empty() {
        table.push_str("\n  (no GPU nodes in the KFD topology)");
    }
    for device in devices {
        table.push_str(&format!(
            "\n  {:<5}  {:<4}  {:<4}  {:<8}  {:<20}  {}  {}",
            device.index,
            device.rocr_index,
            device.node,
            device.arch,
            device.uuid().unwrap_or_else(|| "(none)".into()),
            device.bdf,
            device.lock_file_name(),
        ));
    }
    table
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DeviceSelector {
    Index(usize),
    Arch(String),
    Uuid(u64),
    Bdf { bdf: PciBdf, has_domain: bool },
}

impl fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(index) => write!(f, "{index}"),
            Self::Arch(arch) => f.write_str(arch),
            Self::Uuid(id) => write!(f, "GPU-{id:016x}"),
            Self::Bdf { bdf, has_domain: true } => write!(f, "{bdf}"),
            Self::Bdf { bdf, has_domain: false } => {
                write!(f, "{:02x}:{:02x}.{:x}", bdf.bus, bdf.device, bdf.function)
            }
        }
    }
}

impl DeviceSelector {
    fn matches(&self, device: &GpuDevice) -> bool {
        match self {
            Self::Index(index) => device.index == *index,
            Self::Arch(arch) => device.arch == *arch,
            Self::Uuid(id) => device.unique_id == Some(*id),
            Self::Bdf { bdf, has_domain } => {
                (!has_domain || device.bdf.domain == bdf.domain)
                    && (device.bdf.bus, device.bdf.device, device.bdf.function)
                        == (bdf.bus, bdf.device, bdf.function)
            }
        }
    }
}

fn parse_selector(token: &str) -> std::result::Result<DeviceSelector, String> {
    if token.is_empty() {
        return Err("empty entry in the device list".into());
    }
    if token.bytes().all(|byte| byte.is_ascii_digit()) {
        return token
            .parse()
            .map(DeviceSelector::Index)
            .map_err(|error| format!("device index {token:?}: {error}"));
    }
    let lower = token.to_ascii_lowercase();
    if let Some(isa) = lower.strip_prefix("gfx") {
        if !isa.is_empty() && isa.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Ok(DeviceSelector::Arch(lower));
        }
    }
    if let Some(hex) = lower.strip_prefix("gpu-") {
        return match u64::from_str_radix(hex, 16) {
            Ok(0) => Err(format!("{token:?} is the no-UUID placeholder; select that card by PCI address")),
            Ok(id) if hex.len() <= 16 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()) => {
                Ok(DeviceSelector::Uuid(id))
            }
            _ => Err(format!("{token:?} is not a GPU-<hex> UUID")),
        };
    }
    if token.contains(':') {
        return PciBdf::parse_parts(token)
            .map(|(bdf, has_domain)| DeviceSelector::Bdf { bdf, has_domain })
            .ok_or_else(|| format!("{token:?} is not a DDDD:BB:DD.F or BB:DD.F PCI address"));
    }
    Err(format!(
        "unknown device selector {token:?}; expected an index, gfxNNNN, GPU-<uuid>, or a PCI address"
    ))
}

/// Outcome of trying to reserve a card during resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Claim {
    Claimed,
    /// Held elsewhere; the text names the holder.
    Busy(String),
}

/// Reserve callback. The daemon takes the card's flock here, which makes the
/// arch selector's "first free card" choice and the reservation one step.
pub type ClaimFn<'a> = dyn FnMut(&GpuDevice) -> std::result::Result<Claim, String> + 'a;

/// Resolve a comma-separated selector list against `devices` (BDF order).
///
/// Exact selectors (index, UUID, PCI address) are matched and claimed first,
/// in list order; each `gfxNNNN` then takes the first card of that arch in BDF
/// order that no other entry selected and whose claim succeeds. The result
/// keeps list order. Any unknown selector, miss, duplicate, busy exact card,
/// or exhausted arch fails with the device table.
pub fn resolve_device_selectors(
    spec: &str,
    devices: &[GpuDevice],
    claim: &mut ClaimFn<'_>,
) -> Result<Vec<GpuDevice>> {
    let fail = |message: String| ConfigError::InvalidValue {
        key: KEY.into(),
        message: format!("{spec:?}: {message}\n{}", device_table(devices)),
    };
    let selectors = spec
        .split(',')
        .map(|token| parse_selector(token.trim()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(fail)?;
    let mut chosen: Vec<Option<usize>> = vec![None; selectors.len()];

    for (slot, selector) in selectors.iter().enumerate() {
        if matches!(selector, DeviceSelector::Arch(_)) {
            continue;
        }
        let mut matches = devices
            .iter()
            .enumerate()
            .filter(|(_, device)| selector.matches(device));
        let Some((position, device)) = matches.next() else {
            return Err(fail(format!("selector {selector} matches no GPU")));
        };
        if matches.next().is_some() {
            return Err(fail(format!(
                "selector {selector} matches several GPUs; give the PCI domain"
            )));
        }
        if chosen.contains(&Some(position)) {
            return Err(fail(format!(
                "selector {selector} names {} more than once",
                device.describe()
            )));
        }
        chosen[slot] = Some(position);
    }
    for (slot, selector) in selectors.iter().enumerate() {
        let Some(position) = chosen[slot] else {
            continue;
        };
        let device = &devices[position];
        match claim(device).map_err(&fail)? {
            Claim::Claimed => {}
            Claim::Busy(holder) => {
                return Err(fail(format!(
                    "selector {selector} resolves to {}, which is {holder}",
                    device.describe()
                )))
            }
        }
    }

    for (slot, selector) in selectors.iter().enumerate() {
        let DeviceSelector::Arch(arch) = selector else {
            continue;
        };
        let mut busy = Vec::new();
        let mut taken = 0;
        for (position, device) in devices.iter().enumerate() {
            if device.arch != *arch {
                continue;
            }
            if chosen.contains(&Some(position)) {
                taken += 1;
                continue;
            }
            match claim(device).map_err(&fail)? {
                Claim::Claimed => {
                    chosen[slot] = Some(position);
                    break;
                }
                Claim::Busy(holder) => busy.push(format!("index {} {holder}", device.index)),
            }
        }
        if chosen[slot].is_none() {
            return Err(fail(if taken == 0 && busy.is_empty() {
                format!("no {arch} GPU on this host")
            } else {
                format!(
                    "no free {arch} GPU for entry {}: {taken} already selected by this list, busy: [{}]",
                    slot + 1,
                    busy.join("; ")
                )
            }));
        }
    }

    Ok(chosen
        .into_iter()
        .map(|position| devices[position.expect("every selector resolved")].clone())
        .collect())
}

/// ROCr and HIP filters for one resolved or inherited device set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceVisibility {
    /// Physical selectors consumed by ROCr.
    pub rocr: String,
    /// Logical selectors inside the ROCr-filtered set consumed by HIP.
    pub hip: String,
}

impl DeviceVisibility {
    fn for_devices(devices: &[GpuDevice]) -> Self {
        Self {
            rocr: devices
                .iter()
                .map(GpuDevice::rocr_selector)
                .collect::<Vec<_>>()
                .join(","),
            hip: logical_device_list(devices.len()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceSelection {
    /// `hardware.devices` resolved to physical cards, in logical order.
    Resolved {
        devices: Vec<GpuDevice>,
        visibility: DeviceVisibility,
    },
    /// Expert override: raw inherited HIP/ROCr filters, normalized.
    Inherited(DeviceVisibility),
    /// No filter: every GPU ROCr exposes is visible.
    Unfiltered,
}

impl DeviceSelection {
    pub fn visibility(&self) -> Option<&DeviceVisibility> {
        match self {
            Self::Resolved { visibility, .. } | Self::Inherited(visibility) => Some(visibility),
            Self::Unfiltered => None,
        }
    }
}

/// Resolve one physical GPU set for both ROCm frontends.
///
/// Explicit `hardware.devices` wins and is resolved against the KFD topology
/// at `topology`, calling `claim` for every card it reserves. Without it, raw
/// inherited HIP/ROCr filters are the expert override: compatible pairs are
/// normalized, ambiguous pairs fail closed. HIP always receives logical
/// `0..N-1` inside the ROCr-filtered set; giving both runtimes the same
/// non-zero index can compound their filters into an empty device set.
pub fn synchronized_device_visibility(
    config: &ProcessConfig,
    hip_visible: Option<&str>,
    rocr_visible: Option<&str>,
    topology: &Path,
    claim: &mut ClaimFn<'_>,
) -> Result<DeviceSelection> {
    if let Some(spec) = config.legacy_value("HIPFIRE_DEVICES") {
        let devices = resolve_device_selectors(&spec, &enumerate_gpus(topology)?, claim)?;
        let visibility = DeviceVisibility::for_devices(&devices);
        return Ok(DeviceSelection::Resolved {
            devices,
            visibility,
        });
    }

    let hip = hip_visible.map(normalize_device_visibility).transpose()?;
    let rocr = rocr_visible.map(normalize_device_visibility).transpose()?;
    match (hip, rocr) {
        (Some(hip), Some(rocr)) => {
            let expected_hip = logical_device_list(device_count(&rocr));
            if hip == expected_hip || hip == rocr {
                Ok(DeviceSelection::Inherited(DeviceVisibility {
                    rocr,
                    hip: expected_hip,
                }))
            } else {
                Err(ConfigError::InvalidValue {
                    key: KEY.into(),
                    message: format!(
                        "{ROCR_VISIBLE_DEVICES}={rocr:?} requires {HIP_VISIBLE_DEVICES}={expected_hip:?}, but inherited {hip:?}; set hardware.devices instead"
                    ),
                })
            }
        }
        (Some(physical), None) | (None, Some(physical)) => {
            Ok(DeviceSelection::Inherited(DeviceVisibility {
                hip: logical_device_list(device_count(&physical)),
                rocr: physical,
            }))
        }
        (None, None) => Ok(DeviceSelection::Unfiltered),
    }
}

/// Resolve and install synchronized HIP/ROCr visibility before either GPU
/// runtime initializes, and publish the resolved cards for
/// [`active_devices`]. Callers must invoke this once, during single-threaded
/// startup.
pub fn apply_device_visibility(
    config: &ProcessConfig,
    claim: &mut ClaimFn<'_>,
) -> Result<DeviceSelection> {
    let hip = unicode_environment(HIP_VISIBLE_DEVICES)?;
    let rocr = unicode_environment(ROCR_VISIBLE_DEVICES)?;
    let selection = synchronized_device_visibility(
        config,
        hip.as_deref(),
        rocr.as_deref(),
        Path::new(KFD_TOPOLOGY_NODES),
        claim,
    )?;
    if let Some(visibility) = selection.visibility() {
        std::env::set_var(HIP_VISIBLE_DEVICES, &visibility.hip);
        std::env::set_var(ROCR_VISIBLE_DEVICES, &visibility.rocr);
    }
    if let DeviceSelection::Resolved { devices, .. } = &selection {
        let active = ActiveDevices {
            devices: devices.clone(),
            hip_arch_override: unicode_environment(HSA_OVERRIDE_GFX_VERSION)?
                .as_deref()
                .and_then(override_arch),
        };
        ACTIVE_DEVICES.set(active).map_err(|_| ConfigError::InvalidValue {
            key: KEY.into(),
            message: "device selection was already applied in this process".into(),
        })?;
    }
    Ok(selection)
}

static ACTIVE_DEVICES: OnceLock<ActiveDevices> = OnceLock::new();

/// The cards `hardware.devices` resolved to in this process, if any.
pub fn active_devices() -> Option<&'static ActiveDevices> {
    ACTIVE_DEVICES.get()
}

/// What HIP reported for one logical device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedDevice {
    pub logical: usize,
    /// `gcnArchName` without target features.
    pub arch: String,
    /// `hipDeviceGetPCIBusId`.
    pub pci_bus_id: String,
}

/// Resolved cards in logical order plus the arch HIP must report for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveDevices {
    pub devices: Vec<GpuDevice>,
    /// Arch forced by `HSA_OVERRIDE_GFX_VERSION`, which HIP reports instead
    /// of the physical ISA.
    pub hip_arch_override: Option<String>,
}

impl ActiveDevices {
    /// Check the complete HIP-visible set: its size and every logical device.
    pub fn verify_visible(&self, observed: &[ObservedDevice]) -> std::result::Result<(), String> {
        if observed.len() != self.devices.len() {
            return Err(self.mismatch(
                observed,
                &format!(
                    "HIP exposes {} devices but hardware.devices resolved {}",
                    observed.len(),
                    self.devices.len()
                ),
            ));
        }
        self.verify(observed)
    }

    /// Check that each observed logical device is the card resolved for it.
    pub fn verify(&self, observed: &[ObservedDevice]) -> std::result::Result<(), String> {
        for seen in observed {
            let Some(device) = self.devices.get(seen.logical) else {
                return Err(self.mismatch(
                    observed,
                    &format!("logical device {} is outside the resolved set", seen.logical),
                ));
            };
            let expected_arch = self.hip_arch_override.as_deref().unwrap_or(&device.arch);
            let arch_ok = seen.arch == expected_arch;
            let bus_ok = PciBdf::parse(&seen.pci_bus_id) == Some(device.bdf);
            if !arch_ok || !bus_ok {
                return Err(self.mismatch(
                    observed,
                    &format!(
                        "logical device {} should be {expected_arch} PCI {}, HIP reports {} PCI {}",
                        seen.logical, device.bdf, seen.arch, seen.pci_bus_id
                    ),
                ));
            }
        }
        Ok(())
    }

    fn mismatch(&self, observed: &[ObservedDevice], reason: &str) -> String {
        let mut message = format!(
            "GPU identity mismatch after HIP init: {reason}\nresolved hardware.devices:\n  logical  index  arch      uuid                  pci"
        );
        for (logical, device) in self.devices.iter().enumerate() {
            message.push_str(&format!(
                "\n  {logical:<7}  {:<5}  {:<8}  {:<20}  {}",
                device.index,
                device.arch,
                device.uuid().unwrap_or_else(|| "(none)".into()),
                device.bdf
            ));
        }
        if let Some(arch) = &self.hip_arch_override {
            message.push_str(&format!("\n  ({HSA_OVERRIDE_GFX_VERSION} expects HIP arch {arch})"));
        }
        message.push_str("\nHIP reports:\n  logical  arch      pci");
        for seen in observed {
            message.push_str(&format!(
                "\n  {:<7}  {:<8}  {}",
                seen.logical, seen.arch, seen.pci_bus_id
            ));
        }
        message
    }
}

/// `HSA_OVERRIDE_GFX_VERSION=11.0.0` → `gfx1100`.
fn override_arch(value: &str) -> Option<String> {
    let mut parts = value.trim().split('.').map(|part| part.parse::<u64>().ok());
    let (Some(Some(major)), Some(Some(minor)), Some(Some(stepping)), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return None;
    };
    (minor < 16 && stepping < 16).then(|| gfx_name(major * 10000 + minor * 100 + stepping))
}

fn unicode_environment(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(ConfigError::InvalidValue {
            key: KEY.into(),
            message: format!("{name} is not valid Unicode"),
        }),
    }
}

fn normalize_device_visibility(value: &str) -> Result<String> {
    let devices = value.split(',').map(str::trim).collect::<Vec<_>>();
    if devices.iter().any(|device| device.is_empty()) {
        return Err(ConfigError::InvalidValue {
            key: KEY.into(),
            message: "expected a non-empty comma-separated physical device list".into(),
        });
    }
    Ok(devices.join(","))
}

fn device_count(value: &str) -> usize {
    value.split(',').count()
}

fn logical_device_list(count: usize) -> String {
    (0..count)
        .map(|device| device.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{resolve, ConfigLayer, ConfigSource, NamedLayer};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Topology(PathBuf);

    impl Drop for Topology {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Nodes: (gfx_target_version, unique_id, location_id); node 0 is a CPU.
    fn topology(gpus: &[(u64, u64, u32)]) -> Topology {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "hipfire-kfd-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let write = |node: usize, properties: String| {
            let dir = root.join(node.to_string());
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join("properties"), properties).unwrap();
        };
        write(0, "cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\nlocation_id 0\ndomain 0\n".into());
        for (offset, (version, unique_id, location)) in gpus.iter().enumerate() {
            write(
                offset + 1,
                format!(
                    "cpu_cores_count 0\nsimd_count 128\ngfx_target_version {version}\nlocation_id {location}\ndomain 0\nunique_id {unique_id}\n"
                ),
            );
        }
        Topology(root)
    }

    /// This workstation: five gfx1201 in KFD order A, E, C, D, B.
    fn five_gfx1201() -> Topology {
        topology(&[
            (120001, 0x9eb7aeda51c88ffd, 0x0300),
            (120001, 0x05f92432f2312a0e, 0xc300),
            (120001, 0x085289909a86cc63, 0xe300),
            (120001, 0x6109a4cb5f833235, 0x7b00),
            (120001, 0xe475645fe0200397, 0x1300),
        ])
    }

    /// hipx: gfx1100, gfx1151 (no UUID), gfx1030, gfx1010 (no UUID).
    fn hipx() -> Topology {
        topology(&[
            (110000, 0x43390a851e296ee5, 26112),
            (110501, 0, 48896),
            (100300, 0xc7ff6b154d0128bc, 28160),
            (100100, 0, 39168),
        ])
    }

    fn free(_: &GpuDevice) -> std::result::Result<Claim, String> {
        Ok(Claim::Claimed)
    }

    fn resolve_free(spec: &str, topology: &Topology) -> Result<Vec<GpuDevice>> {
        resolve_device_selectors(spec, &enumerate_gpus(&topology.0).unwrap(), &mut free)
    }

    fn uuids(devices: &[GpuDevice]) -> Vec<String> {
        devices.iter().map(|device| device.rocr_selector()).collect()
    }

    fn message(error: ConfigError) -> String {
        error.to_string()
    }

    #[test]
    fn index_is_pci_order_not_rocr_order() {
        let host = five_gfx1201();
        let devices = enumerate_gpus(&host.0).unwrap();
        let order = devices
            .iter()
            .map(|device| (device.index, device.rocr_index, device.bdf.to_string()))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                (0, 0, "0000:03:00.0".into()),
                (1, 4, "0000:13:00.0".into()),
                (2, 3, "0000:7b:00.0".into()),
                (3, 1, "0000:c3:00.0".into()),
                (4, 2, "0000:e3:00.0".into()),
            ]
        );
        assert_eq!(
            uuids(&resolve_free("4,3", &host).unwrap()),
            ["GPU-085289909a86cc63", "GPU-05f92432f2312a0e"],
            "indices are BDF positions; list order is logical order"
        );
    }

    #[test]
    fn exact_selectors_accept_uuid_and_pci_spellings() {
        let host = five_gfx1201();
        for spec in [
            "GPU-085289909a86cc63",
            "gpu-085289909A86CC63",
            "GPU-85289909a86cc63",
            "0000:e3:00.0",
            "E3:00.0",
        ] {
            assert_eq!(
                uuids(&resolve_free(spec, &host).unwrap()),
                ["GPU-085289909a86cc63"],
                "{spec}"
            );
        }
    }

    #[test]
    fn repeated_arch_picks_distinct_free_cards_in_bdf_order() {
        let host = five_gfx1201();
        let devices = enumerate_gpus(&host.0).unwrap();
        // A (index 0) is locked elsewhere; B is named explicitly later in the list.
        let mut claimed = Vec::new();
        let mut claim = |device: &GpuDevice| {
            if device.index == 0 {
                return Ok(Claim::Busy("reserved by holder PID 7".into()));
            }
            claimed.push(device.index);
            Ok(Claim::Claimed)
        };
        let picked =
            resolve_device_selectors("gfx1201, gfx1201, 13:00.0", &devices, &mut claim).unwrap();
        assert_eq!(
            picked.iter().map(|device| device.index).collect::<Vec<_>>(),
            [2, 3, 1]
        );
        assert_eq!(claimed, [1, 2, 3], "exact entries are claimed before arch entries");
    }

    #[test]
    fn no_uuid_cards_lower_to_rocr_ordinals_and_pci_lock_keys() {
        let host = hipx();
        let devices = enumerate_gpus(&host.0).unwrap();
        let arches = devices
            .iter()
            .map(|device| (device.arch.as_str(), device.rocr_index))
            .collect::<Vec<_>>();
        // BDF order 66, 6e, 99, bf.
        assert_eq!(
            arches,
            [("gfx1100", 0), ("gfx1030", 2), ("gfx1010", 3), ("gfx1151", 1)]
        );
        let picked = resolve_free("gfx1010,gfx1151,gfx1030", &host).unwrap();
        assert_eq!(
            DeviceVisibility::for_devices(&picked),
            DeviceVisibility {
                rocr: "3,1,GPU-c7ff6b154d0128bc".into(),
                hip: "0,1,2".into(),
            }
        );
        assert_eq!(
            picked
                .iter()
                .map(GpuDevice::lock_file_name)
                .collect::<Vec<_>>(),
            [
                "gpu-pci-0000:99:00.0.lock",
                "gpu-pci-0000:bf:00.0.lock",
                "gpu-GPU-c7ff6b154d0128bc.lock",
            ]
        );
    }

    #[test]
    fn failures_are_closed_and_carry_the_device_table() {
        let host = five_gfx1201();
        for (spec, reason) in [
            ("5", "matches no GPU"),
            ("gfx1100", "no gfx1100 GPU"),
            ("GPU-0000000000000001", "matches no GPU"),
            ("2,GPU-6109a4cb5f833235", "more than once"),
            ("card0", "unknown device selector"),
            ("0,,1", "empty entry"),
            ("GPU-0", "no-UUID placeholder"),
            ("03:00", "not a DDDD:BB:DD.F"),
            ("gfx1201,gfx1201,gfx1201,gfx1201,gfx1201,gfx1201", "no free gfx1201 GPU for entry 6: 5 already selected"),
        ] {
            let error = message(resolve_free(spec, &host).unwrap_err());
            assert!(error.contains(reason), "{spec}: {error}");
            assert!(error.contains("GPU-e475645fe0200397  0000:13:00.0"), "{spec}: {error}");
        }

        let devices = enumerate_gpus(&host.0).unwrap();
        let mut busy = |_: &GpuDevice| Ok(Claim::Busy("reserved by holder PID 9".into()));
        let error = message(resolve_device_selectors("1", &devices, &mut busy).unwrap_err());
        assert!(error.contains("which is reserved by holder PID 9"), "{error}");
        let error = message(resolve_device_selectors("gfx1201", &devices, &mut busy).unwrap_err());
        assert!(error.contains("busy: [index 0 reserved by holder PID 9;"), "{error}");
    }

    #[test]
    fn domainless_pci_address_must_be_unique() {
        let root = topology(&[(120001, 1, 0x0300)]);
        let second = root.0.join("2");
        fs::create_dir_all(&second).unwrap();
        fs::write(
            second.join("properties"),
            "simd_count 128\ngfx_target_version 120001\nlocation_id 768\ndomain 1\nunique_id 2\n",
        )
        .unwrap();
        assert!(message(resolve_free("03:00.0", &root).unwrap_err()).contains("give the PCI domain"));
        assert_eq!(uuids(&resolve_free("0001:03:00.0", &root).unwrap()), ["GPU-0000000000000002"]);
    }

    fn process_with_devices(spec: &str) -> ProcessConfig {
        let mut layer = ConfigLayer::default();
        layer.set_cli("hardware.devices", spec).unwrap();
        ProcessConfig::from_resolved(
            &resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: PathBuf::from("config.toml"),
                },
                layer,
            }])
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn configured_list_overrides_inherited_filters_and_legacy_pairs_stay_checked() {
        let host = five_gfx1201();
        let selection = synchronized_device_visibility(
            &process_with_devices(" 3, 1 "),
            Some("0"),
            Some("2"),
            &host.0,
            &mut free,
        )
        .unwrap();
        assert_eq!(
            selection.visibility(),
            Some(&DeviceVisibility {
                rocr: "GPU-05f92432f2312a0e,GPU-e475645fe0200397".into(),
                hip: "0,1".into(),
            })
        );

        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        let missing = Path::new("/nonexistent-kfd-topology");
        let inherited = |hip, rocr| {
            synchronized_device_visibility(&defaults, hip, rocr, missing, &mut free)
        };
        assert_eq!(
            inherited(Some("2"), None).unwrap(),
            DeviceSelection::Inherited(DeviceVisibility {
                rocr: "2".into(),
                hip: "0".into(),
            }),
            "a lone raw filter is lowered to ROCr physical plus HIP logical without topology"
        );
        assert_eq!(
            inherited(Some("0"), Some("2")).unwrap().visibility(),
            Some(&DeviceVisibility {
                rocr: "2".into(),
                hip: "0".into(),
            })
        );
        assert!(inherited(Some("1"), Some("2")).is_err());
        assert_eq!(inherited(None, None).unwrap(), DeviceSelection::Unfiltered);
    }

    fn seen(logical: usize, arch: &str, pci: &str) -> ObservedDevice {
        ObservedDevice {
            logical,
            arch: arch.into(),
            pci_bus_id: pci.into(),
        }
    }

    #[test]
    fn post_init_check_compares_arch_and_bus_per_logical_device() {
        let host = hipx();
        let active = ActiveDevices {
            devices: resolve_free("gfx1010,gfx1030", &host).unwrap(),
            hip_arch_override: None,
        };
        active
            .verify_visible(&[seen(0, "gfx1010", "0000:99:00.0"), seen(1, "gfx1030", "0000:6E:00.0")])
            .unwrap();
        let swapped = active
            .verify_visible(&[seen(0, "gfx1030", "0000:6e:00.0"), seen(1, "gfx1010", "0000:99:00.0")])
            .unwrap_err();
        assert!(swapped.contains("logical device 0 should be gfx1010 PCI 0000:99:00.0"), "{swapped}");
        assert!(swapped.contains("HIP reports:"), "{swapped}");
        assert!(active.verify_visible(&[seen(0, "gfx1010", "0000:99:00.0")]).is_err());
        assert!(active.verify(&[seen(1, "gfx1030", "0000:6e:00.0")]).is_ok());
        assert!(active.verify(&[seen(2, "gfx1030", "0000:6e:00.0")]).is_err());

        let overridden = ActiveDevices {
            hip_arch_override: override_arch("10.3.0"),
            ..active
        };
        overridden
            .verify(&[seen(0, "gfx1030", "0000:99:00.0")])
            .unwrap();
    }

    #[test]
    fn gfx_names_decode_hex_minor_and_stepping() {
        assert_eq!(gfx_name(90010), "gfx90a");
        assert_eq!(gfx_name(110501), "gfx1151");
        assert_eq!(override_arch("9.0.10").as_deref(), Some("gfx90a"));
        assert_eq!(override_arch("11.0"), None);
    }
}
