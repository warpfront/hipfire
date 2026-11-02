// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Machine facts for `hipfire fit`, read from sysfs, `/proc` and the
//! `hipfire calibrate` record. Nothing here initializes HIP or spawns the
//! daemon.
//!
//! Every figure goes through the function the engine itself uses, fed from
//! files under an injectable root so the tests run against a fake tree:
//!
//! - GPU set and logical order: `hipfire_config::devices` (KFD topology,
//!   `resolve_device_selectors`), in the order the daemon gives
//!   `hardware.devices` ([`select_devices`]).
//! - Free device bytes: `rdna_compute::uma_memory::device_free_from`, the clamp
//!   `Gpu::device_mem_info` applies (crates/rdna-compute/src/dispatch.rs:1750),
//!   fed `hipMemGetInfo`'s two numbers as amdgpu's sysfs counters give them
//!   ([`device_free`]).
//! - Host RAM, TTM pool, GTT cap: the `uma_memory` meminfo helpers and a
//!   root-injectable mirror of `expert_residency::{ttm_pool_estimate,
//!   gtt_budget}` ([`read_memory`]); those two read the live system, so they
//!   cannot be pointed at a test tree.
//! - Link: `calibration::lookup_in` + `host_link::link_from_calibration`
//!   exactly as `calibrate.rs` `routes_for` feeds them ([`link_facts`]).

use std::path::{Path, PathBuf};

use hipfire_arch_qwen4::host_link::{link_from_calibration, HostLink, NARROW_LINK_GBPS};
use hipfire_config::calibration::{self, Calibration, DeviceIdentity, HostFacts, Lookup};
use hipfire_config::devices::{
    enumerate_gpus, override_arch, resolve_device_selectors, Claim, GpuDevice, KFD_TOPOLOGY_NODES,
};
use rdna_compute::uma_memory::{self, DeviceMemory};

use super::types::{FitDevice, FitHost, LinkFacts};

/// Process inputs the machine facts depend on, gathered by [`live_inputs`] and
/// injected directly by tests.
#[derive(Clone, Debug, Default)]
pub(crate) struct HostInputs {
    /// `hardware.devices` (`HIPFIRE_DEVICES`) from the resolved config.
    pub devices_spec: Option<String>,
    pub rocr_visible: Option<String>,
    pub hip_visible: Option<String>,
    /// `HSA_OVERRIDE_GFX_VERSION`: HIP reports this arch for every card.
    pub hsa_override_gfx_version: Option<String>,
    /// `HSA_USERPTR_FOR_PAGED_MEM`, see [`read_memory`].
    pub userptr_for_paged_mem: Option<String>,
    /// The `[calibration]` record of `config.toml`.
    pub calibration: Option<Calibration>,
    /// `config.toml` has a `[calibration]` table that does not parse.
    pub calibration_unreadable: bool,
    /// The config (or env layer) does not resolve, so the CLI could not even
    /// build the daemon's `ProcessConfig`: the full `load refused: ...` note.
    pub config_refusal: Option<String>,
}

/// Machine facts of this host plus report-only lines. A note that starts with `load refused:`
/// means the daemon would refuse the load because `hardware.devices` (or an
/// inherited GPU filter, or the config itself) does not resolve; `devices` is
/// then empty and the note carries the fix. Other notes are unreadable counters.
///
/// Production overrides the sysfs/proc-derived memory facts with the engine's own
/// live readers: `uma_memory::{host_total_bytes, mem_available_bytes}` always,
/// and `expert_residency::{ttm_pool_estimate, gtt_budget}` when
/// `HSA_USERPTR_FOR_PAGED_MEM=0` is already in the environment. Those two read
/// that variable, which the daemon sets to `0` only inside the Qwen4 load
/// process, so while it is unset the GTT-backed answer is the root-injectable
/// mirror in [`read_memory`] (the same formulas, evaluated as that load would).
pub(crate) fn detect_with_notes(paths: &crate::Paths) -> (FitHost, Vec<String>) {
    let inputs = live_inputs(paths);
    let (mut host, notes) = detect_at(Path::new("/"), &inputs);
    host.ram_total = uma_memory::host_total_bytes();
    host.mem_available = uma_memory::mem_available_bytes();
    if inputs
        .userptr_for_paged_mem
        .as_deref()
        .is_some_and(|value| value.trim() == "0")
    {
        host.ttm_pool = hipfire_arch_qwen4::expert_residency::ttm_pool_estimate();
        let budget = hipfire_arch_qwen4::expert_residency::gtt_budget();
        host.gtt_limit = budget.map(|budget| budget.limit_bytes);
        host.gtt_used = budget.map(|budget| budget.used_bytes);
    }
    (host, notes)
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// The device selector and calibration record the daemon and engine would see.
/// `hardware.devices` is read as `HIPFIRE_DEVICES` (like `ple::load_target_arch`)
/// from the `ProcessConfig` the CLI hands the daemon (main.rs `run_command`):
/// the one already installed for the requested model
/// (`hipfire_config::active_process_config`, so a per-model `hardware.devices`
/// matches what the daemon gets), else the resolved global config + env layers.
/// A config that does not resolve stops `hipfire run` before the daemon starts,
/// so it is a refusal.
fn live_inputs(paths: &crate::Paths) -> HostInputs {
    let (devices_spec, config_refusal) = match devices_spec_from_config(paths) {
        Ok(spec) => (spec, None),
        Err(error) => {
            // Keep the home directory out of notes that reach the JSON report.
            let error = error.replace(&paths.root.display().to_string(), "<HIPFIRE_HOME>");
            (
                None,
                Some(format!(
                    "load refused: config does not resolve ({}); fix: correct the key it names with `hipfire config set <key> <value>`",
                    first_line(&error)
                )),
            )
        }
    };
    let (calibration, calibration_unreadable) = match calibration::load(&paths.config) {
        Ok(record) => (record, false),
        Err(_) => (None, true),
    };
    HostInputs {
        devices_spec,
        rocr_visible: env_var("ROCR_VISIBLE_DEVICES"),
        hip_visible: env_var("HIP_VISIBLE_DEVICES"),
        hsa_override_gfx_version: env_var("HSA_OVERRIDE_GFX_VERSION"),
        userptr_for_paged_mem: env_var("HSA_USERPTR_FOR_PAGED_MEM"),
        calibration,
        calibration_unreadable,
        config_refusal,
    }
}

/// `HIPFIRE_DEVICES` from the installed process config, resolving the global
/// config only when none is installed. `Err` is the resolution error text.
fn devices_spec_from_config(paths: &crate::Paths) -> Result<Option<String>, String> {
    if let Some(active) = hipfire_config::active_process_config() {
        return Ok(active.legacy_value("HIPFIRE_DEVICES"));
    }
    let (_, resolved) = crate::resolved_global(paths, true).map_err(|error| error.to_string())?;
    hipfire_config::ProcessConfig::from_resolved(&resolved)
        .map(|config| config.legacy_value("HIPFIRE_DEVICES"))
        .map_err(|error| error.to_string())
}

/// [`detect`] against a sysfs/proc tree rooted at `root`.
pub(crate) fn detect_at(root: &Path, inputs: &HostInputs) -> (FitHost, Vec<String>) {
    let mut notes = Vec::new();
    let memory = read_memory(root, inputs);
    // `calibration::detect_gpus_at` hides the ROCr ordinal and UUID the device
    // selectors match on, so enumerate the topology directly (it is what
    // `detect_gpus_at` calls).
    let gpus = enumerate_gpus(&rooted(root, KFD_TOPOLOGY_NODES)).unwrap_or_default();
    let arch_override = inputs
        .hsa_override_gfx_version
        .as_deref()
        .and_then(override_arch);
    let live_host = inputs
        .calibration
        .as_ref()
        .map(|_| calibration::detect_host_at(root));
    // A config or selector the daemon refuses leaves no device the load could
    // take: report none, with the refusal first in the notes.
    let selected = match &inputs.config_refusal {
        Some(refusal) => Err(refusal.clone()),
        None => select_devices(gpus, inputs),
    };
    let selected = selected.unwrap_or_else(|refusal| {
        notes.push(refusal);
        Vec::new()
    });
    let devices = selected
        .iter()
        .enumerate()
        .map(|(index, gpu)| {
            let arch = arch_override.clone().unwrap_or_else(|| gpu.arch.clone());
            let device = fit_device(
                root,
                index,
                gpu,
                arch,
                memory.apu_available,
                inputs,
                live_host.as_ref(),
            );
            if device.free.is_none() {
                notes.push(format!(
                    "GPU{index}: free device memory is not readable from sysfs{}",
                    if device.unified_memory {
                        " (or MemAvailable is missing)"
                    } else {
                        ""
                    }
                ));
            }
            device
        })
        .collect();
    (
        FitHost {
            ram_total: memory.ram_total,
            mem_available: memory.mem_available,
            ttm_pool: memory.ttm_pool,
            gtt_limit: memory.gtt_limit,
            gtt_used: memory.gtt_used,
            devices,
        },
        notes,
    )
}

fn rooted(root: &Path, absolute: &str) -> PathBuf {
    root.join(absolute.trim_start_matches('/'))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or_default()
}

/// The GPUs a load uses, in logical order (logical 0 = the card a single-GPU
/// load takes). Mirrors the daemon's resolution:
///
/// - `hardware.devices` (`HIPFIRE_DEVICES`): `resolve_device_selectors` against
///   the KFD topology, result in list order
///   (`devices::synchronized_device_visibility`, devices.rs:808-822), nothing
///   claimed (as `ple::device_arch_in`);
/// - else an inherited `ROCR_VISIBLE_DEVICES`, or `HIP_VISIBLE_DEVICES` alone
///   (which the daemon treats as the physical filter, devices.rs:843-848): ROCr
///   ordinals and UUIDs in list order (`devices::rocr_visible`, devices.rs:968);
/// - else every GPU in ROCr agent order (`Unfiltered`, devices.rs:849), whose
///   first entry is the lowest ordinal (`ple::device_arch_in`).
///
/// A selector that does not resolve makes the daemon refuse to load, so this
/// returns `Err(note)` (prefixed `load refused:`) and the caller reports no
/// devices. Falling back to another GPU set would report a FITS the load never
/// reaches. A host with no GPU at all is not a refusal: there is nothing to
/// select from and the caller's "no GPU" finding says so.
fn select_devices(gpus: Vec<GpuDevice>, inputs: &HostInputs) -> Result<Vec<GpuDevice>, String> {
    if gpus.is_empty() {
        return Ok(gpus);
    }
    let mut rocr_order = gpus.clone();
    rocr_order.sort_by_key(|gpu| gpu.rocr_index);
    if let Some(spec) = non_empty(inputs.devices_spec.as_deref()) {
        let mut no_claim = |_: &GpuDevice| Ok(Claim::Claimed);
        return resolve_device_selectors(spec, &gpus, &mut no_claim).map_err(|error| {
            format!(
                "load refused: hardware.devices {spec:?} does not resolve ({}); fix: `hipfire config set hardware.devices <index|gfxNNNN|GPU-uuid|PCI address>` naming a GPU `hipfire doctor` lists",
                first_line(&error.to_string())
            )
        });
    }
    let filter = non_empty(inputs.rocr_visible.as_deref())
        .map(|filter| ("ROCR_VISIBLE_DEVICES", filter))
        .or_else(|| {
            non_empty(inputs.hip_visible.as_deref()).map(|filter| ("HIP_VISIBLE_DEVICES", filter))
        });
    match filter {
        Some((name, filter)) => rocr_filtered(&rocr_order, filter).ok_or_else(|| {
            format!(
                "load refused: {name}={filter:?} names no GPU the KFD topology lists; fix: unset {name} or set it to a listed GPU ordinal"
            )
        }),
        None => Ok(rocr_order),
    }
}

/// The devices ROCr exposes under `filter` (ordinals and `GPU-<hex>` UUIDs, in
/// list order); `None` when an entry matches no GPU. Mirrors
/// `hipfire_config::devices::rocr_visible` (private, devices.rs:968-985).
fn rocr_filtered(devices: &[GpuDevice], filter: &str) -> Option<Vec<GpuDevice>> {
    filter
        .split(',')
        .map(|token| {
            let token = token.trim();
            devices
                .iter()
                .find(|device| match token.parse::<usize>() {
                    Ok(ordinal) => device.rocr_index == ordinal,
                    Err(_) => device
                        .uuid()
                        .is_some_and(|uuid| uuid.eq_ignore_ascii_case(token)),
                })
                .cloned()
        })
        .collect()
}

/// Host memory the admission gates read.
struct Memory {
    ram_total: Option<u64>,
    mem_available: Option<u64>,
    ttm_pool: u64,
    gtt_limit: Option<u64>,
    gtt_used: Option<u64>,
    /// `MemAvailable - CMA` plus the TTM pool, the host side of the
    /// unified-memory free clamp (`uma_memory::host_available_from`).
    apu_available: Option<u64>,
}

/// `mem_info_gtt_used` summed over every `/sys/class/drm/card<N>` amdgpu
/// device; mirrors `uma_memory::amdgpu_gtt_used_bytes` (uma_memory.rs:173) with
/// an injectable root. `None` when `/sys/class/drm` cannot be listed.
fn amdgpu_gtt_used_at(root: &Path) -> Option<u64> {
    let mut used_bytes = 0u64;
    for card in std::fs::read_dir(root.join("sys/class/drm"))
        .ok()?
        .flatten()
    {
        let is_card = card
            .file_name()
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_card {
            used_bytes += uma_memory::read_sysfs_u64(&card.path().join("device/mem_info_gtt_used"))
                .unwrap_or(0);
        }
    }
    Some(used_bytes)
}

/// Host RAM, TTM pool and GTT cap.
///
/// `ttm_pool` and `gtt_limit`/`gtt_used` mirror
/// `expert_residency::ttm_pool_estimate` (expert_residency.rs:2425) and
/// `gtt_budget` (:2443): both apply only while host-mapped memory is
/// GTT-backed, i.e. `HSA_USERPTR_FOR_PAGED_MEM=0`. hip-bridge sets that to `0`
/// for a process that loads a Qwen4 model unless the operator already set it
/// (hip-bridge ffi.rs:703-714), and the load fit describes is that process, so
/// an unset variable counts as `0`; any other operator value means userptr
/// memory and neither applies. The unified-memory free clamp counts the pool
/// ungated, as `Gpu::device_mem_info` does.
fn read_memory(root: &Path, inputs: &HostInputs) -> Memory {
    let meminfo = std::fs::read_to_string(root.join("proc/meminfo")).ok();
    let zoneinfo = std::fs::read_to_string(root.join("proc/zoneinfo")).ok();
    let read_pages = |absolute: &str| uma_memory::read_sysfs_u64(&rooted(root, absolute));
    let pool_pages = read_pages(uma_memory::TTM_PAGE_POOL_SIZE);
    let gtt_used = amdgpu_gtt_used_at(root);
    let gtt_backed = inputs
        .userptr_for_paged_mem
        .as_deref()
        .is_none_or(|value| value.trim() == "0");
    let ttm_pool = if gtt_backed {
        (|| uma_memory::ttm_pool_estimate_from(meminfo.as_deref()?, gtt_used?, pool_pages?))()
            .unwrap_or(0)
    } else {
        0
    };
    let (gtt_limit, gtt_used_reported) = if gtt_backed {
        let budget = (|| {
            let limit =
                read_pages(uma_memory::TTM_PAGES_LIMIT)?.checked_mul(uma_memory::TTM_PAGE_BYTES)?;
            Some((limit, gtt_used?))
        })();
        (budget.map(|(limit, _)| limit), budget.map(|(_, used)| used))
    } else {
        (None, None)
    };
    Memory {
        ram_total: meminfo
            .as_deref()
            .and_then(|text| uma_memory::meminfo_field(text, "MemTotal")),
        mem_available: meminfo
            .as_deref()
            .and_then(|text| uma_memory::meminfo_available_bytes(text, zoneinfo.as_deref())),
        ttm_pool,
        gtt_limit,
        gtt_used: gtt_used_reported,
        apu_available: meminfo.as_deref().and_then(|text| {
            uma_memory::host_available_from(
                text,
                zoneinfo.as_deref(),
                gtt_used.unwrap_or(0),
                pool_pages.unwrap_or(0),
            )
        }),
    }
}

fn fit_device(
    root: &Path,
    index: usize,
    gpu: &GpuDevice,
    arch: String,
    apu_available: Option<u64>,
    inputs: &HostInputs,
    live_host: Option<&HostFacts>,
) -> FitDevice {
    let bdf = gpu.bdf.to_string();
    let unified_memory = hipfire_config::is_unified_memory_arch(&arch);
    let (vram_total, free) = device_free(
        &root.join("sys/bus/pci/devices").join(&bdf),
        unified_memory,
        apu_available,
    );
    let link = link_facts(root, &bdf, &arch, unified_memory, inputs, live_host);
    FitDevice {
        index,
        arch,
        unified_memory,
        vram_total,
        free,
        link: Some(link),
        capacity: None,
        capacity_shown: None,
        env_selector: gpu.rocr_selector(),
        config_selector: gpu.uuid().unwrap_or(bdf),
    }
}

/// `(mem_info_vram_total, free device bytes)` of the PCI device at `sysfs`,
/// as `Gpu::device_mem_info` would report them before the load.
///
/// `hipMemGetInfo` is rebuilt from amdgpu's counters: a discrete GPU, or an APU
/// whose pool is its carve-out, reports `vram_total - vram_used`; an APU whose
/// GTT is larger than the carve-out reports TTM's cap minus live GTT
/// (`gtt_total - gtt_used`; docs/CONFIG.md:824-830). Those two numbers go
/// through `uma_memory::device_free_from`, the clamp `device_mem_info` applies
/// (dispatch.rs:1763-1777): a GTT-backed APU takes
/// `min(that, MemAvailable - CMA + TTM pool)`. HIP's own figure is slightly
/// lower once a process has opened the GPU (its context allocations).
fn device_free(
    sysfs: &Path,
    unified: bool,
    apu_available: Option<u64>,
) -> (Option<u64>, Option<u64>) {
    let counter = |name: &str| uma_memory::read_sysfs_u64(&sysfs.join(name));
    let vram_total = counter("mem_info_vram_total");
    let gtt_total = counter("mem_info_gtt_total");
    let pool = |total: Option<u64>, used: &str| -> Option<(u64, u64)> {
        let total = total?;
        Some((total, total.saturating_sub(counter(used)?)))
    };
    let vram_pool = || pool(vram_total, "mem_info_vram_used");
    let (device_total, device_free) = match (unified, gtt_total) {
        (true, Some(gtt)) if vram_total.is_none_or(|vram| gtt > vram) => {
            match pool(gtt_total, "mem_info_gtt_used") {
                Some(pair) => pair,
                None => return (vram_total, None),
            }
        }
        _ => match vram_pool() {
            Some(pair) => pair,
            None => return (vram_total, None),
        },
    };
    let free = uma_memory::device_free_from(DeviceMemory {
        device_free,
        device_total,
        uma: unified,
        carveout_total: vram_total,
        host_available: uma_memory::gtt_backed(device_total, vram_total)
            .then_some(apu_available)
            .flatten(),
    });
    (vram_total, free)
}

const NOT_CALIBRATED: &str = "run `hipfire calibrate` (the engine probes the link at load)";

/// The host link of one GPU as the calibration record describes it. The lookup
/// is `calibration::lookup_in` over the identity `hipfire calibrate` and the
/// engine build (`DeviceIdentity::live_at`: sysfs VRAM total and PCIe path); the
/// rate is `host_link::link_from_calibration`, as `calibrate.rs` `routes_for`
/// uses it. Only a fresh record states a link.
fn link_facts(
    root: &Path,
    bdf: &str,
    arch: &str,
    unified: bool,
    inputs: &HostInputs,
    live_host: Option<&HostFacts>,
) -> LinkFacts {
    let uncalibrated = |detail: String| LinkFacts {
        calibrated: false,
        gbps: None,
        narrow: None,
        detail,
    };
    if unified {
        return uncalibrated("unified memory: no host link".to_owned());
    }
    if inputs.calibration_unreadable {
        return uncalibrated(format!(
            "not calibrated: config.toml has an unreadable [calibration] record; {NOT_CALIBRATED}"
        ));
    }
    let (Some(record), Some(live_host)) = (inputs.calibration.as_ref(), live_host) else {
        return uncalibrated(format!("not calibrated: {NOT_CALIBRATED}"));
    };
    let identity = DeviceIdentity::live_at(root, bdf, arch, None);
    let lookup = calibration::lookup_in(Some(record), live_host, &identity);
    match &lookup {
        Lookup::Missing(reason) => {
            uncalibrated(format!("not calibrated ({reason}): {NOT_CALIBRATED}"))
        }
        Lookup::Stale(reason) => {
            uncalibrated(format!("calibration is stale ({reason}): {NOT_CALIBRATED}"))
        }
        Lookup::Fresh { measured_at, .. } => {
            let date = measured_at.split('T').next().unwrap_or(measured_at);
            match link_from_calibration(&lookup) {
                Some(HostLink::Discrete { gbps }) => {
                    let narrow = HostLink::Discrete { gbps }.is_narrow_discrete();
                    LinkFacts {
                        calibrated: true,
                        gbps: Some(gbps),
                        narrow: Some(narrow),
                        detail: if narrow {
                            format!(
                                "PCIe {gbps:.1} GB/s (calibrated {date}; narrow, below {NARROW_LINK_GBPS:.0} GB/s)"
                            )
                        } else {
                            format!("PCIe {gbps:.1} GB/s (calibrated {date})")
                        },
                    }
                }
                Some(HostLink::Unified) => uncalibrated("unified memory: no host link".to_owned()),
                Some(HostLink::Unmeasured) | None => LinkFacts {
                    calibrated: true,
                    gbps: None,
                    narrow: None,
                    detail: format!(
                        "calibrated {date} without a pinned host-to-device rate (the engine probes the link at load)"
                    ),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::calibration::{DeviceCalibration, PROBE_VERSION};
    use std::sync::atomic::{AtomicUsize, Ordering};

    const GIB: u64 = 1 << 30;
    const KIB: u64 = 1 << 10;

    /// A temp directory standing in for `/`.
    struct FakeRoot(PathBuf);

    impl FakeRoot {
        fn new(tag: &str) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "hipfire-fit-host-{tag}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn write(&self, relative: &str, contents: impl AsRef<str>) {
            let path = self.0.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents.as_ref()).unwrap();
        }

        /// A KFD GPU node; `bus` is the PCI bus, device and function are 0.
        fn kfd_gpu(&self, node: u32, bus: u32, gfx_target_version: u32) {
            self.write(
                &format!("sys/class/kfd/kfd/topology/nodes/{node}/properties"),
                format!(
                    "simd_count 64\nlocation_id {}\ndomain 0\ngfx_target_version {gfx_target_version}\nunique_id 0\n",
                    bus << 8
                ),
            );
        }

        fn pci(&self, bus: u32, counters: &[(&str, u64)]) {
            for (name, value) in counters {
                self.write(
                    &format!("sys/bus/pci/devices/0000:{bus:02x}:00.0/{name}"),
                    value.to_string(),
                );
            }
        }

        /// `/proc/meminfo` where only `MemFree`, caches and anonymous memory are
        /// tracked, so a large TTM pool estimate is possible.
        fn meminfo(
            &self,
            total_gib: u64,
            available_gib: u64,
            cma_total_gib: u64,
            cma_free_gib: u64,
        ) {
            let kib = |gib: u64| gib * GIB / KIB;
            self.write(
                "proc/meminfo",
                format!(
                    "MemTotal: {} kB\nMemFree: {} kB\nMemAvailable: {} kB\nBuffers: 0 kB\nCached: {} kB\nAnonPages: {} kB\nSlab: {} kB\nKReclaimable: {} kB\nSReclaimable: {} kB\nCmaTotal: {} kB\nCmaFree: {} kB\n",
                    kib(total_gib),
                    kib(8),
                    kib(available_gib),
                    kib(4),
                    kib(4),
                    kib(1),
                    kib(1),
                    kib(1),
                    kib(cma_total_gib),
                    kib(cma_free_gib),
                ),
            );
        }

        fn ttm(&self, pages_limit: u64, page_pool_size: u64) {
            self.write(
                "sys/module/ttm/parameters/pages_limit",
                pages_limit.to_string(),
            );
            self.write(
                "sys/module/ttm/parameters/page_pool_size",
                page_pool_size.to_string(),
            );
        }

        fn drm_gtt_used(&self, card: u32, bytes: u64) {
            self.write(
                &format!("sys/class/drm/card{card}/device/mem_info_gtt_used"),
                bytes.to_string(),
            );
        }
    }

    impl Drop for FakeRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Node 0 is a gfx1151 APU on bus 0xc0, node 1 a gfx1201 card on bus 0x64:
    /// BDF order is [gfx1201, gfx1151], ROCr order [gfx1151, gfx1201].
    fn mixed_root(tag: &str) -> FakeRoot {
        let root = FakeRoot::new(tag);
        root.kfd_gpu(0, 0xc0, 110501);
        root.kfd_gpu(1, 0x64, 120001);
        root.pci(
            0x64,
            &[
                ("mem_info_vram_total", 32 * GIB),
                ("mem_info_vram_used", GIB),
            ],
        );
        root.pci(
            0xc0,
            &[
                ("mem_info_vram_total", GIB),
                ("mem_info_vram_used", GIB / 2),
                ("mem_info_gtt_total", 120 * GIB),
                ("mem_info_gtt_used", 10 * GIB),
            ],
        );
        root.meminfo(128, 60, 2, 1);
        // 120 GiB cap; the pool is capped at 2 GiB (524288 pages).
        root.ttm(120 * GIB / 4096, 524_288);
        root.drm_gtt_used(0, 10 * GIB);
        root
    }

    fn archs(host: &FitHost) -> Vec<&str> {
        host.devices.iter().map(|d| d.arch.as_str()).collect()
    }

    #[test]
    fn default_order_is_rocr_agent_order() {
        let root = mixed_root("order");
        let (host, notes) = detect_at(&root.0, &HostInputs::default());
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(archs(&host), ["gfx1151", "gfx1201"]);
        assert_eq!(host.devices[0].index, 0);
        assert_eq!(host.devices[1].index, 1);
    }

    #[test]
    fn discrete_free_is_vram_total_minus_used() {
        let root = mixed_root("discrete");
        let (host, _) = detect_at(&root.0, &HostInputs::default());
        let card = &host.devices[1];
        assert!(!card.unified_memory);
        assert_eq!(card.vram_total, Some(32 * GIB));
        assert_eq!(card.free, Some(31 * GIB));
        let link = card.link.as_ref().unwrap();
        assert!(!link.calibrated);
        assert!(
            link.detail
                .starts_with("not calibrated: run `hipfire calibrate`"),
            "{}",
            link.detail
        );
    }

    #[test]
    fn gtt_backed_apu_free_is_clamped_by_host_memory() {
        let root = mixed_root("apu");
        let (host, _) = detect_at(&root.0, &HostInputs::default());
        let apu = &host.devices[0];
        assert!(apu.unified_memory);
        assert_eq!(apu.vram_total, Some(GIB));
        // GTT says 110 GiB free; MemAvailable 60 - CMA 2 = 58 GiB plus the
        // 2 GiB TTM pool is 60 GiB.
        assert_eq!(apu.free, Some(60 * GIB));
        assert_eq!(host.mem_available, Some(58 * GIB));
        assert_eq!(host.ram_total, Some(128 * GIB));
        assert_eq!(host.ttm_pool, 2 * GIB);
        assert_eq!(host.gtt_limit, Some(120 * GIB));
        assert_eq!(host.gtt_used, Some(10 * GIB));
        assert_eq!(
            apu.link.as_ref().unwrap().detail,
            "unified memory: no host link"
        );
        assert!(!apu.link.as_ref().unwrap().calibrated);
    }

    #[test]
    fn apu_with_ample_host_memory_keeps_the_gtt_figure() {
        let root = mixed_root("apu-ample");
        root.meminfo(128, 125, 0, 0);
        let (host, _) = detect_at(&root.0, &HostInputs::default());
        // min(110 GiB GTT free, 125 GiB + 2 GiB pool) = 110 GiB.
        assert_eq!(host.devices[0].free, Some(110 * GIB));
    }

    #[test]
    fn carveout_apu_keeps_the_carveout_figure() {
        let root = FakeRoot::new("carveout");
        root.kfd_gpu(0, 0xc0, 110501);
        root.pci(
            0xc0,
            &[
                ("mem_info_vram_total", 32 * GIB),
                ("mem_info_vram_used", 2 * GIB),
                ("mem_info_gtt_total", 16 * GIB),
                ("mem_info_gtt_used", 0),
            ],
        );
        root.meminfo(64, 4, 0, 0);
        let (host, _) = detect_at(&root.0, &HostInputs::default());
        assert_eq!(host.devices[0].free, Some(30 * GIB));
    }

    #[test]
    fn userptr_host_memory_drops_the_gtt_terms_but_not_the_clamp() {
        let root = mixed_root("userptr");
        let inputs = HostInputs {
            userptr_for_paged_mem: Some("1".into()),
            ..HostInputs::default()
        };
        let (host, _) = detect_at(&root.0, &inputs);
        assert_eq!(host.gtt_limit, None);
        assert_eq!(host.gtt_used, None);
        assert_eq!(host.ttm_pool, 0);
        // The unified-memory clamp counts the pool regardless.
        assert_eq!(host.devices[0].free, Some(60 * GIB));
        // An explicit `0` is the GTT-backed default.
        let inputs = HostInputs {
            userptr_for_paged_mem: Some(" 0 ".into()),
            ..HostInputs::default()
        };
        assert_eq!(detect_at(&root.0, &inputs).0.gtt_limit, Some(120 * GIB));
    }

    #[test]
    fn unreadable_apu_host_memory_leaves_free_unknown() {
        let root = mixed_root("blind");
        std::fs::remove_file(root.0.join("proc/meminfo")).unwrap();
        let (host, notes) = detect_at(&root.0, &HostInputs::default());
        assert_eq!(host.ram_total, None);
        assert_eq!(host.mem_available, None);
        assert_eq!(host.devices[0].free, None);
        assert_eq!(host.devices[1].free, Some(31 * GIB));
        assert!(
            notes.iter().any(|note| note.starts_with("GPU0:")),
            "{notes:?}"
        );
    }

    #[test]
    fn device_selection_follows_the_daemon() {
        let root = mixed_root("select");
        let select = |spec: Option<&str>, rocr: Option<&str>, hip: Option<&str>| {
            let inputs = HostInputs {
                devices_spec: spec.map(str::to_owned),
                rocr_visible: rocr.map(str::to_owned),
                hip_visible: hip.map(str::to_owned),
                ..HostInputs::default()
            };
            let (host, notes) = detect_at(&root.0, &inputs);
            (
                host.devices
                    .iter()
                    .map(|d| d.arch.clone())
                    .collect::<Vec<_>>(),
                notes,
            )
        };
        // hardware.devices: BDF indexes, in list order.
        assert_eq!(select(Some("0"), None, None).0, ["gfx1201"]);
        assert_eq!(select(Some("0,1"), None, None).0, ["gfx1201", "gfx1151"]);
        assert_eq!(
            select(Some("gfx1151,gfx1201"), None, None).0,
            ["gfx1151", "gfx1201"]
        );
        // hardware.devices wins over an inherited ROCr filter.
        assert_eq!(select(Some("0"), Some("0"), None).0, ["gfx1201"]);
        // ROCR_VISIBLE_DEVICES names ROCr ordinals, in list order.
        assert_eq!(select(None, Some("1"), None).0, ["gfx1201"]);
        assert_eq!(select(None, Some("1,0"), None).0, ["gfx1201", "gfx1151"]);
        // HIP_VISIBLE_DEVICES alone is the physical filter.
        assert_eq!(select(None, None, Some("1")).0, ["gfx1201"]);
        // Blank values filter nothing.
        assert_eq!(select(Some("  "), Some(""), None).0, ["gfx1151", "gfx1201"]);
    }

    #[test]
    fn unresolvable_selectors_refuse_the_load_with_no_devices() {
        let root = mixed_root("badselect");
        let (host, notes) = detect_at(
            &root.0,
            &HostInputs {
                devices_spec: Some("7".into()),
                ..HostInputs::default()
            },
        );
        assert!(host.devices.is_empty());
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].starts_with("load refused: hardware.devices \"7\" does not resolve"),
            "{notes:?}"
        );
        assert!(
            notes[0].contains("fix: `hipfire config set hardware.devices"),
            "{notes:?}"
        );
        // Memory facts survive a refusal.
        assert_eq!(host.ram_total, Some(128 * GIB));

        let (host, notes) = detect_at(
            &root.0,
            &HostInputs {
                rocr_visible: Some("5".into()),
                ..HostInputs::default()
            },
        );
        assert!(host.devices.is_empty());
        assert!(
            notes[0].starts_with("load refused: ROCR_VISIBLE_DEVICES=\"5\""),
            "{notes:?}"
        );
        let (host, notes) = detect_at(
            &root.0,
            &HostInputs {
                hip_visible: Some("5".into()),
                ..HostInputs::default()
            },
        );
        assert!(host.devices.is_empty());
        assert!(
            notes[0].starts_with("load refused: HIP_VISIBLE_DEVICES"),
            "{notes:?}"
        );
    }

    #[test]
    fn config_refusal_reports_no_devices() {
        let root = mixed_root("badconfig");
        let refusal = "load refused: config does not resolve (bad key); fix: x";
        let (host, notes) = detect_at(
            &root.0,
            &HostInputs {
                config_refusal: Some(refusal.into()),
                ..HostInputs::default()
            },
        );
        assert!(host.devices.is_empty());
        assert_eq!(notes, [refusal]);
    }

    #[test]
    fn selector_on_a_host_without_gpus_is_not_a_refusal() {
        let root = FakeRoot::new("nogpu-spec");
        let (host, notes) = detect_at(
            &root.0,
            &HostInputs {
                devices_spec: Some("0".into()),
                ..HostInputs::default()
            },
        );
        assert!(host.devices.is_empty());
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn hsa_override_changes_the_reported_arch() {
        let root = mixed_root("override");
        let (host, _) = detect_at(
            &root.0,
            &HostInputs {
                devices_spec: Some("0".into()),
                hsa_override_gfx_version: Some("11.0.0".into()),
                ..HostInputs::default()
            },
        );
        assert_eq!(host.devices[0].arch, "gfx1100");
        assert!(!host.devices[0].unified_memory);
    }

    #[test]
    fn host_without_gpus_has_memory_facts_only() {
        let root = FakeRoot::new("nogpu");
        root.meminfo(64, 40, 0, 0);
        let (host, notes) = detect_at(&root.0, &HostInputs::default());
        assert!(host.devices.is_empty());
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(host.ram_total, Some(64 * GIB));
        assert_eq!(host.mem_available, Some(40 * GIB));
        assert_eq!(host.gtt_limit, None);
        assert_eq!(host.gtt_used, None);
        assert_eq!(host.ttm_pool, 0);
        // Not even /proc: nothing is known, nothing panics.
        let bare = FakeRoot::new("bare");
        let (host, _) = detect_at(&bare.0, &HostInputs::default());
        assert!(host.devices.is_empty());
        assert_eq!(host.ram_total, None);
        assert_eq!(host.mem_available, None);
    }

    /// A record measured on this fake host for the bus-0x64 card.
    fn record(root: &Path, measured_at: &str, pinned: Option<f64>) -> Calibration {
        Calibration {
            probe_version: PROBE_VERSION,
            measured_at: measured_at.to_owned(),
            hipfire_version: "test".to_owned(),
            duration_s: 1.0,
            host: calibration::detect_host_at(root),
            cpu_read: None,
            devices: vec![DeviceCalibration {
                pci_bdf: "0000:64:00.0".to_owned(),
                arch: "gfx1201".to_owned(),
                vram_bytes: 32 * GIB,
                unified: false,
                link: None,
                pinned_h2d_gbps: pinned,
                zero_copy_gbps: None,
                engine: Vec::new(),
            }],
            unmeasured: Vec::new(),
        }
    }

    fn card_link(root: &FakeRoot, calibration: Option<Calibration>, unreadable: bool) -> LinkFacts {
        let inputs = HostInputs {
            devices_spec: Some("0".into()),
            calibration,
            calibration_unreadable: unreadable,
            ..HostInputs::default()
        };
        detect_at(&root.0, &inputs).0.devices[0]
            .link
            .clone()
            .unwrap()
    }

    #[test]
    fn fresh_calibration_states_the_link() {
        let root = mixed_root("link-fresh");
        let wide = card_link(
            &root,
            Some(record(&root.0, "2026-10-17T10:00:00Z", Some(37.1))),
            false,
        );
        assert!(wide.calibrated);
        assert_eq!(wide.gbps, Some(37.1));
        assert_eq!(wide.narrow, Some(false));
        assert_eq!(wide.detail, "PCIe 37.1 GB/s (calibrated 2026-10-17)");

        let narrow = card_link(
            &root,
            Some(record(&root.0, "2026-10-17T10:00:00Z", Some(12.0))),
            false,
        );
        assert!(narrow.calibrated);
        assert_eq!(narrow.narrow, Some(true));
        assert_eq!(
            narrow.detail,
            "PCIe 12.0 GB/s (calibrated 2026-10-17; narrow, below 16 GB/s)"
        );

        let unrated = card_link(
            &root,
            Some(record(&root.0, "2026-10-17T10:00:00Z", None)),
            false,
        );
        assert!(unrated.calibrated);
        assert_eq!(unrated.gbps, None);
        assert_eq!(unrated.narrow, None);
        assert!(unrated
            .detail
            .contains("without a pinned host-to-device rate"));
    }

    #[test]
    fn stale_or_missing_calibration_states_no_link() {
        let root = mixed_root("link-stale");
        let mut stale = record(&root.0, "2026-10-17T10:00:00Z", Some(37.1));
        stale.host.cpu_model = "some other cpu".to_owned();
        let link = card_link(&root, Some(stale), false);
        assert!(!link.calibrated);
        assert_eq!(link.gbps, None);
        assert!(
            link.detail.starts_with("calibration is stale (CPU changed"),
            "{}",
            link.detail
        );
        assert!(link.detail.contains("hipfire calibrate"));

        // A record that never listed this card.
        let mut other = record(&root.0, "2026-10-17T10:00:00Z", Some(37.1));
        other.devices[0].pci_bdf = "0000:99:00.0".to_owned();
        let link = card_link(&root, Some(other), false);
        assert!(!link.calibrated);
        assert!(link
            .detail
            .starts_with("not calibrated (device 0000:64:00.0 was not calibrated)"));

        let link = card_link(&root, None, true);
        assert!(!link.calibrated);
        assert!(link.detail.contains("unreadable [calibration] record"));
    }
}
