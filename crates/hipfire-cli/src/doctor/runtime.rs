// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hipfire doctor` without `--npu`: a runtime/driver/version preflight. It
//! reads sysfs, `/dev`, the ROCm tree, the installed kernel pack and the
//! daemon's `--version` line, then names one precise `fix:` for every failed or
//! warned check, before any model is loaded. No GPU or HIP call is made and
//! nothing touches the network.
//!
//! Three layers, so every decision is testable on a CPU-only host:
//!
//! * [`gather`] reads the host under an injected root (`/` in production) and
//!   injected environment, daemon probe and device-node predicates;
//! * [`assess`] is pure: facts in, ordered [`Check`]s out;
//! * [`render_text`] / [`render_json`] print the result.
//!
//! Like the NPU report, nothing printed carries a hostname, user name,
//! filesystem path or PCI address: GPUs are `GPU<i> gfxNNNN (1002:<device id>)`.
//! A fix may name canonical system paths (`/dev/kfd`) and commands.

use super::checks::HostProbe;
use super::probe::SystemProbe;
use super::report::{Check, CheckStatus};
use crate::Paths;
use hipfire_config::devices::{enumerate_gpus, override_arch, PciBdf};
use hipfire_config::rocm;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::fmt::Write as _;
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

/// JSON schema identifier; bump on any field change.
pub(crate) const SCHEMA: &str = "hipfire-doctor-runtime-1";

const AMD_VENDOR: &str = "0x1002";
const HSA_OVERRIDE_VAR: &str = "HSA_OVERRIDE_GFX_VERSION";
/// Canonical paths named in unreadable-source details.
const PCI_DEVICES_PATH: &str = "/sys/bus/pci/devices";
const KFD_NODES_PATH: &str = "/sys/class/kfd/kfd/topology/nodes";

const SETUP_FIX: &str = "`hipfire setup --source ~/.hipfire/src`";
const RENDER_GROUP_FIX: &str = "`sudo usermod -aG render,video $USER`, then log out and back in";
const KFD_MISSING_FIX: &str =
    "amdgpu started without KFD: check `journalctl -k -b | grep -i kfd`; \
     kernels need CONFIG_HSA_AMD=y (every distro kernel has it); a missing or too-old MES/MEC \
     firmware also stops KFD — install the current linux-firmware";
const ROCM_MINIMUMS: &str = "documented minimums: ROCm 6.4 for gfx1200/gfx1201, \
     7.2 for gfx1150/gfx1151/gfx1152, 6.0 otherwise";

// ---- facts -----------------------------------------------------------------

/// The render node of one GPU (`drm/renderD*` in sysfs, `/dev/dri/renderD*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RenderNode {
    #[default]
    Missing,
    /// The node exists; `rw` is whether this user can read and write it.
    Present { rw: bool },
}

/// One AMD display device on the PCI bus.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct PciGpu {
    /// Match key against the KFD topology. Never printed.
    pub bdf: Option<PciBdf>,
    /// PCI device id, lowercase hex without `0x` (`7551`).
    pub device_id: String,
    /// Basename of the `driver` symlink; `None` = unbound.
    pub driver: Option<String>,
    pub render: RenderNode,
    /// `fw_version/*` as `(name without _fw_version, compact hex)`, sorted by
    /// name; `None` when the kernel does not expose the directory.
    pub firmware: Option<Vec<(String, String)>>,
}

/// One KFD topology GPU node.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct KfdNode {
    pub arch: String,
    /// Never printed.
    pub bdf: PciBdf,
}

/// `/dev/kfd` as this user sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum KfdAccess {
    /// Absent, or not a character device.
    #[default]
    Missing,
    /// A character device this user cannot read and write.
    Denied,
    Ready,
}

/// The ROCm install the loader would use. Real resolution lives in
/// [`rocm_facts`]; tests build this directly.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct RocmFacts {
    /// The selected root; `None` when nothing usable is found or the install
    /// set is ambiguous. Never printed.
    pub root: Option<PathBuf>,
    /// Several complete roots and no explicit selection.
    pub ambiguous: bool,
    /// The variable that supplied an explicit root (`HIPFIRE_ROCM_PATH`, ...).
    pub configured_var: Option<&'static str>,
    /// `<root>/.info/version`.
    pub version: Option<String>,
}

/// Runtime libraries inside the selected root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeLibs {
    pub hip: bool,
    pub hsa: bool,
}

/// Whether the ROCm root ships the device library of one arch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bitcode {
    Found,
    /// No bitcode directory at all: a runtime-only install.
    NoBitcodeDir,
    /// Bitcode exists but not this arch's `oclc_isa_version_*.bc`.
    ArchMissing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IsaFacts {
    pub arch: String,
    pub bitcode: Bitcode,
}

/// The recorded manifest of an installed kernel pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackInfo {
    pub tag: String,
    pub commit: String,
    pub rocm_min: String,
    pub rocm_max_exclusive: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackFacts {
    pub arch: String,
    /// `None` = no manifest recorded for this arch.
    pub pack: Option<PackInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct DaemonFacts {
    pub found: bool,
    /// First line of `<daemon> --version` (`hipfire-daemon <ver> <commit>`).
    pub version_line: Option<String>,
}

/// This CLI binary's build identity.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct BuildId {
    pub version: String,
    pub commit: String,
}

impl BuildId {
    fn this_build() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            commit: crate::BUILD_COMMIT.to_owned(),
        }
    }
}

/// Everything [`assess`] decides from. Plain data.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct RuntimeFacts {
    /// `/proc/sys/kernel/osrelease`.
    pub kernel: String,
    /// `/sys/module/amdgpu` exists.
    pub amdgpu_module_loaded: bool,
    /// `/sys/module/amdgpu/version`: present only on out-of-tree (DKMS) builds.
    pub amdgpu_dkms_version: Option<String>,
    /// AMD display devices, in PCI address order; the position is the GPU index.
    pub gpus: Vec<PciGpu>,
    pub kfd: KfdAccess,
    pub kfd_nodes: Vec<KfdNode>,
    pub rocm: RocmFacts,
    /// `None` when no ROCm root is selected.
    pub libs: Option<RuntimeLibs>,
    /// One entry per distinct KFD arch; empty when no ROCm root is selected.
    pub isa: Vec<IsaFacts>,
    /// `HSA_OVERRIDE_GFX_VERSION`, trimmed; `None` when unset or empty.
    pub hsa_override: Option<String>,
    /// One entry per distinct KFD arch, read beside the found daemon.
    pub packs: Vec<PackFacts>,
    pub daemon: DaemonFacts,
    pub cli: BuildId,
    /// Kexec HandOver scratch reservation in bytes. `None` when
    /// `/proc/meminfo` or `/proc/zoneinfo` could not be read; `Some(0)` is a
    /// successful read with no reservation.
    pub kho_scratch_bytes: Option<u64>,
    /// The io error kind when `/sys/bus/pci/devices` exists but cannot be
    /// listed; `None` when it was listed or is absent.
    pub pci_read_error: Option<String>,
    /// Why the KFD topology could not be read (an io error kind); `None` when
    /// it was read or is absent. `kfd_nodes` is empty on error.
    pub kfd_topology_error: Option<String>,
}

/// The preflight result consumed by `hipfire doctor` and `hipfire fit`.
pub(crate) struct RuntimePreflight {
    pub checks: Vec<Check>,
    pub facts: RuntimeFacts,
}

impl RuntimePreflight {
    /// Any check failed.
    pub(crate) fn blocked(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == CheckStatus::Fail)
    }
}

// ---- gathering ---------------------------------------------------------------

/// Everything [`gather`] needs besides the injected filesystem root.
pub(crate) struct Inputs<'a> {
    pub env: &'a dyn Fn(&str) -> Option<String>,
    /// The found daemon binary (`crate::find_daemon`).
    pub daemon: Option<PathBuf>,
    /// First line of `<daemon> --version`.
    pub daemon_version: &'a dyn Fn(&Path) -> Option<String>,
    pub rocm: RocmFacts,
    /// Whether a path is a character device (tests allow a regular file).
    pub is_char_device: &'a dyn Fn(&Path) -> bool,
    /// Whether this user can read and write a path.
    pub access_rw: &'a dyn Fn(&Path) -> bool,
    pub cli: BuildId,
}

fn real_is_char_device(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.file_type().is_char_device())
}

fn real_access_rw(path: &Path) -> bool {
    let Ok(path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
        return false;
    };
    // SAFETY: `path` is a NUL-terminated string that outlives the call.
    unsafe { libc::access(path.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
}

/// The ROCm install the real environment resolves to. Not root-injected.
fn rocm_facts() -> RocmFacts {
    let ambiguous = !rocm::ambiguous_roots().is_empty();
    let root = rocm::root();
    let version = root.as_deref().and_then(rocm::version_for_root);
    RocmFacts {
        root,
        ambiguous,
        configured_var: rocm::configured_root().map(|(var, _)| var),
        version,
    }
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Entry names of `dir`, sorted.
fn read_names(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut names = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .collect::<Vec<_>>();
    names.sort_unstable();
    Ok(names)
}

/// Entry names of `dir`, sorted; empty when unreadable.
fn entry_names(dir: &Path) -> Vec<String> {
    read_names(dir).unwrap_or_default()
}

/// KHO scratch bytes from `<root>/proc/meminfo` and `<root>/proc/zoneinfo`;
/// `None` when either is unreadable. `Some(0)` is a successful read that found
/// no scratch reservation.
fn kho_scratch_bytes(root: &Path) -> Option<u64> {
    let meminfo = std::fs::read_to_string(root.join("proc/meminfo")).ok()?;
    let zoneinfo = std::fs::read_to_string(root.join("proc/zoneinfo")).ok()?;
    Some(rdna_compute::uma_memory::kho_scratch_from(&meminfo, Some(&zoneinfo)).unwrap_or(0))
}

/// Sub-directories (symlinks followed) of `dir`, sorted; one level only.
fn child_dirs(dir: &Path) -> Vec<PathBuf> {
    entry_names(dir)
        .into_iter()
        .map(|name| dir.join(name))
        .filter(|path| path.is_dir())
        .collect()
}

/// `0x00000086` -> `0x86`; anything that is not `0x` + hex is kept as is.
fn compact_hex(value: &str) -> String {
    match value.strip_prefix("0x") {
        Some(digits) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_hexdigit()) => {
            let trimmed = digits.trim_start_matches('0');
            format!("0x{}", if trimmed.is_empty() { "0" } else { trimmed })
        }
        _ => value.to_owned(),
    }
}

fn firmware_versions(device_dir: &Path) -> Option<Vec<(String, String)>> {
    let dir = device_dir.join("fw_version");
    std::fs::read_dir(&dir).ok()?;
    let mut versions = Vec::new();
    for name in entry_names(&dir) {
        let Some(value) = read_trimmed(&dir.join(&name)) else {
            continue;
        };
        let label = name.strip_suffix("_fw_version").unwrap_or(&name).to_owned();
        versions.push((label, compact_hex(&value)));
    }
    Some(versions)
}

fn render_node(root: &Path, device_dir: &Path, inputs: &Inputs<'_>) -> RenderNode {
    let Some(name) = entry_names(&device_dir.join("drm"))
        .into_iter()
        .find(|name| name.starts_with("renderD"))
    else {
        return RenderNode::Missing;
    };
    let node = root.join("dev/dri").join(name);
    if node.exists() {
        RenderNode::Present {
            rw: (inputs.access_rw)(&node),
        }
    } else {
        RenderNode::Missing
    }
}

/// Why `/sys/bus/pci/devices` could not be listed: the io error kind. An
/// absent directory is a known-empty bus, not an error.
fn pci_gpus(root: &Path, inputs: &Inputs<'_>) -> (Vec<PciGpu>, Option<String>) {
    let devices = root.join("sys/bus/pci/devices");
    let names = match read_names(&devices) {
        Ok(names) => names,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return (Vec::new(), Some(error.kind().to_string())),
    };
    let mut gpus = Vec::new();
    for name in names {
        let dir = devices.join(&name);
        let vendor = read_trimmed(&dir.join("vendor")).map(|v| v.to_ascii_lowercase());
        let class = read_trimmed(&dir.join("class")).map(|c| c.to_ascii_lowercase());
        let is_display = class
            .as_deref()
            .is_some_and(|class| class.starts_with("0x0300") || class.starts_with("0x0380"));
        if vendor.as_deref() != Some(AMD_VENDOR) || !is_display {
            continue;
        }
        let device_id = read_trimmed(&dir.join("device"))
            .map(|id| id.to_ascii_lowercase())
            .map(|id| id.strip_prefix("0x").unwrap_or(&id).to_owned())
            .unwrap_or_else(|| "unknown".to_owned());
        let driver = std::fs::read_link(dir.join("driver"))
            .ok()
            .and_then(|target| target.file_name().map(|n| n.to_string_lossy().into_owned()));
        gpus.push(PciGpu {
            bdf: PciBdf::parse(&name),
            device_id,
            driver,
            render: render_node(root, &dir, inputs),
            firmware: firmware_versions(&dir),
        });
    }
    gpus.sort_by_key(|gpu| gpu.bdf);
    (gpus, None)
}

fn kfd_access(root: &Path, inputs: &Inputs<'_>) -> KfdAccess {
    let node = root.join("dev/kfd");
    if !node.exists() || !(inputs.is_char_device)(&node) {
        KfdAccess::Missing
    } else if (inputs.access_rw)(&node) {
        KfdAccess::Ready
    } else {
        KfdAccess::Denied
    }
}

/// ROCm runtimes under the selected root: the HIP runtime library, and the HSA
/// runtime (`libhsa-runtime64.so*` in `lib`/`lib64`).
fn runtime_libs(rocm_root: &Path) -> RuntimeLibs {
    let hsa = ["lib", "lib64"].iter().any(|dir| {
        entry_names(&rocm_root.join(dir))
            .iter()
            .any(|name| name.starts_with("libhsa-runtime64.so"))
    });
    RuntimeLibs {
        hip: rocm::runtime_library(rocm_root).is_some(),
        hsa,
    }
}

/// Existing device-library directories of a ROCm root, most specific layout
/// first. Bounded one-level globs only.
fn bitcode_dirs(rocm_root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![
        rocm_root.join("amdgcn/bitcode"),
        rocm_root.join("lib/llvm/amdgcn/bitcode"),
        rocm_root.join("llvm/amdgcn/bitcode"),
    ];
    dirs.extend(
        child_dirs(&rocm_root.join("lib/llvm/lib/clang"))
            .into_iter()
            .map(|version| version.join("lib/amdgcn/bitcode")),
    );
    dirs.extend(
        child_dirs(rocm_root)
            .into_iter()
            .map(|child| child.join("lib/llvm/amdgcn/bitcode")),
    );
    dirs.retain(|dir| dir.is_dir());
    dirs
}

/// Does `rocm_root` ship `oclc_isa_version_<arch without "gfx">.bc`?
fn bitcode_state(rocm_root: &Path, arch: &str) -> Bitcode {
    let dirs = bitcode_dirs(rocm_root);
    if dirs.is_empty() {
        return Bitcode::NoBitcodeDir;
    }
    let file = format!(
        "oclc_isa_version_{}.bc",
        arch.strip_prefix("gfx").unwrap_or(arch)
    );
    if dirs.iter().any(|dir| dir.join(&file).is_file()) {
        Bitcode::Found
    } else {
        Bitcode::ArchMissing
    }
}

fn distinct_arches(nodes: &[KfdNode]) -> Vec<String> {
    nodes
        .iter()
        .map(|node| node.arch.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Read the host under `root` (`/` in production).
pub(crate) fn gather(root: &Path, inputs: &Inputs<'_>) -> RuntimeFacts {
    let module = root.join("sys/module/amdgpu");
    let nodes_dir = root.join("sys/class/kfd/kfd/topology/nodes");
    let (kfd_nodes, kfd_topology_error) = match enumerate_gpus(&nodes_dir) {
        Ok(devices) => (
            devices
                .into_iter()
                .map(|device| KfdNode {
                    arch: device.arch,
                    bdf: device.bdf,
                })
                .collect::<Vec<_>>(),
            None,
        ),
        // No KFD topology at all (the kfd module is absent): a known absence.
        Err(hipfire_config::ConfigError::Read { path, source })
            if path == nodes_dir && source.kind() == std::io::ErrorKind::NotFound =>
        {
            (Vec::new(), None)
        }
        Err(hipfire_config::ConfigError::Read { source, .. }) => {
            (Vec::new(), Some(source.kind().to_string()))
        }
        Err(_) => (Vec::new(), Some("malformed node properties".to_owned())),
    };
    let (gpus, pci_read_error) = pci_gpus(root, inputs);
    let arches = distinct_arches(&kfd_nodes);
    let rocm_root = inputs.rocm.root.as_deref();
    let exe_dir = inputs.daemon.as_deref().and_then(Path::parent);
    RuntimeFacts {
        kernel: read_trimmed(&root.join("proc/sys/kernel/osrelease")).unwrap_or_default(),
        amdgpu_module_loaded: module.exists(),
        amdgpu_dkms_version: read_trimmed(&module.join("version")),
        gpus,
        kfd: kfd_access(root, inputs),
        libs: rocm_root.map(runtime_libs),
        isa: rocm_root
            .map(|rocm_root| {
                arches
                    .iter()
                    .map(|arch| IsaFacts {
                        arch: arch.clone(),
                        bitcode: bitcode_state(rocm_root, arch),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        hsa_override: (inputs.env)(HSA_OVERRIDE_VAR)
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
        packs: exe_dir
            .map(|exe_dir| {
                arches
                    .iter()
                    .map(|arch| PackFacts {
                        arch: arch.clone(),
                        pack: rdna_compute::kernel_pack::installed_manifest(
                            &rdna_compute::kernel_pack::pack_arch_dir(exe_dir, arch),
                        )
                        .map(|manifest| PackInfo {
                            tag: manifest.tag,
                            commit: manifest.commit,
                            rocm_min: manifest.rocm_min,
                            rocm_max_exclusive: manifest.rocm_max_exclusive,
                        }),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        daemon: DaemonFacts {
            found: inputs.daemon.is_some(),
            version_line: inputs
                .daemon
                .as_deref()
                .and_then(|daemon| (inputs.daemon_version)(daemon)),
        },
        cli: inputs.cli.clone(),
        kho_scratch_bytes: kho_scratch_bytes(root),
        pci_read_error,
        kfd_topology_error,
        kfd_nodes,
        rocm: inputs.rocm.clone(),
    }
}

/// Gather this machine's facts on `/` with the real environment and daemon,
/// then assess them.
pub(crate) fn preflight(paths: &Paths) -> RuntimePreflight {
    let probe = SystemProbe::new(paths);
    let env = |key: &str| std::env::var(key).ok();
    let daemon_version = |daemon: &Path| probe.daemon_version(daemon);
    let inputs = Inputs {
        env: &env,
        daemon: crate::find_daemon(paths),
        daemon_version: &daemon_version,
        rocm: rocm_facts(),
        is_char_device: &real_is_char_device,
        access_rw: &real_access_rw,
        cli: BuildId::this_build(),
    };
    let facts = gather(Path::new("/"), &inputs);
    let checks = assess(&facts);
    RuntimePreflight { checks, facts }
}

/// Print the preflight (text, or one JSON object) and return the exit status:
/// 0 ok (warnings allowed), 1 when any check failed.
pub(crate) fn runtime_command(paths: &Paths, json: bool) -> anyhow::Result<i32> {
    let preflight = preflight(paths);
    if json {
        println!("{}", render_json(&preflight.facts, &preflight.checks)?);
    } else {
        print!("{}", render_text(&preflight.checks));
    }
    Ok(i32::from(preflight.blocked()))
}

// ---- assessment ----------------------------------------------------------------

fn rank(status: CheckStatus) -> u8 {
    match status {
        CheckStatus::Pass => 0,
        CheckStatus::Info => 1,
        CheckStatus::Warn => 2,
        CheckStatus::Fail => 3,
    }
}

/// Findings of one check (one per GPU or arch), merged into a single [`Check`]:
/// the worst status, the distinct details joined, the distinct fixes of every
/// failed or warned row on separate lines.
#[derive(Default)]
struct Rows(Vec<(CheckStatus, String, Option<String>)>);

impl Rows {
    fn pass(&mut self, detail: impl Into<String>) {
        self.0.push((CheckStatus::Pass, detail.into(), None));
    }

    fn info(&mut self, detail: impl Into<String>) {
        self.0.push((CheckStatus::Info, detail.into(), None));
    }

    fn warn(&mut self, detail: impl Into<String>, fix: impl Into<String>) {
        self.0
            .push((CheckStatus::Warn, detail.into(), Some(fix.into())));
    }

    fn fail(&mut self, detail: impl Into<String>, fix: impl Into<String>) {
        self.0
            .push((CheckStatus::Fail, detail.into(), Some(fix.into())));
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn finish(self, id: &'static str) -> Check {
        let status = self
            .0
            .iter()
            .map(|row| row.0)
            .max_by_key(|status| rank(*status))
            .unwrap_or(CheckStatus::Pass);
        let mut details: Vec<String> = Vec::new();
        let mut fixes: Vec<String> = Vec::new();
        for (row_status, detail, fix) in self.0 {
            if !details.contains(&detail) {
                details.push(detail);
            }
            if let Some(fix) = fix {
                let blocking = matches!(row_status, CheckStatus::Warn | CheckStatus::Fail);
                if blocking && !fixes.contains(&fix) {
                    fixes.push(fix);
                }
            }
        }
        Check {
            id,
            status,
            detail: details.join("; "),
            fix: (!fixes.is_empty()).then(|| fixes.join("\n")),
        }
    }
}

fn single(
    id: &'static str,
    status: CheckStatus,
    detail: impl Into<String>,
    fix: Option<String>,
) -> Check {
    Check {
        id,
        status,
        detail: detail.into(),
        fix,
    }
}

/// `GPU0 gfx1201 (1002:7551)`, or `GPU0 (1002:7551)` when the arch is unknown.
fn gpu_name(index: usize, arch: Option<&str>, device_id: &str) -> String {
    match arch {
        Some(arch) => format!("GPU{index} {arch} (1002:{device_id})"),
        None => format!("GPU{index} (1002:{device_id})"),
    }
}

fn kfd_arch<'a>(facts: &'a RuntimeFacts, gpu: &PciGpu) -> Option<&'a str> {
    let bdf = gpu.bdf?;
    facts
        .kfd_nodes
        .iter()
        .find(|node| node.bdf == bdf)
        .map(|node| node.arch.as_str())
}

/// amdgpu-bound GPUs as `(index, gpu, KFD arch)`.
fn bound_gpus(facts: &RuntimeFacts) -> Vec<(usize, &PciGpu, Option<&str>)> {
    facts
        .gpus
        .iter()
        .enumerate()
        .filter(|(_, gpu)| gpu.driver.as_deref() == Some("amdgpu"))
        .map(|(index, gpu)| (index, gpu, kfd_arch(facts, gpu)))
        .collect()
}

fn rocm_version(facts: &RuntimeFacts) -> &str {
    facts.rocm.version.as_deref().unwrap_or("unknown")
}

fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}

fn commit_known(commit: &str) -> bool {
    !commit.is_empty() && commit != "unknown"
}

/// Both known and neither a prefix of the other (the shorter one is compared).
fn commits_differ(a: &str, b: &str) -> bool {
    commit_known(a) && commit_known(b) && !(a.starts_with(b) || b.starts_with(a))
}

/// `(version, commit)` of `hipfire-daemon <ver> <commit>`.
fn parse_daemon_line(line: &str) -> (Option<&str>, Option<&str>) {
    let mut words = line.split_whitespace();
    words.next();
    (words.next(), words.next())
}

fn install_guidance_line() -> String {
    rocm::install_guidance().join(" ")
}

fn check_amdgpu_driver(facts: &RuntimeFacts) -> Check {
    const ID: &str = "amdgpu_driver";
    let mut rows = Rows::default();
    if facts.gpus.is_empty() {
        if let Some(error) = &facts.pci_read_error {
            rows.fail(
                format!("cannot read the PCI device list at {PCI_DEVICES_PATH} ({error})"),
                "check that /sys is mounted and readable (run `hipfire doctor` as a normal user \
                 on the host, not in a restricted container)",
            );
            return rows.finish(ID);
        }
        rows.fail(
            "no AMD GPU found on the PCI bus",
            "hipfire needs an AMD GPU (Radeon/Instinct/Ryzen AI APU)",
        );
        return rows.finish(ID);
    }
    let release = if facts.kernel.is_empty() {
        "unknown"
    } else {
        facts.kernel.as_str()
    };
    let module_note = if facts.amdgpu_module_loaded {
        ""
    } else {
        " (the amdgpu module is not loaded)"
    };
    for (index, gpu) in facts.gpus.iter().enumerate() {
        let device = &gpu.device_id;
        let label = gpu_name(index, kfd_arch(facts, gpu), device);
        match gpu.driver.as_deref() {
            Some("amdgpu") => {}
            None => rows.fail(
                format!("{label} has no driver bound (kernel {release}){module_note}"),
                format!(
                    "amdgpu did not bind GPU{index} (1002:{device}) on kernel {release}: run \
                     `journalctl -k -b | grep -iE 'amdgpu|firmware'`; a 'failed to load firmware' \
                     line means install the current linux-firmware package; no amdgpu line at all \
                     means this kernel predates the GPU: boot a newer kernel or install \
                     amdgpu-dkms from the same ROCm release as your ROCm userspace"
                ),
            ),
            Some("vfio-pci") => rows.fail(
                format!("{label} is bound to vfio-pci, not amdgpu"),
                "remove this GPU from the vfio-pci ids= list (kernel cmdline or \
                 /etc/modprobe.d) and reboot",
            ),
            Some("radeon") => rows.fail(
                format!("{label} is bound to radeon, not amdgpu"),
                "boot with `radeon.si_support=0 radeon.cik_support=0 amdgpu.si_support=1 \
                 amdgpu.cik_support=1`",
            ),
            Some(other) => rows.fail(
                format!("{label} is bound to {other}, not amdgpu"),
                format!("unbind {other} from this GPU and let amdgpu bind it"),
            ),
        }
    }
    if rows.is_empty() {
        let count = facts.gpus.len();
        let build = match &facts.amdgpu_dkms_version {
            Some(version) => format!("amdgpu-dkms {version}"),
            None => "in-tree".to_owned(),
        };
        rows.pass(format!(
            "{count} GPU(s) on amdgpu ({build}, kernel {release})"
        ));
    }
    rows.finish(ID)
}

fn check_kfd(facts: &RuntimeFacts) -> Check {
    const ID: &str = "kfd";
    match facts.kfd {
        KfdAccess::Missing => single(
            ID,
            CheckStatus::Fail,
            "no /dev/kfd: the amdgpu compute interface (KFD) is not available",
            Some(KFD_MISSING_FIX.to_owned()),
        ),
        KfdAccess::Denied => single(
            ID,
            CheckStatus::Fail,
            "/dev/kfd exists but this user cannot read and write it",
            Some(RENDER_GROUP_FIX.to_owned()),
        ),
        KfdAccess::Ready => single(
            ID,
            CheckStatus::Pass,
            "/dev/kfd is readable and writable",
            None,
        ),
    }
}

fn check_kfd_gpus(facts: &RuntimeFacts) -> Check {
    const ID: &str = "kfd_gpus";
    if let Some(error) = &facts.kfd_topology_error {
        return single(
            ID,
            CheckStatus::Fail,
            format!("cannot read the KFD topology at {KFD_NODES_PATH} ({error})"),
            Some(
                "check /sys/class/kfd/kfd/topology is readable; if /dev/kfd exists but the \
                 topology does not, reload amdgpu (`sudo modprobe -r amdgpu && sudo modprobe \
                 amdgpu`) or reboot"
                    .to_owned(),
            ),
        );
    }
    let bound = bound_gpus(facts);
    if bound.is_empty() {
        return single(
            ID,
            CheckStatus::Info,
            "no amdgpu-bound GPU to match against KFD",
            None,
        );
    }
    let mut rows = Rows::default();
    for (index, gpu, arch) in &bound {
        if arch.is_none() {
            rows.fail(
                format!(
                    "GPU{index} 1002:{} is driven by amdgpu but KFD did not register it, \
                     so ROCm cannot use it",
                    gpu.device_id
                ),
                KFD_MISSING_FIX,
            );
        }
    }
    if rows.is_empty() {
        let listed = bound
            .iter()
            .map(|(index, _, arch)| format!("GPU{index} {}", arch.unwrap_or("unknown")))
            .collect::<Vec<_>>()
            .join(", ");
        rows.pass(listed);
    }
    rows.finish(ID)
}

fn check_render_node(facts: &RuntimeFacts) -> Option<Check> {
    let with_kfd = bound_gpus(facts)
        .into_iter()
        .filter(|(_, _, arch)| arch.is_some())
        .collect::<Vec<_>>();
    if with_kfd.is_empty() {
        return None;
    }
    let mut rows = Rows::default();
    let mut usable = Vec::new();
    for (index, gpu, _) in with_kfd {
        match gpu.render {
            RenderNode::Missing => rows.fail(
                format!("no render node for GPU{index}"),
                "amdgpu created no DRM render node; check `journalctl -k -b | grep -i amdgpu`",
            ),
            RenderNode::Present { rw: false } => rows.fail(
                format!("this user cannot read and write the render node of GPU{index}"),
                RENDER_GROUP_FIX,
            ),
            RenderNode::Present { rw: true } => usable.push(format!("GPU{index}")),
        }
    }
    // One line for every usable node: a four-GPU host reads as one fact.
    match usable.len() {
        0 => {}
        1 => rows.pass(format!(
            "{} render node is readable and writable",
            usable[0]
        )),
        _ => rows.pass(format!(
            "{} render nodes are readable and writable",
            usable.join(", ")
        )),
    }
    Some(rows.finish("render_node"))
}

fn check_firmware(facts: &RuntimeFacts) -> Option<Check> {
    let bound = bound_gpus(facts);
    if bound.is_empty() {
        return None;
    }
    // GPUs reporting the same arch and the same versions share one line, so
    // a host of identical cards lists its firmware once.
    let mut groups: Vec<(Vec<String>, String, String)> = Vec::new();
    for (index, gpu, arch) in bound {
        let kind = match arch {
            Some(arch) => arch.to_owned(),
            None => format!("(1002:{})", gpu.device_id),
        };
        let versions = match &gpu.firmware {
            None => "not exposed by this kernel".to_owned(),
            Some(versions) if versions.is_empty() => "no firmware versions listed".to_owned(),
            Some(versions) => versions
                .iter()
                .map(|(label, hex)| format!("{label} {hex}"))
                .collect::<Vec<_>>()
                .join(", "),
        };
        match groups.iter_mut().find(|(_, group_kind, group_versions)| {
            *group_kind == kind && *group_versions == versions
        }) {
            Some((members, _, _)) => members.push(format!("GPU{index}")),
            None => groups.push((vec![format!("GPU{index}")], kind, versions)),
        }
    }
    let mut rows = Rows::default();
    for (members, kind, versions) in groups {
        rows.info(format!("{} {kind}: {versions}", members.join(", ")));
    }
    Some(rows.finish("firmware"))
}

fn check_rocm_root(facts: &RuntimeFacts) -> Check {
    const ID: &str = "rocm_root";
    let rocm = &facts.rocm;
    if rocm.ambiguous {
        return single(
            ID,
            CheckStatus::Fail,
            "several complete ROCm installs were found and none is selected",
            Some(
                "set one root: `export HIPFIRE_ROCM_PATH=/opt/rocm-<version>` \
                 (see docs/GETTING_STARTED.md)"
                    .to_owned(),
            ),
        );
    }
    if rocm.root.is_none() {
        let (detail, fix) = match rocm.configured_var {
            Some(var) => (
                format!("{var} is set but names no usable ROCm install"),
                format!(
                    "point {var} at a ROCm root (or unset it to auto-detect). {}",
                    install_guidance_line()
                ),
            ),
            None => ("no ROCm install found".to_owned(), install_guidance_line()),
        };
        return single(ID, CheckStatus::Fail, detail, Some(fix));
    }
    single(
        ID,
        CheckStatus::Pass,
        format!("ROCm {}", rocm_version(facts)),
        None,
    )
}

fn check_hip_runtime(facts: &RuntimeFacts) -> Option<Check> {
    const ID: &str = "hip_runtime";
    let libs = facts.libs?;
    let mut missing = Vec::new();
    if !libs.hip {
        missing.push("the HIP runtime (libamdhip64)");
    }
    if !libs.hsa {
        missing.push("the HSA runtime (libhsa-runtime64)");
    }
    if missing.is_empty() {
        return Some(single(
            ID,
            CheckStatus::Pass,
            "libamdhip64 and libhsa-runtime64 are present",
            None,
        ));
    }
    let version = rocm_version(facts);
    Some(single(
        ID,
        CheckStatus::Fail,
        format!("ROCm {version} lacks {}", missing.join(" and ")),
        Some(format!(
            "install the ROCm HIP runtime packages for ROCm {version}. {}",
            install_guidance_line()
        )),
    ))
}

/// The installed kernel pack of `arch`, when it admits the local ROCm (an
/// unknown local version gives the pack the benefit of the doubt).
fn admitting_pack<'a>(facts: &'a RuntimeFacts, arch: &str) -> Option<&'a PackInfo> {
    let pack = facts
        .packs
        .iter()
        .find(|pack| pack.arch == arch)?
        .pack
        .as_ref()?;
    let admitted = match facts.rocm.version.as_deref() {
        None => true,
        Some(version) => rdna_compute::kernel_pack::rocm_admitted(
            version,
            &pack.rocm_min,
            &pack.rocm_max_exclusive,
        ),
    };
    admitted.then_some(pack)
}

fn check_rocm_isa(facts: &RuntimeFacts) -> Option<Check> {
    if facts.isa.is_empty() {
        return None;
    }
    let version = rocm_version(facts);
    let mut rows = Rows::default();
    for isa in &facts.isa {
        let arch = &isa.arch;
        match isa.bitcode {
            Bitcode::Found => rows.pass(format!("ROCm {version} has the {arch} device library")),
            Bitcode::NoBitcodeDir => rows.warn(
                "no device compiler libraries: kernels outside an installed kernel pack \
                 cannot be compiled (JIT)",
                format!(
                    "install the ROCm HIP SDK (hipcc + device libraries) for ROCm {version}, or \
                     use the prebuilt kernel pack: `hipfire kernel-pack install --tag <tag> \
                     --arch {arch} --source ~/.hipfire/src` \
                     (see docs/GETTING_STARTED.md#prebuilt-kernel-packs)"
                ),
            ),
            Bitcode::ArchMissing => {
                let fix = format!(
                    "ROCm {version} does not know {arch}: install a newer ROCm ({ROCM_MINIMUMS})"
                );
                if admitting_pack(facts, arch).is_some() {
                    rows.warn(
                        format!(
                            "ROCm {version} ships no device library for {arch}; the installed \
                             kernel pack still runs registry kernels"
                        ),
                        fix,
                    );
                } else {
                    rows.fail(
                        format!(
                            "ROCm {version} ships no device library for {arch}: this ROCm \
                             release predates the GPU"
                        ),
                        fix,
                    );
                }
            }
        }
    }
    Some(rows.finish("rocm_isa"))
}

/// hipfire's documented minimum ROCm for an arch (README.md, GETTING_STARTED.md).
fn documented_floor(arch: &str) -> &'static str {
    match arch {
        "gfx1200" | "gfx1201" => "6.4",
        "gfx1150" | "gfx1151" | "gfx1152" => "7.2",
        _ => "6.0",
    }
}

fn check_rocm_version_floor(facts: &RuntimeFacts) -> Option<Check> {
    const ID: &str = "rocm_version_floor";
    let arches = distinct_arches(&facts.kfd_nodes);
    if facts.rocm.root.is_none() || arches.is_empty() {
        return None;
    }
    let Some(version) = facts.rocm.version.as_deref() else {
        return Some(single(
            ID,
            CheckStatus::Info,
            "ROCm version unknown (no .info/version)",
            None,
        ));
    };
    let mut rows = Rows::default();
    for arch in &arches {
        let floor = documented_floor(arch);
        if rdna_compute::kernel_pack::rocm_admitted(version, floor, "1000") {
            rows.pass(format!(
                "ROCm {version} meets the documented minimum {floor} for {arch}"
            ));
        } else {
            rows.warn(
                format!("ROCm {version} is below hipfire's documented minimum {floor} for {arch}"),
                format!("install ROCm {floor} or newer (hipfire's documented minimum for {arch})"),
            );
        }
    }
    Some(rows.finish(ID))
}

fn check_hsa_override(facts: &RuntimeFacts) -> Check {
    const ID: &str = "hsa_override";
    const FIX: &str = "`unset HSA_OVERRIDE_GFX_VERSION` (and remove it from your shell profile or \
         service unit) unless you rely on it for a GPU ROCm does not support";
    let Some(value) = facts.hsa_override.as_deref() else {
        return single(ID, CheckStatus::Pass, "not set", None);
    };
    let Some(forced) = override_arch(value) else {
        return single(
            ID,
            CheckStatus::Warn,
            "HSA_OVERRIDE_GFX_VERSION is set but is not a valid major.minor.stepping \
             version (for example 11.0.0)",
            Some(FIX.to_owned()),
        );
    };
    let arches = distinct_arches(&facts.kfd_nodes);
    match arches.iter().find(|arch| **arch != forced) {
        Some(real) => single(
            ID,
            CheckStatus::Warn,
            format!(
                "{HSA_OVERRIDE_VAR}={} makes ROCm report {forced} for the {real} GPU: \
                 hipfire then builds and runs {forced} kernels on it",
                value.trim()
            ),
            Some(FIX.to_owned()),
        ),
        None if arches.is_empty() && facts.kfd_topology_error.is_some() => single(
            ID,
            CheckStatus::Info,
            format!("set to {forced}; the KFD topology is unreadable, so it was not compared against the GPU"),
            None,
        ),
        None if arches.is_empty() => single(
            ID,
            CheckStatus::Info,
            format!("set to {forced}; no KFD GPU to compare against"),
            None,
        ),
        None => single(
            ID,
            CheckStatus::Info,
            format!("set to the GPU's own ISA ({forced}): redundant"),
            None,
        ),
    }
}

fn check_kernel_pack(facts: &RuntimeFacts) -> Option<Check> {
    const ID: &str = "kernel_pack";
    let arches = distinct_arches(&facts.kfd_nodes);
    if arches.is_empty() {
        return None;
    }
    let daemon_commit = facts
        .daemon
        .version_line
        .as_deref()
        .and_then(|line| parse_daemon_line(line).1);
    let mut rows = Rows::default();
    for arch in &arches {
        let pack = facts
            .packs
            .iter()
            .find(|pack| &pack.arch == arch)
            .and_then(|pack| pack.pack.as_ref());
        let Some(pack) = pack else {
            rows.info(format!(
                "no kernel pack for {arch}: kernels JIT-compile on first use"
            ));
            continue;
        };
        let (min, max) = (&pack.rocm_min, &pack.rocm_max_exclusive);
        let tag = &pack.tag;
        let version = facts.rocm.version.as_deref();
        match version {
            None => {
                rows.info(format!(
                    "kernel pack {tag} for {arch} supports ROCm [{min}, {max}); the installed \
                     ROCm version is unknown, so the range was not checked"
                ));
                continue;
            }
            Some(version) if !rdna_compute::kernel_pack::rocm_admitted(version, min, max) => {
                rows.fail(
                    format!(
                        "kernel pack {tag} for {arch} supports ROCm [{min}, {max}); \
                         ROCm {version} is installed"
                    ),
                    format!(
                        "`hipfire setup --source ~/.hipfire/src --compile-kernels` rebuilds \
                         kernels for ROCm {version} (needs the ROCm HIP SDK), or reinstall a \
                         ROCm in [{min}, {max})"
                    ),
                );
                continue;
            }
            Some(_) => {}
        }
        if daemon_commit.is_some_and(|commit| commits_differ(&pack.commit, commit)) {
            rows.warn(
                format!(
                    "kernel pack {tag} for {arch} was built from {} but the daemon is {}: any \
                     object whose recorded key no longer matches the daemon's kernel source, \
                     flags, toolchain or cache ABI is rejected and recompiled with hipcc on \
                     first use (without a device compiler it fails as an 'unverified kernel \
                     image')",
                    short(&pack.commit),
                    short(daemon_commit.unwrap_or_default())
                ),
                format!(
                    "`hipfire update`, or reinstall the pack that matches this daemon: \
                     `hipfire kernel-pack install --tag <tag> --arch {arch} \
                     --source ~/.hipfire/src`"
                ),
            );
            continue;
        }
        rows.pass(format!(
            "pack {tag} (ROCm [{min}, {max})) matches ROCm {}",
            version.unwrap_or("unknown")
        ));
    }
    Some(rows.finish(ID))
}

fn check_daemon(facts: &RuntimeFacts) -> Check {
    const ID: &str = "daemon";
    if !facts.daemon.found {
        return single(
            ID,
            CheckStatus::Fail,
            "hipfire-daemon is not installed",
            Some(SETUP_FIX.to_owned()),
        );
    }
    let Some(line) = facts.daemon.version_line.as_deref() else {
        return single(
            ID,
            CheckStatus::Warn,
            "daemon did not answer --version",
            Some(SETUP_FIX.to_owned()),
        );
    };
    let (version, commit) = parse_daemon_line(line);
    let cli = &facts.cli;
    let different = version != Some(cli.version.as_str())
        || commit.is_some_and(|commit| commits_differ(&cli.commit, commit));
    let shown = |commit: &str| {
        if commit_known(commit) {
            format!(" ({})", short(commit))
        } else {
            " (unknown)".to_owned()
        }
    };
    if different {
        return single(
            ID,
            CheckStatus::Warn,
            format!(
                "CLI {}{} and daemon {}{} are different builds",
                cli.version,
                shown(&cli.commit),
                version.unwrap_or("unknown"),
                shown(commit.unwrap_or(""))
            ),
            Some(
                "`hipfire update` (or re-run the installer) so the CLI and the daemon come from \
                 one build"
                    .to_owned(),
            ),
        );
    }
    let version = version.unwrap_or("unknown");
    let detail = match commit.filter(|commit| commit_known(commit)) {
        Some(commit) => format!("hipfire-daemon {version} ({})", short(commit)),
        None => format!("hipfire-daemon {version}"),
    };
    single(ID, CheckStatus::Pass, detail, None)
}

fn check_kho_scratch(facts: &RuntimeFacts) -> Option<Check> {
    const ID: &str = "kho_scratch";
    if !facts
        .kfd_nodes
        .iter()
        .any(|node| hipfire_config::is_unified_memory_arch(&node.arch))
    {
        return None;
    }
    let Some(bytes) = facts.kho_scratch_bytes else {
        return Some(single(
            ID,
            CheckStatus::Info,
            "KHO scratch unknown: /proc/meminfo or /proc/zoneinfo unreadable",
            None,
        ));
    };
    Some(match rdna_compute::uma_memory::kho_scratch_warning(bytes) {
        Some(warning) => single(
            ID,
            CheckStatus::Warn,
            warning,
            Some(
                "boot with `kho=off` on the kernel command line (docs/KNOWN-ISSUES.md)".to_owned(),
            ),
        ),
        None => single(
            ID,
            CheckStatus::Pass,
            "no significant KHO scratch reservation",
            None,
        ),
    })
}

/// Every check, in the stable documented order. Pure: no I/O.
pub(crate) fn assess(facts: &RuntimeFacts) -> Vec<Check> {
    let mut checks = vec![
        check_amdgpu_driver(facts),
        check_kfd(facts),
        check_kfd_gpus(facts),
    ];
    checks.extend(check_render_node(facts));
    checks.extend(check_firmware(facts));
    checks.push(check_rocm_root(facts));
    checks.extend(check_hip_runtime(facts));
    checks.extend(check_rocm_isa(facts));
    checks.extend(check_rocm_version_floor(facts));
    checks.push(check_hsa_override(facts));
    checks.extend(check_kernel_pack(facts));
    checks.push(check_daemon(facts));
    checks.extend(check_kho_scratch(facts));
    checks
}

// ---- output ------------------------------------------------------------------------

fn status_text(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "PASS",
        CheckStatus::Fail => "FAIL",
        CheckStatus::Info => "INFO",
        CheckStatus::Warn => "WARN",
    }
}

/// Header, one line per check with its indented `fix:`, and the result line.
pub(crate) fn render_text(checks: &[Check]) -> String {
    let mut text = String::new();
    let _ = writeln!(text, "hipfire doctor: runtime preflight (no GPU work)");
    text.push('\n');
    for check in checks {
        let _ = writeln!(
            text,
            "{} {}: {}",
            status_text(check.status),
            check.id,
            check.detail
        );
        if let Some(fix) = &check.fix {
            for (index, line) in fix.lines().enumerate() {
                let label = if index == 0 { "fix:" } else { "    " };
                let _ = writeln!(text, "       {label} {line}");
            }
        }
    }
    let failed = checks
        .iter()
        .filter(|check| check.status == CheckStatus::Fail)
        .count();
    text.push('\n');
    if failed == 0 {
        let _ = writeln!(text, "result: ok");
    } else {
        let _ = writeln!(text, "result: blocked ({failed} failed)");
    }
    text
}

#[derive(Serialize)]
struct JsonBuild<'a> {
    version: &'a str,
    commit: &'a str,
}

#[derive(Serialize)]
struct JsonAmdgpu<'a> {
    source: &'static str,
    version: Option<&'a str>,
}

#[derive(Serialize)]
struct JsonRocm<'a> {
    version: Option<&'a str>,
}

#[derive(Serialize)]
struct JsonGpu<'a> {
    index: usize,
    arch: Option<&'a str>,
    pci_id: String,
    driver: Option<&'a str>,
    firmware: BTreeMap<&'a str, &'a str>,
}

#[derive(Serialize)]
struct JsonHost<'a> {
    kernel: &'a str,
    amdgpu: JsonAmdgpu<'a>,
    rocm: JsonRocm<'a>,
    gpus: Vec<JsonGpu<'a>>,
}

#[derive(Serialize)]
struct JsonReport<'a> {
    schema: &'static str,
    hipfire: JsonBuild<'a>,
    result: &'static str,
    host: JsonHost<'a>,
    checks: &'a [Check],
}

/// The `hipfire-doctor-runtime-1` report: one pretty JSON object.
pub(crate) fn render_json(facts: &RuntimeFacts, checks: &[Check]) -> serde_json::Result<String> {
    let blocked = checks.iter().any(|check| check.status == CheckStatus::Fail);
    let report = JsonReport {
        schema: SCHEMA,
        hipfire: JsonBuild {
            version: &facts.cli.version,
            commit: short(&facts.cli.commit),
        },
        result: if blocked { "blocked" } else { "ok" },
        host: JsonHost {
            kernel: &facts.kernel,
            amdgpu: JsonAmdgpu {
                source: if facts.amdgpu_dkms_version.is_some() {
                    "dkms"
                } else {
                    "in-tree"
                },
                version: facts.amdgpu_dkms_version.as_deref(),
            },
            rocm: JsonRocm {
                version: facts.rocm.version.as_deref(),
            },
            gpus: facts
                .gpus
                .iter()
                .enumerate()
                .map(|(index, gpu)| JsonGpu {
                    index,
                    arch: kfd_arch(facts, gpu),
                    pci_id: format!("1002:{}", gpu.device_id),
                    driver: gpu.driver.as_deref(),
                    firmware: gpu
                        .firmware
                        .iter()
                        .flatten()
                        .map(|(name, hex)| (name.as_str(), hex.as_str()))
                        .collect(),
                })
                .collect(),
        },
        checks,
    };
    serde_json::to_string_pretty(&report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Appears in the fake ROCm root path; must never reach a report.
    const FAKE_HOME: &str = "/home/fakeuser";
    const FAKE_BDF: &str = "0000:c3:00.0";

    fn bdf() -> PciBdf {
        PciBdf::parse(FAKE_BDF).unwrap()
    }

    fn pack(commit: &str) -> PackInfo {
        PackInfo {
            tag: "v0.4.3".to_owned(),
            commit: commit.to_owned(),
            rocm_min: "7.2".to_owned(),
            rocm_max_exclusive: "8.0".to_owned(),
        }
    }

    /// A healthy single-GPU gfx1201 host: nothing fails or warns.
    fn healthy() -> RuntimeFacts {
        RuntimeFacts {
            kernel: "6.14.0-fake".to_owned(),
            amdgpu_module_loaded: true,
            amdgpu_dkms_version: None,
            gpus: vec![PciGpu {
                bdf: Some(bdf()),
                device_id: "7551".to_owned(),
                driver: Some("amdgpu".to_owned()),
                render: RenderNode::Present { rw: true },
                firmware: Some(vec![("mes".to_owned(), "0x86".to_owned())]),
            }],
            kfd: KfdAccess::Ready,
            kfd_nodes: vec![KfdNode {
                arch: "gfx1201".to_owned(),
                bdf: bdf(),
            }],
            rocm: RocmFacts {
                root: Some(PathBuf::from(format!("{FAKE_HOME}/rocm"))),
                ambiguous: false,
                configured_var: None,
                version: Some("7.2.0".to_owned()),
            },
            libs: Some(RuntimeLibs {
                hip: true,
                hsa: true,
            }),
            isa: vec![IsaFacts {
                arch: "gfx1201".to_owned(),
                bitcode: Bitcode::Found,
            }],
            hsa_override: None,
            packs: vec![PackFacts {
                arch: "gfx1201".to_owned(),
                pack: Some(pack("0123456789abcdef0123")),
            }],
            daemon: DaemonFacts {
                found: true,
                version_line: Some("hipfire-daemon 0.4.3 0123456789abcdef0123".to_owned()),
            },
            cli: BuildId {
                version: "0.4.3".to_owned(),
                commit: "0123456789abcdef0123".to_owned(),
            },
            kho_scratch_bytes: Some(0),
            pci_read_error: None,
            kfd_topology_error: None,
        }
    }

    fn find<'a>(checks: &'a [Check], id: &str) -> &'a Check {
        checks
            .iter()
            .find(|check| check.id == id)
            .unwrap_or_else(|| panic!("no check {id}"))
    }

    fn only(facts: &RuntimeFacts, id: &str) -> Check {
        find(&assess(facts), id).clone()
    }

    /// `id` has `status`; a Fail/Warn carries a fix and nothing private leaks.
    fn expect(facts: &RuntimeFacts, id: &str, status: CheckStatus) -> Check {
        let checks = assess(facts);
        for check in &checks {
            let text = format!("{} {}", check.detail, check.fix.as_deref().unwrap_or(""));
            for private in ["fakeuser", FAKE_BDF, "c3:00.0"] {
                assert!(
                    !text.contains(private),
                    "{} leaks {private}: {text}",
                    check.id
                );
            }
            if matches!(check.status, CheckStatus::Fail | CheckStatus::Warn) {
                assert!(
                    check.fix.is_some(),
                    "{} {:?} has no fix",
                    check.id,
                    check.status
                );
            }
        }
        let check = find(&checks, id).clone();
        assert_eq!(check.status, status, "{id}: {}", check.detail);
        check
    }

    fn fix_of(check: &Check) -> &str {
        check.fix.as_deref().expect("fix")
    }

    // ---- all-pass host -----------------------------------------------------

    #[test]
    fn healthy_host_passes_in_the_documented_order() {
        let facts = healthy();
        let checks = assess(&facts);
        let ids = checks.iter().map(|check| check.id).collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "amdgpu_driver",
                "kfd",
                "kfd_gpus",
                "render_node",
                "firmware",
                "rocm_root",
                "hip_runtime",
                "rocm_isa",
                "rocm_version_floor",
                "hsa_override",
                "kernel_pack",
                "daemon",
            ]
        );
        for check in &checks {
            let want = if check.id == "firmware" {
                CheckStatus::Info
            } else {
                CheckStatus::Pass
            };
            assert_eq!(check.status, want, "{}: {}", check.id, check.detail);
            assert_eq!(check.fix, None, "{}", check.id);
        }
        let blocked = RuntimePreflight {
            checks: checks.clone(),
            facts,
        };
        assert!(!blocked.blocked());
        assert_eq!(
            find(&checks, "amdgpu_driver").detail,
            "1 GPU(s) on amdgpu (in-tree, kernel 6.14.0-fake)"
        );
        assert_eq!(find(&checks, "kfd_gpus").detail, "GPU0 gfx1201");
        assert_eq!(find(&checks, "firmware").detail, "GPU0 gfx1201: mes 0x86");
        assert_eq!(find(&checks, "rocm_root").detail, "ROCm 7.2.0");
        assert_eq!(
            find(&checks, "rocm_isa").detail,
            "ROCm 7.2.0 has the gfx1201 device library"
        );
        assert_eq!(find(&checks, "hsa_override").detail, "not set");
        assert_eq!(
            find(&checks, "kernel_pack").detail,
            "pack v0.4.3 (ROCm [7.2, 8.0)) matches ROCm 7.2.0"
        );
        assert_eq!(
            find(&checks, "daemon").detail,
            "hipfire-daemon 0.4.3 (0123456789ab)"
        );
    }

    #[test]
    fn dkms_build_is_named_in_the_pass_detail() {
        let mut facts = healthy();
        facts.amdgpu_dkms_version = Some("6.12.12".to_owned());
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Pass);
        assert_eq!(
            check.detail,
            "1 GPU(s) on amdgpu (amdgpu-dkms 6.12.12, kernel 6.14.0-fake)"
        );
    }

    // ---- amdgpu_driver -----------------------------------------------------

    #[test]
    fn no_amd_gpu_fails() {
        let mut facts = healthy();
        facts.gpus.clear();
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert_eq!(check.detail, "no AMD GPU found on the PCI bus");
        assert!(fix_of(&check).contains("hipfire needs an AMD GPU"));
    }

    #[test]
    fn unbound_gpu_fails_with_the_journal_and_firmware_fix() {
        let mut facts = healthy();
        facts.gpus[0].driver = None;
        facts.amdgpu_module_loaded = false;
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert!(check
            .detail
            .contains("GPU0 gfx1201 (1002:7551) has no driver bound"));
        assert!(check.detail.contains("amdgpu module is not loaded"));
        let fix = fix_of(&check);
        assert!(fix.contains("amdgpu did not bind GPU0 (1002:7551) on kernel 6.14.0-fake"));
        assert!(fix.contains("journalctl -k -b | grep -iE 'amdgpu|firmware'"));
        assert!(fix.contains("linux-firmware"));
        assert!(fix.contains("amdgpu-dkms"));
    }

    #[test]
    fn foreign_drivers_get_their_own_fix() {
        let mut facts = healthy();
        facts.gpus[0].driver = Some("vfio-pci".to_owned());
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert!(check.detail.contains("vfio-pci"));
        assert!(fix_of(&check).contains("vfio-pci ids= list"));

        facts.gpus[0].driver = Some("radeon".to_owned());
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert!(fix_of(&check).contains("radeon.si_support=0"));
        assert!(fix_of(&check).contains("amdgpu.cik_support=1"));

        facts.gpus[0].driver = Some("nouveau".to_owned());
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert_eq!(
            fix_of(&check),
            "unbind nouveau from this GPU and let amdgpu bind it"
        );
    }

    // ---- kfd ---------------------------------------------------------------

    #[test]
    fn kfd_missing_and_denied_fail() {
        let mut facts = healthy();
        facts.kfd = KfdAccess::Missing;
        let check = expect(&facts, "kfd", CheckStatus::Fail);
        assert!(check.detail.starts_with("no /dev/kfd"));
        assert!(fix_of(&check).contains("CONFIG_HSA_AMD=y"));
        assert!(fix_of(&check).contains("MES/MEC"));

        facts.kfd = KfdAccess::Denied;
        let check = expect(&facts, "kfd", CheckStatus::Fail);
        assert_eq!(
            fix_of(&check),
            "`sudo usermod -aG render,video $USER`, then log out and back in"
        );
    }

    #[test]
    fn amdgpu_gpu_without_a_kfd_node_fails() {
        let mut facts = healthy();
        facts.kfd_nodes.clear();
        let check = expect(&facts, "kfd_gpus", CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "GPU0 1002:7551 is driven by amdgpu but KFD did not register it, so ROCm cannot use it"
        );
        assert!(fix_of(&check).contains("journalctl -k -b | grep -i kfd"));
    }

    #[test]
    fn kfd_gpus_lists_every_gpu_by_pci_order() {
        let mut facts = healthy();
        let second = PciBdf::parse("0000:e3:00.0").unwrap();
        facts.gpus.push(PciGpu {
            bdf: Some(second),
            device_id: "744c".to_owned(),
            driver: Some("amdgpu".to_owned()),
            render: RenderNode::Present { rw: true },
            firmware: None,
        });
        facts.kfd_nodes.push(KfdNode {
            arch: "gfx1100".to_owned(),
            bdf: second,
        });
        let check = expect(&facts, "kfd_gpus", CheckStatus::Pass);
        assert_eq!(check.detail, "GPU0 gfx1201, GPU1 gfx1100");
        let firmware = only(&facts, "firmware");
        assert!(firmware
            .detail
            .contains("GPU1 gfx1100: not exposed by this kernel"));
    }

    #[test]
    fn identical_gpus_share_one_firmware_and_render_line() {
        let mut facts = healthy();
        let second = PciBdf::parse("0000:e3:00.0").unwrap();
        facts.gpus.push(PciGpu {
            bdf: Some(second),
            ..facts.gpus[0].clone()
        });
        facts.kfd_nodes.push(KfdNode {
            arch: "gfx1201".to_owned(),
            bdf: second,
        });
        assert_eq!(
            only(&facts, "firmware").detail,
            "GPU0, GPU1 gfx1201: mes 0x86"
        );
        assert_eq!(
            expect(&facts, "render_node", CheckStatus::Pass).detail,
            "GPU0, GPU1 render nodes are readable and writable"
        );
    }

    // ---- render_node -------------------------------------------------------

    #[test]
    fn render_node_missing_and_denied_fail() {
        let mut facts = healthy();
        facts.gpus[0].render = RenderNode::Missing;
        let check = expect(&facts, "render_node", CheckStatus::Fail);
        assert_eq!(check.detail, "no render node for GPU0");
        assert!(fix_of(&check).contains("created no DRM render node"));

        facts.gpus[0].render = RenderNode::Present { rw: false };
        let check = expect(&facts, "render_node", CheckStatus::Fail);
        assert!(fix_of(&check).contains("usermod -aG render,video"));
    }

    // ---- rocm_root / hip_runtime -------------------------------------------

    #[test]
    fn missing_rocm_fails_with_the_install_guidance() {
        let mut facts = healthy();
        facts.rocm = RocmFacts::default();
        facts.libs = None;
        facts.isa.clear();
        let check = expect(&facts, "rocm_root", CheckStatus::Fail);
        assert_eq!(check.detail, "no ROCm install found");
        assert!(fix_of(&check).contains(rocm::ROCM_INSTALL_DOCS));
        let ids = assess(&facts).iter().map(|c| c.id).collect::<Vec<_>>();
        assert!(!ids.contains(&"hip_runtime") && !ids.contains(&"rocm_isa"));
    }

    #[test]
    fn a_configured_but_unusable_root_names_the_variable() {
        let mut facts = healthy();
        facts.rocm = RocmFacts {
            configured_var: Some("HIPFIRE_ROCM_PATH"),
            ..RocmFacts::default()
        };
        let check = expect(&facts, "rocm_root", CheckStatus::Fail);
        assert!(check.detail.starts_with("HIPFIRE_ROCM_PATH is set"));
        assert!(fix_of(&check).contains("unset it to auto-detect"));
    }

    #[test]
    fn ambiguous_rocm_roots_fail() {
        let mut facts = healthy();
        facts.rocm = RocmFacts {
            ambiguous: true,
            ..RocmFacts::default()
        };
        let check = expect(&facts, "rocm_root", CheckStatus::Fail);
        assert!(check.detail.contains("none is selected"));
        assert!(fix_of(&check).contains("export HIPFIRE_ROCM_PATH=/opt/rocm-<version>"));
    }

    #[test]
    fn missing_runtime_libraries_are_listed() {
        let mut facts = healthy();
        facts.libs = Some(RuntimeLibs {
            hip: false,
            hsa: true,
        });
        let check = expect(&facts, "hip_runtime", CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "ROCm 7.2.0 lacks the HIP runtime (libamdhip64)"
        );
        assert!(fix_of(&check).contains("ROCm 7.2.0"));

        facts.libs = Some(RuntimeLibs {
            hip: false,
            hsa: false,
        });
        let check = expect(&facts, "hip_runtime", CheckStatus::Fail);
        assert!(check.detail.contains("libamdhip64") && check.detail.contains("libhsa-runtime64"));
    }

    // ---- rocm_isa ----------------------------------------------------------

    #[test]
    fn runtime_only_rocm_warns_about_the_missing_compiler_libraries() {
        let mut facts = healthy();
        facts.isa[0].bitcode = Bitcode::NoBitcodeDir;
        let check = expect(&facts, "rocm_isa", CheckStatus::Warn);
        assert!(check.detail.starts_with("no device compiler libraries"));
        assert!(fix_of(&check).contains("hipfire kernel-pack install --tag <tag> --arch gfx1201"));
    }

    #[test]
    fn rocm_that_predates_the_gpu_fails_without_a_pack_and_warns_with_one() {
        let mut facts = healthy();
        facts.isa[0].bitcode = Bitcode::ArchMissing;
        facts.packs.clear();
        let check = expect(&facts, "rocm_isa", CheckStatus::Fail);
        assert!(check.detail.contains("predates the GPU"));
        assert!(fix_of(&check).contains("ROCm 7.2.0 does not know gfx1201"));
        assert!(fix_of(&check).contains("6.4 for gfx1200/gfx1201"));

        // A pack that does not admit this ROCm does not rescue it.
        facts.packs = vec![PackFacts {
            arch: "gfx1201".to_owned(),
            pack: Some(PackInfo {
                rocm_min: "10.1".to_owned(),
                rocm_max_exclusive: "11.0".to_owned(),
                ..pack("0123456789abcdef0123")
            }),
        }];
        expect(&facts, "rocm_isa", CheckStatus::Fail);

        facts.packs = healthy().packs;
        let check = expect(&facts, "rocm_isa", CheckStatus::Warn);
        assert!(check
            .detail
            .contains("kernel pack still runs registry kernels"));
    }

    // ---- rocm_version_floor ------------------------------------------------

    #[test]
    fn rocm_below_the_documented_floor_warns_never_fails() {
        let mut facts = healthy();
        facts.rocm.version = Some("6.3.1".to_owned());
        let check = expect(&facts, "rocm_version_floor", CheckStatus::Warn);
        assert!(check.detail.contains("minimum 6.4 for gfx1201"));
        assert_eq!(
            fix_of(&check),
            "install ROCm 6.4 or newer (hipfire's documented minimum for gfx1201)"
        );
    }

    #[test]
    fn floors_follow_the_arch_and_compare_numerically() {
        assert_eq!(documented_floor("gfx1200"), "6.4");
        assert_eq!(documented_floor("gfx1151"), "7.2");
        assert_eq!(documented_floor("gfx1152"), "7.2");
        assert_eq!(documented_floor("gfx90a"), "6.0");
        // 10.1 > 7.2 although "10.1" < "7.2" lexically.
        let mut facts = healthy();
        facts.kfd_nodes[0].arch = "gfx1151".to_owned();
        facts.rocm.version = Some("10.1.0".to_owned());
        expect(&facts, "rocm_version_floor", CheckStatus::Pass);
        facts.rocm.version = Some("7.1.9".to_owned());
        expect(&facts, "rocm_version_floor", CheckStatus::Warn);
    }

    #[test]
    fn unknown_rocm_version_is_info() {
        let mut facts = healthy();
        facts.rocm.version = None;
        let check = expect(&facts, "rocm_version_floor", CheckStatus::Info);
        assert_eq!(check.detail, "ROCm version unknown (no .info/version)");
    }

    // ---- hsa_override ------------------------------------------------------

    #[test]
    fn hsa_override_cases() {
        let mut facts = healthy();
        expect(&facts, "hsa_override", CheckStatus::Pass);

        facts.hsa_override = Some("12.0.1".to_owned());
        let check = expect(&facts, "hsa_override", CheckStatus::Info);
        assert_eq!(
            check.detail,
            "set to the GPU's own ISA (gfx1201): redundant"
        );

        facts.hsa_override = Some("11.0.0".to_owned());
        let check = expect(&facts, "hsa_override", CheckStatus::Warn);
        assert_eq!(
            check.detail,
            "HSA_OVERRIDE_GFX_VERSION=11.0.0 makes ROCm report gfx1100 for the gfx1201 GPU: \
             hipfire then builds and runs gfx1100 kernels on it"
        );
        assert!(fix_of(&check).starts_with("`unset HSA_OVERRIDE_GFX_VERSION`"));

        facts.hsa_override = Some("not-a-version /home/fakeuser".to_owned());
        let check = expect(&facts, "hsa_override", CheckStatus::Warn);
        assert!(check.detail.contains("not a valid major.minor.stepping"));
        assert!(fix_of(&check).starts_with("`unset HSA_OVERRIDE_GFX_VERSION`"));
    }

    // ---- kernel_pack -------------------------------------------------------

    #[test]
    fn missing_pack_is_info() {
        let mut facts = healthy();
        facts.packs[0].pack = None;
        let check = expect(&facts, "kernel_pack", CheckStatus::Info);
        assert_eq!(
            check.detail,
            "no kernel pack for gfx1201: kernels JIT-compile on first use"
        );
    }

    #[test]
    fn pack_outside_the_rocm_range_fails() {
        let mut facts = healthy();
        facts.rocm.version = Some("8.1.0".to_owned());
        let check = expect(&facts, "kernel_pack", CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "kernel pack v0.4.3 for gfx1201 supports ROCm [7.2, 8.0); ROCm 8.1.0 is installed"
        );
        assert!(fix_of(&check).contains("hipfire setup --source ~/.hipfire/src --compile-kernels"));
        assert!(fix_of(&check).contains("reinstall a ROCm in [7.2, 8.0)"));
    }

    #[test]
    fn pack_from_another_commit_warns_with_the_daemon_consequence() {
        let mut facts = healthy();
        facts.packs[0].pack = Some(pack("fedcba9876543210ffff"));
        let check = expect(&facts, "kernel_pack", CheckStatus::Warn);
        assert!(check
            .detail
            .contains("built from fedcba987654 but the daemon is 0123456789ab"));
        assert!(check.detail.contains("recompiled with hipcc on first use"));
        assert!(check.detail.contains("unverified kernel image"));
        assert!(fix_of(&check).starts_with("`hipfire update`"));

        // A short commit compares by prefix.
        facts.packs[0].pack = Some(pack("0123456789ab"));
        expect(&facts, "kernel_pack", CheckStatus::Pass);
    }

    // ---- daemon ------------------------------------------------------------

    #[test]
    fn daemon_cases() {
        let mut facts = healthy();
        facts.daemon.found = false;
        facts.daemon.version_line = None;
        let check = expect(&facts, "daemon", CheckStatus::Fail);
        assert_eq!(fix_of(&check), "`hipfire setup --source ~/.hipfire/src`");

        facts.daemon.found = true;
        let check = expect(&facts, "daemon", CheckStatus::Warn);
        assert_eq!(check.detail, "daemon did not answer --version");
        assert_eq!(fix_of(&check), "`hipfire setup --source ~/.hipfire/src`");

        facts.daemon.version_line = Some("hipfire-daemon 0.4.2 0123456789abcdef0123".to_owned());
        let check = expect(&facts, "daemon", CheckStatus::Warn);
        assert_eq!(
            check.detail,
            "CLI 0.4.3 (0123456789ab) and daemon 0.4.2 (0123456789ab) are different builds"
        );
        assert!(fix_of(&check).starts_with("`hipfire update`"));

        facts.daemon.version_line = Some("hipfire-daemon 0.4.3 ffffffffffffffffffff".to_owned());
        expect(&facts, "daemon", CheckStatus::Warn);

        // An unknown commit on either side is not a mismatch.
        facts.cli.commit = "unknown".to_owned();
        expect(&facts, "daemon", CheckStatus::Pass);
        facts.daemon.version_line = Some("hipfire-daemon 0.4.3".to_owned());
        let check = expect(&facts, "daemon", CheckStatus::Pass);
        assert_eq!(check.detail, "hipfire-daemon 0.4.3");
    }

    // ---- kho_scratch -------------------------------------------------------

    #[test]
    fn kho_scratch_is_checked_only_on_unified_memory_hosts() {
        let mut facts = healthy();
        facts.kho_scratch_bytes = Some(8 << 30);
        assert!(!assess(&facts).iter().any(|check| check.id == "kho_scratch"));

        facts.kfd_nodes[0].arch = "gfx1151".to_owned();
        let check = expect(&facts, "kho_scratch", CheckStatus::Warn);
        assert!(check.detail.contains("KHO scratch reserves 8.0 GiB"));
        assert!(fix_of(&check).contains("kho=off"));

        facts.kho_scratch_bytes = Some(0);
        expect(&facts, "kho_scratch", CheckStatus::Pass);
    }

    #[test]
    fn unreadable_kho_sources_are_unknown_not_a_pass() {
        let mut facts = healthy();
        facts.kfd_nodes[0].arch = "gfx1151".to_owned();
        facts.kho_scratch_bytes = None;
        let check = expect(&facts, "kho_scratch", CheckStatus::Info);
        assert_eq!(
            check.detail,
            "KHO scratch unknown: /proc/meminfo or /proc/zoneinfo unreadable"
        );
        assert!(check.fix.is_none());
        facts.kho_scratch_bytes = Some(0);
        expect(&facts, "kho_scratch", CheckStatus::Pass);
    }

    #[test]
    fn unreadable_pci_list_fails_without_claiming_no_gpu() {
        let mut facts = healthy();
        facts.gpus.clear();
        facts.pci_read_error = Some("permission denied".to_owned());
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "cannot read the PCI device list at /sys/bus/pci/devices (permission denied)"
        );
        assert!(fix_of(&check).contains("check that /sys is mounted and readable"));
        assert!(!check.detail.contains("no AMD GPU"));
        assert!(!fix_of(&check).contains("needs an AMD GPU"));

        // A genuinely empty bus keeps its result.
        facts.pci_read_error = None;
        let check = expect(&facts, "amdgpu_driver", CheckStatus::Fail);
        assert_eq!(check.detail, "no AMD GPU found on the PCI bus");
    }

    #[test]
    fn unreadable_kfd_topology_fails_without_claiming_unregistered() {
        let mut facts = healthy();
        facts.kfd_nodes.clear();
        facts.isa.clear();
        facts.packs.clear();
        facts.hsa_override = Some("11.0.0".to_owned());
        facts.kfd_topology_error = Some("not a directory".to_owned());
        let check = expect(&facts, "kfd_gpus", CheckStatus::Fail);
        assert_eq!(
            check.detail,
            "cannot read the KFD topology at /sys/class/kfd/kfd/topology/nodes (not a directory)"
        );
        assert!(fix_of(&check).contains("check /sys/class/kfd/kfd/topology is readable"));
        assert!(fix_of(&check).contains("modprobe -r amdgpu"));
        assert!(!check.detail.contains("did not register"));
        assert!(!fix_of(&check).contains("linux-firmware"));

        // Arch-dependent checks are unknown, never a pass or a "no GPU" claim.
        let checks = assess(&facts);
        for id in [
            "render_node",
            "rocm_isa",
            "rocm_version_floor",
            "kernel_pack",
            "kho_scratch",
        ] {
            assert!(
                !checks.iter().any(|check| check.id == id),
                "{id} was reported"
            );
        }
        let hsa = find(&checks, "hsa_override");
        assert_eq!(hsa.status, CheckStatus::Info);
        assert!(hsa.detail.contains("KFD topology is unreadable"));
        assert!(!hsa.detail.contains("no KFD GPU"));
    }

    // ---- aggregation and output --------------------------------------------

    #[test]
    fn per_gpu_findings_merge_with_distinct_fixes() {
        let mut facts = healthy();
        let second = PciBdf::parse("0000:e3:00.0").unwrap();
        let mut gpu = facts.gpus[0].clone();
        gpu.bdf = Some(second);
        gpu.render = RenderNode::Present { rw: false };
        facts.gpus[0].render = RenderNode::Present { rw: false };
        facts.gpus.push(gpu);
        facts.kfd_nodes.push(KfdNode {
            arch: "gfx1201".to_owned(),
            bdf: second,
        });
        let check = expect(&facts, "render_node", CheckStatus::Fail);
        assert!(check.detail.contains("GPU0") && check.detail.contains("GPU1"));
        // The same fix twice is printed once.
        assert_eq!(fix_of(&check).lines().count(), 1);
    }

    #[test]
    fn text_has_header_fix_lines_and_result() {
        let mut facts = healthy();
        facts.kfd = KfdAccess::Missing;
        facts.hsa_override = Some("11.0.0".to_owned());
        let text = render_text(&assess(&facts));
        assert!(text.starts_with("hipfire doctor: runtime preflight (no GPU work)\n"));
        assert!(text.contains("FAIL kfd: no /dev/kfd"));
        assert!(text.contains("\n       fix: amdgpu started without KFD"));
        assert!(text.contains("WARN hsa_override: HSA_OVERRIDE_GFX_VERSION=11.0.0"));
        assert!(text.contains("PASS amdgpu_driver: "));
        assert!(text.contains("INFO firmware: "));
        assert!(text.ends_with("result: blocked (1 failed)\n"));

        let ok = render_text(&assess(&healthy()));
        assert!(ok.ends_with("result: ok\n"));
        // Warnings alone do not block.
        let mut warned = healthy();
        warned.hsa_override = Some("11.0.0".to_owned());
        assert!(render_text(&assess(&warned)).ends_with("result: ok\n"));
    }

    #[test]
    fn json_follows_the_schema_and_leaks_nothing() {
        let mut facts = healthy();
        facts.amdgpu_dkms_version = Some("6.12.12".to_owned());
        facts.hsa_override = Some("11.0.0".to_owned());
        let checks = assess(&facts);
        let text = render_json(&facts, &checks).unwrap();
        for private in ["fakeuser", FAKE_BDF, "c3:00.0"] {
            assert!(!text.contains(private), "json leaks {private}");
        }
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["schema"], "hipfire-doctor-runtime-1");
        assert_eq!(json["hipfire"]["version"], "0.4.3");
        assert_eq!(json["hipfire"]["commit"], "0123456789ab");
        assert_eq!(json["result"], "ok");
        assert_eq!(json["host"]["kernel"], "6.14.0-fake");
        assert_eq!(json["host"]["amdgpu"]["source"], "dkms");
        assert_eq!(json["host"]["amdgpu"]["version"], "6.12.12");
        assert_eq!(json["host"]["rocm"]["version"], "7.2.0");
        let gpu = &json["host"]["gpus"][0];
        assert_eq!(gpu["index"], 0);
        assert_eq!(gpu["arch"], "gfx1201");
        assert_eq!(gpu["pci_id"], "1002:7551");
        assert_eq!(gpu["driver"], "amdgpu");
        assert_eq!(gpu["firmware"]["mes"], "0x86");
        let listed = json["checks"].as_array().unwrap();
        assert_eq!(listed.len(), checks.len());
        let hsa = listed.iter().find(|c| c["id"] == "hsa_override").unwrap();
        assert_eq!(hsa["status"], "warn");
        assert!(hsa["fix"].as_str().unwrap().contains("unset"));
        assert!(listed[0].get("fix").is_none());

        facts.kfd = KfdAccess::Missing;
        let blocked = render_json(&facts, &assess(&facts)).unwrap();
        let json: serde_json::Value = serde_json::from_str(&blocked).unwrap();
        assert_eq!(json["result"], "blocked");

        facts.amdgpu_dkms_version = None;
        let json: serde_json::Value =
            serde_json::from_str(&render_json(&facts, &assess(&facts)).unwrap()).unwrap();
        assert_eq!(json["host"]["amdgpu"]["source"], "in-tree");
        assert!(json["host"]["amdgpu"]["version"].is_null());
    }

    // ---- gather over a fake root --------------------------------------------

    static NEXT: AtomicU32 = AtomicU32::new(0);

    /// A temp directory standing in for `/`, removed on drop.
    struct FakeRoot(PathBuf);

    impl FakeRoot {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "hipfire-doctor-runtime-fake-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        fn path(&self, relative: &str) -> PathBuf {
            self.0.join(relative)
        }

        fn put(&self, relative: &str, text: &str) {
            let path = self.path(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }

        fn dir(&self, relative: &str) {
            std::fs::create_dir_all(self.path(relative)).unwrap();
        }
    }

    impl Drop for FakeRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const PACK_MANIFEST: &str = r#"{"format":1,"tag":"v0.4.3","commit":"0123456789abcdef0123","arch":"gfx1201","modules":1,"kernel_cache_abi":1,"extra_flags":"","toolchain_id":"hipcc 7.2","hip_version":"7.2","rocm_version":"7.2.0","code_object_version":6,"rocm_min":"7.2","rocm_max_exclusive":"8.0"}"#;

    /// Two AMD GPUs (03:00.0 bound to amdgpu with a KFD node and firmware,
    /// c3:00.0 unbound), a non-AMD device, an AMD non-display function, DKMS
    /// version, osrelease, `/dev/kfd`, a ROCm tree and an installed pack.
    fn fake_host() -> FakeRoot {
        let root = FakeRoot::new();
        let pci = "sys/bus/pci/devices";
        root.put(&format!("{pci}/0000:03:00.0/vendor"), "0x1002\n");
        root.put(&format!("{pci}/0000:03:00.0/class"), "0x030000\n");
        root.put(&format!("{pci}/0000:03:00.0/device"), "0x7551\n");
        root.dir(&format!("{pci}/0000:03:00.0/drm/renderD128"));
        root.put(
            &format!("{pci}/0000:03:00.0/fw_version/mes_fw_version"),
            "0x00000086\n",
        );
        root.put(
            &format!("{pci}/0000:03:00.0/fw_version/mec_fw_version"),
            "0x00000012\n",
        );
        symlink(
            "../../../../bus/pci/drivers/amdgpu",
            root.path(&format!("{pci}/0000:03:00.0/driver")),
        )
        .unwrap();
        root.put(&format!("{pci}/0000:c3:00.0/vendor"), "0x1002\n");
        root.put(&format!("{pci}/0000:c3:00.0/class"), "0x038000\n");
        root.put(&format!("{pci}/0000:c3:00.0/device"), "0x744C\n");
        root.put(&format!("{pci}/0000:00:00.0/vendor"), "0x8086\n");
        root.put(&format!("{pci}/0000:00:00.0/class"), "0x060000\n");
        root.put(&format!("{pci}/0000:00:08.1/vendor"), "0x1002\n");
        root.put(&format!("{pci}/0000:00:08.1/class"), "0x048000\n");
        root.put("sys/module/amdgpu/version", "6.12.12\n");
        root.put("proc/sys/kernel/osrelease", "6.14.0-fake\n");
        root.put("proc/meminfo", "MemAvailable:  1024 kB\nCmaTotal:  0 kB\n");
        root.put("proc/zoneinfo", "Node 0, zone Normal\n  cma 0\n");
        root.put("dev/kfd", "");
        root.put("dev/dri/renderD128", "");
        let nodes = "sys/class/kfd/kfd/topology/nodes";
        root.put(
            &format!("{nodes}/0/properties"),
            "simd_count 0\nlocation_id 0\n",
        );
        // location_id 0x300 -> bus 03, device 0, function 0.
        root.put(
            &format!("{nodes}/1/properties"),
            "simd_count 64\nlocation_id 768\ndomain 0\ngfx_target_version 120001\n",
        );
        root.put("opt/rocm/.info/version", "7.2.0\n");
        root.put("opt/rocm/lib/libamdhip64.so", "");
        root.put("opt/rocm/lib/libhsa-runtime64.so.1", "");
        root.put(
            "opt/rocm/lib/llvm/amdgcn/bitcode/oclc_isa_version_1201.bc",
            "",
        );
        root.put("bin/daemon", "");
        root.put("bin/kernels/compiled/gfx1201.manifest.json", PACK_MANIFEST);
        root
    }

    fn gather_fake(root: &FakeRoot, env_value: Option<&str>) -> RuntimeFacts {
        let env = |key: &str| {
            (key == "HSA_OVERRIDE_GFX_VERSION")
                .then(|| env_value.map(str::to_owned))
                .flatten()
        };
        let daemon_version =
            |_: &Path| Some("hipfire-daemon 0.4.3 0123456789abcdef0123".to_owned());
        let is_char_device = |_: &Path| true;
        let access_rw = |path: &Path| path.exists();
        let rocm_root = root.path("opt/rocm");
        let inputs = Inputs {
            env: &env,
            daemon: Some(root.path("bin/daemon")),
            daemon_version: &daemon_version,
            rocm: RocmFacts {
                version: rocm::version_for_root(&rocm_root),
                root: Some(rocm_root),
                ambiguous: false,
                configured_var: None,
            },
            is_char_device: &is_char_device,
            access_rw: &access_rw,
            cli: BuildId {
                version: "0.4.3".to_owned(),
                commit: "0123456789abcdef0123".to_owned(),
            },
        };
        gather(&root.0, &inputs)
    }

    #[test]
    fn gather_reads_a_fake_root() {
        let root = fake_host();
        let facts = gather_fake(&root, Some(" 12.0.1 "));
        assert_eq!(facts.kernel, "6.14.0-fake");
        assert!(facts.amdgpu_module_loaded);
        assert_eq!(facts.amdgpu_dkms_version.as_deref(), Some("6.12.12"));
        // Only the two AMD display devices, in PCI order.
        assert_eq!(facts.gpus.len(), 2);
        let bound = &facts.gpus[0];
        assert_eq!(bound.bdf, PciBdf::parse("0000:03:00.0"));
        assert_eq!(bound.device_id, "7551");
        assert_eq!(bound.driver.as_deref(), Some("amdgpu"));
        assert_eq!(bound.render, RenderNode::Present { rw: true });
        assert_eq!(
            bound.firmware,
            Some(vec![
                ("mec".to_owned(), "0x12".to_owned()),
                ("mes".to_owned(), "0x86".to_owned()),
            ])
        );
        let unbound = &facts.gpus[1];
        assert_eq!(unbound.device_id, "744c");
        assert_eq!(unbound.driver, None);
        assert_eq!(unbound.render, RenderNode::Missing);
        assert_eq!(unbound.firmware, None);
        assert_eq!(facts.kfd, KfdAccess::Ready);
        assert_eq!(
            facts.kfd_nodes,
            vec![KfdNode {
                arch: "gfx1201".to_owned(),
                bdf: PciBdf::parse("0000:03:00.0").unwrap(),
            }]
        );
        assert_eq!(
            facts.libs,
            Some(RuntimeLibs {
                hip: true,
                hsa: true
            })
        );
        assert_eq!(
            facts.isa,
            vec![IsaFacts {
                arch: "gfx1201".to_owned(),
                bitcode: Bitcode::Found
            }]
        );
        assert_eq!(facts.hsa_override.as_deref(), Some("12.0.1"));
        assert_eq!(
            facts.packs,
            vec![PackFacts {
                arch: "gfx1201".to_owned(),
                pack: Some(pack("0123456789abcdef0123")),
            }]
        );
        assert_eq!(
            facts.daemon,
            DaemonFacts {
                found: true,
                version_line: Some("hipfire-daemon 0.4.3 0123456789abcdef0123".to_owned()),
            }
        );

        // The unbound GPU is the only blocker.
        let checks = assess(&facts);
        let failing = checks
            .iter()
            .filter(|check| check.status == CheckStatus::Fail)
            .map(|check| check.id)
            .collect::<Vec<_>>();
        assert_eq!(failing, ["amdgpu_driver"]);
        let check = find(&checks, "amdgpu_driver");
        assert!(check
            .detail
            .contains("GPU1 (1002:744c) has no driver bound"));
        assert!(fix_of(check).contains("amdgpu did not bind GPU1 (1002:744c)"));
    }

    #[test]
    fn gather_without_any_device_is_empty_not_an_error() {
        let root = FakeRoot::new();
        let env = |_: &str| -> Option<String> { None };
        let daemon_version = |_: &Path| -> Option<String> { None };
        let never = |_: &Path| false;
        let inputs = Inputs {
            env: &env,
            daemon: None,
            daemon_version: &daemon_version,
            rocm: RocmFacts::default(),
            is_char_device: &never,
            access_rw: &never,
            cli: BuildId::default(),
        };
        let facts = gather(&root.0, &inputs);
        assert!(facts.gpus.is_empty() && facts.kfd_nodes.is_empty());
        assert_eq!(facts.kfd, KfdAccess::Missing);
        assert!(facts.libs.is_none() && facts.isa.is_empty() && facts.packs.is_empty());
        assert_eq!(facts.daemon, DaemonFacts::default());
        let checks = assess(&facts);
        assert_eq!(find(&checks, "amdgpu_driver").status, CheckStatus::Fail);
        assert_eq!(find(&checks, "kfd").status, CheckStatus::Fail);
        assert_eq!(find(&checks, "rocm_root").status, CheckStatus::Fail);
        assert_eq!(find(&checks, "daemon").status, CheckStatus::Fail);
    }

    #[test]
    fn kfd_that_is_not_a_char_device_or_not_writable_is_classified() {
        let root = fake_host();
        let env = |_: &str| -> Option<String> { None };
        let daemon_version = |_: &Path| -> Option<String> { None };
        let make = |char_device: bool, writable: bool| {
            let is_char_device = move |_: &Path| char_device;
            let access_rw = move |_: &Path| writable;
            let inputs = Inputs {
                env: &env,
                daemon: None,
                daemon_version: &daemon_version,
                rocm: RocmFacts::default(),
                is_char_device: &is_char_device,
                access_rw: &access_rw,
                cli: BuildId::default(),
            };
            gather(&root.0, &inputs)
        };
        assert_eq!(make(false, true).kfd, KfdAccess::Missing);
        assert_eq!(make(true, false).kfd, KfdAccess::Denied);
        assert_eq!(make(true, true).kfd, KfdAccess::Ready);
        // Render node access follows the same predicate.
        assert_eq!(
            make(true, false).gpus[0].render,
            RenderNode::Present { rw: false }
        );
    }

    /// `gather` over `root` with no daemon, no ROCm and no device nodes.
    fn gather_bare(root: &FakeRoot) -> RuntimeFacts {
        let env = |_: &str| -> Option<String> { None };
        let daemon_version = |_: &Path| -> Option<String> { None };
        let never = |_: &Path| false;
        let inputs = Inputs {
            env: &env,
            daemon: None,
            daemon_version: &daemon_version,
            rocm: RocmFacts::default(),
            is_char_device: &never,
            access_rw: &never,
            cli: BuildId::default(),
        };
        gather(&root.0, &inputs)
    }

    #[test]
    fn gather_reports_an_unreadable_pci_list_not_a_missing_gpu() {
        let root = FakeRoot::new();
        // A regular file where the directory belongs: read_dir fails.
        root.put("sys/bus/pci/devices", "");
        let facts = gather_bare(&root);
        assert!(facts.gpus.is_empty());
        assert!(facts
            .pci_read_error
            .as_deref()
            .is_some_and(|kind| !kind.is_empty()));
        let check = only(&facts, "amdgpu_driver");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check
            .detail
            .starts_with("cannot read the PCI device list at /sys/bus/pci/devices ("));
        assert!(!check.detail.contains("no AMD GPU"));
        assert!(fix_of(&check).contains("check that /sys is mounted and readable"));
        let root_text = root.0.to_string_lossy().into_owned();
        assert!(!check.detail.contains(&root_text));
    }

    #[test]
    fn gather_reports_an_unreadable_kfd_topology_not_an_unregistered_gpu() {
        let host = fake_host();
        let nodes = host.path("sys/class/kfd/kfd/topology/nodes");
        std::fs::remove_dir_all(&nodes).unwrap();
        std::fs::write(&nodes, "").unwrap();
        let facts = gather_fake(&host, None);
        assert!(facts.kfd_nodes.is_empty());
        assert!(facts
            .kfd_topology_error
            .as_deref()
            .is_some_and(|kind| !kind.is_empty()));
        assert!(facts.isa.is_empty() && facts.packs.is_empty());
        let checks = assess(&facts);
        let check = find(&checks, "kfd_gpus");
        assert_eq!(check.status, CheckStatus::Fail);
        assert!(check
            .detail
            .starts_with("cannot read the KFD topology at /sys/class/kfd/kfd/topology/nodes ("));
        assert!(!check.detail.contains("did not register"));
        assert!(fix_of(check).contains("modprobe -r amdgpu"));
        let root_text = host.0.to_string_lossy().into_owned();
        assert!(!check.detail.contains(&root_text));
        for id in [
            "render_node",
            "rocm_isa",
            "rocm_version_floor",
            "kernel_pack",
            "kho_scratch",
        ] {
            assert!(
                !checks.iter().any(|check| check.id == id),
                "{id} was reported"
            );
        }
    }

    #[test]
    fn gather_keeps_known_empty_and_known_absent_results() {
        // Absent PCI and KFD trees: known absence, no read error.
        let facts = gather_bare(&FakeRoot::new());
        assert_eq!(facts.pci_read_error, None);
        assert_eq!(facts.kfd_topology_error, None);
        assert_eq!(
            only(&facts, "amdgpu_driver").detail,
            "no AMD GPU found on the PCI bus"
        );

        // Empty (listed) PCI bus: the same result.
        let root = FakeRoot::new();
        root.dir("sys/bus/pci/devices");
        let facts = gather_bare(&root);
        assert_eq!(facts.pci_read_error, None);
        assert_eq!(
            only(&facts, "amdgpu_driver").detail,
            "no AMD GPU found on the PCI bus"
        );

        // Empty (listed) KFD topology next to a bound GPU: KFD did not register it.
        let host = fake_host();
        let nodes = host.path("sys/class/kfd/kfd/topology/nodes");
        std::fs::remove_dir_all(&nodes).unwrap();
        std::fs::create_dir_all(&nodes).unwrap();
        let facts = gather_fake(&host, None);
        assert_eq!(facts.kfd_topology_error, None);
        assert!(facts.kfd_nodes.is_empty());
        assert!(only(&facts, "kfd_gpus")
            .detail
            .contains("KFD did not register"));

        // A readable host reports known zero scratch.
        let facts = gather_fake(&fake_host(), None);
        assert_eq!(facts.pci_read_error, None);
        assert_eq!(facts.kfd_topology_error, None);
        assert_eq!(facts.kho_scratch_bytes, Some(0));
    }

    #[test]
    fn gather_distinguishes_unreadable_kho_sources_from_zero_and_nonzero() {
        // Unreadable: neither file, or only one of them.
        let root = FakeRoot::new();
        assert_eq!(gather_bare(&root).kho_scratch_bytes, None);
        root.put("proc/meminfo", "CmaTotal:  0 kB\n");
        assert_eq!(gather_bare(&root).kho_scratch_bytes, None);

        // Readable and empty: known zero.
        let root = FakeRoot::new();
        root.put("proc/meminfo", "");
        root.put("proc/zoneinfo", "");
        assert_eq!(gather_bare(&root).kho_scratch_bytes, Some(0));

        // Readable with an explicit zero reservation: known zero.
        let root = FakeRoot::new();
        root.put("proc/meminfo", "CmaTotal:  0 kB\n");
        root.put("proc/zoneinfo", "Node 0, zone Normal\n  cma 0\n");
        assert_eq!(gather_bare(&root).kho_scratch_bytes, Some(0));

        // Readable with scratch outside any registered CMA area.
        root.put("proc/zoneinfo", "Node 0, zone Normal\n  cma 524288\n");
        assert_eq!(gather_bare(&root).kho_scratch_bytes, Some(2 << 30));
    }

    // ---- ISA bitcode search ---------------------------------------------------

    #[test]
    fn bitcode_is_found_in_the_core_versioned_layout() {
        let root = FakeRoot::new();
        let layout = "core-10.1/lib/llvm/amdgcn/bitcode";
        root.dir(layout);
        // A runtime-only-looking sibling must not matter.
        root.dir("lib");
        assert_eq!(bitcode_state(&root.0, "gfx1201"), Bitcode::ArchMissing);
        root.put(&format!("{layout}/oclc_isa_version_1201.bc"), "");
        root.put(&format!("{layout}/oclc_isa_version_90a.bc"), "");
        assert_eq!(bitcode_state(&root.0, "gfx1201"), Bitcode::Found);
        assert_eq!(bitcode_state(&root.0, "gfx90a"), Bitcode::Found);
        assert_eq!(bitcode_state(&root.0, "gfx1100"), Bitcode::ArchMissing);
    }

    #[test]
    fn bitcode_is_found_in_the_flat_amdgcn_layout() {
        let root = FakeRoot::new();
        assert_eq!(bitcode_state(&root.0, "gfx1201"), Bitcode::NoBitcodeDir);
        root.put("amdgcn/bitcode/oclc_isa_version_1201.bc", "");
        assert_eq!(bitcode_state(&root.0, "gfx1201"), Bitcode::Found);
        assert_eq!(bitcode_state(&root.0, "gfx942"), Bitcode::ArchMissing);
    }

    #[test]
    fn bitcode_is_found_in_the_other_documented_layouts() {
        for layout in [
            "lib/llvm/amdgcn/bitcode",
            "llvm/amdgcn/bitcode",
            "lib/llvm/lib/clang/20/lib/amdgcn/bitcode",
        ] {
            let root = FakeRoot::new();
            root.put(&format!("{layout}/oclc_isa_version_1100.bc"), "");
            assert_eq!(
                bitcode_state(&root.0, "gfx1100"),
                Bitcode::Found,
                "{layout}"
            );
        }
    }

    #[test]
    fn compact_hex_strips_leading_zeros_only_for_hex() {
        assert_eq!(compact_hex("0x00000086"), "0x86");
        assert_eq!(compact_hex("0x00000000"), "0x0");
        assert_eq!(compact_hex("0x1a2b"), "0x1a2b");
        assert_eq!(compact_hex("86"), "86");
        assert_eq!(compact_hex("0xzz"), "0xzz");
    }
}
