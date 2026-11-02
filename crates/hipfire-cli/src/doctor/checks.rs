// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free fitness checks for the Strix Halo NPU trunk route.
//!
//! Every fact comes through [`HostProbe`]: file facts are read under
//! [`HostProbe::root`] (a fake sysfs/proc/dev root in tests), everything else
//! (rlimits, capabilities, access(2), the GPU lock, `sudo -n`, the daemon, local
//! models) is a probe method. Nothing here opens the GPU or the NPU.

use super::power;
use super::report::{Check, CheckStatus, Fclk, Host, Iommu, Memlock, Model};
use hipfire_arch_qwen4::npu_trunk::{self, fclk};
use hipfire_config::devices::{enumerate_gpus, GpuDevice};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The registry model the built-in calibrations are measured on.
pub(crate) const FLASH_NEXT_TAG: &str = "qwen3.8:flash-next";
const STRIX_HALO_ARCH: &str = "gfx1151";

/// What the calibration and soak runs need, produced only when nothing blocks.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Target {
    pub model_path: PathBuf,
    pub model_tag: Option<String>,
    pub daemon: PathBuf,
    /// PCI address of the Halo GPU. Used to find sysfs/lock files; never reported.
    pub gpu_pci: String,
}

/// A local model file, preferred candidates first.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LocalModelFile {
    pub path: PathBuf,
    pub tag: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    ReadWrite,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RocmState {
    Ready,
    NoRoot,
    NoRuntime,
}

/// Result of the non-blocking GPU lock probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum GpuLock {
    /// No lock file, or the lock was taken and released at once.
    Free,
    Busy {
        holder: Option<u32>,
    },
    /// The lock directory or file could not be probed; never blocks.
    Unknown(String),
}

/// Every host fact the checks read. The real implementation is
/// [`super::probe::SystemProbe`].
pub(crate) trait HostProbe {
    /// Root of the sysfs/proc/dev tree (`/` on a real host).
    fn root(&self) -> &Path;
    /// `RLIMIT_MEMLOCK` soft and hard in bytes; `None` is unlimited.
    fn memlock_limits(&self) -> (Option<u64>, Option<u64>);
    /// `CAP_IPC_LOCK` is in this process's effective set.
    fn cap_ipc_lock(&self) -> bool;
    /// Bytes this process already pins (`VmPin`).
    fn vm_pin_bytes(&self) -> u64;
    /// `access(2)` on `path` for the current user.
    fn access(&self, path: &Path, mode: Access) -> bool;
    fn is_char_device(&self, path: &Path) -> bool;
    /// `/dev/accel/accel0` firmware probe the engine itself uses.
    fn npu_firmware(&self) -> Result<String, String>;
    fn rocm(&self) -> RocmState;
    fn find_daemon(&self) -> Option<PathBuf>;
    /// First line of `<daemon> --version`.
    fn daemon_version(&self, daemon: &Path) -> Option<String>;
    /// `(tag, full commit)` of the kernel pack installed beside `exe_dir`.
    fn installed_pack(&self, exe_dir: &Path) -> Option<(String, String)>;
    /// Effective `fusions.npu_trunk`.
    fn effective_npu_trunk(&self) -> String;
    /// A live `serve.pid`.
    fn serve_running(&self) -> bool;
    fn gpu_lock(&self, device: &GpuDevice) -> GpuLock;
    fn sudo_n_available(&self) -> bool;
    /// Local Flash-Next-class model files, the registry tag's file first.
    fn local_models(&self) -> Vec<LocalModelFile>;
    /// NPU trunk index digest (`model_trunk_index_sha256`, lowercase hex) of a model file.
    fn model_trunk_index_sha256(&self, path: &Path) -> Option<String>;
    /// `(model_trunk_index_sha256, "file" | "builtin")` of every calibration the
    /// engine could pick for the stock and overlay routes.
    fn calibration_records(&self) -> Result<Vec<(String, &'static str)>, String>;
}

/// Everything the checks learned.
pub(crate) struct Outcome {
    pub checks: Vec<Check>,
    pub host: Host,
    pub model: Option<Model>,
    /// Effective `fusions.npu_trunk`.
    pub npu_trunk: String,
    /// Present only when no blocking check failed.
    pub target: Option<Target>,
}

impl Outcome {
    pub(crate) fn blocked(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == CheckStatus::Fail)
    }

    /// Index sha of the matched model, for the profile comparison.
    pub(crate) fn model_sha(&self) -> Option<&str> {
        self.model
            .as_ref()
            .map(|model| model.model_trunk_index_sha256.as_str())
    }
}

fn pass(id: &'static str, detail: impl Into<String>) -> Check {
    Check {
        id,
        status: CheckStatus::Pass,
        detail: detail.into(),
        fix: None,
    }
}

fn fail(id: &'static str, detail: impl Into<String>, fix: impl Into<String>) -> Check {
    Check {
        id,
        status: CheckStatus::Fail,
        detail: detail.into(),
        fix: Some(fix.into()),
    }
}

fn info(id: &'static str, detail: impl Into<String>) -> Check {
    Check {
        id,
        status: CheckStatus::Info,
        detail: detail.into(),
        fix: None,
    }
}

fn read_trimmed(path: &Path) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn short(commit: &str) -> &str {
    commit.get(..12).unwrap_or(commit)
}

fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn limit(value: Option<u64>) -> String {
    value.map_or_else(|| "unlimited".to_owned(), mib)
}

fn no_write(_: &Path, _: &str) -> Result<(), String> {
    Ok(())
}

fn gpu_arch(gpus: &[GpuDevice]) -> Check {
    if gpus.iter().any(|gpu| gpu.arch == STRIX_HALO_ARCH) {
        return pass("gpu_arch", format!("{STRIX_HALO_ARCH} (Strix Halo)"));
    }
    let detail = if gpus.is_empty() {
        "no GPU found in the KFD topology".to_owned()
    } else {
        let arches = gpus.iter().map(|gpu| gpu.arch.as_str()).collect::<Vec<_>>();
        format!("found {}, not {STRIX_HALO_ARCH}", arches.join(", "))
    };
    fail(
        "gpu_arch",
        detail,
        "The NPU trunk route needs a Strix Halo (gfx1151) GPU.",
    )
}

/// `(check, npu is xdna2)`.
fn npu_xdna2(root: &Path) -> (Check, bool) {
    let device = root.join("sys/class/accel/accel0/device");
    let driver = std::fs::read_link(device.join("driver"))
        .ok()
        .and_then(|link| {
            link.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        });
    let vendor = read_trimmed(&device.join("vendor"));
    let id = read_trimmed(&device.join("device"));
    let xdna2 = driver.as_deref() == Some("amdxdna")
        && vendor.as_deref() == Some("0x1022")
        && id.as_deref() == Some("0x17f0");
    if xdna2 {
        return (
            pass("npu_xdna2", "amdxdna driver bound to PCI 1022:17f0"),
            true,
        );
    }
    let detail = if driver.is_none() && vendor.is_none() && id.is_none() {
        "no NPU (accel0) found".to_owned()
    } else {
        format!(
            "accel0 driver {}, PCI {}:{}",
            driver.as_deref().unwrap_or("none"),
            vendor
                .as_deref()
                .map_or("?", |v| v.trim_start_matches("0x")),
            id.as_deref().map_or("?", |v| v.trim_start_matches("0x")),
        )
    };
    let check = fail(
        "npu_xdna2",
        detail,
        "Boot a kernel with the amdxdna driver (in-tree since Linux 6.14) and enable the NPU \
         (IPU) in the firmware setup.",
    );
    (check, false)
}

fn accel_node(probe: &dyn HostProbe) -> Check {
    let node = probe.root().join("dev/accel/accel0");
    if !node.exists() || !probe.is_char_device(&node) {
        return fail(
            "accel_node",
            "accel0 device node is missing or is not a character device",
            "The node appears once the amdxdna driver binds the NPU: see npu_xdna2.",
        );
    }
    if !probe.access(&node, Access::ReadWrite) {
        return fail(
            "accel_node",
            "accel0 exists but this user cannot read and write it",
            "sudo usermod -aG render $USER, then log out and in.",
        );
    }
    pass("accel_node", "accel0 is readable and writable")
}

fn npu_firmware(probe: &dyn HostProbe) -> Check {
    match probe.npu_firmware() {
        Ok(version) => pass("npu_firmware", format!("NPU firmware {}", version.trim())),
        Err(error) => fail(
            "npu_firmware",
            error,
            "Install linux-firmware with amdnpu/17f0_* and reload the driver: \
             sudo modprobe -r amdxdna && sudo modprobe amdxdna",
        ),
    }
}

fn memlock(probe: &dyn HostProbe) -> (Check, Memlock) {
    let (soft, hard) = probe.memlock_limits();
    let cap = probe.cap_ipc_lock();
    let required = probe.vm_pin_bytes() + npu_trunk::trunk_pin_bytes();
    let covers = |value: Option<u64>| value.is_none_or(|value| value >= required);
    // The trunk raises the soft limit itself when the hard limit allows it.
    let ok = covers(soft) || covers(hard) || cap;
    let facts = Memlock {
        soft,
        hard,
        required,
        cap_ipc_lock: cap,
    };
    let detail = format!(
        "RLIMIT_MEMLOCK soft {}, hard {}; the trunk pins {}{}",
        limit(soft),
        limit(hard),
        mib(required),
        if cap { "; CAP_IPC_LOCK held" } else { "" },
    );
    if ok {
        return (pass("memlock", detail), facts);
    }
    let bounded = npu_trunk::bounded_memlock_bytes(required);
    let kib = bounded / 1024;
    let fix = format!(
        "Grant a bounded memlock limit of {bounded} B ({kib} KiB), not infinity: \
         systemd `LimitMEMLOCK={bounded}`; /etc/security/limits.d `<user> soft memlock {kib}` and \
         `<user> hard memlock {kib}`; or `ulimit -l {kib}` in a shell whose hard limit allows it."
    );
    (fail("memlock", detail, fix), facts)
}

fn rocm_runtime(probe: &dyn HostProbe) -> Check {
    let detail = match probe.rocm() {
        RocmState::Ready => return pass("rocm_runtime", "ROCm root with the HIP runtime library"),
        RocmState::NoRoot => "no ROCm root found",
        RocmState::NoRuntime => "the ROCm root has no HIP runtime library (libamdhip64)",
    };
    fail(
        "rocm_runtime",
        detail,
        hipfire_config::rocm::install_guidance().join("\n"),
    )
}

/// Kernel set identity for `Host.kernels`.
fn kernel_set(probe: &dyn HostProbe, daemon: Option<&Path>) -> String {
    let Some(daemon) = daemon else {
        return "unknown".to_owned();
    };
    if let Some((tag, commit)) = daemon.parent().and_then(|dir| probe.installed_pack(dir)) {
        return format!("pack:{tag}@{}", short(&commit));
    }
    match probe.daemon_version(daemon).and_then(|line| {
        line.split_whitespace()
            .nth(2)
            .map(|commit| short(commit).to_owned())
    }) {
        Some(commit) => format!("jit:{commit}"),
        None => "unknown".to_owned(),
    }
}

struct Matched {
    file: LocalModelFile,
    model_trunk_index_sha256: String,
    calibration: &'static str,
}

fn model(probe: &dyn HostProbe) -> (Check, Option<Matched>) {
    let fix = format!("hipfire pull {FLASH_NEXT_TAG}");
    let records = match probe.calibration_records() {
        Ok(records) => records,
        Err(error) => {
            let check = fail("model", format!("calibration unreadable: {error}"), fix);
            return (check, None);
        }
    };
    let candidates = probe.local_models();
    let mut seen = Vec::new();
    for file in &candidates {
        let Some(sha) = probe.model_trunk_index_sha256(&file.path) else {
            continue;
        };
        if let Some((_, calibration)) = records.iter().find(|(admitted, _)| *admitted == sha) {
            let shown = file.tag.as_deref().unwrap_or("local model");
            let detail = format!(
                "{shown}, trunk index sha256 {}…, {calibration} calibration",
                &sha[..8]
            );
            let matched = Matched {
                file: file.clone(),
                model_trunk_index_sha256: sha,
                calibration: *calibration,
            };
            return (pass("model", detail), Some(matched));
        }
        seen.push(format!("{}…", sha.get(..8).unwrap_or(&sha)));
    }
    let detail = if seen.is_empty() {
        "no local Flash-Next (arch 16) model".to_owned()
    } else {
        format!(
            "{} local Flash-Next model(s), none matches a calibration (trunk index sha256 {})",
            seen.len(),
            seen.join(", ")
        )
    };
    (fail("model", detail, fix), None)
}

/// `gpu` is `None` when there is no gfx1151 GPU to probe.
fn gpu_free(probe: &dyn HostProbe, gpu: Option<&GpuDevice>) -> Check {
    let mut busy = Vec::new();
    if probe.serve_running() {
        busy.push("hipfire serve is running".to_owned());
    }
    let mut note = None;
    match gpu.map(|gpu| probe.gpu_lock(gpu)) {
        None => return info("gpu_free", "skipped: no gfx1151 GPU to probe"),
        Some(GpuLock::Free) => {}
        Some(GpuLock::Busy { holder }) => busy.push(match holder {
            Some(pid) => format!("the GPU is reserved by another process (holder PID {pid})"),
            None => "the GPU is reserved by another process".to_owned(),
        }),
        Some(GpuLock::Unknown(why)) => note = Some(why),
    }
    if busy.is_empty() {
        let detail = match note {
            Some(why) => format!("no serve process; GPU reservation not probed ({why})"),
            None => "no serve process and the GPU reservation is free".to_owned(),
        };
        return pass("gpu_free", detail);
    }
    fail("gpu_free", busy.join("; "), "hipfire stop")
}

fn fclk_check(probe: &dyn HostProbe, gpu: Option<&GpuDevice>) -> (Check, Fclk) {
    const POLICY: &str =
        "Serving runs the fabric clock on automatic; only lab tools need it pinned.";
    let Some(gpu) = gpu else {
        let check = info("fclk", "skipped: no gfx1151 GPU to read");
        return (check, Fclk::default());
    };
    let dir = probe
        .root()
        .join("sys/bus/pci/devices")
        .join(gpu.bdf.to_string());
    // The no-op writer makes this handle read-only by construction.
    let state = fclk::Sysfs::at(dir.clone(), no_write, Duration::ZERO).state();
    let pinned = state.as_ref().ok().map(|state| state.pinned());
    let attrs_writable = [fclk::PERF, fclk::FCLK]
        .iter()
        .all(|attr| probe.access(&dir.join(attr), Access::Write));
    let pinnable = attrs_writable || probe.sudo_n_available();
    let clock = match pinned {
        Some(true) => "pinned",
        Some(false) => "on automatic (not pinned)",
        None => "unreadable",
    };
    let detail = format!(
        "fabric clock {clock}; {} pin it without a password. {POLICY}",
        if pinnable { "can" } else { "cannot" },
    );
    (info("fclk", detail), Fclk { pinned, pinnable })
}

/// `gpu` is `None` when there is no gfx1151 GPU to read.
fn power_check(probe: &dyn HostProbe, gpu: Option<&GpuDevice>) -> Check {
    let Some(gpu) = gpu else {
        return info("power", "skipped: no gfx1151 GPU to read");
    };
    let device = probe
        .root()
        .join("sys/bus/pci/devices")
        .join(gpu.bdf.to_string());
    power::assess(&power::read(probe.root(), &device))
}

/// Run every check. Order is the report order.
pub(crate) fn run(probe: &dyn HostProbe) -> Outcome {
    let root = probe.root();
    let gpus = enumerate_gpus(&root.join("sys/class/kfd/kfd/topology/nodes")).unwrap_or_default();
    let gpu = gpus.iter().find(|gpu| gpu.arch == STRIX_HALO_ARCH);
    let gpu_pci = gpu.map(|gpu| gpu.bdf.to_string());
    let facts = npu_trunk::HostFacts::read(root, gpu_pci.as_deref());
    let daemon = probe.find_daemon();

    let mut checks = vec![gpu_arch(&gpus)];
    let (xdna2_check, xdna2) = npu_xdna2(root);
    checks.push(xdna2_check);
    checks.push(accel_node(probe));
    checks.push(npu_firmware(probe));
    let (memlock_check, memlock_facts) = memlock(probe);
    checks.push(memlock_check);
    checks.push(rocm_runtime(probe));
    checks.push(match &daemon {
        Some(path) => {
            let version = probe.daemon_version(path);
            pass(
                "daemon",
                version.unwrap_or_else(|| "hipfire-daemon found".to_owned()),
            )
        }
        None => fail("daemon", "hipfire-daemon not found", "hipfire setup"),
    });
    let (model_check, matched) = model(probe);
    checks.push(model_check);
    checks.push(gpu_free(probe, gpu));
    let npu_trunk_config = probe.effective_npu_trunk();
    checks.push(info(
        "npu_trunk_config",
        format!("fusions.npu_trunk = {npu_trunk_config}"),
    ));
    let (fclk_info, fclk_facts) = fclk_check(probe, gpu);
    checks.push(fclk_info);
    checks.push(power_check(probe, gpu));

    let unknown = |value: Option<String>| value.unwrap_or_else(|| "unknown".to_owned());
    let host = Host {
        gpu_arch: gpu
            .map(|gpu| gpu.arch.clone())
            .or_else(|| gpus.first().map(|gpu| gpu.arch.clone())),
        npu: xdna2.then(|| "xdna2".to_owned()),
        kernel: unknown(facts.kernel),
        iommu: Iommu {
            gpu: unknown(facts.iommu_gpu),
            npu: unknown(facts.iommu_npu),
        },
        amdxdna: unknown(facts.amdxdna),
        npu_fw: unknown(facts.npu_fw),
        kernels: kernel_set(probe, daemon.as_deref()),
        memlock: memlock_facts,
        fclk: fclk_facts,
    };
    let model = matched.as_ref().map(|matched| Model {
        tag: matched.file.tag.clone(),
        model_trunk_index_sha256: matched.model_trunk_index_sha256.clone(),
        calibration: matched.calibration,
    });
    let mut outcome = Outcome {
        checks,
        host,
        model,
        npu_trunk: npu_trunk_config,
        target: None,
    };
    if !outcome.blocked() {
        if let (Some(matched), Some(daemon), Some(gpu_pci)) = (matched, daemon, gpu_pci) {
            outcome.target = Some(Target {
                model_path: matched.file.path,
                model_tag: matched.file.tag,
                daemon,
                gpu_pci,
            });
        }
    }
    outcome
}

#[cfg(test)]
pub(crate) mod testing {
    //! A fake host: a temp directory standing in for `/` plus scripted answers.

    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU32, Ordering};

    pub(crate) const REFERENCE_INDEX: &str =
        "098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6";
    /// Appears in every fake model/daemon path; must never reach a report.
    pub(crate) const FAKE_USER: &str = "fakeuser";

    pub(crate) fn fake_daemon() -> PathBuf {
        PathBuf::from(format!("/home/{FAKE_USER}/.hipfire/bin/daemon"))
    }

    pub(crate) struct FakeProbe {
        root: PathBuf,
        pub memlock: (Option<u64>, Option<u64>),
        pub cap: bool,
        pub vm_pin: u64,
        pub accessible: bool,
        pub char_device: bool,
        pub firmware: Result<String, String>,
        pub rocm: RocmState,
        pub daemon: Option<PathBuf>,
        pub daemon_version: Option<String>,
        pub pack: Option<(String, String)>,
        pub npu_trunk: String,
        pub serve: bool,
        pub lock: GpuLock,
        pub sudo: bool,
        pub models: Vec<(LocalModelFile, Option<String>)>,
        pub records: Result<Vec<(String, &'static str)>, String>,
        pub writable: RefCell<BTreeSet<String>>,
    }

    static NEXT: AtomicU32 = AtomicU32::new(0);

    fn put(root: &Path, relative: &str, text: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    impl FakeProbe {
        /// A healthy hipx-like host: every check passes.
        pub(crate) fn healthy() -> Self {
            let root = std::env::temp_dir().join(format!(
                "hipfire-doctor-fake-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            let node = "sys/class/kfd/kfd/topology/nodes/1/properties";
            // location_id 0xc000 -> bus c0, device 0, function 0.
            put(
                &root,
                node,
                "simd_count 64\nlocation_id 49152\ndomain 0\ngfx_target_version 110501\n",
            );
            put(&root, "sys/class/accel/accel0/device/vendor", "0x1022\n");
            put(&root, "sys/class/accel/accel0/device/device", "0x17f0\n");
            put(
                &root,
                "sys/class/accel/accel0/device/fw_version",
                "1.1.2.65\n",
            );
            put(
                &root,
                "sys/class/accel/accel0/device/iommu_group/type",
                "identity\n",
            );
            symlink(
                "../../../../bus/pci/drivers/amdxdna",
                root.join("sys/class/accel/accel0/device/driver"),
            )
            .unwrap();
            put(
                &root,
                "sys/bus/pci/devices/0000:c0:00.0/iommu_group/type",
                "identity\n",
            );
            put(
                &root,
                "sys/bus/pci/devices/0000:c0:00.0/power_dpm_force_performance_level",
                "auto\n",
            );
            put(
                &root,
                "sys/bus/pci/devices/0000:c0:00.0/pp_dpm_fclk",
                "0: 400Mhz\n1: 1000Mhz\n2: 2000Mhz *\n",
            );
            put(&root, "sys/module/amdxdna/.keep", "");
            put(&root, "proc/sys/kernel/osrelease", "7.3.0-rc7-perfopt+\n");
            put(&root, "dev/accel/accel0", "");
            let model = LocalModelFile {
                path: PathBuf::from(format!("/home/{FAKE_USER}/.hipfire/models/flash-next.mq4")),
                tag: Some(FLASH_NEXT_TAG.to_owned()),
            };
            Self {
                root,
                memlock: (Some(1_287_168 * 1024), Some(1_287_168 * 1024)),
                cap: false,
                vm_pin: 0,
                accessible: true,
                char_device: true,
                firmware: Ok("1.1.2.65\n".to_owned()),
                rocm: RocmState::Ready,
                daemon: Some(fake_daemon()),
                daemon_version: Some("hipfire-daemon 0.4.3 0123456789abcdef0123".to_owned()),
                pack: Some(("v0.4.3".to_owned(), "fedcba9876543210ffff".to_owned())),
                npu_trunk: "auto".to_owned(),
                serve: false,
                lock: GpuLock::Free,
                sudo: true,
                models: vec![(model, Some(REFERENCE_INDEX.to_owned()))],
                records: Ok(vec![(REFERENCE_INDEX.to_owned(), "builtin")]),
                writable: RefCell::new(BTreeSet::new()),
            }
        }

        pub(crate) fn put(&self, relative: &str, text: &str) {
            put(&self.root, relative, text);
        }

        pub(crate) fn remove(&self, relative: &str) {
            let path = self.root.join(relative);
            if path.is_dir() {
                std::fs::remove_dir_all(path).unwrap();
            } else {
                std::fs::remove_file(path).unwrap();
            }
        }
    }

    impl Drop for FakeProbe {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    impl HostProbe for FakeProbe {
        fn root(&self) -> &Path {
            &self.root
        }
        fn memlock_limits(&self) -> (Option<u64>, Option<u64>) {
            self.memlock
        }
        fn cap_ipc_lock(&self) -> bool {
            self.cap
        }
        fn vm_pin_bytes(&self) -> u64 {
            self.vm_pin
        }
        fn access(&self, path: &Path, mode: Access) -> bool {
            match mode {
                Access::ReadWrite => self.accessible,
                Access::Write => self
                    .writable
                    .borrow()
                    .iter()
                    .any(|attr| path.ends_with(attr)),
            }
        }
        fn is_char_device(&self, _: &Path) -> bool {
            self.char_device
        }
        fn npu_firmware(&self) -> Result<String, String> {
            self.firmware.clone()
        }
        fn rocm(&self) -> RocmState {
            self.rocm
        }
        fn find_daemon(&self) -> Option<PathBuf> {
            self.daemon.clone()
        }
        fn daemon_version(&self, _: &Path) -> Option<String> {
            self.daemon_version.clone()
        }
        fn installed_pack(&self, _: &Path) -> Option<(String, String)> {
            self.pack.clone()
        }
        fn effective_npu_trunk(&self) -> String {
            self.npu_trunk.clone()
        }
        fn serve_running(&self) -> bool {
            self.serve
        }
        fn gpu_lock(&self, _: &GpuDevice) -> GpuLock {
            self.lock.clone()
        }
        fn sudo_n_available(&self) -> bool {
            self.sudo
        }
        fn local_models(&self) -> Vec<LocalModelFile> {
            self.models.iter().map(|(file, _)| file.clone()).collect()
        }
        fn model_trunk_index_sha256(&self, path: &Path) -> Option<String> {
            self.models
                .iter()
                .find(|(file, _)| file.path == path)
                .and_then(|(_, sha)| sha.clone())
        }
        fn calibration_records(&self) -> Result<Vec<(String, &'static str)>, String> {
            self.records.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn check<'a>(outcome: &'a Outcome, id: &str) -> &'a Check {
        outcome.checks.iter().find(|check| check.id == id).unwrap()
    }

    fn failed(outcome: &Outcome) -> Vec<&'static str> {
        outcome
            .checks
            .iter()
            .filter(|check| check.status == CheckStatus::Fail)
            .map(|check| check.id)
            .collect()
    }

    /// The only failed check is `id`; returns it.
    fn only_failure<'a>(outcome: &'a Outcome, id: &str) -> &'a Check {
        assert_eq!(failed(outcome), [id], "{:#?}", outcome.checks);
        assert!(outcome.target.is_none());
        check(outcome, id)
    }

    #[test]
    fn healthy_host_passes_everything_and_yields_a_target() {
        let probe = FakeProbe::healthy();
        let outcome = run(&probe);
        assert!(failed(&outcome).is_empty(), "{:#?}", outcome.checks);
        let ids = outcome
            .checks
            .iter()
            .map(|check| check.id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "gpu_arch",
                "npu_xdna2",
                "accel_node",
                "npu_firmware",
                "memlock",
                "rocm_runtime",
                "daemon",
                "model",
                "gpu_free",
                "npu_trunk_config",
                "fclk",
                "power"
            ]
        );
        let target = outcome.target.as_ref().unwrap();
        assert_eq!(target.gpu_pci, "0000:c0:00.0");
        assert_eq!(target.model_tag.as_deref(), Some(FLASH_NEXT_TAG));
        assert_eq!(target.daemon, fake_daemon());
        let host = &outcome.host;
        assert_eq!(host.gpu_arch.as_deref(), Some("gfx1151"));
        assert_eq!(host.npu.as_deref(), Some("xdna2"));
        assert_eq!(host.kernel, "7.3.0-rc7-perfopt+");
        assert_eq!(host.iommu.gpu, "identity");
        assert_eq!(host.iommu.npu, "identity");
        assert_eq!(host.amdxdna, "in-tree");
        assert_eq!(host.npu_fw, "1.1.2.65");
        assert_eq!(host.kernels, "pack:v0.4.3@fedcba987654");
        assert_eq!(
            host.fclk,
            Fclk {
                pinned: Some(false),
                pinnable: true
            }
        );
        let model = outcome.model.as_ref().unwrap();
        assert_eq!(model.model_trunk_index_sha256, REFERENCE_INDEX);
        assert_eq!(model.calibration, "builtin");
        assert_eq!(outcome.npu_trunk, "auto");
    }

    #[test]
    fn gpu_arch_blocks_without_a_halo() {
        let probe = FakeProbe::healthy();
        probe.put(
            "sys/class/kfd/kfd/topology/nodes/1/properties",
            "simd_count 96\nlocation_id 256\ndomain 0\ngfx_target_version 110000\n",
        );
        let outcome = run(&probe);
        let gpu = only_failure(&outcome, "gpu_arch");
        assert_eq!(gpu.detail, "found gfx1100, not gfx1151");
        assert_eq!(
            gpu.fix.as_deref(),
            Some("The NPU trunk route needs a Strix Halo (gfx1151) GPU.")
        );
        assert_eq!(outcome.host.gpu_arch.as_deref(), Some("gfx1100"));
        assert_eq!(check(&outcome, "gpu_free").status, CheckStatus::Info);
        assert_eq!(check(&outcome, "fclk").status, CheckStatus::Info);
        let power = check(&outcome, "power");
        assert_eq!(power.status, CheckStatus::Info);
        assert_eq!(power.detail, "skipped: no gfx1151 GPU to read");
    }

    #[test]
    fn healthy_host_without_power_files_reports_power_as_information() {
        let outcome = run(&FakeProbe::healthy());
        let power = check(&outcome, "power");
        assert_eq!(power.status, CheckStatus::Info);
        assert!(power.fix.is_none());
    }

    #[test]
    fn full_power_host_passes_the_power_check() {
        let probe = FakeProbe::healthy();
        probe.put("sys/firmware/acpi/platform_profile", "performance\n");
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/hwmon/hwmon5/power1_cap",
            "120000000\n",
        );
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/hwmon/hwmon5/power1_cap_default",
            "120000000\n",
        );
        let outcome = run(&probe);
        let power = check(&outcome, "power");
        assert_eq!(power.status, CheckStatus::Pass, "{}", power.detail);
        assert!(failed(&outcome).is_empty());
    }

    #[test]
    fn low_power_host_warns_without_blocking_or_changing_the_target() {
        let probe = FakeProbe::healthy();
        probe.put("sys/firmware/acpi/platform_profile", "balanced\n");
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/hwmon/hwmon5/power1_cap",
            "55000000\n",
        );
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/hwmon/hwmon5/power1_cap_default",
            "120000000\n",
        );
        let outcome = run(&probe);
        let power = check(&outcome, "power");
        assert_eq!(power.status, CheckStatus::Warn, "{}", power.detail);
        assert!(power.detail.contains("system power mode is balanced"));
        assert!(power.detail.contains("55 W"));
        assert!(power.detail.contains("assume full power"));
        assert!(power.fix.as_deref().unwrap().contains("performance"));
        assert!(!outcome.blocked());
        assert!(outcome.target.is_some());
    }

    #[test]
    fn npu_xdna2_blocks_on_wrong_driver_or_device() {
        let probe = FakeProbe::healthy();
        probe.put("sys/class/accel/accel0/device/device", "0x1502\n");
        let outcome = run(&probe);
        let check = only_failure(&outcome, "npu_xdna2");
        assert!(check.detail.contains("1022:1502"), "{}", check.detail);
        let fix = check.fix.as_deref().unwrap();
        assert!(fix.contains("amdxdna") && fix.contains("6.14") && fix.contains("firmware setup"));
        assert_eq!(outcome.host.npu, None);

        let probe = FakeProbe::healthy();
        probe.remove("sys/class/accel");
        let outcome = run(&probe);
        assert_eq!(
            only_failure(&outcome, "npu_xdna2").detail,
            "no NPU (accel0) found"
        );
    }

    #[test]
    fn accel_node_blocks_without_access_with_the_group_fix() {
        let mut probe = FakeProbe::healthy();
        probe.accessible = false;
        let outcome = run(&probe);
        let check = only_failure(&outcome, "accel_node");
        assert_eq!(
            check.fix.as_deref(),
            Some("sudo usermod -aG render $USER, then log out and in.")
        );
    }

    #[test]
    fn accel_node_blocks_when_missing_or_not_a_char_device() {
        let probe = FakeProbe::healthy();
        probe.remove("dev/accel");
        let outcome = run(&probe);
        only_failure(&outcome, "accel_node");

        let mut probe = FakeProbe::healthy();
        probe.char_device = false;
        let outcome = run(&probe);
        only_failure(&outcome, "accel_node");
    }

    #[test]
    fn npu_firmware_blocks_with_the_reload_fix() {
        let mut probe = FakeProbe::healthy();
        probe.firmware = Err("NPU fw_version: No such file or directory (os error 2)".into());
        let outcome = run(&probe);
        let check = only_failure(&outcome, "npu_firmware");
        let fix = check.fix.as_deref().unwrap();
        assert!(fix.contains("amdnpu/17f0_*"));
        assert!(fix.contains("sudo modprobe -r amdxdna && sudo modprobe amdxdna"));
    }

    #[test]
    fn memlock_blocks_on_the_default_limit_and_quotes_the_bounded_value() {
        let mut probe = FakeProbe::healthy();
        probe.memlock = (Some(8 << 20), Some(8 << 20));
        let outcome = run(&probe);
        let check = only_failure(&outcome, "memlock");
        let required = npu_trunk::trunk_pin_bytes();
        let bounded = npu_trunk::bounded_memlock_bytes(required);
        let kib = bounded / 1024;
        let fix = check.fix.as_deref().unwrap();
        assert!(fix.contains(&format!("LimitMEMLOCK={bounded}")), "{fix}");
        assert!(fix.contains(&format!("<user> soft memlock {kib}")), "{fix}");
        assert!(fix.contains(&format!("<user> hard memlock {kib}")), "{fix}");
        assert!(fix.contains(&format!("ulimit -l {kib}")), "{fix}");
        assert_eq!(outcome.host.memlock.required, required);
        assert_eq!(outcome.host.memlock.soft, Some(8 << 20));
    }

    #[test]
    fn memlock_counts_vm_pin_and_passes_on_hard_limit_or_capability() {
        let required = npu_trunk::trunk_pin_bytes();
        // Hard limit alone is enough: the trunk raises the soft limit itself.
        let mut probe = FakeProbe::healthy();
        probe.memlock = (Some(8 << 20), Some(required));
        assert!(failed(&run(&probe)).is_empty());
        // Already pinned bytes count against the same limit.
        probe.vm_pin = 1 << 20;
        probe.memlock = (Some(8 << 20), Some(required));
        only_failure(&run(&probe), "memlock");
        // CAP_IPC_LOCK or an unlimited limit passes.
        probe.cap = true;
        assert!(failed(&run(&probe)).is_empty());
        probe.cap = false;
        probe.memlock = (None, None);
        assert!(failed(&run(&probe)).is_empty());
    }

    #[test]
    fn rocm_runtime_blocks_with_install_guidance() {
        let mut probe = FakeProbe::healthy();
        probe.rocm = RocmState::NoRuntime;
        let outcome = run(&probe);
        let check = only_failure(&outcome, "rocm_runtime");
        assert_eq!(
            check.fix.as_deref(),
            Some(hipfire_config::rocm::install_guidance().join("\n").as_str())
        );
        probe.rocm = RocmState::NoRoot;
        only_failure(&run(&probe), "rocm_runtime");
    }

    #[test]
    fn daemon_blocks_with_hipfire_setup() {
        let mut probe = FakeProbe::healthy();
        probe.daemon = None;
        let outcome = run(&probe);
        let check = only_failure(&outcome, "daemon");
        assert_eq!(check.fix.as_deref(), Some("hipfire setup"));
        assert_eq!(outcome.host.kernels, "unknown");
    }

    #[test]
    fn model_blocks_without_an_admitted_local_model() {
        let mut probe = FakeProbe::healthy();
        probe.models[0].1 = Some("f".repeat(64));
        let outcome = run(&probe);
        let check = only_failure(&outcome, "model");
        assert_eq!(
            check.fix.as_deref(),
            Some("hipfire pull qwen3.8:flash-next")
        );
        assert!(
            check.detail.contains("none matches a calibration"),
            "{}",
            check.detail
        );
        assert!(outcome.model.is_none());

        probe.models.clear();
        let outcome = run(&probe);
        let check = only_failure(&outcome, "model");
        assert_eq!(check.detail, "no local Flash-Next (arch 16) model");
    }

    #[test]
    fn model_blocks_on_an_unreadable_calibration_file() {
        let mut probe = FakeProbe::healthy();
        probe.records = Err("unsupported calibration schema_version".into());
        let outcome = run(&probe);
        let check = only_failure(&outcome, "model");
        assert!(check.detail.contains("calibration unreadable"));
    }

    #[test]
    fn model_reports_the_calibration_source_and_picks_the_first_admitted_file() {
        let mut probe = FakeProbe::healthy();
        probe.records = Ok(vec![(REFERENCE_INDEX.to_owned(), "file")]);
        let other = LocalModelFile {
            path: PathBuf::from("/fake/models/other.mq4"),
            tag: None,
        };
        probe.models.insert(0, (other, Some("e".repeat(64))));
        let outcome = run(&probe);
        assert!(failed(&outcome).is_empty());
        let model = outcome.model.unwrap();
        assert_eq!(model.calibration, "file");
        assert_eq!(model.tag.as_deref(), Some(FLASH_NEXT_TAG));
    }

    #[test]
    fn gpu_free_blocks_on_serve_or_a_held_lock_with_hipfire_stop() {
        let mut probe = FakeProbe::healthy();
        probe.serve = true;
        let outcome = run(&probe);
        let check = only_failure(&outcome, "gpu_free");
        assert_eq!(check.fix.as_deref(), Some("hipfire stop"));
        assert!(check.detail.contains("hipfire serve is running"));

        probe.serve = false;
        probe.lock = GpuLock::Busy { holder: Some(4242) };
        let outcome = run(&probe);
        let check = only_failure(&outcome, "gpu_free");
        assert!(check.detail.contains("holder PID 4242"), "{}", check.detail);

        probe.lock = GpuLock::Unknown("lock dir unreadable".into());
        let outcome = run(&probe);
        assert!(failed(&outcome).is_empty());
        assert_eq!(check_status(&outcome, "gpu_free"), CheckStatus::Pass);
    }

    fn check_status(outcome: &Outcome, id: &str) -> CheckStatus {
        check(outcome, id).status
    }

    #[test]
    fn info_checks_never_block() {
        let mut probe = FakeProbe::healthy();
        probe.npu_trunk = "off".into();
        probe.sudo = false;
        let outcome = run(&probe);
        assert!(failed(&outcome).is_empty());
        assert_eq!(
            check(&outcome, "npu_trunk_config").status,
            CheckStatus::Info
        );
        assert_eq!(outcome.npu_trunk, "off");
        let clock = check(&outcome, "fclk");
        assert_eq!(clock.status, CheckStatus::Info);
        assert!(clock.detail.contains("on automatic"), "{}", clock.detail);
        assert!(clock.detail.contains("only lab tools need it pinned"));
        assert_eq!(
            outcome.host.fclk,
            Fclk {
                pinned: Some(false),
                pinnable: false
            }
        );
        // Writable attributes make it pinnable without sudo.
        probe
            .writable
            .borrow_mut()
            .extend([fclk::PERF.to_owned(), fclk::FCLK.to_owned()]);
        assert!(run(&probe).host.fclk.pinnable);
    }

    #[test]
    fn fclk_reports_a_pinned_clock() {
        let probe = FakeProbe::healthy();
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/power_dpm_force_performance_level",
            "manual\n",
        );
        probe.put(
            "sys/bus/pci/devices/0000:c0:00.0/pp_dpm_fclk",
            "0: 400Mhz\n1: 1000Mhz\n2: 2000Mhz *\n",
        );
        assert_eq!(run(&probe).host.fclk.pinned, Some(true));
    }

    #[test]
    fn kernel_set_falls_back_to_jit_then_unknown() {
        let mut probe = FakeProbe::healthy();
        probe.pack = None;
        assert_eq!(run(&probe).host.kernels, "jit:0123456789ab");
        probe.daemon_version = None;
        assert_eq!(run(&probe).host.kernels, "unknown");
        probe.daemon_version = Some("hipfire-daemon 0.4.3".into());
        assert_eq!(run(&probe).host.kernels, "unknown");
    }

    #[test]
    fn two_failures_are_both_reported() {
        let mut probe = FakeProbe::healthy();
        probe.daemon = None;
        probe.serve = true;
        let outcome = run(&probe);
        assert_eq!(failed(&outcome), ["daemon", "gpu_free"]);
        assert!(outcome.blocked());
    }
}
