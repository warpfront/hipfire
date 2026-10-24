// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The real [`HostProbe`]: this machine's `/`, rlimits, capabilities, `sudo -n`,
//! the installed daemon and the local model files. No GPU or NPU is opened.

use super::checks::{Access, GpuLock, HostProbe, LocalModelFile, RocmState, FLASH_NEXT_TAG};
use crate::Paths;
use hipfire_arch_qwen4::npu_trunk::{self, load_calibration};
use hipfire_config::devices::GpuDevice;
use hipfire_runtime::hfq::HfqFile;
use std::ffi::CString;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Calibration routes the engine can arm on a Halo (`stock` GPU trunk or the
/// shipped overlay), in the order the built-in records are tried.
const ROUTES: [&str; 2] = ["stock", "overlay:gfx1151-zab-v1"];
/// HFQ architecture id of Qwen3.8-Flash-Next.
const FLASH_NEXT_ARCH_ID: u32 = 16;
const CAP_IPC_LOCK: u32 = 14;
const DAEMON_VERSION_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct SystemProbe<'a> {
    paths: &'a Paths,
    root: PathBuf,
}

impl<'a> SystemProbe<'a> {
    pub(crate) fn new(paths: &'a Paths) -> Self {
        Self {
            paths,
            root: PathBuf::from("/"),
        }
    }
}

fn proc_status() -> String {
    std::fs::read_to_string("/proc/self/status").unwrap_or_default()
}

/// `<key>:  <n> kB` of `/proc/self/status`, in bytes.
pub(crate) fn status_bytes(text: &str, key: &str) -> Option<u64> {
    let rest = text
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix(':'))?;
    let mut fields = rest.split_whitespace();
    let value = fields.next()?.parse::<u64>().ok()?;
    match fields.next() {
        Some("kB") => value.checked_mul(1024),
        None => Some(value),
        Some(_) => None,
    }
}

/// `CapEff` has `CAP_IPC_LOCK`.
pub(crate) fn cap_eff_has_ipc_lock(text: &str) -> bool {
    text.lines()
        .find_map(|line| line.strip_prefix("CapEff:"))
        .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok())
        .is_some_and(|mask| (mask >> CAP_IPC_LOCK) & 1 == 1)
}

fn rlimit_value(value: libc::rlim_t) -> Option<u64> {
    (value != libc::RLIM_INFINITY).then_some(value)
}

fn holder_pid(file: &mut File) -> Option<u32> {
    let mut text = String::new();
    file.read_to_string(&mut text).ok()?;
    text.split_whitespace().next()?.parse().ok()
}

/// Take the card's lock without blocking and release it at once, like
/// `gpu_lock.rs`: exclusive, or shared under fabric admission. A missing file is
/// a free card.
pub(crate) fn probe_lock_file(path: &Path, shared: bool) -> GpuLock {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return GpuLock::Free,
        Err(error) => return GpuLock::Unknown(error.to_string()),
    };
    let mode = if shared { libc::LOCK_SH } else { libc::LOCK_EX };
    // SAFETY: `file` owns a valid descriptor for the whole call.
    if unsafe { libc::flock(file.as_raw_fd(), mode | libc::LOCK_NB) } == 0 {
        // SAFETY: same descriptor. Unlock explicitly rather than on close: a
        // child forked by another thread holds a copy of this descriptor, and
        // with it the lock, until it execs.
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        return GpuLock::Free;
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        GpuLock::Busy {
            holder: holder_pid(&mut file),
        }
    } else {
        GpuLock::Unknown(error.to_string())
    }
}

/// Lock directories the daemon would use, first usable one wins. The fabric
/// admission (`HIPFIRE_WEIGHT_HOST=1`) always uses the machine-wide directory.
fn lock_dirs(shared: bool, override_dir: Option<PathBuf>) -> Vec<PathBuf> {
    let machine = PathBuf::from("/run/lock/hipfire");
    if shared {
        return vec![machine];
    }
    match override_dir {
        Some(dir) => vec![dir],
        None => vec![machine, PathBuf::from("/tmp/hipfire-locks")],
    }
}

impl HostProbe for SystemProbe<'_> {
    fn root(&self) -> &Path {
        &self.root
    }

    fn memlock_limits(&self) -> (Option<u64>, Option<u64>) {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: `limit` is a valid out-pointer for the call.
        if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) } != 0 {
            return (Some(0), Some(0));
        }
        (rlimit_value(limit.rlim_cur), rlimit_value(limit.rlim_max))
    }

    fn cap_ipc_lock(&self) -> bool {
        cap_eff_has_ipc_lock(&proc_status())
    }

    fn vm_pin_bytes(&self) -> u64 {
        status_bytes(&proc_status(), "VmPin").unwrap_or(0)
    }

    fn access(&self, path: &Path, mode: Access) -> bool {
        let Ok(path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
            return false;
        };
        let mode = match mode {
            Access::ReadWrite => libc::R_OK | libc::W_OK,
            Access::Write => libc::W_OK,
        };
        // SAFETY: `path` is a NUL-terminated string that outlives the call.
        unsafe { libc::access(path.as_ptr(), mode) == 0 }
    }

    fn is_char_device(&self, path: &Path) -> bool {
        std::fs::metadata(path).is_ok_and(|meta| meta.file_type().is_char_device())
    }

    fn npu_firmware(&self) -> Result<String, String> {
        npu_trunk::npu_firmware()
    }

    fn rocm(&self) -> RocmState {
        let Some(root) = hipfire_config::rocm::root() else {
            return RocmState::NoRoot;
        };
        if hipfire_config::rocm::runtime_library(&root).is_some() {
            RocmState::Ready
        } else {
            RocmState::NoRuntime
        }
    }

    fn find_daemon(&self) -> Option<PathBuf> {
        crate::find_daemon(self.paths)
    }

    fn daemon_version(&self, daemon: &Path) -> Option<String> {
        // An older daemon without `--version` would start its protocol loop;
        // stdin is closed and the wait is bounded, so it cannot linger.
        let mut child = Command::new(daemon)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let started = Instant::now();
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() < DAEMON_VERSION_TIMEOUT => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
            }
        };
        let mut stdout = String::new();
        child.stdout.take()?.read_to_string(&mut stdout).ok()?;
        let line = stdout.lines().next()?.trim();
        (status.success() && line.starts_with("hipfire-daemon ")).then(|| line.to_owned())
    }

    fn installed_pack(&self, exe_dir: &Path) -> Option<(String, String)> {
        let arch_dir = rdna_compute::kernel_pack::pack_arch_dir(exe_dir, "gfx1151");
        let manifest = rdna_compute::kernel_pack::installed_manifest(&arch_dir)?;
        Some((manifest.tag, manifest.commit))
    }

    fn effective_npu_trunk(&self) -> String {
        crate::resolved_global(self.paths, true)
            .ok()
            .and_then(|(_, resolved)| {
                resolved
                    .get("fusions.npu_trunk")
                    .map(|resolved| resolved.value.to_string())
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }

    fn serve_running(&self) -> bool {
        std::fs::read_to_string(self.paths.root.join("serve.pid"))
            .ok()
            .and_then(|raw| crate::serve::parse_pid_record(&raw))
            .is_some_and(|record| crate::process_identity::pid_alive(record.pid))
    }

    fn gpu_lock(&self, device: &GpuDevice) -> GpuLock {
        let shared = std::env::var("HIPFIRE_WEIGHT_HOST").as_deref() == Ok("1");
        let override_dir = std::env::var_os("HIPFIRE_LOCK_DIR").map(PathBuf::from);
        let name = device.lock_file_name();
        lock_dirs(shared, override_dir)
            .into_iter()
            .find(|dir| dir.is_dir())
            .map_or(GpuLock::Free, |dir| {
                probe_lock_file(&dir.join(&name), shared)
            })
    }

    fn sudo_n_available(&self) -> bool {
        Command::new("sudo")
            .args(["-n", "true"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn local_models(&self) -> Vec<LocalModelFile> {
        let registry = hipfire_registry::load(&self.paths.registry).registry;
        let models = crate::list_local_models(self.paths, &registry).unwrap_or_default();
        let (mut preferred, mut rest) = (Vec::new(), Vec::new());
        for model in models {
            if HfqFile::probe_arch_id(&model.path).ok() != Some(FLASH_NEXT_ARCH_ID) {
                continue;
            }
            let file = LocalModelFile {
                path: model.path,
                tag: model.registry_tag,
            };
            if file.tag.as_deref() == Some(FLASH_NEXT_TAG) {
                preferred.push(file);
            } else {
                rest.push(file);
            }
        }
        preferred.append(&mut rest);
        preferred
    }

    fn model_trunk_index_sha256(&self, path: &Path) -> Option<String> {
        npu_trunk::model_trunk_index_sha256(&HfqFile::open(path).ok()?).ok()
    }

    fn calibration_records(&self) -> Result<Vec<(String, &'static str)>, String> {
        let path = hipfire_config::npu_trunk_calibration_path();
        let mut records: Vec<(String, &'static str)> = Vec::new();
        for route in ROUTES {
            let (calibration, source) = load_calibration(&path, route)?;
            let record = (
                calibration
                    .key
                    .model_trunk_index_sha256
                    .to_ascii_lowercase(),
                source,
            );
            if !records.contains(&record) {
                records.push(record);
            }
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-doctor-probe-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Unlock, then close. Closing alone frees a flock only once every
    /// descriptor of the open file description is gone, and a child that
    /// another test thread is spawning holds a copy until it execs.
    fn release(holder: File) {
        // SAFETY: valid descriptor owned by `holder`.
        assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_UN) }, 0);
    }

    #[test]
    fn status_parsing_reads_vmpin_and_cap_ipc_lock() {
        let text =
            "Name:\tx\nVmPin:\t    1024 kB\nVmLck:\t       0 kB\nCapEff:\t0000000000004000\n";
        assert_eq!(status_bytes(text, "VmPin"), Some(1024 * 1024));
        assert_eq!(status_bytes(text, "VmLck"), Some(0));
        assert_eq!(status_bytes(text, "VmHWM"), None);
        assert!(cap_eff_has_ipc_lock(text));
        assert!(!cap_eff_has_ipc_lock("CapEff:\t0000000000000000\n"));
        assert!(!cap_eff_has_ipc_lock("CapEff:\t0000000000003fff\n"));
        assert!(cap_eff_has_ipc_lock("CapEff:\t000001ffffffffff\n"));
        assert!(!cap_eff_has_ipc_lock("no capability line"));
    }

    #[test]
    fn rlimit_infinity_is_unlimited() {
        assert_eq!(rlimit_value(libc::RLIM_INFINITY), None);
        assert_eq!(rlimit_value(8 << 20), Some(8 << 20));
    }

    #[test]
    fn lock_probe_missing_file_is_free() {
        let dir = scratch("missing");
        assert_eq!(probe_lock_file(&dir.join("gpu.lock"), false), GpuLock::Free);
    }

    #[test]
    fn lock_probe_names_the_holder_and_releases_immediately() {
        let dir = scratch("busy");
        let path = dir.join("gpu.lock");
        let mut holder = File::create(&path).unwrap();
        writeln!(holder, "4242 0000:c0:00.0").unwrap();
        assert_eq!(probe_lock_file(&path, false), GpuLock::Free);
        // SAFETY: valid descriptor owned by `holder`.
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        assert_eq!(
            probe_lock_file(&path, false),
            GpuLock::Busy { holder: Some(4242) }
        );
        assert_eq!(
            probe_lock_file(&path, true),
            GpuLock::Busy { holder: Some(4242) }
        );
        release(holder);
        assert_eq!(probe_lock_file(&path, false), GpuLock::Free);
    }

    /// The race behind the flaky release check, made deterministic: another
    /// thread forks a child that holds copies of every descriptor until it
    /// execs. Closing a holder leaves its lock held by that copy; unlocking
    /// frees it at once; CLOEXEC drops the copy at exec.
    #[test]
    fn lock_release_survives_a_child_forked_mid_spawn() {
        use std::os::unix::process::CommandExt;

        let dir = scratch("forked");
        let closed_path = dir.join("closed.lock");
        let unlocked_path = dir.join("unlocked.lock");
        let closed = File::create(&closed_path).unwrap();
        let unlocked = File::create(&unlocked_path).unwrap();
        for holder in [&closed, &unlocked] {
            // SAFETY: valid descriptor owned by `holder`.
            assert_eq!(
                unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                0
            );
        }
        let (mut ready_r, ready_w) = std::io::pipe().unwrap();
        let (go_r, go_w) = std::io::pipe().unwrap();
        let (ready_fd, go_fd, go_w_fd) = (ready_w.as_raw_fd(), go_r.as_raw_fd(), go_w.as_raw_fd());
        let mut command = Command::new("sleep");
        command.arg("30");
        // SAFETY: the hook runs in the forked child before exec and only calls
        // the async-signal-safe `close`, `write` and `read`.
        unsafe {
            command.pre_exec(move || {
                libc::close(go_w_fd);
                libc::write(ready_fd, [1u8].as_ptr().cast(), 1);
                let mut byte = 0u8;
                libc::read(go_fd, (&mut byte as *mut u8).cast(), 1);
                Ok(())
            });
        }
        // `spawn` returns only after the child execs, so it runs on its own
        // thread, like a concurrent test starting a process.
        let spawner = std::thread::spawn(move || {
            let child = command.spawn();
            drop(ready_w);
            child
        });
        let mut byte = [0u8];
        assert_eq!(ready_r.read(&mut byte).unwrap(), 1, "child never forked");

        drop(closed);
        assert_eq!(
            probe_lock_file(&closed_path, false),
            GpuLock::Busy { holder: None }
        );
        release(unlocked);
        assert_eq!(probe_lock_file(&unlocked_path, false), GpuLock::Free);

        drop(go_w);
        let mut child = spawner.join().unwrap().unwrap();
        let leaked = std::fs::read_dir(format!("/proc/{}/fd", child.id()))
            .unwrap()
            .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
            .filter(|target| *target == closed_path || *target == unlocked_path)
            .collect::<Vec<_>>();
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(
            leaked.is_empty(),
            "lock descriptors survived exec: {leaked:?}"
        );
    }

    #[test]
    fn lock_probe_shared_coexists_with_a_shared_holder() {
        let dir = scratch("shared");
        let path = dir.join("gpu.lock");
        let holder = File::create(&path).unwrap();
        // SAFETY: valid descriptor owned by `holder`.
        assert_eq!(
            unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) },
            0
        );
        assert_eq!(probe_lock_file(&path, true), GpuLock::Free);
        assert_eq!(
            probe_lock_file(&path, false),
            GpuLock::Busy { holder: None }
        );
    }

    #[test]
    fn lock_dirs_follow_the_daemon_rules() {
        let machine = PathBuf::from("/run/lock/hipfire");
        let tmp = PathBuf::from("/tmp/hipfire-locks");
        assert_eq!(lock_dirs(false, None), [machine.clone(), tmp]);
        assert_eq!(
            lock_dirs(false, Some(PathBuf::from("/x"))),
            [PathBuf::from("/x")]
        );
        // Fabric admission ignores a private override.
        assert_eq!(lock_dirs(true, Some(PathBuf::from("/x"))), [machine]);
    }
}
