// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Machine-wide per-GPU reservations.
//!
//! One lock file per card, keyed by [`GpuDevice::lock_identity`]:
//! `gpu-GPU-<uuid>.lock`, or `gpu-pci-<bdf>.lock` for cards without a UUID.
//! The holder takes an exclusive, non-blocking OS lock on the file and writes
//! `"<pid> <bdf>"` into it, so a busy claim can name its holder. The OS drops
//! the lock when the process exits, however it exits. Unix uses `flock`;
//! Windows uses `LockFileEx` on a byte range past the PID text, because
//! Windows byte-range locks are mandatory and would otherwise stop the loser
//! from reading the holder PID. Everything above the two OS primitives is
//! shared, so the Linux unit tests exercise the Windows semantics too.

use hipfire_config::devices::{Claim, GpuDevice};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
mod os {
    use std::fs::File;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::os::unix::io::AsRawFd;
    use std::path::{Path, PathBuf};

    pub(super) fn prepare_dir(path: &Path) -> Result<(), String> {
        if !path.is_absolute() {
            return Err(format!("GPU lock directory must be absolute: {}", path.display()));
        }
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o1777))
                .map_err(|e| format!("set permissions on {}: {e}", path.display()))?;
        }
        if !path.is_dir() {
            return Err(format!("GPU lock path is not a directory: {}", path.display()));
        }
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|e| format!("invalid GPU lock directory {}: {e}", path.display()))?;
        if unsafe { libc::access(c_path.as_ptr(), libc::W_OK | libc::X_OK) } != 0 {
            return Err(format!("GPU lock directory is not writable: {}", path.display()));
        }
        Ok(())
    }

    pub(super) fn default_dirs() -> Vec<PathBuf> {
        vec![PathBuf::from("/run/lock/hipfire"), PathBuf::from("/tmp/hipfire-locks")]
    }

    pub(super) fn open(path: &Path) -> std::io::Result<File> {
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o666)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
    }

    /// `Ok(false)` when another open file description holds the lock.
    pub(super) fn try_lock(file: &File) -> std::io::Result<bool> {
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.kind() == std::io::ErrorKind::WouldBlock {
            Ok(false)
        } else {
            Err(error)
        }
    }

    /// Loosen files created under a restrictive umask for other HOME/users.
    pub(super) fn after_claim(path: &Path) {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666));
    }

    pub(super) fn home() -> Result<String, String> {
        std::env::var("HOME").map_err(|e| format!("HOME: {e}"))
    }
}

#[cfg(windows)]
mod os {
    use std::ffi::c_void;
    use std::fs::File;
    use std::os::windows::io::AsRawHandle;
    use std::path::{Path, PathBuf};

    const LOCKFILE_FAIL_IMMEDIATELY: u32 = 0x1;
    const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x2;
    const ERROR_LOCK_VIOLATION: i32 = 33;
    /// Locked byte: far past the PID text, so the holder's mandatory lock
    /// never blocks a contender from reading who holds the card.
    const LOCK_OFFSET: u64 = 1 << 62;

    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        event: *mut c_void,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn LockFileEx(
            file: *mut c_void,
            flags: u32,
            reserved: u32,
            bytes_low: u32,
            bytes_high: u32,
            overlapped: *mut Overlapped,
        ) -> i32;
    }

    pub(super) fn prepare_dir(path: &Path) -> Result<(), String> {
        if !path.is_absolute() {
            return Err(format!("GPU lock directory must be absolute: {}", path.display()));
        }
        std::fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        if !path.is_dir() {
            return Err(format!("GPU lock path is not a directory: {}", path.display()));
        }
        Ok(())
    }

    /// `%ProgramData%\hipfire\locks` is machine-wide; the per-user temp
    /// fallback only contends with daemons of the same user.
    pub(super) fn default_dirs() -> Vec<PathBuf> {
        let program_data = std::env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"));
        vec![
            program_data.join("hipfire").join("locks"),
            std::env::temp_dir().join("hipfire-locks"),
        ]
    }

    /// std opens with FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    /// so a contender can open the file and read the holder PID.
    pub(super) fn open(path: &Path) -> std::io::Result<File> {
        std::fs::OpenOptions::new().read(true).write(true).create(true).open(path)
    }

    /// `Ok(false)` when another handle holds the lock.
    pub(super) fn try_lock(file: &File) -> std::io::Result<bool> {
        let mut overlapped = Overlapped {
            internal: 0,
            internal_high: 0,
            offset: LOCK_OFFSET as u32,
            offset_high: (LOCK_OFFSET >> 32) as u32,
            event: std::ptr::null_mut(),
        };
        // SAFETY: the handle is owned by `file` for the whole call; the
        // OVERLAPPED is a stack local only used for this synchronous request.
        let ok = unsafe {
            LockFileEx(
                file.as_raw_handle(),
                LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0,
                1,
                0,
                &mut overlapped,
            )
        };
        if ok != 0 {
            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(ERROR_LOCK_VIOLATION) {
            Ok(false)
        } else {
            Err(error)
        }
    }

    pub(super) fn after_claim(_: &Path) {}

    pub(super) fn home() -> Result<String, String> {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|e| format!("HOME/USERPROFILE: {e}"))
    }
}

fn gpu_lock_dir() -> Result<PathBuf, String> {
    // Bootstrap read: the lock directory is resolved before the process
    // config exists (scripts/check-env-docs.py BOOTSTRAP_ENV).
    if let Some(override_dir) = std::env::var_os("HIPFIRE_LOCK_DIR") {
        let dir = PathBuf::from(override_dir);
        os::prepare_dir(&dir)?;
        return Ok(dir);
    }
    let mut last_error = String::from("no GPU lock directory candidates");
    for dir in os::default_dirs() {
        match os::prepare_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

/// Machine-wide GPU reservations held until process exit. The per-HOME PID
/// file is advisory discovery for uninstall tooling, not a mutex: two daemons
/// using different cards in one HOME must both run.
pub(crate) struct GpuLocks {
    dir: PathBuf,
    held: Vec<(String, File)>,
}

impl GpuLocks {
    pub(crate) fn open() -> Result<Self, String> {
        let dir = gpu_lock_dir()?;
        eprintln!("[gpu-lock] directory={}", dir.display());
        Ok(Self::in_dir(dir))
    }

    fn in_dir(dir: PathBuf) -> Self {
        Self { dir, held: Vec::new() }
    }

    /// Non-blocking reservation; a busy card reports its holder so arch
    /// selectors can move on to the next card. Never waits, so the claim
    /// order cannot deadlock against another daemon.
    pub(crate) fn try_claim(&mut self, device: &GpuDevice) -> Result<Claim, String> {
        let identity = device.lock_identity();
        if self.held.iter().any(|(held, _)| *held == identity) {
            return Ok(Claim::Claimed);
        }
        let path = self.dir.join(device.lock_file_name());
        let mut file = os::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if !os::try_lock(&file).map_err(|e| format!("lock {}: {e}", path.display()))? {
            let mut holder = String::new();
            let _ = file.read_to_string(&mut holder);
            return Ok(Claim::Busy(format!(
                "already reserved by holder PID {}",
                holder.split_whitespace().next().unwrap_or("<unknown>")
            )));
        }
        // The lock file stays on disk forever; never unlink an active inode.
        os::after_claim(&path);
        file.set_len(0).map_err(|e| e.to_string())?;
        file.seek(std::io::SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        writeln!(file, "{} {}", std::process::id(), device.bdf).map_err(|e| e.to_string())?;
        file.flush().map_err(|e| e.to_string())?;
        eprintln!("[gpu-lock] reserved {}", path.display());
        self.held.push((identity, file));
        Ok(Claim::Claimed)
    }

    /// Reserve every card in `devices`, sorted by identity; any busy card
    /// fails the whole set (held handles close on process exit).
    pub(crate) fn claim_all(&mut self, devices: &[GpuDevice]) -> Result<(), String> {
        let mut sorted = devices.iter().collect::<Vec<_>>();
        sorted.sort_by_key(|device| device.lock_identity());
        for device in sorted {
            if let Claim::Busy(holder) = self.try_claim(device)? {
                return Err(format!(
                    "GPU {} (PCI {}) {holder}",
                    device.lock_identity(),
                    device.bdf
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn write_pid_file(&self) -> Result<(), String> {
        if self.held.is_empty() {
            return Err("no visible GPUs to reserve".into());
        }
        let mut identities = self.held.iter().map(|(identity, _)| identity.as_str()).collect::<Vec<_>>();
        identities.sort_unstable();
        let hipfire_dir = Path::new(&os::home()?).join(".hipfire");
        std::fs::create_dir_all(&hipfire_dir).map_err(|e| e.to_string())?;
        let pid_path = hipfire_dir.join(format!("daemon-{}.pid", identities.join("_")));
        std::fs::write(&pid_path, format!("{}\n", std::process::id()))
            .map_err(|e| format!("write {}: {e}", pid_path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::GpuLocks;
    use hipfire_config::devices::{devices_from_hip, Claim, GpuDevice, ObservedDevice};

    struct LockDir(std::path::PathBuf);

    impl Drop for LockDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn cards() -> Vec<GpuDevice> {
        devices_from_hip(&[
            ObservedDevice {
                logical: 0,
                arch: "gfx1201".into(),
                pci_bus_id: "0000:03:00.0".into(),
                uuid: Some("GPU-9eb7aeda51c88ffd".into()),
            },
            ObservedDevice {
                logical: 1,
                arch: "gfx1201".into(),
                pci_bus_id: "0000:13:00.0".into(),
                uuid: None,
            },
        ])
        .unwrap()
    }

    /// Two reservation sets in one process stand in for two daemons: flock
    /// and LockFileEx both conflict per open file handle.
    #[test]
    fn claims_are_exclusive_name_the_holder_and_release_on_drop() {
        let dir = LockDir(std::env::temp_dir().join(format!("hipfire-gpu-lock-{}", std::process::id())));
        std::fs::create_dir_all(&dir.0).unwrap();
        let cards = cards();
        let mut first = GpuLocks::in_dir(dir.0.clone());
        let mut second = GpuLocks::in_dir(dir.0.clone());

        assert_eq!(first.try_claim(&cards[0]).unwrap(), Claim::Claimed);
        assert_eq!(first.try_claim(&cards[0]).unwrap(), Claim::Claimed, "re-claim of a held card");
        let holder = format!("already reserved by holder PID {}", std::process::id());
        assert_eq!(second.try_claim(&cards[0]).unwrap(), Claim::Busy(holder.clone()));
        assert_eq!(
            std::fs::read_to_string(dir.0.join("gpu-GPU-9eb7aeda51c88ffd.lock")).unwrap(),
            format!("{} 0000:03:00.0\n", std::process::id())
        );

        // all-or-nothing: a set containing a busy card fails
        let error = second.claim_all(&cards).unwrap_err();
        assert_eq!(error, format!("GPU GPU-9eb7aeda51c88ffd (PCI 0000:03:00.0) {holder}"));
        assert_eq!(second.try_claim(&cards[1]).unwrap(), Claim::Claimed);
        assert!(dir.0.join("gpu-pci-0000:13:00.0.lock").exists(), "no-UUID card locks by PCI address");
        assert_eq!(
            first.claim_all(&cards).unwrap_err(),
            format!("GPU pci-0000:13:00.0 (PCI 0000:13:00.0) {holder}")
        );

        drop(second);
        drop(first);
        let mut third = GpuLocks::in_dir(dir.0.clone());
        third.claim_all(&cards).unwrap();
    }
}
