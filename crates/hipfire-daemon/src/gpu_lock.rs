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
            return Err(format!(
                "GPU lock directory must be absolute: {}",
                path.display()
            ));
        }
        if !path.exists() {
            std::fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o1777))
                .map_err(|e| format!("set permissions on {}: {e}", path.display()))?;
        }
        if !path.is_dir() {
            return Err(format!(
                "GPU lock path is not a directory: {}",
                path.display()
            ));
        }
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|e| format!("invalid GPU lock directory {}: {e}", path.display()))?;
        if unsafe { libc::access(c_path.as_ptr(), libc::W_OK | libc::X_OK) } != 0 {
            return Err(format!(
                "GPU lock directory is not writable: {}",
                path.display()
            ));
        }
        Ok(())
    }

    pub(super) fn default_dirs() -> Vec<PathBuf> {
        vec![
            PathBuf::from("/run/lock/hipfire"),
            PathBuf::from("/tmp/hipfire-locks"),
        ]
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
        flock_nb(file, libc::LOCK_EX)
    }

    /// Shared variant for opted-in weight-host tenants: coexists with other
    /// shared holders (the fabric wrapper, the weight host, other tenants)
    /// and with nothing exclusive.
    pub(super) fn try_lock_shared(file: &File) -> std::io::Result<bool> {
        flock_nb(file, libc::LOCK_SH)
    }

    fn flock_nb(file: &File, mode: libc::c_int) -> std::io::Result<bool> {
        if unsafe { libc::flock(file.as_raw_fd(), mode | libc::LOCK_NB) } == 0 {
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
            return Err(format!(
                "GPU lock directory must be absolute: {}",
                path.display()
            ));
        }
        std::fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
        if !path.is_dir() {
            return Err(format!(
                "GPU lock path is not a directory: {}",
                path.display()
            ));
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
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
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

/// How this process holds its cards.
enum ClaimKind {
    /// Default: `flock` exclusive, holder PID written into the lock file and
    /// the per-HOME PID file.
    Exclusive,
    /// `HIPFIRE_WEIGHT_HOST=1`: `flock` shared on the machine-wide lock file
    /// that the admitting fabric wrapper already holds shared. Never writes
    /// the lock file or the PID file: several shared holders coexist.
    #[cfg(target_os = "linux")]
    Shared(fabric::FabricAdmission),
}

/// Machine-wide GPU reservations held until process exit. The per-HOME PID
/// file is advisory discovery for uninstall tooling, not a mutex: two daemons
/// using different cards in one HOME must both run.
pub(crate) struct GpuLocks {
    dir: PathBuf,
    held: Vec<(String, File)>,
    kind: ClaimKind,
}

impl GpuLocks {
    pub(crate) fn open() -> Result<Self, String> {
        let dir = gpu_lock_dir()?;
        eprintln!("[gpu-lock] directory={}", dir.display());
        Ok(Self::in_dir(dir))
    }

    /// Shared-correctness claim for weight-host participants (host and
    /// tenants). `token` is the fabric admission (`HIPFIRE_FABRIC_ADMISSION`);
    /// it must name a running correct-class entry whose session contains this
    /// process. The claim always uses the machine-wide lock directory, the
    /// one exclusive daemons and the fabric contend on: a private
    /// `HIPFIRE_LOCK_DIR` (which the fabric hands every correct job for
    /// exclusive daemons) would make a shared lock meaningless.
    #[cfg(target_os = "linux")]
    pub(crate) fn open_shared(token: &str) -> Result<Self, String> {
        let admission = fabric::FabricAdmission::verify_for_pid(token, std::process::id())?;
        let dir = fabric::machine_lock_dir();
        if !dir.is_dir() {
            return Err(format!(
                "weight-host shared claim needs the machine-wide lock directory {}",
                dir.display()
            ));
        }
        if let Some(private) = std::env::var_os("HIPFIRE_LOCK_DIR") {
            eprintln!(
                "[gpu-lock] shared claim uses machine-wide {}; HIPFIRE_LOCK_DIR={} applies to exclusive claims only",
                dir.display(),
                PathBuf::from(private).display()
            );
        }
        eprintln!(
            "[gpu-lock] directory={} shared claim admitted by fabric entry {} ({})",
            dir.display(),
            admission.entry_id,
            admission.state_path.display()
        );
        Ok(Self {
            dir,
            held: Vec::new(),
            kind: ClaimKind::Shared(admission),
        })
    }

    fn in_dir(dir: PathBuf) -> Self {
        Self {
            dir,
            held: Vec::new(),
            kind: ClaimKind::Exclusive,
        }
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
        #[cfg(target_os = "linux")]
        if let ClaimKind::Shared(admission) = &self.kind {
            let file = os::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            // The admitting wrapper must hold this very card's lock shared;
            // that binds the admission to the physical card.
            admission.require_wrapper_holds(&path)?;
            if !os::try_lock_shared(&file).map_err(|e| format!("lock {}: {e}", path.display()))? {
                return Ok(Claim::Busy(
                    "held exclusively; shared weight-host claim refused".into(),
                ));
            }
            eprintln!("[gpu-lock] shared {}", path.display());
            self.held.push((identity, file));
            return Ok(Claim::Claimed);
        }
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
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|e| e.to_string())?;
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
        // Shared holders coexist on one card: a per-HOME PID file would be
        // clobbered by every co-tenant, and uninstall tooling must never
        // signal a weight-host tenant through it.
        #[cfg(target_os = "linux")]
        if matches!(self.kind, ClaimKind::Shared(_)) {
            return Ok(());
        }
        let mut identities = self
            .held
            .iter()
            .map(|(identity, _)| identity.as_str())
            .collect::<Vec<_>>();
        identities.sort_unstable();
        let hipfire_dir = Path::new(&os::home()?).join(".hipfire");
        std::fs::create_dir_all(&hipfire_dir).map_err(|e| e.to_string())?;
        let pid_path = hipfire_dir.join(format!("daemon-{}.pid", identities.join("_")));
        std::fs::write(&pid_path, format!("{}\n", std::process::id()))
            .map_err(|e| format!("write {}: {e}", pid_path.display()))
    }
}

/// Dev-fabric admission adapter (`/home/kaden/qcal/locks/fabric/fabric.py`).
///
/// A correct-class fabric job exports `HIPFIRE_FABRIC_ADMISSION=<state>#<id>`:
/// the card's state file and the job's entry id. The token is accepted only
/// when the entry is a running correct job, its child session is alive with
/// its recorded start time, the checked process belongs to that session, and
/// (per card) the wrapper holds that card's machine-wide lock shared. This is
/// same-user trust for the development fabric, not security isolation.
#[cfg(target_os = "linux")]
pub(crate) mod fabric {
    use std::os::linux::fs::MetadataExt;
    use std::path::{Path, PathBuf};

    pub(crate) const ADMISSION_ENV: &str = "HIPFIRE_FABRIC_ADMISSION";

    /// Matches fabric.py `HIPFIRE_LOCK_DIR`, the first Linux default.
    pub(crate) fn machine_lock_dir() -> PathBuf {
        PathBuf::from("/run/lock/hipfire")
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) struct FabricAdmission {
        pub(crate) state_path: PathBuf,
        pub(crate) entry_id: u64,
        /// Fabric wrapper: the process holding the card's shared locks.
        pub(crate) wrapper_pid: u32,
        /// Session leader of the admitted job (fabric `child`).
        pub(crate) session: u32,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Entry {
        wrapper_pid: u32,
        wrapper_start: u64,
        child: u32,
        child_start: u64,
    }

    impl FabricAdmission {
        /// Verify that `token` admits process `pid` right now.
        pub(crate) fn verify_for_pid(token: &str, pid: u32) -> Result<Self, String> {
            let (state_path, entry_id) = parse_token(token)?;
            let meta = std::fs::metadata(&state_path)
                .map_err(|e| format!("fabric admission: {}: {e}", state_path.display()))?;
            let uid = unsafe { libc::geteuid() };
            // fabric.py writes 0664 under the user's group umask; refuse only
            // foreign owners and world-writable state.
            if meta.st_uid() != uid || meta.st_mode() & 0o002 != 0 {
                return Err(format!(
                    "fabric admission: {} must be owned by uid {uid} and not world-writable",
                    state_path.display()
                ));
            }
            let raw = std::fs::read_to_string(&state_path)
                .map_err(|e| format!("fabric admission: read {}: {e}", state_path.display()))?;
            let entry = running_entry(&raw, entry_id)?;
            if proc_start(entry.wrapper_pid) != Some(entry.wrapper_start) {
                return Err(format!(
                    "fabric admission: entry {entry_id} wrapper {} is gone",
                    entry.wrapper_pid
                ));
            }
            if proc_start(entry.child) != Some(entry.child_start) {
                return Err(format!(
                    "fabric admission: entry {entry_id} session {} is gone",
                    entry.child
                ));
            }
            let session = proc_session(pid)
                .ok_or_else(|| format!("fabric admission: process {pid} is gone"))?;
            if session != entry.child {
                return Err(format!(
                    "fabric admission: process {pid} is in session {session}, entry {entry_id} admitted session {}",
                    entry.child
                ));
            }
            Ok(Self {
                state_path,
                entry_id,
                wrapper_pid: entry.wrapper_pid,
                session,
            })
        }

        /// The admitting wrapper must hold a shared `flock` on `lock_path`.
        pub(crate) fn require_wrapper_holds(&self, lock_path: &Path) -> Result<(), String> {
            let meta = std::fs::metadata(lock_path)
                .map_err(|e| format!("fabric admission: {}: {e}", lock_path.display()))?;
            let locks = std::fs::read_to_string("/proc/locks")
                .map_err(|e| format!("fabric admission: /proc/locks: {e}"))?;
            if holds_shared_flock(&locks, self.wrapper_pid, meta.st_dev(), meta.st_ino()) {
                Ok(())
            } else {
                Err(format!(
                    "fabric admission: entry {} wrapper {} does not hold {} shared (wrong card?)",
                    self.entry_id,
                    self.wrapper_pid,
                    lock_path.display()
                ))
            }
        }
    }

    fn parse_token(token: &str) -> Result<(PathBuf, u64), String> {
        let (path, id) = token.rsplit_once('#').ok_or_else(|| {
            format!("fabric admission token {token:?}: expected <state-json>#<entry-id>")
        })?;
        let path = PathBuf::from(path);
        if !path.is_absolute() || path.extension().and_then(|e| e.to_str()) != Some("json") {
            return Err(format!(
                "fabric admission token {token:?}: state path must be an absolute .json file"
            ));
        }
        let id = id.parse::<u64>().map_err(|_| {
            format!("fabric admission token {token:?}: entry id is not a decimal integer")
        })?;
        Ok((path, id))
    }

    fn running_entry(raw: &str, entry_id: u64) -> Result<Entry, String> {
        let state: serde_json::Value =
            serde_json::from_str(raw).map_err(|e| format!("fabric admission: state JSON: {e}"))?;
        let entry = state
            .get("entries")
            .and_then(|v| v.as_array())
            .and_then(|entries| {
                entries
                    .iter()
                    .find(|e| e.get("id").and_then(|v| v.as_u64()) == Some(entry_id))
            })
            .ok_or_else(|| format!("fabric admission: entry {entry_id} not found"))?;
        let text = |key: &str| entry.get(key).and_then(|v| v.as_str());
        let number = |key: &str| {
            entry
                .get(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("fabric admission: entry {entry_id} lacks {key}"))
        };
        if text("kind") != Some("correct") || text("state") != Some("running") {
            return Err(format!(
                "fabric admission: entry {entry_id} is not a running correct job"
            ));
        }
        let pid = |key: &str| {
            number(key)
                .and_then(|v| u32::try_from(v).map_err(|_| format!("fabric admission: bad {key}")))
        };
        Ok(Entry {
            wrapper_pid: pid("pid")?,
            wrapper_start: number("start")?,
            child: pid("child")?,
            child_start: number("child_start")?,
        })
    }

    /// `/proc/<pid>/stat` field after the command: index 3 = session,
    /// index 19 = starttime (the same parse fabric.py uses).
    fn stat_field(pid: u32, index: usize) -> Option<u64> {
        let raw = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        raw.get(raw.rfind(')')? + 2..)?
            .split_whitespace()
            .nth(index)?
            .parse()
            .ok()
    }

    pub(crate) fn proc_start(pid: u32) -> Option<u64> {
        stat_field(pid, 19)
    }

    fn proc_session(pid: u32) -> Option<u32> {
        stat_field(pid, 3).and_then(|v| u32::try_from(v).ok())
    }

    /// glibc `dev_t` split, as printed (hex) in `/proc/locks`.
    fn dev_major_minor(dev: u64) -> (u64, u64) {
        let major = ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff);
        let minor = ((dev >> 12) & 0xffff_ff00) | (dev & 0xff);
        (major, minor)
    }

    /// True when `/proc/locks` lists a granted (not blocked) FLOCK READ by
    /// `pid` on the inode `(dev, ino)`.
    fn holds_shared_flock(locks: &str, pid: u32, dev: u64, ino: u64) -> bool {
        let want = dev_major_minor(dev);
        locks.lines().any(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.contains(&"->") || fields.len() < 6 {
                return false;
            }
            if fields[1] != "FLOCK"
                || fields[3] != "READ"
                || fields[4].parse::<u32>().ok() != Some(pid)
            {
                return false;
            }
            let mut id = fields[5].split(':');
            let parsed = (|| {
                let major = u64::from_str_radix(id.next()?, 16).ok()?;
                let minor = u64::from_str_radix(id.next()?, 16).ok()?;
                let inode = id.next()?.parse::<u64>().ok()?;
                Some(((major, minor), inode))
            })();
            parsed == Some((want, ino))
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn token_shape_is_strict() {
            assert_eq!(
                parse_token("/q/state/hiptrx-card-B.json#75").unwrap(),
                (PathBuf::from("/q/state/hiptrx-card-B.json"), 75)
            );
            for bad in [
                "",
                "75",
                "state.json#75",
                "/q/state.txt#75",
                "/q/s.json#",
                "/q/s.json#-1",
                "/q/s.json#7x",
            ] {
                assert!(parse_token(bad).is_err(), "{bad:?} must be refused");
            }
        }

        #[test]
        fn only_a_running_correct_entry_admits() {
            let state = r#"{"seq":9,"entries":[
                {"id":8,"kind":"timing","state":"running","pid":10,"start":11,"child":12,"child_start":13},
                {"id":9,"kind":"correct","state":"waiting","pid":20,"start":21,"child":null,"child_start":null},
                {"id":7,"kind":"correct","state":"running","pid":30,"start":31,"child":32,"child_start":33}]}"#;
            assert_eq!(
                running_entry(state, 7).unwrap(),
                Entry {
                    wrapper_pid: 30,
                    wrapper_start: 31,
                    child: 32,
                    child_start: 33
                }
            );
            assert!(
                running_entry(state, 8).is_err(),
                "timing entries never admit shared claims"
            );
            assert!(
                running_entry(state, 9).is_err(),
                "waiting entries are not admitted"
            );
            assert!(running_entry(state, 6).is_err(), "unknown entry");
            assert!(running_entry("{not json", 7).is_err());
        }

        #[test]
        fn proc_locks_match_needs_granted_read_by_the_wrapper_on_the_inode() {
            let dev = (0x1au64 << 8) | 0x03; // major 0x1a, minor 3
            let locks = "1: FLOCK  ADVISORY  WRITE 500 1a:03:77 0 EOF\n\
                         2: FLOCK  ADVISORY  READ  600 1a:03:78 0 EOF\n\
                         2: -> FLOCK  ADVISORY  READ  700 1a:03:77 0 EOF\n\
                         3: FLOCK  ADVISORY  READ  600 1a:03:77 0 EOF\n";
            assert!(holds_shared_flock(locks, 600, dev, 77));
            assert!(!holds_shared_flock(locks, 500, dev, 77), "exclusive holder");
            assert!(!holds_shared_flock(locks, 700, dev, 77), "blocked waiter");
            assert!(!holds_shared_flock(locks, 600, dev, 79), "other inode");
            assert!(!holds_shared_flock(locks, 600, dev + 1, 77), "other device");
        }

        #[test]
        fn live_session_check_uses_proc() {
            let me = std::process::id();
            assert!(proc_start(me).is_some());
            let sid = unsafe { libc::getsid(0) } as u32;
            assert_eq!(proc_session(me), Some(sid));
        }
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
        let dir =
            LockDir(std::env::temp_dir().join(format!("hipfire-gpu-lock-{}", std::process::id())));
        std::fs::create_dir_all(&dir.0).unwrap();
        let cards = cards();
        let mut first = GpuLocks::in_dir(dir.0.clone());
        let mut second = GpuLocks::in_dir(dir.0.clone());

        assert_eq!(first.try_claim(&cards[0]).unwrap(), Claim::Claimed);
        assert_eq!(
            first.try_claim(&cards[0]).unwrap(),
            Claim::Claimed,
            "re-claim of a held card"
        );
        let holder = format!("already reserved by holder PID {}", std::process::id());
        assert_eq!(
            second.try_claim(&cards[0]).unwrap(),
            Claim::Busy(holder.clone())
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("gpu-GPU-9eb7aeda51c88ffd.lock")).unwrap(),
            format!("{} 0000:03:00.0\n", std::process::id())
        );

        // all-or-nothing: a set containing a busy card fails
        let error = second.claim_all(&cards).unwrap_err();
        assert_eq!(
            error,
            format!("GPU GPU-9eb7aeda51c88ffd (PCI 0000:03:00.0) {holder}")
        );
        assert_eq!(second.try_claim(&cards[1]).unwrap(), Claim::Claimed);
        assert!(
            dir.0.join("gpu-pci-0000:13:00.0.lock").exists(),
            "no-UUID card locks by PCI address"
        );
        assert_eq!(
            first.claim_all(&cards).unwrap_err(),
            format!("GPU pci-0000:13:00.0 (PCI 0000:13:00.0) {holder}")
        );

        drop(second);
        drop(first);
        let mut third = GpuLocks::in_dir(dir.0.clone());
        third.claim_all(&cards).unwrap();
    }

    /// Shared weight-host claims coexist with each other and with the
    /// fabric wrapper's shared hold, conflict with exclusive daemons in both
    /// directions, never write the lock file, and need the wrapper's hold.
    #[cfg(target_os = "linux")]
    #[test]
    fn shared_claims_coexist_exclude_exclusive_and_need_the_wrapper_hold() {
        use super::{fabric::FabricAdmission, os, ClaimKind};
        let dir = LockDir(
            std::env::temp_dir().join(format!("hipfire-gpu-lock-shared-{}", std::process::id())),
        );
        std::fs::create_dir_all(&dir.0).unwrap();
        let cards = cards();
        let shared = |dir: &std::path::Path| GpuLocks {
            dir: dir.to_path_buf(),
            held: Vec::new(),
            // This test process stands in for the fabric wrapper.
            kind: ClaimKind::Shared(FabricAdmission {
                state_path: "/unused/state.json".into(),
                entry_id: 1,
                wrapper_pid: std::process::id(),
                session: 0,
            }),
        };
        let lock_path = dir.0.join(cards[0].lock_file_name());

        let error = shared(&dir.0).try_claim(&cards[0]).unwrap_err();
        assert!(error.contains("does not hold"), "no wrapper hold: {error}");

        let wrapper = os::open(&lock_path).unwrap();
        assert!(os::try_lock_shared(&wrapper).unwrap());
        let mut host = shared(&dir.0);
        let mut tenant = shared(&dir.0);
        assert_eq!(host.try_claim(&cards[0]).unwrap(), Claim::Claimed);
        assert_eq!(tenant.try_claim(&cards[0]).unwrap(), Claim::Claimed);
        tenant.write_pid_file().unwrap();
        assert_eq!(
            std::fs::read_to_string(&lock_path).unwrap(),
            "",
            "shared claims never write the lock file"
        );

        let mut exclusive = GpuLocks::in_dir(dir.0.clone());
        assert!(matches!(
            exclusive.try_claim(&cards[0]).unwrap(),
            Claim::Busy(_)
        ));

        drop(host);
        drop(tenant);
        drop(wrapper);
        assert_eq!(exclusive.try_claim(&cards[0]).unwrap(), Claim::Claimed);
        let wrapper = os::open(&lock_path).unwrap();
        assert!(
            !os::try_lock_shared(&wrapper).unwrap(),
            "exclusive daemon blocks the fabric hold"
        );
    }
}
