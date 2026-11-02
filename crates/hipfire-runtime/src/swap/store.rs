// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// SwapStore — where a swapped-out session's snapshot lives.
//
// Two tiers: host memory up to a budget, then scratch files up to a second
// budget. Both are bounded caches: a snapshot is only an accelerator for the
// next turn, and the session's tokens are authoritative, so evicting the LRU
// snapshot costs a re-prefill and never correctness.
//
// Spill files are SCRATCH: stale `*.snap` files are removed on construction,
// and each file is removed when taken, dropped, or evicted. The directory
// itself is operator-configurable, so unrelated files are never touched.
//
// The directory is also SECURITY-SENSITIVE: it holds conversation tokens and
// KV for a live session. It is created 0700, must be owned by this user, is
// locked for the store's lifetime (one daemon per directory), and spill files
// are created 0600 with `create_new` so a planted symlink is never followed.
// They still carry a full stamp, so making them durable later is a policy
// change rather than a format change.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use crate::swap::snapshot::SlotSnapshot;
use crate::swap::SwapError;

/// Held for the store's lifetime so two processes can never share `dir`.
/// Left in place after Drop: unlinking a lock file while a `try_lock` waiter
/// holds the new inode would let two processes believe they own the directory.
const LOCK_FILE: &str = ".hipfire-swap.lock";

enum Entry {
    Host(SlotSnapshot),
    Disk { path: PathBuf },
}

struct Stored {
    entry: Entry,
    /// Charged bytes: payload length for a host snapshot, encoded file length
    /// for a disk snapshot.
    bytes: u64,
    /// Monotonic insertion stamp; the lowest live stamp is the LRU victim.
    last_used: u64,
}

pub struct SwapStore {
    dir: PathBuf,
    /// True when this store created `dir`; only then does shutdown remove it.
    created_dir: bool,
    /// Exclusive ownership of `dir` for this process. `None` means the disk
    /// tier is disabled (a read-only directory cannot be locked or written).
    _lock: Option<File>,
    host_budget: u64,
    host_used: u64,
    disk_budget: u64,
    disk_used: u64,
    clock: u64,
    entries: HashMap<u64, Stored>,
}

fn is_spill_name(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("snap")
}

/// A stale spill or an interrupted write's temp file. Both are scratch and
/// are removed on open/drop; unrelated files are never touched.
fn is_stale_spill(path: &Path) -> bool {
    is_spill_name(path)
        || path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".snap.tmp"))
}

/// Remove only this store's spill files. Best effort by design: a stale
/// `*.snap` is unreachable (the entry map starts empty), so a file that
/// cannot be removed is a warning, never a load failure.
fn remove_spill_files(dir: &Path) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!(
                "WARNING swap: cannot read {} ({e}); leaving stale spills",
                dir.display()
            );
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!(
                    "WARNING swap: cannot read an entry in {} ({e})",
                    dir.display()
                );
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(e) => {
                eprintln!("WARNING swap: cannot stat {} ({e})", path.display());
                continue;
            }
        };
        // Remove stale spills and stale spill *symlinks*. Removing a symlink
        // never touches its target, and leaving one behind is misleading.
        if is_stale_spill(&path) && (file_type.is_file() || file_type.is_symlink()) {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!(
                    "WARNING swap: cannot remove stale spill {} ({e})",
                    path.display()
                );
            }
        }
    }
}

/// Remove a spill file, reporting whether its disk charge is now free.
/// `NotFound` counts as free (someone else removed it); any other error keeps
/// the bytes charged, so a failed unlink can never under-count the budget.
fn remove_spill(path: &Path) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::NotFound => true,
        Err(e) => {
            eprintln!(
                "WARNING swap: cannot remove {} ({e}); its bytes stay charged",
                path.display()
            );
            false
        }
    }
}

#[cfg(unix)]
fn current_uid() -> u32 {
    // SAFETY: `geteuid` is a pure syscall wrapper with no memory arguments.
    unsafe { libc::geteuid() }
}

/// Create or validate `dir` as a private directory owned by this user.
///
/// Returns `(created, disk_budget)`. A `disk_budget` of 0 means the disk tier
/// is disabled because the directory could not be made private; the host tier
/// and the rest of serve stay usable.
fn prepare_dir(dir: &Path, disk_budget: u64) -> Result<(bool, u64), SwapError> {
    // Refuse a symlinked root. `serve.swap_dir` is operator-supplied and the
    // fallback lives in a predictable temp path, so following a link would
    // let another user redirect the spill files.
    let existed = match std::fs::symlink_metadata(dir) {
        Ok(meta) => {
            if meta.file_type().is_symlink() {
                return Err(SwapError::Io(format!(
                    "swap directory {} is a symlink; pass the resolved directory",
                    dir.display()
                )));
            }
            if !meta.is_dir() {
                return Err(SwapError::Io(format!(
                    "swap directory {} exists and is not a directory",
                    dir.display()
                )));
            }
            true
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        Err(e) => return Err(SwapError::Io(e.to_string())),
    };
    if !existed {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)
                .map_err(|e| SwapError::Io(e.to_string()))?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(dir).map_err(|e| SwapError::Io(e.to_string()))?;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let meta = std::fs::symlink_metadata(dir).map_err(|e| SwapError::Io(e.to_string()))?;
        if meta.uid() != current_uid() {
            return Err(SwapError::Io(format!(
                "swap directory {} is owned by uid {} (this process is uid {}); \
                 choose a private directory",
                dir.display(),
                meta.uid(),
                current_uid()
            )));
        }
        // Strip group/other access. Keep an owner who deliberately removed
        // write permission (a read-only dir must stay read-only); the lock
        // then fails and the disk tier is disabled.
        if meta.mode() & 0o077 != 0 {
            let mut perms = meta.permissions();
            perms.set_mode(meta.mode() & 0o700);
            if let Err(e) = std::fs::set_permissions(dir, perms) {
                eprintln!(
                    "WARNING swap: cannot make {} private ({e}); disk tier disabled",
                    dir.display()
                );
                return Ok((!existed, 0));
            }
        }
    }

    Ok((!existed, disk_budget))
}

/// Take an exclusive, non-blocking lock on `dir`. A read-only directory
/// (permission denied) degrades to `None` — the caller disables the disk
/// tier — while a lock held by another process is a hard error.
#[cfg(unix)]
fn lock_dir(dir: &Path) -> Result<Option<File>, SwapError> {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join(LOCK_FILE);
    if let Ok(meta) = std::fs::symlink_metadata(&path) {
        if meta.file_type().is_symlink() {
            return Err(SwapError::Io(format!(
                "swap lock {} is a symlink; refusing to follow it",
                path.display()
            )));
        }
    }
    let file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(e)
            if e.kind() == io::ErrorKind::PermissionDenied
                || e.raw_os_error() == Some(libc::EROFS) =>
        {
            eprintln!(
                "WARNING swap: cannot create {} ({e}); disk tier disabled",
                path.display()
            );
            return Ok(None);
        }
        Err(e) => return Err(SwapError::Io(e.to_string())),
    };
    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(e) => Err(SwapError::Io(format!(
            "swap directory {} is already in use by another hipfire process ({e:?})",
            dir.display()
        ))),
    }
}

#[cfg(not(unix))]
fn lock_dir(_dir: &Path) -> Result<Option<File>, SwapError> {
    Ok(None)
}

/// Write a snapshot to `path` without following a symlink and without
/// materialising a second copy of the payload.
fn write_snapshot(path: &Path, snap: &SlotSnapshot) -> io::Result<()> {
    // A stale spill or a planted symlink is unlinked (not followed); a
    // directory is left alone and fails the write with a clear error.
    match std::fs::symlink_metadata(path) {
        Ok(meta) => {
            if meta.file_type().is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "swap spill path is a directory",
                ));
            }
            std::fs::remove_file(path)?;
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    // `create_new` is O_CREAT|O_EXCL: it never follows a symlink, including
    // one planted between the unlink above and this open.
    let mut file = options.open(path)?;
    snap.write_to(&mut file)
}

impl SwapStore {
    /// Open a store rooted at `dir`, removing stale spill files.
    ///
    /// The directory is created 0700, must be owned by this user, and is
    /// locked exclusively for this store's lifetime. Stale snapshots are
    /// unreachable from this process (`entries` starts empty), so a cleanup
    /// failure is warned about and the disk tier is disabled rather than
    /// failing the model load.
    pub fn new(
        dir: PathBuf,
        host_budget_bytes: u64,
        disk_budget_bytes: u64,
    ) -> Result<Self, SwapError> {
        let (created_dir, disk_budget) = prepare_dir(&dir, disk_budget_bytes)?;
        // Lock before cleanup: a second process must not delete this one's
        // live spill files on its way to failing.
        let lock = lock_dir(&dir)?;
        #[cfg(unix)]
        let disk_budget = if lock.is_some() { disk_budget } else { 0 };
        #[cfg(not(unix))]
        let disk_budget = disk_budget;
        remove_spill_files(&dir);
        Ok(SwapStore {
            dir,
            created_dir,
            _lock: lock,
            host_budget: host_budget_bytes,
            host_used: 0,
            disk_budget,
            disk_used: 0,
            clock: 0,
            entries: HashMap::new(),
        })
    }

    pub fn host_used(&self) -> u64 {
        self.host_used
    }

    /// Effective host-tier ceiling (bytes).
    pub fn host_budget(&self) -> u64 {
        self.host_budget
    }

    pub fn disk_used(&self) -> u64 {
        self.disk_used
    }

    /// Effective disk-tier ceiling: 0 when the tier is disabled because the
    /// directory could not be made private or locked.
    pub fn disk_budget(&self) -> u64 {
        self.disk_budget
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `"host"`, `"disk"`, or `None` when the key is not stored.
    pub fn tier_of(&self, key: u64) -> Option<&'static str> {
        self.entries.get(&key).map(|e| match e.entry {
            Entry::Host(_) => "host",
            Entry::Disk { .. } => "disk",
        })
    }

    fn path_for(&self, key: u64) -> PathBuf {
        self.dir.join(format!("{key}.snap"))
    }

    /// Evict the least-recently-used disk snapshot. Returns false when the disk
    /// tier holds nothing to evict.
    fn evict_lru_disk(&mut self) -> bool {
        let victim = self
            .entries
            .iter()
            .filter(|(_, s)| matches!(s.entry, Entry::Disk { .. }))
            .min_by_key(|(_, s)| s.last_used)
            .map(|(key, _)| *key);
        match victim {
            Some(key) => {
                self.drop_key(key);
                true
            }
            None => false,
        }
    }

    /// Store a snapshot, in host memory if it fits the budget, else on disk.
    /// A snapshot larger than the whole disk budget is refused *before* any
    /// eviction, so it cannot destroy the cache on its way out.
    pub fn put(&mut self, key: u64, snap: SlotSnapshot) -> Result<(), SwapError> {
        self.drop_key(key);
        self.clock += 1;
        let stamp = self.clock;
        let bytes = snap.payload.len() as u64;
        if bytes <= self.host_budget && self.host_used.saturating_add(bytes) <= self.host_budget {
            self.host_used += bytes;
            self.entries.insert(
                key,
                Stored {
                    entry: Entry::Host(snap),
                    bytes,
                    last_used: stamp,
                },
            );
            return Ok(());
        }
        let n = snap.encoded_len() as u64;
        if n > self.disk_budget {
            return Err(SwapError::Io(format!(
                "swap disk budget {} B cannot hold {n} B snapshot",
                self.disk_budget
            )));
        }
        // Refuse before touching the cache when even evicting every disk
        // entry could not make room. The write goes to a temp file first:
        // a failed write then costs no parked snapshot.
        let evictable: u64 = self
            .entries
            .values()
            .filter(|stored| matches!(stored.entry, Entry::Disk { .. }))
            .map(|stored| stored.bytes)
            .fold(0u64, u64::saturating_add);
        let free = self.disk_budget.saturating_sub(self.disk_used);
        if n > free.saturating_add(evictable) {
            return Err(SwapError::Io(format!(
                "swap disk budget {} B cannot hold {n} B snapshot",
                self.disk_budget
            )));
        }
        let path = self.path_for(key);
        let tmp = path.with_extension("snap.tmp");
        if let Err(e) = write_snapshot(&tmp, &snap) {
            let _ = std::fs::remove_file(&tmp);
            return Err(SwapError::Io(e.to_string()));
        }
        while self.disk_used.saturating_add(n) > self.disk_budget {
            if !self.evict_lru_disk() {
                let _ = std::fs::remove_file(&tmp);
                return Err(SwapError::Io(format!(
                    "swap disk budget {} B cannot hold {n} B snapshot",
                    self.disk_budget
                )));
            }
        }
        if let Err(e) = std::fs::rename(&tmp, &path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(SwapError::Io(e.to_string()));
        }
        self.disk_used += n;
        self.entries.insert(
            key,
            Stored {
                entry: Entry::Disk { path },
                bytes: n,
                last_used: stamp,
            },
        );
        Ok(())
    }

    /// Remove and return a snapshot. The key does not survive a successful
    /// take: a session is either resident or swapped, never both.
    pub fn take(&mut self, key: u64) -> Result<SlotSnapshot, SwapError> {
        match self.entries.remove(&key) {
            None => Err(SwapError::Io(format!("no snapshot for key {key}"))),
            Some(Stored {
                entry: Entry::Host(snap),
                bytes,
                ..
            }) => {
                self.host_used = self.host_used.saturating_sub(bytes);
                Ok(snap)
            }
            Some(Stored {
                entry: Entry::Disk { path },
                bytes,
                ..
            }) => {
                let data = std::fs::read(&path);
                // Remove and uncharge regardless of whether the read or parse
                // succeeds: a corrupt file is never going to parse, and leaving
                // it behind strands both the inode and the disk budget. A
                // failed unlink keeps the charge, so the budget never lies.
                if remove_spill(&path) {
                    self.disk_used = self.disk_used.saturating_sub(bytes);
                }
                SlotSnapshot::from_bytes(&data.map_err(|e| SwapError::Io(e.to_string()))?)
            }
        }
    }

    /// Forget a key, freeing its budget or deleting its file. Idempotent.
    pub fn drop_key(&mut self, key: u64) {
        if let Some(stored) = self.entries.remove(&key) {
            match stored.entry {
                Entry::Host(_) => {
                    self.host_used = self.host_used.saturating_sub(stored.bytes);
                }
                Entry::Disk { path } => {
                    if remove_spill(&path) {
                        self.disk_used = self.disk_used.saturating_sub(stored.bytes);
                    }
                }
            }
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl Drop for SwapStore {
    fn drop(&mut self) {
        // Scratch means scratch, but only the files this store created. Remove
        // the directory itself only when this store created it. The lock is
        // released here; `try_lock` means no process waits on the old inode.
        remove_spill_files(&self.dir);
        self._lock = None;
        let _ = std::fs::remove_file(self.dir.join(LOCK_FILE));
        if self.created_dir {
            let _ = std::fs::remove_dir(&self.dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swap::snapshot::{checksum_of, SnapshotStamp};

    fn tmpdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hipfire-swap-test-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn small_snap() -> SlotSnapshot {
        let payload = vec![3u8; 128];
        let checksum = checksum_of(&payload);
        SlotSnapshot {
            stamp: SnapshotStamp {
                model_hash: 1,
                instance_id: 1,
                kv_dtype_tag: 1,
                per_pos_bytes: 4,
                per_pos_v_bytes: 4,
                n_fa_layers: 2,
                dn_layout_version: 1,
                cap: 128,
                dn_bytes: 64,
            },
            seq_len: 8,
            tokens: vec![4, 5, 6],
            payload,
            checksum,
        }
    }

    fn spill_entries(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| is_spill_name(p))
            .collect()
    }

    #[test]
    fn a_snapshot_under_budget_stays_in_host_memory() {
        let mut s = SwapStore::new(tmpdir("host"), 1 << 20, 1 << 30).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), Some("host"));
        assert!(s.host_used() > 0);
    }

    #[test]
    fn exceeding_the_budget_spills_to_disk() {
        let mut s = SwapStore::new(tmpdir("disk"), 64, 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), Some("disk"));
        assert_eq!(s.host_used(), 0, "a spilled snapshot holds no host budget");
    }

    #[test]
    fn take_returns_the_same_bytes_from_either_tier() {
        for (i, budget) in [1u64 << 20, 64].into_iter().enumerate() {
            let mut s = SwapStore::new(tmpdir(&format!("rt{i}")), budget, 1 << 20).unwrap();
            let original = small_snap();
            s.put(1, original.clone()).unwrap();
            let back = s.take(1).unwrap();
            assert_eq!(back.payload, original.payload);
            assert_eq!(back.tokens, original.tokens);
            assert_eq!(back.seq_len, original.seq_len);
            assert_eq!(back.stamp, original.stamp);
        }
    }

    #[test]
    fn take_frees_the_budget_and_the_key() {
        let mut s = SwapStore::new(tmpdir("free"), 1 << 20, 1 << 30).unwrap();
        s.put(1, small_snap()).unwrap();
        let _ = s.take(1).unwrap();
        assert_eq!(s.host_used(), 0);
        assert!(s.take(1).is_err(), "a taken key must not resolve twice");
    }

    #[test]
    fn a_truncated_spill_file_is_an_error_not_a_panic() {
        let dir = tmpdir("trunc");
        let mut s = SwapStore::new(dir.clone(), 64, 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        let f = spill_entries(&dir)
            .into_iter()
            .find(|p| p.file_name().unwrap() == "1.snap")
            .expect("1.snap");
        assert!(s.disk_used() > 0);
        std::fs::write(&f, b"short").unwrap();
        assert!(s.take(1).is_err());
        assert_eq!(s.disk_used(), 0, "a corrupt spill must release its budget");
        assert!(!f.exists(), "a corrupt spill must be removed");
    }

    #[test]
    fn new_removes_stale_spills_and_keeps_other_files() {
        let dir = tmpdir("stale");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("1.snap"), b"stale").unwrap();
        std::fs::write(dir.join("keep.txt"), b"keep").unwrap();
        std::fs::create_dir_all(dir.join("keepdir")).unwrap();
        let s = SwapStore::new(dir.clone(), 64, 1 << 20).unwrap();
        assert!(!dir.join("1.snap").exists(), "a stale spill is removed");
        assert!(dir.join("keep.txt").exists(), "operator data is preserved");
        assert!(dir.join("keepdir").is_dir(), "subdirectories are preserved");
        drop(s);
        assert!(
            dir.join("keep.txt").exists(),
            "shutdown must not remove operator data"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn new_removes_a_stale_spill_symlink_without_touching_its_target() {
        let dir = tmpdir("stale-link");
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("target.txt");
        std::fs::write(&target, b"keep").unwrap();
        std::os::unix::fs::symlink(&target, dir.join("1.snap")).unwrap();
        let s = SwapStore::new(dir.clone(), 64, 1 << 20).unwrap();
        assert!(!dir.join("1.snap").exists(), "stale spill symlink removed");
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"keep",
            "the symlink target is not touched"
        );
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_rejects_a_path_that_is_a_file() {
        let dir = tmpdir("notdir");
        std::fs::write(&dir, b"file").unwrap();
        let err = SwapStore::new(dir.clone(), 64, 1 << 20)
            .err()
            .expect("a file at the swap path is refused");
        assert!(format!("{err}").contains("not a directory"));
        assert!(dir.is_file(), "the existing file must not be deleted");
        let _ = std::fs::remove_file(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn new_rejects_a_symlinked_directory() {
        let dir = tmpdir("root-link");
        let target = tmpdir("root-link-target");
        std::fs::create_dir_all(&target).unwrap();
        std::os::unix::fs::symlink(&target, &dir).unwrap();
        let err = SwapStore::new(dir.clone(), 64, 1 << 20)
            .err()
            .expect("a symlinked swap dir is refused");
        assert!(format!("{err}").contains("symlink"), "{err}");
        assert!(target.is_dir(), "the symlink target must not be touched");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[test]
    fn a_directory_created_by_the_store_is_removed_on_drop() {
        let dir = tmpdir("created-drop");
        let s = SwapStore::new(dir.clone(), 64, 1 << 20).unwrap();
        assert!(dir.is_dir());
        drop(s);
        assert!(!dir.exists());
    }

    #[test]
    fn host_budget_accepts_an_exact_fit_and_spills_one_byte_over() {
        let payload = small_snap().payload.len() as u64;
        let mut exact = SwapStore::new(tmpdir("hostfit"), payload, 1 << 20).unwrap();
        exact.put(1, small_snap()).unwrap();
        assert_eq!(exact.tier_of(1), Some("host"));
        assert_eq!(exact.host_used(), payload);

        let mut over = SwapStore::new(tmpdir("hostover"), payload - 1, 1 << 20).unwrap();
        over.put(1, small_snap()).unwrap();
        assert_eq!(over.tier_of(1), Some("disk"));
        assert_eq!(over.host_used(), 0);
    }

    #[test]
    fn disk_budget_accepts_an_exact_snapshot_and_refuses_one_byte_less() {
        let encoded = small_snap().encoded_len() as u64;
        let mut exact = SwapStore::new(tmpdir("diskfit"), 0, encoded).unwrap();
        exact.put(1, small_snap()).unwrap();
        assert_eq!(exact.tier_of(1), Some("disk"));
        assert_eq!(exact.disk_used(), encoded);

        let mut under = SwapStore::new(tmpdir("diskunder"), 0, encoded - 1).unwrap();
        assert!(under.put(1, small_snap()).is_err());
        assert!(under.is_empty());
        assert_eq!(under.disk_used(), 0);
    }

    #[test]
    fn re_parking_a_key_refreshes_its_lru_stamp() {
        let encoded = small_snap().encoded_len() as u64;
        let mut s = SwapStore::new(tmpdir("relru"), 0, encoded * 2).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(2, small_snap()).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(3, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), Some("disk"), "re-parked key is fresh");
        assert_eq!(s.tier_of(2), None, "the untouched key is the LRU");
        assert_eq!(s.tier_of(3), Some("disk"));
    }

    #[test]
    fn re_parking_the_same_host_key_does_not_double_charge() {
        let payload = small_snap().payload.len() as u64;
        let mut s = SwapStore::new(tmpdir("rehost"), payload * 2, 0).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(s.host_used(), payload);
    }

    #[test]
    fn an_evicted_key_does_not_resolve() {
        let encoded = small_snap().encoded_len() as u64;
        let mut s = SwapStore::new(tmpdir("evicted"), 0, encoded).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(2, small_snap()).unwrap();
        assert!(s.take(1).is_err());
    }

    #[test]
    fn a_failed_spill_write_does_not_charge_or_insert() {
        let dir = tmpdir("writefail");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir(dir.join("1.snap")).unwrap();
        let mut s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        assert!(s.put(1, small_snap()).is_err());
        assert_eq!(s.disk_used(), 0);
        assert!(s.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn drop_key_frees_budget_and_is_idempotent() {
        let mut s = SwapStore::new(tmpdir("dropk"), 1 << 20, 1 << 30).unwrap();
        s.put(1, small_snap()).unwrap();
        s.drop_key(1);
        assert_eq!(s.host_used(), 0);
        s.drop_key(1);
        assert_eq!(s.tier_of(1), None);
    }

    #[test]
    fn disk_budget_evicts_the_lru_snapshot() {
        let encoded = small_snap().encoded_len() as u64;
        let mut s = SwapStore::new(tmpdir("disklru"), 0, encoded * 2).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(2, small_snap()).unwrap();
        s.put(3, small_snap()).unwrap();
        assert_eq!(s.tier_of(1), None, "the oldest disk snapshot is evicted");
        assert_eq!(s.tier_of(2), Some("disk"));
        assert_eq!(s.tier_of(3), Some("disk"));
        assert_eq!(s.disk_used(), encoded * 2);
        assert_eq!(
            spill_entries(s.dir()).len(),
            2,
            "an evicted snapshot must not leave its spill file behind"
        );
    }

    #[test]
    fn a_zero_disk_budget_refuses_spill() {
        let mut s = SwapStore::new(tmpdir("nodisk"), 0, 0).unwrap();
        assert!(s.put(1, small_snap()).is_err());
        assert!(s.is_empty());
        assert_eq!(s.disk_used(), 0);
    }

    #[test]
    fn take_frees_disk_budget_and_removes_the_file() {
        let mut s = SwapStore::new(tmpdir("diskfree"), 0, 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        assert!(s.disk_used() > 0);
        let _ = s.take(1).unwrap();
        assert_eq!(s.disk_used(), 0);
        assert_eq!(spill_entries(s.dir()).len(), 0);
    }

    #[test]
    fn an_oversized_snapshot_does_not_evict_existing_spills() {
        let encoded = small_snap().encoded_len() as u64;
        let mut s = SwapStore::new(tmpdir("oversize"), 0, encoded * 2).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(2, small_snap()).unwrap();
        assert_eq!(s.len(), 2);
        // Three times the per-snapshot size: it cannot fit the 2x budget at
        // all, so it must be refused before the LRU loop runs.
        let mut big = small_snap();
        big.payload = vec![9u8; small_snap().payload.len() * 3];
        big.checksum = checksum_of(&big.payload);
        big.stamp.dn_bytes = big.payload.len() as u64 - 8;
        assert!(s.put(3, big).is_err());
        assert_eq!(s.tier_of(1), Some("disk"), "entry 1 survived");
        assert_eq!(s.tier_of(2), Some("disk"), "entry 2 survived");
        assert_eq!(s.disk_used(), encoded * 2);
        assert_eq!(spill_entries(s.dir()).len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn a_planted_spill_symlink_is_replaced_without_touching_its_target() {
        use std::os::unix::fs::symlink;
        let dir = tmpdir("planted");
        std::fs::create_dir_all(&dir).unwrap();
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, b"IMPORTANT").unwrap();
        let mut s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        // Planted after construction: the startup cleanup cannot see it.
        symlink(&victim, dir.join("1.snap")).unwrap();
        s.put(1, small_snap()).unwrap();
        assert_eq!(std::fs::read(&victim).unwrap(), b"IMPORTANT");
        assert!(!std::fs::symlink_metadata(dir.join("1.snap"))
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(s.take(1).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn the_swap_dir_and_spill_files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmpdir("perms");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        assert_eq!(
            std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        s.put(1, small_snap()).unwrap();
        assert_eq!(
            std::fs::metadata(dir.join("1.snap"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_second_store_on_the_same_directory_is_refused() {
        let dir = tmpdir("onedaemon");
        std::fs::create_dir_all(&dir).unwrap();
        let first = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        let err = SwapStore::new(dir.clone(), 0, 1 << 20)
            .err()
            .expect("a second store must not share the directory");
        assert!(format!("{err}").contains("in use"), "{err}");
        drop(first);
        let third = SwapStore::new(dir.clone(), 0, 1 << 20);
        assert!(third.is_ok(), "the lock must be released on drop");
        drop(third);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_directory_degrades_to_host_only_and_keeps_stale_files() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmpdir("readonly");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("1.snap"), b"stale").unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        assert!(dir.join("1.snap").exists(), "the stale file is left alone");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_unlink_keeps_its_disk_charge() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmpdir("unlinkfail");
        std::fs::create_dir_all(&dir).unwrap();
        let mut s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        let charged = s.disk_used();
        assert!(charged > 0);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        s.drop_key(1);
        assert_eq!(s.disk_used(), charged, "an unremovable spill stays charged");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_spill_write_does_not_evict_cached_snapshots() {
        let dir = tmpdir("writefail-cache");
        std::fs::create_dir_all(&dir).unwrap();
        let encoded = small_snap().encoded_len() as u64;
        let mut s = SwapStore::new(dir.clone(), 0, encoded * 2).unwrap();
        s.put(1, small_snap()).unwrap();
        s.put(2, small_snap()).unwrap();
        // Make the temp path unwritable as a *file* target; the write fails
        // before the LRU loop, so both cached snapshots must survive.
        std::fs::create_dir(dir.join("3.snap.tmp")).unwrap();
        assert!(s.put(3, small_snap()).is_err());
        assert_eq!(s.tier_of(1), Some("disk"));
        assert_eq!(s.tier_of(2), Some("disk"));
        assert_eq!(s.disk_used(), encoded * 2);
        assert_eq!(spill_entries(&dir).len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_successful_disk_put_leaves_no_temp_file() {
        let dir = tmpdir("tmpclean");
        let mut s = SwapStore::new(dir.clone(), 0, 1 << 20).unwrap();
        s.put(1, small_snap()).unwrap();
        assert!(dir.join("1.snap").exists());
        assert!(!dir.join("1.snap.tmp").exists());
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
