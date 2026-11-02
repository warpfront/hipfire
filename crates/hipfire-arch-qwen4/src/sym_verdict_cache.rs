// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-file verdict cache of the Qwen4 routed-expert symmetry scan
//! ([`crate::artifact::routed_headers_symmetric`]). The scan reads every
//! routed trunk payload byte of the artifact (about 13 s on Flash-Next); its
//! verdict is a pure function of those bytes, so it is paid once per file.
//!
//! One JSON file under the hipfire home
//! (`<HIPFIRE_HOME>/cache/qwen4-sym-verdicts.json`) holds one record per
//! canonical artifact path: the file's [`StatKey`] (device, inode, size,
//! mtime), the [`SCAN_VERSION`] that produced the verdict, the digest of the
//! routed tensor names, shapes and byte ranges the scan reads, and the
//! verdict. A record answers a lookup only when all of those equal the
//! current ones; any difference is a miss. Only completed scans are stored
//! (a scan error is never cached). Writes go through
//! [`write_json_atomically`] and a failed write leaves the cache as it was; a
//! missing, unreadable, corrupt or foreign-version file reads as empty.
//!
//! A process-wide [`Memo`] keyed the same way makes a second scan of the same
//! file in one process free.

use hipfire_runtime::verified_cache::{write_json_atomically, StatKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

/// Developer knob: `0` disables the verdict cache (file and in-process memo),
/// so every call scans.
pub const VERDICT_CACHE_ENV: &str = "HIPFIRE_QWEN4_SYM_VERDICT_CACHE";

/// Version of the scan's semantics. Bump it whenever the scan can return a
/// different verdict for the same bytes; every stored verdict then misses.
pub const SCAN_VERSION: u32 = 1;

const FILE_VERSION: u32 = 1;

/// The cache file of the hipfire home `root`.
pub fn cache_file_in_home(root: &Path) -> PathBuf {
    root.join("cache").join("qwen4-sym-verdicts.json")
}

/// What a verdict is keyed by: the artifact (canonical path plus stat) and the
/// routed tensor layout the scan reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// Canonical artifact path.
    pub path: PathBuf,
    /// Stat of the artifact taken before the scan.
    pub stat: StatKey,
    /// Lowercase hex SHA-256 over the sorted routed tensor names, quant types,
    /// shapes and payload offsets and lengths.
    pub names_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Record {
    stat: StatKey,
    scan_version: u32,
    routed_names_digest: String,
    symmetric: bool,
}

impl Record {
    fn of(identity: &Identity, symmetric: bool) -> Self {
        Self {
            stat: identity.stat,
            scan_version: SCAN_VERSION,
            routed_names_digest: identity.names_digest.clone(),
            symmetric,
        }
    }

    fn answers(&self, identity: &Identity) -> bool {
        self.stat == identity.stat
            && self.scan_version == SCAN_VERSION
            && self.routed_names_digest == identity.names_digest
    }
}

#[derive(Default, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    records: BTreeMap<String, Record>,
}

fn key_of(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn read_records(file: &Path) -> BTreeMap<String, Record> {
    std::fs::read(file)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<CacheFile>(&bytes).ok())
        .filter(|cache| cache.version == FILE_VERSION)
        .map(|cache| cache.records)
        .unwrap_or_default()
}

/// In-process verdict memo, keyed like the file.
pub struct Memo(Mutex<BTreeMap<String, Record>>);

impl Memo {
    pub const fn new() -> Self {
        Self(Mutex::new(BTreeMap::new()))
    }

    fn get(&self, identity: &Identity) -> Option<bool> {
        let records = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        records
            .get(&key_of(&identity.path))
            .filter(|record| record.answers(identity))
            .map(|record| record.symmetric)
    }

    fn put(&self, identity: &Identity, symmetric: bool) {
        let mut records = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        records.insert(key_of(&identity.path), Record::of(identity, symmetric));
    }
}

impl Default for Memo {
    fn default() -> Self {
        Self::new()
    }
}

/// The memo every product caller shares.
pub static PROCESS_MEMO: Memo = Memo::new();

/// Where a verdict came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Origin {
    /// An earlier call of this process.
    Memo,
    /// The cache file.
    File,
    /// A scan that ran now; `stored` says whether the verdict reached the file.
    Scanned { seconds: f64, stored: bool },
}

/// The verdict of `identity`: from `memo`, else from the cache `file`, else
/// from `scan` (stored in the file and memo on success). `None` for `file`
/// or `memo` skips that layer entirely (no read, no write). A scan error is
/// returned as is and stores nothing.
pub fn verdict<E>(
    file: Option<&Path>,
    memo: Option<&Memo>,
    identity: &Identity,
    scan: impl FnOnce() -> Result<bool, E>,
) -> Result<(bool, Origin), E> {
    if let Some(symmetric) = memo.and_then(|memo| memo.get(identity)) {
        return Ok((symmetric, Origin::Memo));
    }
    if let Some(symmetric) = file.and_then(|file| {
        read_records(file)
            .remove(&key_of(&identity.path))
            .filter(|record| record.answers(identity))
            .map(|record| record.symmetric)
    }) {
        if let Some(memo) = memo {
            memo.put(identity, symmetric);
        }
        return Ok((symmetric, Origin::File));
    }
    let started = Instant::now();
    let symmetric = scan()?;
    let seconds = started.elapsed().as_secs_f64();
    if let Some(memo) = memo {
        memo.put(identity, symmetric);
    }
    let stored = file.is_some_and(|file| {
        let mut records = read_records(file);
        records.insert(key_of(&identity.path), Record::of(identity, symmetric));
        write_json_atomically(
            file,
            &CacheFile {
                version: FILE_VERSION,
                records,
            },
        )
        .is_ok()
    });
    Ok((symmetric, Origin::Scanned { seconds, stored }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-sym-verdict-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn identity() -> Identity {
        Identity {
            path: PathBuf::from("/models/flash-next.hfq"),
            stat: StatKey {
                dev: 7,
                ino: 11,
                size: 1 << 30,
                mtime_sec: 1_700_000_000,
                mtime_nsec: 5,
            },
            names_digest: "ab".repeat(32),
        }
    }

    /// One `verdict` call over the cache file only, counting scans.
    fn run(file: &Path, identity: &Identity, answer: bool, scans: &Cell<u32>) -> (bool, Origin) {
        verdict::<()>(Some(file), None, identity, || {
            scans.set(scans.get() + 1);
            Ok(answer)
        })
        .unwrap()
    }

    #[test]
    fn a_stored_verdict_answers_only_an_identical_identity() {
        let home = scratch("keys");
        let file = cache_file_in_home(&home);
        let scans = Cell::new(0);
        let base = identity();
        assert!(matches!(
            run(&file, &base, true, &scans),
            (true, Origin::Scanned { stored: true, .. })
        ));
        assert_eq!(run(&file, &base, false, &scans), (true, Origin::File));
        assert_eq!(scans.get(), 1);

        let changed = [
            Identity {
                stat: StatKey {
                    dev: 8,
                    ..base.stat
                },
                ..base.clone()
            },
            Identity {
                stat: StatKey {
                    ino: 12,
                    ..base.stat
                },
                ..base.clone()
            },
            Identity {
                stat: StatKey {
                    size: base.stat.size + 1,
                    ..base.stat
                },
                ..base.clone()
            },
            Identity {
                stat: StatKey {
                    mtime_sec: base.stat.mtime_sec + 1,
                    ..base.stat
                },
                ..base.clone()
            },
            Identity {
                stat: StatKey {
                    mtime_nsec: base.stat.mtime_nsec + 1,
                    ..base.stat
                },
                ..base.clone()
            },
            Identity {
                names_digest: "cd".repeat(32),
                ..base.clone()
            },
        ];
        for (index, other) in changed.iter().enumerate() {
            let before = scans.get();
            let (symmetric, origin) = run(&file, other, false, &scans);
            assert!(!symmetric, "changed identity {index} must rescan");
            assert!(matches!(origin, Origin::Scanned { .. }), "identity {index}");
            assert_eq!(scans.get(), before + 1, "identity {index}");
        }
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_changed_scan_version_misses() {
        let home = scratch("version");
        let file = cache_file_in_home(&home);
        let base = identity();
        let stale = CacheFile {
            version: FILE_VERSION,
            records: BTreeMap::from([(
                key_of(&base.path),
                Record {
                    scan_version: SCAN_VERSION + 1,
                    ..Record::of(&base, true)
                },
            )]),
        };
        write_json_atomically(&file, &stale).unwrap();
        let scans = Cell::new(0);
        assert!(matches!(
            run(&file, &base, false, &scans),
            (false, Origin::Scanned { stored: true, .. })
        ));
        assert_eq!(scans.get(), 1);
        // The rescan replaced the stale record in place.
        assert_eq!(run(&file, &base, true, &scans), (false, Origin::File));
        assert_eq!(read_records(&file).len(), 1);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_foreign_file_version_is_a_miss() {
        let home = scratch("file-version");
        let file = cache_file_in_home(&home);
        let base = identity();
        write_json_atomically(
            &file,
            &CacheFile {
                version: FILE_VERSION + 1,
                records: BTreeMap::from([(key_of(&base.path), Record::of(&base, true))]),
            },
        )
        .unwrap();
        let scans = Cell::new(0);
        assert!(matches!(
            run(&file, &base, false, &scans),
            (false, Origin::Scanned { .. })
        ));
        assert_eq!(scans.get(), 1);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_corrupt_file_is_a_miss_and_is_replaced() {
        let home = scratch("corrupt");
        let file = cache_file_in_home(&home);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, b"{ not json").unwrap();
        let scans = Cell::new(0);
        let base = identity();
        assert!(matches!(
            run(&file, &base, true, &scans),
            (true, Origin::Scanned { stored: true, .. })
        ));
        assert_eq!(run(&file, &base, false, &scans), (true, Origin::File));
        assert_eq!(scans.get(), 1);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_scan_error_is_returned_and_never_stored() {
        let home = scratch("error");
        let file = cache_file_in_home(&home);
        let base = identity();
        let failed = verdict(Some(&file), None, &base, || Err("short payload"));
        assert_eq!(failed, Err("short payload"));
        assert!(!file.exists());
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn without_a_file_nothing_is_read_or_written() {
        let home = scratch("off");
        let file = cache_file_in_home(&home);
        let base = identity();
        // A record is on disk, but the layer is skipped: the scan runs.
        write_json_atomically(
            &file,
            &CacheFile {
                version: FILE_VERSION,
                records: BTreeMap::from([(key_of(&base.path), Record::of(&base, true))]),
            },
        )
        .unwrap();
        let before = fs::read(&file).unwrap();
        let scans = Cell::new(0);
        let (symmetric, origin) = verdict::<()>(None, None, &base, || {
            scans.set(scans.get() + 1);
            Ok(false)
        })
        .unwrap();
        assert!(!symmetric);
        assert!(matches!(origin, Origin::Scanned { stored: false, .. }));
        assert_eq!(scans.get(), 1);
        assert_eq!(fs::read(&file).unwrap(), before);

        // And a scan with no file creates none.
        let empty = scratch("off-empty");
        let target = cache_file_in_home(&empty);
        verdict::<()>(None, None, &base, || Ok(true)).unwrap();
        assert!(!target.exists());
        assert!(!empty.join("cache").exists());
        fs::remove_dir_all(&home).unwrap();
        fs::remove_dir_all(&empty).unwrap();
    }

    #[test]
    fn an_unwritable_cache_never_fails_the_verdict() {
        let home = scratch("unwritable");
        let blocker = home.join("blocker");
        fs::write(&blocker, b"a file, not a directory").unwrap();
        let file = cache_file_in_home(&blocker);
        let scans = Cell::new(0);
        assert!(matches!(
            run(&file, &identity(), true, &scans),
            (true, Origin::Scanned { stored: false, .. })
        ));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn the_memo_answers_a_second_call_and_follows_the_identity() {
        let memo = Memo::new();
        let base = identity();
        let scans = Cell::new(0);
        let call = |identity: &Identity, answer: bool| {
            verdict::<()>(None, Some(&memo), identity, || {
                scans.set(scans.get() + 1);
                Ok(answer)
            })
            .unwrap()
        };
        assert!(matches!(call(&base, true), (true, Origin::Scanned { .. })));
        assert_eq!(call(&base, false), (true, Origin::Memo));
        assert_eq!(scans.get(), 1);
        let touched = Identity {
            stat: StatKey {
                mtime_sec: base.stat.mtime_sec + 1,
                ..base.stat
            },
            ..base.clone()
        };
        assert!(matches!(
            call(&touched, false),
            (false, Origin::Scanned { .. })
        ));
        assert_eq!(scans.get(), 2);
    }

    #[test]
    fn a_file_hit_fills_the_memo() {
        let home = scratch("fill");
        let file = cache_file_in_home(&home);
        let base = identity();
        let scans = Cell::new(0);
        run(&file, &base, true, &scans);
        let memo = Memo::new();
        let (symmetric, origin) =
            verdict::<()>(Some(&file), Some(&memo), &base, || Ok(false)).unwrap();
        assert_eq!((symmetric, origin), (true, Origin::File));
        fs::remove_file(&file).unwrap();
        let (symmetric, origin) =
            verdict::<()>(Some(&file), Some(&memo), &base, || Ok(false)).unwrap();
        assert_eq!((symmetric, origin), (true, Origin::Memo));
        fs::remove_dir_all(&home).unwrap();
    }
}
