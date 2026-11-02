// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared metadata verification authority for local artifacts. The CLI records
//! digests after verifying artifact bytes; CLI and runtime callers reuse those
//! records only while the path's complete stat key still matches.
//!
//! One JSON file under the hipfire home (`<HIPFIRE_HOME>/cache/verified-artifacts.json`)
//! holds one record per verified path: device, inode, size, mtime (seconds and
//! nanoseconds), the verified SHA-256 and when it was verified. [`VerifiedCache::satisfies`]
//! requires both an exact stat match and the caller's expected digest, so a
//! record for another pin never satisfies. On a miss, the CLI verifies the bytes
//! before calling [`VerifiedCache::remember`], which writes atomically (temporary
//! file + rename). Missing, corrupt or unwritable cache files never fail a
//! cache operation; callers decide whether a miss requires verification or refusal.
//!
//! The loader reuses receipts without hashing through
//! [`VerifiedCache::verified_digest`]: one read-only lookup per path at load
//! time, returning the recorded digest only on exact stat equality plus a
//! syntactically valid SHA-256.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const CACHE_VERSION: u32 = 1;

/// The stat fields a record is keyed by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatKey {
    pub dev: u64,
    pub ino: u64,
    pub size: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
}

impl StatKey {
    /// The [`StatKey`] of live file metadata: device, inode, size and mtime.
    pub fn of(metadata: &fs::Metadata) -> Self {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                dev: metadata.dev(),
                ino: metadata.ino(),
                size: metadata.len(),
                mtime_sec: metadata.mtime(),
                mtime_nsec: metadata.mtime_nsec(),
            }
        }
        #[cfg(not(unix))]
        {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok());
            Self {
                dev: 0,
                ino: 0,
                size: metadata.len(),
                mtime_sec: modified.map_or(0, |time| time.as_secs() as i64),
                mtime_nsec: modified.map_or(0, |time| i64::from(time.subsec_nanos())),
            }
        }
    }
}

/// One verified path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Record {
    #[serde(flatten)]
    stat: StatKey,
    sha256: String,
    /// Unix seconds when the bytes were hashed.
    verified_at: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct CacheFile {
    version: u32,
    records: BTreeMap<String, Record>,
}

/// The on-disk verification cache, or a disabled stand-in.
#[derive(Clone, Debug)]
pub struct VerifiedCache {
    file: Option<PathBuf>,
}

impl VerifiedCache {
    /// The cache of the hipfire home `root`: `<root>/cache/verified-artifacts.json`.
    pub fn in_home(root: &Path) -> Self {
        Self::at(root.join("cache").join("verified-artifacts.json"))
    }

    /// A cache stored in `file`.
    pub fn at(file: PathBuf) -> Self {
        Self { file: Some(file) }
    }

    /// A cache that never hits and never writes.
    pub fn disabled() -> Self {
        Self { file: None }
    }

    /// Every record of the cache file; a missing or corrupt file reads as empty.
    fn load(&self) -> BTreeMap<String, Record> {
        let Some(file) = &self.file else {
            return BTreeMap::new();
        };
        fs::read(file)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<CacheFile>(&bytes).ok())
            .filter(|cache| cache.version == CACHE_VERSION)
            .map(|cache| cache.records)
            .unwrap_or_default()
    }

    /// The record of `path`, if any (test and diagnostics access).
    #[cfg(test)]
    fn record_of(&self, path: &Path) -> Option<Record> {
        self.load().remove(&key_of(path))
    }

    /// True when `path` (whose current stat is `current`) was verified against
    /// `expected_sha256`: the record's stat equals `current` and its digest
    /// equals the expected one.
    pub fn satisfies(&self, path: &Path, current: &StatKey, expected_sha256: &str) -> bool {
        self.load().get(&key_of(path)).is_some_and(|record| {
            record.stat == *current && record.sha256.eq_ignore_ascii_case(expected_sha256)
        })
    }

    /// The verified digest of `path` (whose current stat is `current`) with no
    /// expected digest to compare against: `Some` lowercase hex only when the
    /// record's stat equals `current` and its digest is a syntactically valid
    /// SHA-256 (64 hex digits). Read-only: never hashes, writes, or trusts
    /// registry state, so the loader calls this once at load time. Anything
    /// else — missing or corrupt file, unknown path, changed stat, malformed
    /// digest — is a miss (`None`).
    pub fn verified_digest(&self, path: &Path, current: &StatKey) -> Option<String> {
        let record = self.load().remove(&key_of(path))?;
        if record.stat != *current {
            return None;
        }
        let digest = record.sha256.to_ascii_lowercase();
        if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            Some(digest)
        } else {
            None
        }
    }

    /// Record that `path`, whose stat at hashing time was `stat`, hashed to
    /// `sha256`. Written atomically; any failure leaves the cache as it was.
    pub fn remember(&self, path: &Path, stat: StatKey, sha256: &str) {
        static WARNED: AtomicBool = AtomicBool::new(false);
        self.remember_warning_once(path, stat, sha256, &WARNED);
    }

    /// [`Self::remember`] with the once-per-process warning latch injected;
    /// returns true when this call emitted the warning.
    fn remember_warning_once(
        &self,
        path: &Path,
        stat: StatKey,
        sha256: &str,
        warned: &AtomicBool,
    ) -> bool {
        let Some(file) = &self.file else {
            return false;
        };
        let mut records = self.load();
        records.insert(
            key_of(path),
            Record {
                stat,
                sha256: sha256.to_ascii_lowercase(),
                verified_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_secs()),
            },
        );
        let Err(err) = write_json_atomically(
            file,
            &CacheFile {
                version: CACHE_VERSION,
                records,
            },
        ) else {
            return false;
        };
        if warned.swap(true, Ordering::Relaxed) {
            return false;
        }
        eprintln!(
            "warning: cannot write verified-artifact cache {}: {err}; \
             artifact verification will repeat on every load or request until it is writable",
            file.display()
        );
        true
    }
}

fn key_of(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Serialize `cache` as pretty JSON to `file` atomically: parent directories
/// are created, the bytes go to a process-unique temporary file beside `file`,
/// and a rename publishes them. A failure removes the temporary file.
pub fn write_json_atomically<T: Serialize>(file: &Path, cache: &T) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(cache).map_err(std::io::Error::other)?;
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = file.as_os_str().to_owned();
    temporary.push(format!(".tmp.{}", std::process::id()));
    let temporary = PathBuf::from(temporary);
    let result = fs::write(&temporary, bytes).and_then(|()| fs::rename(&temporary, file));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hipfire-verified-cache-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    const SHA: &str = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";

    fn stat(ino: u64) -> StatKey {
        StatKey {
            dev: 7,
            ino,
            size: 125,
            mtime_sec: 1_700_000_000,
            mtime_nsec: 123_456_789,
        }
    }

    #[test]
    fn a_remembered_stat_and_digest_satisfy_only_an_exact_match() {
        let home = scratch("exact");
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/models/x.mq4-xts");
        assert!(!cache.satisfies(model, &stat(1), SHA));
        cache.remember(model, stat(1), SHA);
        assert!(cache.satisfies(model, &stat(1), SHA));
        assert!(cache.satisfies(model, &stat(1), &SHA.to_ascii_uppercase()));
        // Any stat field change, or another expected digest, does not satisfy.
        for changed in [
            StatKey { dev: 8, ..stat(1) },
            stat(2),
            StatKey {
                size: 126,
                ..stat(1)
            },
            StatKey {
                mtime_sec: 1_700_000_001,
                ..stat(1)
            },
            StatKey {
                mtime_nsec: 1,
                ..stat(1)
            },
        ] {
            assert!(!cache.satisfies(model, &changed, SHA), "{changed:?}");
        }
        assert!(!cache.satisfies(model, &stat(1), &"a".repeat(64)));
        assert!(!cache.satisfies(Path::new("/models/other"), &stat(1), SHA));
        let record = cache.record_of(model).unwrap();
        assert_eq!(record.stat, stat(1));
        assert_eq!(record.sha256, SHA);
        assert!(record.verified_at > 0);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn records_are_one_per_path_and_replaced_in_place() {
        let home = scratch("replace");
        let cache = VerifiedCache::in_home(&home);
        let (a, b) = (Path::new("/m/a"), Path::new("/m/b"));
        cache.remember(a, stat(1), SHA);
        cache.remember(b, stat(2), SHA);
        cache.remember(a, stat(3), SHA);
        assert_eq!(cache.record_of(a).unwrap().stat, stat(3));
        assert_eq!(cache.record_of(b).unwrap().stat, stat(2));
        let written: serde_json::Value =
            serde_json::from_slice(&fs::read(home.join("cache/verified-artifacts.json")).unwrap())
                .unwrap();
        assert_eq!(written["records"].as_object().unwrap().len(), 2);
        let leftovers: Vec<_> = fs::read_dir(home.join("cache"))
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_corrupt_or_foreign_cache_file_is_ignored_and_replaced() {
        let home = scratch("corrupt");
        let file = home.join("cache/verified-artifacts.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/m/x");
        for bytes in [
            &b"{ this is not json"[..],
            b"",
            b"[]",
            br#"{"version":99,"records":{}}"#,
            br#"{"version":1,"records":{"/m/x":{"dev":"nope"}}}"#,
        ] {
            fs::write(&file, bytes).unwrap();
            assert!(!cache.satisfies(model, &stat(1), SHA), "{bytes:?}");
            cache.remember(model, stat(1), SHA);
            assert!(cache.satisfies(model, &stat(1), SHA), "{bytes:?}");
        }
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn an_unwritable_cache_never_fails_and_a_disabled_cache_never_hits() {
        let home = scratch("unwritable");
        // The cache directory's parent is a regular file: creating it fails.
        let blocker = home.join("blocker");
        fs::write(&blocker, b"x").unwrap();
        let cache = VerifiedCache::in_home(&blocker);
        cache.remember(Path::new("/m/x"), stat(1), SHA);
        assert!(!cache.satisfies(Path::new("/m/x"), &stat(1), SHA));
        let disabled = VerifiedCache::disabled();
        disabled.remember(Path::new("/m/x"), stat(1), SHA);
        assert!(!disabled.satisfies(Path::new("/m/x"), &stat(1), SHA));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_failing_cache_write_warns_once_and_leaves_the_result_unaffected() {
        let home = scratch("warn-once");
        let blocker = home.join("blocker");
        fs::write(&blocker, b"x").unwrap();
        let cache = VerifiedCache::in_home(&blocker);
        let warned = AtomicBool::new(false);
        let model = Path::new("/m/x");
        assert!(cache.remember_warning_once(model, stat(1), SHA, &warned));
        assert!(!cache.remember_warning_once(model, stat(1), SHA, &warned));
        // The caller's verified result stands; the cache just never hits.
        assert!(!cache.satisfies(model, &stat(1), SHA));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn stat_keys_follow_the_file_metadata() {
        let home = scratch("stat");
        let file = home.join("artifact");
        fs::write(&file, b"payload").unwrap();
        let first = StatKey::of(&fs::metadata(&file).unwrap());
        assert_eq!(first.size, 7);
        assert_eq!(first, StatKey::of(&fs::metadata(&file).unwrap()));
        fs::write(&file, b"longer payload").unwrap();
        assert_ne!(first, StatKey::of(&fs::metadata(&file).unwrap()));
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn verified_digest_needs_no_expected_digest_and_reads_back_lowercase() {
        let home = scratch("digest-hit");
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/models/x.mq4-xts");
        assert_eq!(cache.verified_digest(model, &stat(1)), None);
        cache.remember(model, stat(1), SHA);
        assert_eq!(
            cache.verified_digest(model, &stat(1)),
            Some(SHA.to_string())
        );
        // An uppercase record still reads back lowercase.
        cache.remember(model, stat(2), &SHA.to_ascii_uppercase());
        assert_eq!(
            cache.verified_digest(model, &stat(2)),
            Some(SHA.to_string())
        );
        // Another stat, or another path, is a miss.
        assert_eq!(cache.verified_digest(model, &stat(3)), None);
        assert_eq!(
            cache.verified_digest(Path::new("/models/other"), &stat(1)),
            None
        );
        // Read-only: the lookup leaves no files beside the cache itself.
        let entries: Vec<_> = fs::read_dir(home.join("cache"))
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn verified_digest_rejects_malformed_sha_and_corrupt_files() {
        let home = scratch("digest-bad");
        let file = home.join("cache/verified-artifacts.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let cache = VerifiedCache::in_home(&home);
        let model = Path::new("/m/x");
        let record = |sha: &str| {
            serde_json::json!({
                "version": 1,
                "records": {
                    "/m/x": {
                        "dev": 7,
                        "ino": 1,
                        "size": 125,
                        "mtime_sec": 1_700_000_000,
                        "mtime_nsec": 123_456_789,
                        "sha256": sha,
                        "verified_at": 1,
                    }
                }
            })
        };
        // Empty, short, long, or non-hex digests never satisfy and never digest.
        let malformed = [
            String::from(""),
            "z".repeat(64),
            "a".repeat(63),
            "a".repeat(65),
            "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f20097g".to_string(),
        ];
        for sha in &malformed {
            fs::write(&file, serde_json::to_vec(&record(sha)).unwrap()).unwrap();
            assert!(!cache.satisfies(model, &stat(1), SHA), "{sha:?}");
            assert_eq!(cache.verified_digest(model, &stat(1)), None, "{sha:?}");
        }
        // A well-formed uppercase record digests to lowercase.
        fs::write(
            &file,
            serde_json::to_vec(&record(&SHA.to_ascii_uppercase())).unwrap(),
        )
        .unwrap();
        assert_eq!(
            cache.verified_digest(model, &stat(1)),
            Some(SHA.to_string())
        );
        // A missing or corrupt file is a miss, never an error.
        fs::remove_file(&file).unwrap();
        assert_eq!(cache.verified_digest(model, &stat(1)), None);
        for bytes in [&b"{ this is not json"[..], b"", b"[]"] {
            fs::write(&file, bytes).unwrap();
            assert_eq!(cache.verified_digest(model, &stat(1)), None, "{bytes:?}");
        }
        fs::remove_dir_all(&home).unwrap();
    }
}
