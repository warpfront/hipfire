//! Transparent content-addressed cache for successful M7 certifications.
//!
//! Storage layer only: given the exact M7 inputs (ELF bytes, arch, symbol)
//! this module resolves a cache file path, loads a previously stored success
//! receipt, or stores one. All policy decisions (what counts as success, when
//! to consult the cache) live in `pm_check.rs`; this file never runs the
//! checker.
//!
//! # Identity rationale
//!
//! A cache hit skips re-running the M7 certification, so a hit must imply the
//! checker *would* reach the same verdict. The request key therefore binds:
//!
//! * the exact runtime M7 inputs: ELF bytes, `arch`, and `symbol`
//!   (length-framed into the digest so concatenations cannot collide), plus
//! * the SHA256 of the running Linux `/proc/self/exe` image, memoized
//!   process-wide. The executable image conservatively covers the checker
//!   itself and every transitive input baked into it: lift/IR tables and
//!   analyses, serde, compiler version, feature flags, and build stamps.
//!
//! Deliberately NOT part of the identity: file modification timestamps or
//! version strings. Timestamps change on rebuild/reinstall without changing
//! behavior (false misses) and version strings can stay fixed while behavior
//! changes (false hits). Hashing the executed bytes cannot stale-cache across
//! a behavior change and cannot false-hit across builds. If the identity is
//! unavailable (non-Linux, unreadable `/proc/self/exe`) caching fails closed:
//! [`Cache::for_request`] returns `None` and the caller runs uncached.
//!
//! The executable is hashed with a streamed 8 KiB read loop; it is never
//! loaded fully into memory.
//!
//! ## Executable identity cache (`binid.json`)
//!
//! Hashing the whole image on every fresh CLI process is pure I/O overhead,
//! so the cache root holds one `binid.json` record binding the executable's
//! metadata to its full SHA256: `{dev, ino, size, mtime sec+nsec, ctime
//! sec+nsec}` -> digest. A fresh process `stat`s `/proc/self/exe`; on a
//! metadata match it reuses the recorded digest without hashing, otherwise
//! it hashes once (the digest is memoized process-wide, keyed by metadata,
//! so even a root that cannot persist the record hashes at most once),
//! `fstat`s the same opened image before and after the hash, and atomically
//! rewrites the record. Any metadata change across the hash fails closed
//! (caching disabled for the request).
//!
//! Trust assumption: ordinary OS inode/`ctime` semantics, where a normal
//! write or replacement changes size, `mtime`, `ctime`, or `ino`. Seconds
//! and nanoseconds are stored as separate integers so no `sec * 1e9` product
//! can overflow. The first run in a fresh root hashes and publishes the
//! record; warm runs reuse it with no image I/O. Publishing is best effort:
//! a write failure warns once and still uses the freshly hashed digest, so
//! the record never turns an established identity into a miss; only an
//! unavailable identity disables the cache.
//!
//! # Location / env config
//!
//! The cache root is resolved once per [`Cache::for_request`] call (the live
//! environment is never re-read within one request):
//!
//! * `HIPFIRE_CERT_CACHE=0` disables the cache entirely. This check runs
//!   before the executable is hashed, so a disabled cache costs no I/O.
//! * `HIPFIRE_CERT_CACHE` set to any other value names the cache directory
//!   directly; an empty override disables caching with a warning.
//! * Otherwise `$XDG_CACHE_HOME/hipfire/pm-m7`, else `$HOME/.cache/hipfire/pm-m7`.
//! * If no home directory is available, caching is disabled.
//!
//! Directories are created when publishing either the identity record or a
//! receipt. If the root cannot be created or an entry cannot be written, a
//! warning prefixed `warning: M7 certification cache` is printed once per
//! process and certification continues uncached.
//!
//! # Entry schema
//!
//! Each entry is a JSON object `{ schema, key, receipt, checksum }` written
//! with `deny_unknown_fields` on read. `checksum` is the hex SHA256 over the
//! raw request-key bytes concatenated with the canonical receipt bytes
//! (`serde_json::to_vec` of the stored [`serde_json::Value`]), so the
//! checksum binds the receipt payload to the request key. Entries that are
//! missing fields, carry unknown fields, fail to parse, name a different
//! schema version or key, or whose checksum mismatches are treated as corrupt:
//! the file is deleted (best effort) and the load reports a miss. A load never
//! exposes a partial or untrusted payload.
//!
//! Because both store and load pass through `serde_json::Value` (whose maps
//! have a canonical key order), `serde_json::to_vec` of a cache hit is byte
//! identical to `to_vec` of the value the uncached path produced.
//!
//! Writes are atomic within the cache directory: bytes go to a unique
//! `create_new` temp file (`pid` + process-wide counter) followed by
//! `rename`. All storage/environmental failures degrade to miss/uncached; a
//! failed store always removes its own temp file.
//!
//! The directory holds at most [`MAX_ENTRIES`] entries after a successful
//! eviction pass; on store, the oldest entries by `mtime` are evicted, except
//! the just-published path and the most recent 32 paths used by this process.

use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Entry schema version. Bumped if the on-disk format ever changes; the
/// version is also mixed into the request key so old keys never collide.
const ENTRY_SCHEMA: u32 = 1;

/// Domain separator mixed into the request-key digest.
const KEY_DOMAIN: &[u8] = b"hipfire/pm-m7-cert-cache/v1";

/// Maximum entries kept per cache directory before oldest-by-mtime eviction.
pub(crate) const MAX_ENTRIES: usize = 512;

/// Maximum bytes read from one untrusted cache entry. Larger files are
/// treated as corrupt (deleted, reported as a miss) so a corrupt entry
/// cannot exhaust memory. Success receipts are small JSON; 1 MiB is ample.
const MAX_ENTRY_BYTES: u64 = 1 << 20;

/// Stable stderr prefix for every cache I/O warning. Integration tests match
/// on this string.
pub(crate) const WARN_PREFIX: &str = "warning: M7 certification cache";

/// Schema version for the `binid.json` identity record. Independent of
/// [`ENTRY_SCHEMA`]: the identity record binds a binary to its digest, not
/// a request to a receipt.
const BINID_SCHEMA: u32 = 1;

/// Identity-record filename inside the cache root.
const BINID_FILE: &str = "binid.json";

/// Domain separator mixed into the binid checksum so a receipt entry can
/// never be mistaken for an identity record or vice versa.
const BINID_DOMAIN: &[u8] = b"hipfire/pm-m7-binid/v1";

/// Maximum bytes read from the untrusted identity record. The record is a
/// single small JSON object; 64 KiB is ample.
const MAX_BINID_BYTES: u64 = 64 << 10;

/// Process-wide known identities: metadata key plus full image digest per
/// key. A map (not a single slot) so parallel tests with distinct fixture
/// binaries cannot evict each other's entries; production observes one key.
/// `None`/empty means unavailable; caching fails closed.
static EXE_MEMO: Mutex<Vec<(BinidKey, [u8; 32])>> = Mutex::new(Vec::new());

/// Counter for unique store temp-file names (shared by receipts and binid).
static TMP_CTR: AtomicU64 = AtomicU64::new(0);

/// Whether a cache I/O warning has been emitted this process.
static WARNED_ONCE: AtomicBool = AtomicBool::new(false);

/// Most recently used paths, oldest first; bounded by [`RECENT_CAP`].
/// Eviction protects these in addition to the just-published entry.
static RECENT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Cap on [`RECENT`]; only the newest entries are protected.
const RECENT_CAP: usize = 32;

/// Content-addressed handle for one M7 certification request.
///
/// `None` from [`Cache::for_request`] means caching is disabled for this
/// request; the caller must run the checker uncached.
pub(crate) struct Cache {
    path: PathBuf,
    key: [u8; 32],
    key_hex: String,
}

impl Cache {
    /// Resolve the cache entry for the exact M7 inputs, or `None` when
    /// caching is disabled (env opt-out, no home dir) or the checker
    /// identity is unavailable. Never creates directories.
    pub(crate) fn for_request(bytes: &[u8], arch: &str, symbol: &str) -> Option<Self> {
        // Snapshot the environment once; never re-read it within this request.
        // The opt-out check runs before executable hashing by construction.
        let override_dir = std::env::var_os("HIPFIRE_CERT_CACHE");
        if matches!(override_dir.as_deref(), Some(v) if v == OsStr::new("0")) {
            return None;
        }
        let xdg = std::env::var_os("XDG_CACHE_HOME");
        let home = std::env::var_os("HOME");

        let dir = match override_dir {
            Some(p) if p.is_empty() => {
                warn_once("disabled: HIPFIRE_CERT_CACHE is empty".into());
                return None;
            }
            Some(p) => PathBuf::from(p),
            _ => match xdg {
                Some(p) if !p.is_empty() => PathBuf::from(p).join("hipfire/pm-m7"),
                _ => match home {
                    Some(p) if !p.is_empty() => PathBuf::from(p).join(".cache/hipfire/pm-m7"),
                    _ => return None,
                },
            },
        };
        let exe = resolve_root_identity(&dir)?;
        Some(Self::for_dir(dir, bytes, arch, symbol, &exe))
    }

    /// Build the entry handle for an explicit root directory. Shared by
    /// [`Cache::for_request`] and the `cfg(test)` root helper.
    fn for_dir(dir: PathBuf, bytes: &[u8], arch: &str, symbol: &str, exe: &[u8; 32]) -> Self {
        let key = derive_key(bytes, arch, symbol, exe);
        let key_hex = hex(&key);
        let path = dir.join(format!("{key_hex}.json"));
        Self { path, key, key_hex }
    }

    /// Load the cached success receipt, or `None` on any miss. Reads at most
    /// 1 MiB (+1 probe byte); oversize, unparseable, wrong-schema/key, or
    /// checksum-mismatched entries are deleted best-effort and reported as
    /// misses. The corrupt cleanup removes only this exact requested path.
    pub(crate) fn load(&self) -> Option<serde_json::Value> {
        let file = File::open(&self.path).ok()?;
        let mut bytes = Vec::new();
        if file
            .take(MAX_ENTRY_BYTES + 1)
            .read_to_end(&mut bytes)
            .ok()?
            > MAX_ENTRY_BYTES as usize
        {
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        let entry: CacheEntry = match serde_json::from_slice(&bytes) {
            Ok(e) => e,
            Err(_) => {
                let _ = std::fs::remove_file(&self.path);
                return None;
            }
        };
        if entry.schema != ENTRY_SCHEMA || entry.key != self.key_hex {
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        let canonical = match serde_json::to_vec(&entry.receipt) {
            Ok(v) => v,
            Err(_) => {
                let _ = std::fs::remove_file(&self.path);
                return None;
            }
        };
        if entry.checksum != hex(&checksum(&self.key, &canonical)) {
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        record_recent(&self.path);
        Some(entry.receipt)
    }

    /// Store a success receipt. Any environmental or storage failure
    /// (unwritable root, temp-file/rename errors, serialization errors)
    /// degrades silently to uncached execution; the caller's receipt is
    /// unaffected. Creates the cache directory if needed.
    pub(crate) fn store(&self, receipt: &serde_json::Value) {
        let Some(dir) = self.path.parent() else {
            return;
        };
        if let Err(e) = std::fs::create_dir_all(dir) {
            warn_once(format!("disabled: cannot create {}: {e}", dir.display()));
            return;
        }
        let canonical = match serde_json::to_vec(receipt) {
            Ok(v) => v,
            Err(_) => return,
        };
        let entry = StoredEntry {
            schema: ENTRY_SCHEMA,
            key: &self.key_hex,
            receipt,
            checksum: hex(&checksum(&self.key, &canonical)),
        };
        let bytes = match serde_json::to_vec(&entry) {
            Ok(v) => v,
            Err(_) => return,
        };
        let tmp = dir.join(format!(
            ".{}.tmp-{}-{}",
            self.key_hex,
            std::process::id(),
            TMP_CTR.fetch_add(1, Ordering::Relaxed)
        ));
        let write_result = (|| -> std::io::Result<()> {
            let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
            f.write_all(&bytes)?;
            drop(f);
            std::fs::rename(&tmp, &self.path)?;
            Ok(())
        })();
        if let Err(e) = write_result {
            let _ = std::fs::remove_file(&tmp);
            warn_once(format!("cannot write {}: {e}", self.path.display()));
            return;
        }
        record_recent(&self.path);
        enforce_entry_cap(dir, MAX_ENTRIES, &self.path);
    }

    /// Cache file path (tests only; drives corruption fixtures).
    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Raw request key bytes (tests only; drives key-dimension checks).
    #[cfg(test)]
    pub(crate) fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// Hex request key (tests only).
    #[cfg(test)]
    pub(crate) fn key_hex(&self) -> &str {
        &self.key_hex
    }

    /// Entry handle under an explicit root without touching the environment
    /// (tests only; uses the real executable identity, so `None` when the
    /// identity is unavailable).
    #[cfg(test)]
    pub(crate) fn for_request_in_root(
        root: &Path,
        bytes: &[u8],
        arch: &str,
        symbol: &str,
    ) -> Option<Self> {
        let exe = resolve_root_identity(root)?;
        Some(Self::for_dir(root.to_path_buf(), bytes, arch, symbol, &exe))
    }

    /// Whether the once-only I/O warning has fired this process (tests only).
    #[cfg(test)]
    pub(crate) fn warned_for_tests() -> bool {
        WARNED_ONCE.load(Ordering::Relaxed)
    }

    /// Reset the once-only I/O warning guard (tests only).
    #[cfg(test)]
    pub(crate) fn reset_warned_for_tests() {
        WARNED_ONCE.store(false, Ordering::Relaxed);
    }
}

/// Production request-key derivation: fixed schema and domain constants.
/// See [`derive_key_with`] for the schema dimension.
pub(crate) fn derive_key(elf: &[u8], arch: &str, symbol: &str, exe: &[u8; 32]) -> [u8; 32] {
    derive_key_with(ENTRY_SCHEMA, elf, arch, symbol, exe)
}

/// Request-key derivation with an explicit schema version. Production always
/// passes [`ENTRY_SCHEMA`]; tests pass a bumped version to prove the schema
/// is part of the key. Every field is length-framed so concatenations across
/// field boundaries cannot collide.
pub(crate) fn derive_key_with(
    schema: u32,
    elf: &[u8],
    arch: &str,
    symbol: &str,
    exe: &[u8; 32],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(KEY_DOMAIN);
    h.update(schema.to_le_bytes());
    h.update((elf.len() as u64).to_le_bytes());
    h.update(elf);
    h.update((arch.len() as u64).to_le_bytes());
    h.update(arch.as_bytes());
    h.update((symbol.len() as u64).to_le_bytes());
    h.update(symbol.as_bytes());
    h.update(exe);
    h.finalize().into()
}

/// On-disk entry. `deny_unknown_fields` rejects forward-versioned or
/// hand-edited entries instead of trusting a subset.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheEntry {
    schema: u32,
    key: String,
    receipt: serde_json::Value,
    checksum: String,
}

/// Borrow the receipt while publishing instead of cloning its entire tree.
#[derive(Serialize)]
struct StoredEntry<'a> {
    schema: u32,
    key: &'a str,
    receipt: &'a serde_json::Value,
    checksum: String,
}

/// SHA256 over the raw request key concatenated with the canonical receipt
/// bytes. Binds the payload to the exact request.
fn checksum(key: &[u8; 32], canonical_receipt: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(key);
    h.update(canonical_receipt);
    h.finalize().into()
}

/// Observatory for tests: where the executable digest came from.
/// Production ignores this; tests assert reuse vs rehash through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExeSource {
    /// Digest matched the `binid.json` record for this metadata; no image
    /// bytes were hashed on this path.
    Binid,
    /// Digest came from the in-process memo (an earlier hash or a validated
    /// record adopted this process); republishes best effort.
    Memo,
    /// Image bytes were hashed on this path (cold key, metadata change).
    Hashed,
}

/// Identity key for the running executable: Linux metadata fields that a
/// normal write or replacement changes. Seconds and nanoseconds stay
/// separate so no `sec * 1e9` product can overflow. `Copy` keeps the memo
/// lookup free of clones.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BinidKey {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
}

/// Executable identity for one cache root: full image SHA256.
fn resolve_root_identity(root: &Path) -> Option<[u8; 32]> {
    resolve_exe_identity(root, Path::new("/proc/self/exe")).map(|(_, digest, _)| digest)
}

/// Resolve the executable identity through the binid record, hashing at
/// most once per metadata key per process. A matching record returns its
/// digest without hashing; a memo hit for the same key republishes without
/// hashing. Otherwise hashes once, adopts the digest into the memo, and
/// publishes the record best effort. A binid write failure still returns
/// the known digest; only an unavailable identity returns `None` (caching
/// disabled).
fn resolve_exe_identity(root: &Path, exe_path: &Path) -> Option<(BinidKey, [u8; 32], ExeSource)> {
    let stat_key = stat_key(exe_path)?;
    // Binid first: on a fresh process the memo is empty and a matching
    // record avoids hashing entirely.
    if let Some(recorded) = load_binid(root, &stat_key) {
        memo_adopt(stat_key, recorded);
        return Some((stat_key, recorded, ExeSource::Binid));
    }
    // Memo fallback: these exact bytes were hashed (or a validated record
    // adopted) earlier in this process, so no rehash. Republishes best
    // effort so a deleted record heals and the next resolve observes
    // `Binid` again.
    if let Some(hit) = memo_hit(&stat_key) {
        store_binid(root, &stat_key, &hit);
        return Some((stat_key, hit, ExeSource::Memo));
    }
    let (key, digest) = hash_exe_image(exe_path)?;
    // Defense in depth against a `stat` race: the key must describe the
    // bytes just hashed. Any drift fails closed for this request.
    if key != stat_key {
        return None;
    }
    memo_adopt(key, digest);
    store_binid(root, &key, &digest);
    Some((key, digest, ExeSource::Hashed))
}

/// `stat` the executable path into a [`BinidKey`]. [`std::fs::metadata`]
/// follows the `/proc/self/exe` symlink to the target image.
#[cfg(target_os = "linux")]
fn stat_key(exe_path: &Path) -> Option<BinidKey> {
    Some(binid_key(&std::fs::metadata(exe_path).ok()?))
}

/// Non-Linux: no `/proc/self/exe` identity, so no metadata key.
#[cfg(not(target_os = "linux"))]
fn stat_key(_exe_path: &Path) -> Option<BinidKey> {
    None
}

/// Shared key constructor for `stat` and `fstat` metadata, so path and
/// descriptor observations cannot drift apart. Infallible: metadata reads
/// cannot fail, so callers stay concise.
#[cfg(target_os = "linux")]
fn binid_key(m: &std::fs::Metadata) -> BinidKey {
    use std::os::unix::fs::MetadataExt as _;
    BinidKey {
        dev: m.dev(),
        ino: m.ino(),
        size: m.size(),
        mtime_sec: m.mtime(),
        mtime_nsec: m.mtime_nsec(),
        ctime_sec: m.ctime(),
        ctime_nsec: m.ctime_nsec(),
    }
}
#[cfg(not(target_os = "linux"))]
fn binid_key(_m: &std::fs::Metadata) -> BinidKey {
    unreachable!("non-Linux identity is unavailable; stat_key returns None first")
}

/// Open the executable, `fstat` the same descriptor before and after the
/// streamed 8 KiB hash, and fail closed on any drift or I/O error. Never
/// loads the image fully into memory.
#[cfg(target_os = "linux")]
fn hash_exe_image(exe_path: &Path) -> Option<(BinidKey, [u8; 32])> {
    let mut file = File::open(exe_path).ok()?;
    let before = binid_key(&file.metadata().ok()?);
    let mut h = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let after = binid_key(&file.metadata().ok()?);
    if before != after {
        return None;
    }
    Some((after, h.finalize().into()))
}

/// Non-Linux: the identity is unavailable, so caching fails closed.
#[cfg(not(target_os = "linux"))]
fn hash_exe_image(_exe_path: &Path) -> Option<(BinidKey, [u8; 32])> {
    None
}

/// Process-wide known digest for this exact metadata key, if any.
fn memo_hit(key: &BinidKey) -> Option<[u8; 32]> {
    EXE_MEMO
        .lock()
        .ok()?
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, d)| *d)
}

/// Adopt a freshly hashed or record-loaded digest into the memo.
fn memo_adopt(key: BinidKey, digest: [u8; 32]) {
    if let Ok(mut memo) = EXE_MEMO.lock() {
        if let Some(slot) = memo.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = digest;
        } else {
            memo.push((key, digest));
        }
    }
}

/// On-disk identity record. `deny_unknown_fields` rejects forward-versioned
/// or hand-edited records instead of trusting a subset.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BinidRecord {
    schema: u32,
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
    digest: String,
    checksum: String,
}

/// Borrowed form while publishing (avoids hex-string clones).
#[derive(Serialize)]
struct StoredBinid<'a> {
    schema: u32,
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
    digest: &'a str,
    checksum: String,
}

/// SHA256 over the domain separator, the metadata key fields, and the raw
/// digest bytes. Binds the digest to the exact binary the key describes;
/// receipt fields can never validate as a binid record.
fn binid_checksum(key: &BinidKey, digest: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(BINID_DOMAIN);
    h.update(key.dev.to_le_bytes());
    h.update(key.ino.to_le_bytes());
    h.update(key.size.to_le_bytes());
    h.update(key.mtime_sec.to_le_bytes());
    h.update(key.mtime_nsec.to_le_bytes());
    h.update(key.ctime_sec.to_le_bytes());
    h.update(key.ctime_nsec.to_le_bytes());
    h.update(digest);
    h.finalize().into()
}

/// Raw digest bytes from lowercase hex. Rejects anything but 64 hex chars.
fn unhex_digest(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let hi = hex_val(chunk[0])?;
        let lo = hex_val(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

/// One lowercase hex digit value.
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        _ => None,
    }
}

/// Load the identity record: any miss, from absent to corrupt, is `None`.
/// Corrupt, incomplete, edited, or wrong-key records are deleted best
/// effort so the next resolve rehashes and republishes cleanly.
fn load_binid(root: &Path, key: &BinidKey) -> Option<[u8; 32]> {
    let path = root.join(BINID_FILE);
    let miss = |delete: bool| {
        if delete {
            let _ = std::fs::remove_file(&path);
        }
        None
    };
    let file = File::open(&path).ok()?;
    // Oversize (or exactly at the probe bound) means not our record.
    if file
        .metadata()
        .map(|m| m.len() > MAX_BINID_BYTES)
        .unwrap_or(true)
    {
        return miss(true);
    }
    let mut bytes = Vec::new();
    if file
        .take(MAX_BINID_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?
        > MAX_BINID_BYTES as usize
    {
        return miss(true);
    }
    let record: BinidRecord = match serde_json::from_slice(&bytes) {
        Ok(record) => record,
        Err(_) => return miss(true),
    };
    if record.schema != BINID_SCHEMA {
        return miss(true);
    }
    if record.dev != key.dev
        || record.ino != key.ino
        || record.size != key.size
        || record.mtime_sec != key.mtime_sec
        || record.mtime_nsec != key.mtime_nsec
        || record.ctime_sec != key.ctime_sec
        || record.ctime_nsec != key.ctime_nsec
    {
        // Stale record for a previous binary: keep it (a concurrent fresh
        // process may still match) and report a miss so this process
        // rehashes and republishes.
        return miss(false);
    }
    let Some(digest) = unhex_digest(&record.digest) else {
        return miss(true);
    };
    if record.digest != hex(&digest) {
        return miss(true);
    }
    if record.checksum != hex(&binid_checksum(key, &digest)) {
        return miss(true);
    }
    Some(digest)
}

/// Publish the identity record atomically (`create_new` temp + `rename`).
/// Failures warn once and keep the freshly hashed digest uncached: the
/// record never turns an established identity into a miss.
fn store_binid(root: &Path, key: &BinidKey, digest: &[u8; 32]) {
    if std::fs::create_dir_all(root).is_err() {
        warn_once(format!(
            "disabled: cannot create {}: {}",
            root.display(),
            BINID_FILE
        ));
        return;
    }
    let digest_hex = hex(digest);
    let record = StoredBinid {
        schema: BINID_SCHEMA,
        dev: key.dev,
        ino: key.ino,
        size: key.size,
        mtime_sec: key.mtime_sec,
        mtime_nsec: key.mtime_nsec,
        ctime_sec: key.ctime_sec,
        ctime_nsec: key.ctime_nsec,
        digest: &digest_hex,
        checksum: hex(&binid_checksum(key, digest)),
    };
    let bytes = match serde_json::to_vec(&record) {
        Ok(v) => v,
        Err(_) => return,
    };
    let tmp = root.join(format!(
        ".{}.tmp-{}-{}",
        BINID_FILE,
        std::process::id(),
        TMP_CTR.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| -> std::io::Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(&bytes)?;
        drop(f);
        std::fs::rename(&tmp, root.join(BINID_FILE))?;
        Ok(())
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        warn_once(format!(
            "cannot write {}: {e}",
            root.join(BINID_FILE).display()
        ));
    }
}

/// Lowercase hex encoding (avoids a formatting allocation per byte).
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

/// Print one cache I/O warning per process, with the stable [`WARN_PREFIX`].
fn warn_once(msg: String) {
    if !WARNED_ONCE.swap(true, Ordering::Relaxed) {
        eprintln!("{WARN_PREFIX}: {msg}");
    }
}

/// Remember a successfully loaded/stored path so eviction protects it.
fn record_recent(path: &Path) {
    let Ok(mut recent) = RECENT.lock() else {
        return;
    };
    let owned = if let Some(i) = recent.iter().position(|p| p == path) {
        recent.remove(i)
    } else {
        path.to_path_buf()
    };
    recent.push(owned);
    while recent.len() > RECENT_CAP {
        recent.remove(0);
    }
}
/// Whether `name` is a recognized owned cache entry filename: 64 lowercase
/// hex digits + `.json` (the SHA256 key encoding [`Cache::for_dir`] writes).
fn is_entry_name(name: &str) -> bool {
    name.len() == 64 + ".json".len()
        && name.ends_with(".json")
        && name.as_bytes()[..64]
            .iter()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Evict oldest-by-`mtime` owned entries over `cap`, never removing
/// `protect` (the just-published path) or any path in the process-wide
/// recently-used set. Only recognized owned entry filenames (64 lowercase
/// hex + `.json`) are candidates: unrelated files and our own active temp
/// files are never deleted. Best effort: I/O failures stop eviction
/// silently. `cap` is a parameter (rather than [`MAX_ENTRIES`] directly) so
/// tests can exercise eviction with small fixtures.
pub(crate) fn enforce_entry_cap(dir: &Path, cap: usize, protect: &Path) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<(PathBuf, Option<std::time::SystemTime>)> = Vec::new();
    for entry in read.flatten() {
        let p = entry.path();
        if !is_entry_name(p.file_name().and_then(|n| n.to_str()).unwrap_or("")) {
            continue;
        }
        if !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let mtime = std::fs::metadata(&p).and_then(|m| m.modified()).ok();
        entries.push((p, mtime));
    }
    if entries.len() <= cap {
        return;
    }
    // Oldest first; unknown mtime sorts as oldest so undatable files do not
    // accumulate forever.
    entries.sort_by_key(|entry| entry.1);
    let recent = RECENT.lock().map(|r| r.clone()).unwrap_or_default();
    let mut over = entries.len().saturating_sub(cap);
    for (path, _) in &entries {
        if over == 0 {
            break;
        }
        if path == protect || recent.iter().any(|p| p == path) {
            continue;
        }
        if std::fs::remove_file(path).is_ok() {
            over -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    static TEST_DIR_CTR: AtomicU64 = AtomicU64::new(0);

    /// Unique scratch root per test. No env mutation, no shared fixtures.
    fn test_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pm-m7-certcache-{}-{}-{name}",
            std::process::id(),
            TEST_DIR_CTR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test root");
        dir
    }

    fn cache_in(root: &Path, elf: &[u8], arch: &str, symbol: &str) -> Option<Cache> {
        Cache::for_request_in_root(root, elf, arch, symbol)
    }

    /// `None` (identity unavailable) skips the test: unit tests must not
    /// fail on platforms without `/proc/self/exe`.
    macro_rules! require_cache {
        ($e:expr) => {
            match $e {
                Some(c) => c,
                None => return,
            }
        };
    }

    const EXE_A: [u8; 32] = [0x11; 32];
    const EXE_B: [u8; 32] = [0x22; 32];

    #[test]
    fn key_differs_across_elf_arch_symbol() {
        let root = test_root("keydims");
        let base = require_cache!(cache_in(&root, b"elf-bytes", "gfx1100", "sym"));
        for (elf, arch, sym) in [
            (b"elf-bytez".as_slice(), "gfx1100", "sym"),
            (b"elf-bytes".as_slice(), "gfx1101", "sym"),
            (b"elf-bytes".as_slice(), "gfx1100", "sym2"),
            (b"".as_slice(), "gfx1100", "sym"),
        ] {
            let other = require_cache!(cache_in(&root, elf, arch, sym));
            assert_ne!(base.key(), other.key(), "elf={elf:?} arch={arch} sym={sym}");
            assert_ne!(base.path(), other.path());
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn key_framing_separates_concatenation_collisions() {
        // Naive concatenation collides: "ab"+"c" == "a"+"bc". Length
        // framing must keep these distinct in every field pairing.
        assert_ne!(
            derive_key(b"ab", "c", "s", &EXE_A),
            derive_key(b"a", "bc", "s", &EXE_A),
        );
        assert_ne!(
            derive_key(b"e", "ab", "c", &EXE_A),
            derive_key(b"e", "a", "bc", &EXE_A),
        );
    }

    #[test]
    fn key_differs_across_schema_version() {
        // Production pins ENTRY_SCHEMA; a schema bump must rekey everything.
        assert_eq!(
            derive_key(b"e", "a", "s", &EXE_A),
            derive_key_with(ENTRY_SCHEMA, b"e", "a", "s", &EXE_A),
        );
        assert_ne!(
            derive_key(b"e", "a", "s", &EXE_A),
            derive_key_with(ENTRY_SCHEMA.wrapping_add(1), b"e", "a", "s", &EXE_A),
        );
    }

    #[test]
    fn key_differs_across_executable_identity() {
        // A rebuilt/relocated checker image must never hit a stale entry.
        assert_ne!(
            derive_key(b"e", "a", "s", &EXE_A),
            derive_key(b"e", "a", "s", &EXE_B),
        );
    }

    #[test]
    fn roundtrip_preserves_canonical_bytes() {
        let dir = test_root("roundtrip");
        let c = require_cache!(cache_in(&dir, b"\x7fELF-m7", "gfx1100", "k"));
        assert!(c.load().is_none());
        let receipt =
            serde_json::json!({"ok": true, "arch": "gfx1100", "nested": {"b": 1, "a": [3, 2, 1]}});
        c.store(&receipt);
        let hit = c.load().expect("cache hit");
        assert_eq!(hit, receipt);
        assert_eq!(
            serde_json::to_vec(&hit).unwrap(),
            serde_json::to_vec(&receipt).unwrap(),
            "hit bytes must equal uncached value bytes",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Store a valid entry, rewrite its file, and assert the load outcome.
    /// Returns the entry path for deletion assertions.
    fn tampered_path(c: &Cache, f: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
        c.store(&serde_json::json!({"ok": true, "v": 1}));
        let mut v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(c.path()).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(c.path(), serde_json::to_vec(&v).unwrap()).unwrap();
        c.path().to_path_buf()
    }

    #[test]
    fn load_rejects_truncated_entry_and_deletes() {
        let dir = test_root("trunc");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        std::fs::write(c.path(), b"{not json").unwrap();
        assert!(c.load().is_none());
        assert!(!c.path().exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_unknown_fields_and_deletes() {
        let dir = test_root("unknown");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        let p = tampered_path(&c, |v| v["future_field"] = serde_json::json!(1));
        assert!(c.load().is_none());
        assert!(!p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_missing_checksum_and_deletes() {
        let dir = test_root("missing");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        let p = tampered_path(&c, |v| {
            v.as_object_mut().unwrap().remove("checksum");
        });
        assert!(c.load().is_none());
        assert!(!p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_schema_mismatch_and_deletes() {
        let dir = test_root("schema");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        // Checksum covers key+receipt only, so it stays valid here: the
        // rejection is purely the schema gate.
        let p = tampered_path(&c, |v| v["schema"] = serde_json::json!(ENTRY_SCHEMA + 999));
        assert!(c.load().is_none());
        assert!(!p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_envelope_key_mismatch_and_deletes() {
        let dir = test_root("envkey");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        let other = require_cache!(cache_in(&dir, b"other-elf", "a", "s"));
        let other_hex = other.key_hex().to_owned();
        assert_ne!(c.key_hex(), other_hex);
        let p = tampered_path(&c, |v| v["key"] = serde_json::json!(other_hex));
        // Never exposes the foreign payload.
        assert!(c.load().is_none());
        assert!(!p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_edited_receipt_checksum_mismatch_and_deletes() {
        let dir = test_root("checksum");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        let p = tampered_path(&c, |v| v["receipt"]["ok"] = serde_json::json!(false));
        assert!(c.load().is_none());
        assert!(!p.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_rejects_oversize_entry_and_deletes() {
        let dir = test_root("oversize");
        let c = require_cache!(cache_in(&dir, b"e", "a", "s"));
        // MAX_ENTRY_BYTES of valid JSON prefix plus one byte: forces the
        // oversize path without allocating a huge Value.
        let mut big = vec![b' '; MAX_ENTRY_BYTES as usize + 1];
        big[0] = b'[';
        std::fs::write(c.path(), &big).unwrap();
        assert!(c.load().is_none());
        assert!(!c.path().exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn owned_name(seed: u8) -> String {
        format!("{:02x}", seed).repeat(32) + ".json"
    }

    #[test]
    fn eviction_cap_helper_keeps_protect_and_shrinks_to_cap() {
        let dir = test_root("evict-cap");
        // Owned entry filenames only (64 lowercase hex + .json).
        for seed in [1u8, 2, 3, 4] {
            std::fs::write(dir.join(owned_name(seed)), br#"{"x":1}"#).unwrap();
        }
        // A non-entry temp file and an unrelated JSON file must be ignored
        // by the cap count and never deleted.
        std::fs::write(dir.join(".abc.tmp-1-2"), b"partial").unwrap();
        std::fs::write(dir.join("notes.json"), br#"{"x":1}"#).unwrap();
        let protect = dir.join(owned_name(4));
        enforce_entry_cap(&dir, 2, &protect);
        assert!(protect.exists());
        assert!(dir.join("notes.json").exists());
        assert!(dir.join(".abc.tmp-1-2").exists());
        let count = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| is_entry_name(e.path().file_name().and_then(|n| n.to_str()).unwrap_or("")))
            .count();
        assert_eq!(count, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_never_removes_just_used_entry() {
        let dir = test_root("evict-recent");
        let keep = require_cache!(cache_in(&dir, b"keep-elf", "gfx1100", "keep"));
        let receipt = serde_json::json!({"ok": true});
        keep.store(&receipt);
        assert!(keep.load().is_some());
        // `keep` is now in the process-wide recently-used set.

        for seed in [0xAAu8, 0xBB] {
            std::fs::write(dir.join(owned_name(seed)), br#"{"x":1}"#).unwrap();
        }
        let cur = require_cache!(cache_in(&dir, b"cur-elf", "gfx1100", "cur"));
        cur.store(&receipt);

        enforce_entry_cap(&dir, 1, cur.path());

        assert!(cur.path().exists(), "just-published entry must survive");
        assert!(keep.path().exists(), "just-used entry must survive");
        assert!(!dir.join(owned_name(0xAA)).exists());
        assert!(!dir.join(owned_name(0xBB)).exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unwritable_root_store_is_uncached_and_warns() {
        Cache::reset_warned_for_tests();
        let dir = test_root("unwritable");
        let blocker = dir.join("blocker");
        std::fs::write(&blocker, b"x").unwrap();
        // Entry path is blocker/<hex>.json, so create_dir_all(blocker) fails.
        let c = require_cache!(cache_in(&blocker, b"e", "a", "s"));
        // First failure warns and stays uncached; the repeat stays silent
        // (once-only guard) and uncached. No panic either way.
        c.store(&serde_json::json!({"ok": true}));
        assert!(c.load().is_none());
        assert!(Cache::warned_for_tests());
        c.store(&serde_json::json!({"ok": true}));
        assert!(c.load().is_none());
        assert!(Cache::warned_for_tests());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binid_reuses_digest_until_exe_copy_rewritten() {
        let dir = test_root("binid-reuse");
        // Duplicate the running image into scratch space and rewrite the
        // copy, never the live executable.
        let exe_copy = dir.join("exe-copy");
        if std::fs::copy("/proc/self/exe", &exe_copy).is_err() {
            let _ = std::fs::remove_dir_all(&dir);
            return; // No /proc/self/exe on this platform; nothing to prove.
        }
        let Some((_, first, first_src)) = resolve_exe_identity(&dir, &exe_copy) else {
            let _ = std::fs::remove_dir_all(&dir);
            return; // Identity unavailable here; skip like `require_cache!`.
        };
        assert_eq!(first_src, ExeSource::Hashed, "cold root must hash once");
        assert!(
            dir.join(BINID_FILE).exists(),
            "miss must publish the record"
        );
        let (_, second, second_src) = resolve_exe_identity(&dir, &exe_copy).expect("warm resolve");
        assert_eq!(second, first, "same metadata must reuse the digest");
        assert_eq!(second_src, ExeSource::Binid, "warm resolve must not rehash");
        // Rewrite the copy: new bytes with a different size/mtime.
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&exe_copy)
                .expect("reopen copy");
            f.write_all(b"pm-m7-binid-rewrite-salt").expect("salt");
            f.sync_all().ok();
        }
        let (_, third, third_src) =
            resolve_exe_identity(&dir, &exe_copy).expect("rewritten resolve");
        assert_eq!(third_src, ExeSource::Hashed, "metadata change must rehash");
        assert_ne!(third, first, "new bytes must yield a new digest");
        let (_, fourth, fourth_src) =
            resolve_exe_identity(&dir, &exe_copy).expect("rewarmed resolve");
        assert_eq!(fourth, third);
        assert_eq!(fourth_src, ExeSource::Binid);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binid_corrupt_and_edited_records_rehash() {
        let dir = test_root("binid-corrupt");
        let fake = dir.join("fake-exe");
        std::fs::write(&fake, b"fake-executable-bytes-v1").expect("fixture");
        let Some((_, first, _)) = resolve_exe_identity(&dir, &fake) else {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        };
        // Truncated record: the memo republishes the known digest without a
        // rehash, then warm reuse reads the healed record.
        std::fs::write(dir.join(BINID_FILE), b"{corrupt").expect("corrupt");
        let (_, rehashed, rehashed_src) =
            resolve_exe_identity(&dir, &fake).expect("corrupt resolve");
        assert_eq!(rehashed_src, ExeSource::Memo);
        assert_eq!(rehashed, first);
        let (_, warm, warm_src) = resolve_exe_identity(&dir, &fake).expect("rewarm");
        assert_eq!(warm, first);
        assert_eq!(warm_src, ExeSource::Binid);
        // Edited digest field: the checksum gate must reject it even though
        // the hex itself is well-formed; the memo again carries the digest.
        let mut v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(BINID_FILE)).expect("binid"))
                .expect("json");
        v["digest"] = serde_json::json!("00".repeat(32));
        std::fs::write(
            dir.join(BINID_FILE),
            serde_json::to_vec(&v).expect("edited"),
        )
        .expect("write");
        let (_, edited, edited_src) = resolve_exe_identity(&dir, &fake).expect("edited resolve");
        assert_eq!(edited_src, ExeSource::Memo);
        assert_eq!(edited, first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn binid_ignored_by_entry_cap() {
        let dir = test_root("binid-cap");
        let fake = dir.join("fake-exe");
        std::fs::write(&fake, b"fake-executable-bytes-v1").expect("fixture");
        if resolve_exe_identity(&dir, &fake).is_none() {
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        assert!(dir.join(BINID_FILE).exists());
        for seed in [1u8, 2, 3] {
            std::fs::write(dir.join(owned_name(seed)), br#"{"x":1}"#).expect("entry");
        }
        let protect = dir.join(owned_name(3));
        enforce_entry_cap(&dir, 2, &protect);
        assert!(dir.join(BINID_FILE).exists(), "binid must survive eviction");
        let count = std::fs::read_dir(&dir)
            .expect("read")
            .flatten()
            .filter(|e| is_entry_name(e.path().file_name().and_then(|n| n.to_str()).unwrap_or("")))
            .count();
        assert_eq!(count, 2, "binid must not count toward the receipt cap");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
