// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The per-install store of the driver floor's load snapshots
//! (SPEC-AR-FLOOR S2a-5b, verdict §7.11 item 4).
//!
//! A load-time calibration costs about a second of load time. A load whose
//! [`FloorSnapshotKey`] matches a stored entry freezes the stored prices
//! instead and skips the calibration. The key names everything that moves
//! the prices: the daemon build (binary md5), the target and drafter files
//! (their verified-artifact SHA-256 when the install's cache holds one, else
//! path, size and mtime: never a hash of the model at load), the GPU (arch,
//! device name, VRAM), the HIP runtime version, the AR route, the KV and
//! DeltaNet state formats, the verify block size and the pipeline / tensor
//! parallel degrees.
//!
//! One JSON file under the hipfire home, beside the verified-artifact cache
//! (`<HIPFIRE_HOME>/cache/spec-floor-prices.json`), with a schema version,
//! written by temporary file and rename. Every read, parse, version or
//! validation failure is a miss, never an error. An entry is written only
//! from a tight calibration ([`tight_spread`]) and deleted when a load's live
//! prices drift from it by more than [`SNAPSHOT_STALE_DRIFT`].

use crate::spec_prices::{LoadPrices, PinnedPrices, LOAD_PRICE_WINDOW_ROWS, PRICE_WINDOW};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

/// Schema version of the store file; any other version reads as empty.
pub const SNAPSHOT_STORE_VERSION: u32 = 1;
/// The store's file name under `<HIPFIRE_HOME>/cache`.
pub const SNAPSHOT_STORE_FILE: &str = "spec-floor-prices.json";
/// Largest sample dispersion, `(max − min) ÷ median` of a cell's kept
/// samples, of a calibration that may be written: the AR cell and the
/// full-block window cell must both be within it. A looser calibration is
/// used for its own load and not stored.
pub const SNAPSHOT_TIGHT_SPREAD: f64 = 0.05;
/// Live price drift, `|live median ÷ snapshot − 1|` over a full cell, past
/// which a load's stored entry is deleted (the load keeps its snapshot; the
/// next load recalibrates).
pub const SNAPSHOT_STALE_DRIFT: f64 = 0.10;

/// Everything that moves a load's speculation prices. Two loads share a
/// snapshot only when every field is equal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloorSnapshotKey {
    /// md5 of the daemon binary.
    pub daemon_md5: String,
    /// Target file identity: `sha256:<hex>` (verified-artifact cache) or
    /// `stat:<path>:<size>:<mtime_sec>.<mtime_nsec>`.
    pub target: String,
    /// Drafter file identity, same forms.
    pub drafter: String,
    /// `<arch>|<device name>|<total VRAM bytes>`.
    pub device: String,
    /// Raw `hipRuntimeGetVersion`.
    pub hip_runtime: i32,
    /// The AR executor: route class and transport.
    pub route: String,
    /// KV cache format.
    pub kv: String,
    /// DeltaNet (GDN) state format.
    pub gdn: String,
    /// Verified rows of a full block.
    pub block_size: usize,
    pub pp: usize,
    pub tp: usize,
}

/// `(max − min) ÷ median` within [`SNAPSHOT_TIGHT_SPREAD`] for both cells.
pub fn tight_spread(ar_spread: f64, window_spread: f64) -> bool {
    ar_spread <= SNAPSHOT_TIGHT_SPREAD && window_spread <= SNAPSHOT_TIGHT_SPREAD
}

/// The largest drift `|live median ÷ snapshot − 1|` of a full
/// ([`PRICE_WINDOW`] kept samples) base-bucket cell of `live` (the AR cell
/// and the full-block window cell) from `snapshot`; `None` while neither
/// cell is full. The snapshot is the base bucket's calibration, so only the
/// base bucket is compared.
pub fn snapshot_drift(snapshot: &PinnedPrices, live: &LoadPrices) -> Option<f64> {
    let full = snapshot.full_rows;
    [(0, snapshot.ar_us), (full, snapshot.window_us[full])]
        .into_iter()
        .filter_map(|(rows, priced)| {
            let cell = live.cell(rows, 0)?;
            if cell.kept() != PRICE_WINDOW || priced <= 0.0 {
                return None;
            }
            Some((cell.median()? / f64::from(priced) - 1.0).abs())
        })
        .reduce(f64::max)
}

/// The md5 of a binary, hashed on a background thread so a caller that
/// starts it at process start never waits for it at load. Read with
/// [`BinaryDigest::get`], which joins the thread once; an unreadable file
/// is `None` (the store is then not used).
pub struct BinaryDigest {
    worker: Mutex<Option<JoinHandle<Option<String>>>>,
    value: OnceLock<Option<String>>,
}

impl BinaryDigest {
    /// Start hashing `path`.
    pub fn spawn(path: PathBuf) -> Self {
        let worker = std::thread::Builder::new()
            .name("hipfire-self-md5".into())
            .spawn(move || md5_of(&path))
            .ok();
        Self {
            worker: Mutex::new(worker),
            value: OnceLock::new(),
        }
    }

    /// Start hashing the running executable.
    pub fn of_current_exe() -> Self {
        match std::env::current_exe() {
            Ok(path) => Self::spawn(path),
            Err(_) => Self {
                worker: Mutex::new(None),
                value: OnceLock::new(),
            },
        }
    }

    /// The hex md5, waiting for the thread the first time.
    pub fn get(&self) -> Option<&str> {
        self.value
            .get_or_init(|| {
                let worker = self.worker.lock().ok()?.take()?;
                worker.join().ok().flatten()
            })
            .as_deref()
    }
}

fn md5_of(path: &Path) -> Option<String> {
    let mut file = fs::File::open(path).ok()?;
    let mut context = md5::Context::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        context.consume(&buf[..n]);
    }
    Some(format!("{:x}", context.compute()))
}

/// A file identity of `path` without hashing it: `sha256:<hex>` when
/// `verified` (the install's verified-artifact digest for the file's
/// current stat) is known, else `stat:<path>:<size>:<mtime_sec>.<nsec>`.
/// `None` when the file cannot be stat'ed.
pub fn file_identity(path: &Path, verified: Option<&str>) -> Option<String> {
    if let Some(sha) = verified {
        return Some(format!("sha256:{sha}"));
    }
    let stat = crate::verified_cache::StatKey::of(&fs::metadata(path).ok()?);
    Some(format!(
        "stat:{}:{}:{}.{:09}",
        path.display(),
        stat.size,
        stat.mtime_sec,
        stat.mtime_nsec
    ))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Entry {
    key: FloorSnapshotKey,
    ar_us: f32,
    /// Window price by verified rows, index 0 unused (`0.0` = no price).
    window_us: Vec<f32>,
    full_rows: usize,
    /// The calibration's dispersion, for the record.
    ar_spread: f64,
    window_spread: f64,
    /// Unix seconds.
    written_at: u64,
}

impl Entry {
    /// The entry's prices, if they are a valid snapshot for `block_size`.
    fn prices(&self, block_size: usize) -> Option<PinnedPrices> {
        let valid = |us: f32| us.is_finite() && us > 0.0;
        if self.full_rows != block_size
            || !(1..=LOAD_PRICE_WINDOW_ROWS).contains(&self.full_rows)
            || self.window_us.len() != LOAD_PRICE_WINDOW_ROWS + 1
            || !valid(self.ar_us)
            || !valid(self.window_us[self.full_rows])
            || self.window_us.iter().any(|us| !us.is_finite() || *us < 0.0)
        {
            return None;
        }
        let mut window_us = [0.0; LOAD_PRICE_WINDOW_ROWS + 1];
        window_us.copy_from_slice(&self.window_us);
        Some(PinnedPrices {
            ar_us: self.ar_us,
            window_us,
            full_rows: self.full_rows,
        })
    }
}

#[derive(Default, Serialize, Deserialize)]
struct StoreFile {
    version: u32,
    entries: Vec<Entry>,
}

/// The on-disk snapshot store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloorSnapshotStore {
    file: PathBuf,
}

impl FloorSnapshotStore {
    /// The store of the hipfire home `root`:
    /// `<root>/cache/spec-floor-prices.json`.
    pub fn in_home(root: &Path) -> Self {
        Self::at(root.join("cache").join(SNAPSHOT_STORE_FILE))
    }

    /// A store kept in `file`.
    pub fn at(file: PathBuf) -> Self {
        Self { file }
    }

    pub fn file(&self) -> &Path {
        &self.file
    }

    /// Every entry; a missing, unreadable, corrupt or foreign-version file
    /// reads as empty.
    fn load(&self) -> Vec<Entry> {
        fs::read(&self.file)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<StoreFile>(&bytes).ok())
            .filter(|store| store.version == SNAPSHOT_STORE_VERSION)
            .map(|store| store.entries)
            .unwrap_or_default()
    }

    fn save(&self, entries: Vec<Entry>) -> std::io::Result<()> {
        crate::verified_cache::write_json_atomically(
            &self.file,
            &StoreFile {
                version: SNAPSHOT_STORE_VERSION,
                entries,
            },
        )
    }

    /// The snapshot stored for `key`. A missing file, a read or parse
    /// error, another schema version, no entry for `key`, or an entry whose
    /// prices are not a valid snapshot for the key's block size is a miss.
    pub fn lookup(&self, key: &FloorSnapshotKey) -> Option<PinnedPrices> {
        self.load()
            .iter()
            .find(|entry| entry.key == *key)
            .and_then(|entry| entry.prices(key.block_size))
    }

    /// Store `prices` for `key` (replacing an entry with the same key),
    /// with the calibration's dispersion. Atomic; an error leaves the file as
    /// it was.
    pub fn store(
        &self,
        key: &FloorSnapshotKey,
        prices: &PinnedPrices,
        ar_spread: f64,
        window_spread: f64,
    ) -> std::io::Result<()> {
        let mut entries = self.load();
        entries.retain(|entry| entry.key != *key);
        entries.push(Entry {
            key: key.clone(),
            ar_us: prices.ar_us,
            window_us: prices.window_us.to_vec(),
            full_rows: prices.full_rows,
            ar_spread,
            window_spread,
            written_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        });
        self.save(entries)
    }

    /// Delete `key`'s entry; `Ok(false)` when there was none.
    pub fn remove(&self, key: &FloorSnapshotKey) -> std::io::Result<bool> {
        let mut entries = self.load();
        let before = entries.len();
        entries.retain(|entry| entry.key != *key);
        if entries.len() == before {
            return Ok(false);
        }
        self.save(entries).map(|()| true)
    }
}

/// The stored entry a load's snapshot is tied to: deleted when the load's
/// live prices drift from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedSnapshot {
    pub store: FloorSnapshotStore,
    pub key: FloorSnapshotKey,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(label: &str) -> FloorSnapshotStore {
        let dir = tempfile::Builder::new()
            .prefix(&format!("hipfire-floor-store-{label}-"))
            .tempdir()
            .unwrap()
            .keep();
        FloorSnapshotStore::in_home(&dir)
    }

    pub(crate) fn key() -> FloorSnapshotKey {
        FloorSnapshotKey {
            daemon_md5: "e5270bdae7a164fff43dda40b21197d6".into(),
            target: "sha256:".to_owned() + &"ab".repeat(32),
            drafter: "stat:/m/qwen38-27b-dflash-mq4.hfq:3221225472:1763000000.000000001".into(),
            device: "gfx1201|AMD Radeon AI PRO R9700|34342961152".into(),
            hip_runtime: 60_443_484,
            route: "ArRouteKey(5)|pm4".into(),
            kv: "q8".into(),
            gdn: "Q8".into(),
            block_size: 16,
            pp: 1,
            tp: 1,
        }
    }

    pub(crate) fn prices(ar_us: f32, window_us: f32) -> PinnedPrices {
        let mut rows = [0.0; LOAD_PRICE_WINDOW_ROWS + 1];
        rows[16] = window_us;
        rows[12] = window_us * 0.85;
        PinnedPrices {
            ar_us,
            window_us: rows,
            full_rows: 16,
        }
    }

    #[test]
    fn a_stored_snapshot_round_trips_and_lives_under_the_home_cache() {
        let store = scratch("roundtrip");
        assert!(store.file().ends_with("cache/spec-floor-prices.json"));
        assert_eq!(store.lookup(&key()), None, "missing file is a miss");
        let pinned = prices(23_412.5, 39_050.25);
        store.store(&key(), &pinned, 0.012, 0.031).unwrap();
        assert_eq!(store.lookup(&key()), Some(pinned));
        // Replacing keeps one entry per key.
        let newer = prices(23_000.0, 39_000.0);
        store.store(&key(), &newer, 0.0, 0.0).unwrap();
        assert_eq!(store.lookup(&key()), Some(newer));
        assert_eq!(store.load().len(), 1);
        assert!(store.remove(&key()).unwrap());
        assert!(!store.remove(&key()).unwrap());
        assert_eq!(store.lookup(&key()), None);
    }

    #[test]
    fn every_key_component_is_part_of_the_key() {
        let store = scratch("components");
        store
            .store(&key(), &prices(23_400.0, 39_000.0), 0.0, 0.0)
            .unwrap();
        assert!(store.lookup(&key()).is_some());
        let changes: [(&str, fn(&mut FloorSnapshotKey)); 11] = [
            ("daemon binary", |k| k.daemon_md5.push('0')),
            ("target file", |k| {
                k.target = "sha256:".to_owned() + &"cd".repeat(32)
            }),
            ("drafter file", |k| {
                k.drafter = k.drafter.replace("1763000000", "1763000001")
            }),
            ("device", |k| {
                k.device = "gfx1201|AMD Radeon RX 9070 XT|17095983104".into()
            }),
            ("HIP runtime", |k| k.hip_runtime += 1),
            ("route", |k| k.route = "ArRouteKey(0)|hip".into()),
            ("KV format", |k| k.kv = "asym3".into()),
            ("GDN format", |k| k.gdn = "F32".into()),
            ("block size", |k| k.block_size = 8),
            ("pp", |k| k.pp = 2),
            ("tp", |k| k.tp = 2),
        ];
        for (what, change) in changes {
            let mut other = key();
            change(&mut other);
            assert_ne!(other, key(), "{what}");
            assert_eq!(store.lookup(&other), None, "a {what} change is a miss");
        }
    }

    #[test]
    fn a_corrupt_or_foreign_file_is_a_miss_and_never_an_error() {
        let store = scratch("corrupt");
        fs::create_dir_all(store.file().parent().unwrap()).unwrap();
        for bytes in [
            &b""[..],
            b"{",
            b"not json",
            b"{\"version\":2,\"entries\":[]}",
            b"{\"version\":1,\"entries\":[{\"key\":7}]}",
            b"[1,2,3]",
        ] {
            fs::write(store.file(), bytes).unwrap();
            assert_eq!(store.lookup(&key()), None, "{bytes:?}");
        }
        // A foreign version is overwritten by a store, never merged.
        fs::write(store.file(), b"{\"version\":99,\"entries\":[]}").unwrap();
        store
            .store(&key(), &prices(20_000.0, 40_000.0), 0.0, 0.0)
            .unwrap();
        assert!(store.lookup(&key()).is_some());
        // An unreadable store (a directory in its place) is a miss, and a
        // write into it fails without panicking.
        let blocked = scratch("blocked");
        fs::create_dir_all(blocked.file()).unwrap();
        assert_eq!(blocked.lookup(&key()), None);
        assert!(blocked.store(&key(), &prices(1.0, 2.0), 0.0, 0.0).is_err());
    }

    #[test]
    fn an_invalid_entry_is_a_miss() {
        let store = scratch("invalid");
        let mut bad = prices(23_400.0, 39_000.0);
        bad.ar_us = f32::NAN;
        store.store(&key(), &bad, 0.0, 0.0).unwrap();
        assert_eq!(store.lookup(&key()), None, "invalid AR price");
        let mut bad = prices(23_400.0, 39_000.0);
        bad.window_us[16] = 0.0;
        store.store(&key(), &bad, 0.0, 0.0).unwrap();
        assert_eq!(store.lookup(&key()), None, "no full-block price");
        // An entry stored for another block size never matches this key.
        let mut eight = key();
        eight.block_size = 8;
        store
            .store(&eight, &prices(23_400.0, 39_000.0), 0.0, 0.0)
            .unwrap();
        assert_eq!(store.lookup(&eight), None, "full_rows 16 vs block 8");
    }

    #[test]
    fn tight_means_both_spreads_within_five_percent() {
        assert!(tight_spread(0.0, 0.0));
        assert!(tight_spread(0.05, 0.05));
        assert!(!tight_spread(0.051, 0.0));
        assert!(!tight_spread(0.0, 0.2));
    }

    #[test]
    fn file_identity_prefers_the_verified_digest_and_never_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.hfq");
        fs::write(&file, b"weights").unwrap();
        let sha = "ef".repeat(32);
        assert_eq!(
            file_identity(&file, Some(&sha)),
            Some(format!("sha256:{sha}"))
        );
        let stat = file_identity(&file, None).unwrap();
        assert!(
            stat.starts_with(&format!("stat:{}:7:", file.display())),
            "{stat}"
        );
        // A rewrite of the same size with a new mtime is a different file.
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&file, b"WEIGHTS").unwrap();
        let touched = file_identity(&file, None).unwrap();
        assert_ne!(stat, touched);
        assert_eq!(file_identity(&dir.path().join("missing"), None), None);
    }
}

#[cfg(test)]
mod planted {
    //! Gate 25 (CPU half) and gate 25b: a planted snapshot through the
    //! load's `DriverFloor` lifecycle.
    use super::tests::{key, prices};
    use super::*;
    use crate::spec::{DriverFloor, FloorSnapshotSource};

    /// The calibrated prices of a steady R9700 `qwen3.8:27b` load (b ≈ 1.67).
    const AR_US: f32 = 23_400.0;
    const WINDOW_US: f32 = 39_000.0;

    fn store(label: &str) -> FloorSnapshotStore {
        let dir = tempfile::Builder::new()
            .prefix(&format!("hipfire-floor-planted-{label}-"))
            .tempdir()
            .unwrap()
            .keep();
        FloorSnapshotStore::in_home(&dir)
    }

    /// A load that reads `store`'s entry for `key()`, as `prepare_spec_floor`
    /// does on a hit.
    fn hit_load(store: &FloorSnapshotStore) -> DriverFloor {
        let mut df = DriverFloor::new(16, false);
        let pinned = store.lookup(&key()).expect("the planted entry is a hit");
        assert!(df.freeze_snapshot(pinned, FloorSnapshotSource::Cache));
        df.persist_to(PersistedSnapshot {
            store: store.clone(),
            key: key(),
        });
        df
    }

    #[test]
    fn a_planted_halved_ar_price_is_used_then_deleted_by_drift_and_the_next_load_recalibrates() {
        let store = store("halved");
        store
            .store(&key(), &prices(AR_US / 2.0, WINDOW_US), 0.01, 0.01)
            .unwrap();
        let mut df = hit_load(&store);
        // Used: the load decides on the planted price (b doubled, so prose
        // retires and runs AR tokens).
        let (snapshot, source) = df.snapshot().unwrap();
        assert_eq!(source.name(), "cache");
        assert_eq!(snapshot.ar_us, AR_US / 2.0);
        // Fewer than a full cell of live AR samples: no verdict yet.
        for _ in 0..PRICE_WINDOW - 1 {
            df.prices.record_ar(300, AR_US);
        }
        assert_eq!(df.retire_stale_snapshot(), None);
        assert!(store.lookup(&key()).is_some());
        // A full cell at the true price drifts 100% from the plant: deleted
        // once, the load keeps its snapshot.
        df.prices.record_ar(300, AR_US);
        let drift = df.retire_stale_snapshot().expect("stale");
        assert!((drift - 1.0).abs() < 1e-6, "{drift}");
        assert_eq!(store.lookup(&key()), None, "deleted");
        assert!(df.persisted().is_none());
        assert_eq!(
            df.snapshot().unwrap().0.ar_us,
            AR_US / 2.0,
            "the load keeps it"
        );
        assert_eq!(df.retire_stale_snapshot(), None, "logged once");
        // The next load misses and recalibrates.
        assert_eq!(store.lookup(&key()), None);
    }

    #[test]
    fn live_prices_within_ten_percent_keep_the_entry() {
        let store = store("steady");
        store
            .store(&key(), &prices(AR_US, WINDOW_US), 0.01, 0.01)
            .unwrap();
        let mut df = hit_load(&store);
        for _ in 0..PRICE_WINDOW {
            df.prices.record_ar(300, AR_US * 1.09);
        }
        for _ in 0..=PRICE_WINDOW {
            df.prices.record_window(16, 300, WINDOW_US * 0.92);
        }
        assert_eq!(df.retire_stale_snapshot(), None);
        assert!(store.lookup(&key()).is_some());
        // The window cell alone can also condemn it.
        for _ in 0..PRICE_WINDOW {
            df.prices.record_window(16, 300, WINDOW_US * 1.2);
        }
        assert!(df.retire_stale_snapshot().is_some());
        assert_eq!(store.lookup(&key()), None);
    }

    #[test]
    fn a_long_context_cell_never_condemns_the_base_bucket_snapshot() {
        let store = store("bucket");
        store
            .store(&key(), &prices(AR_US, WINDOW_US), 0.01, 0.01)
            .unwrap();
        let mut df = hit_load(&store);
        for _ in 0..PRICE_WINDOW {
            df.prices.record_ar(30_000, AR_US * 2.0);
        }
        assert_eq!(df.retire_stale_snapshot(), None);
        assert!(store.lookup(&key()).is_some());
    }

    /// Gate 25b, recorded and not fixed: an over-priced AR snapshot (b
    /// halved) retires nothing, so no AR token runs, the AR cell stays empty
    /// and only the window cell can show drift. With windows at their true
    /// price the entry is never deleted: it lives until the key changes.
    #[test]
    fn an_over_priced_ar_snapshot_cannot_self_heal() {
        let store = store("doubled");
        store
            .store(&key(), &prices(AR_US * 2.0, WINDOW_US), 0.01, 0.01)
            .unwrap();
        let mut df = hit_load(&store);
        for _ in 0..4 * PRICE_WINDOW {
            df.prices.record_window(16, 300, WINDOW_US);
        }
        assert_eq!(df.prices.kept(0, 300), 0, "no AR token ran");
        assert_eq!(df.retire_stale_snapshot(), None);
        assert!(store.lookup(&key()).is_some(), "the wrong entry survives");
    }

    #[test]
    fn the_binary_digest_is_the_files_md5() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("hipfire-daemon");
        fs::write(&file, b"daemon bytes").unwrap();
        let digest = BinaryDigest::spawn(file);
        assert_eq!(
            digest.get(),
            Some(format!("{:x}", md5::compute(b"daemon bytes")).as_str())
        );
        assert_eq!(digest.get(), digest.get(), "joined once");
        let missing = BinaryDigest::spawn(dir.path().join("missing"));
        assert_eq!(missing.get(), None);
    }
}
