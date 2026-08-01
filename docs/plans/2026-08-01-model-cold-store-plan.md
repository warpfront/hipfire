# Model Cold Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. In this project the plan is executed by ultracode Workflow agents, one task per agent, with adversarial verify gates between phases.

**Goal:** Cold model quants live on the NAS and transparently copy back to NVMe when requested; a watermark evictor keeps `~/.hipfire/models` under a configured size.

**Architecture:** Tier state (`local`/`cold`) lives in the per-machine **ModelCatalog** (`hipfire-config`: `models.toml`, legacy `models.json` fallback) — NOT in `hipfire-registry` (that crate is the shared remote catalog, `deny_unknown_fields`, generated; do not touch it). A new `hipfire-coldstore` crate owns transfers, verification, locking, manifest, and eviction planning. The CLI integrates at the single host-side resolution choke point `find_model_path` plus the three `engine.load` success sites.

**Tech Stack:** Rust (workspace at repo root), serde/serde_json/toml, blake3 (new dep, coldstore crate only), libc flock, clap 4.6 derive (CLI), hand-rolled tmpdir tests (NO tempfile in these crates — follow `test_paths` pattern at `crates/hipfire-cli/src/main.rs:8294`).

## Global Constraints

- Base branch: `origin/beta` of warpfront/hipfire. Worktree: `.claude/worktrees/model-cold-store`, branch `feat/model-cold-store`.
- `crates/hipfire-registry` and `registry/*.json` are UNTOUCHED by this feature.
- `#[serde(deny_unknown_fields)]` exists on registry types only; catalog TOML parsing is a manual allowlist in `load_catalog_toml` (`crates/hipfire-config/src/lib.rs:3176`) — every new key edits that allowlist AND `write_catalog_toml` (`:3294`).
- All on-disk writes are atomic tmp+rename, same-directory (`atomic_write` pattern, `hipfire-config/src/lib.rs:3754`).
- A model's bytes must exist verified in ≥1 tier at all times; source unlinked only after destination re-read verification (size + blake3).
- Copy destinations use `.partial.<pid>` (NAS side) / `.hydrating.<pid>` (local side) suffixes; startup reconcile deletes orphans.
- All tier transitions hold `flock(LOCK_EX)` on `~/.hipfire/.coldstore.lock`.
- Cold-store failures NEVER block hot-model serving; `enabled = false` (default) must make every code path a no-op.
- Timestamps are unix epoch seconds (`u64`) — no chrono/time dep.
- Commit style: conventional commits (`feat(coldstore): …`, `test(coldstore): …`).
- Test suite to keep green: `cargo test -p hipfire-config -p hipfire-coldstore -p hipfire-cli --locked` plus `cargo build --release --workspace --all-targets --locked`.
- Progress output: throttled `eprint!("\r…")` mirroring `report_progress` (`hipfire-cli/src/main.rs:1702`), closed with `eprintln!()`, suppressed by `quiet`.

---

### Task 0: Worktree + docs commit

**Files:**
- Create: worktree `.claude/worktrees/model-cold-store` on branch `feat/model-cold-store` from `origin/beta`
- Add: `docs/plans/2026-08-01-model-cold-store-design.md`, `docs/plans/2026-08-01-model-cold-store-plan.md` (copy from main checkout; they are untracked there)

**Interfaces:**
- Produces: the working directory every later task runs in. All later paths are relative to the worktree root.

- [ ] **Step 1: Create worktree**

```bash
cd /home/kaden/ClaudeCode/autorocm/hipfire
git fetch origin beta
git worktree add .claude/worktrees/model-cold-store -b feat/model-cold-store origin/beta
cp docs/plans/2026-08-01-model-cold-store-design.md docs/plans/2026-08-01-model-cold-store-plan.md .claude/worktrees/model-cold-store/docs/plans/
```

- [ ] **Step 2: Verify base and commit docs**

```bash
cd .claude/worktrees/model-cold-store
git log --oneline -1   # expect e2f7dd1a4 or newer beta tip
git add docs/plans/2026-08-01-model-cold-store-design.md docs/plans/2026-08-01-model-cold-store-plan.md
git commit -m "docs(coldstore): add model cold store design + implementation plan"
```

---

### Task 1: Catalog schema — tier fields on `LocalModelConfig`

**Files:**
- Modify: `crates/hipfire-config/src/lib.rs` — `LocalModelConfig` (~line 2984), `load_catalog_toml` key allowlist (~3222), `write_catalog_toml` (~3294), plus new `ModelTier` enum next to `LocalModelConfig`
- Test: inline `#[cfg(test)]` mod in same file (existing mod near line 4100)

**Interfaces:**
- Produces (later tasks rely on these exact names):

```rust
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelTier {
    #[default]
    Local,
    Cold,
}
impl ModelTier {
    pub fn as_str(&self) -> &'static str; // "local" | "cold"
    pub fn parse(s: &str) -> Option<Self>;
}

pub struct LocalModelConfig {
    pub path: Option<PathBuf>,
    pub registry_tag: Option<String>,
    pub overrides: ConfigLayer,
    pub tier: ModelTier,             // NEW, default Local
    pub cold_path: Option<PathBuf>,  // NEW
    pub blake3: Option<String>,      // NEW, lowercase hex
    pub last_load_unix: Option<u64>, // NEW
    pub pinned: bool,                // NEW, default false
}
```

- [ ] **Step 1: Write failing round-trip test** (in the existing `#[cfg(test)]` mod of `hipfire-config/src/lib.rs`, using its tmpdir helper at ~4100):

```rust
#[test]
fn catalog_toml_round_trips_cold_store_fields() {
    let paths = test_config_paths("coldstore-roundtrip");
    let mut catalog = ModelCatalog::default();
    catalog.models.insert(
        "qwen3.6-35b-a3b.mq4r".into(),
        LocalModelConfig {
            path: Some(paths.models.join("qwen3.6-35b-a3b.mq4r")),
            registry_tag: None,
            overrides: ConfigLayer::default(),
            tier: ModelTier::Cold,
            cold_path: Some(PathBuf::from("/mnt/nas/kaden/hipfire-cold/qwen3.6-35b-a3b.mq4r")),
            blake3: Some("aa".repeat(32)),
            last_load_unix: Some(1_750_000_000),
            pinned: true,
        },
    );
    write_catalog_toml(&paths, &catalog).unwrap();
    let loaded = load_catalog(&paths).unwrap();
    assert_eq!(loaded.catalog, catalog);
    std::fs::remove_dir_all(&paths.root).unwrap();
}
```

Also a defaults test: a `models.toml` entry with only `path` set loads with `tier == ModelTier::Local`, `pinned == false`, all new options `None`.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p hipfire-config catalog_toml_round_trips_cold_store_fields`
Expected: compile error (fields don't exist) — that counts as the failing state.

- [ ] **Step 3: Implement**

1. Add `ModelTier` enum + impl (`as_str`, `parse`) above `LocalModelConfig`.
2. Add the five fields to `LocalModelConfig` (keep `Default` derive working — all new fields have defaults).
3. In `load_catalog_toml` per-model key loop (~3222): extend the allowlist match with `"tier"` (string, via `ModelTier::parse`, error on other values), `"cold_path"` (string → PathBuf), `"blake3"` (string, require 64 ascii-hex chars lowercase-insensitively, store lowercased), `"last_load_unix"` (integer ≥ 0), `"pinned"` (bool).
4. In `write_catalog_toml`: emit each new key only when non-default (tier only when Cold, pinned only when true, options only when Some) to keep files tidy.
5. Legacy JSON catalog reader (`load_legacy_catalog`) is NOT extended — legacy entries load with defaults; first write persists to `models.toml`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p hipfire-config --locked`
Expected: PASS including pre-existing catalog tests.

- [ ] **Step 5: Commit** — `git commit -m "feat(config): cold-store tier fields on model catalog"`

---

### Task 2: Config schema — `[cold_store]` section

**Files:**
- Modify: `crates/hipfire-config/src/lib.rs` — `ConfigCategory` enum (~111), `FIELDS` table (~456)
- Test: existing `schema_has_unique_keys_and_legacy_keys` must stay green; add a defaults test

**Interfaces:**
- Produces canonical keys (later tasks read them from `ResolvedConfig`): `cold_store.enabled` (bool, default false), `cold_store.cold_dir` (string, default ""), `cold_store.watermark_gb` (int, default 200), `cold_store.target_gb` (int, default 160), `cold_store.min_idle_days` (int, default 7). All `ConfigScope::Process`, `registry_allowed = false`, `env_compat = None`, category `ConfigCategory::ColdStore`.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn cold_store_defaults_resolve() {
    let resolved = resolve(std::iter::empty::<NamedLayer>()).unwrap();
    assert_eq!(resolved.get("cold_store.enabled").unwrap().value, ConfigValue::Bool(false));
    assert_eq!(resolved.get("cold_store.watermark_gb").unwrap().value, ConfigValue::Integer(200));
    assert_eq!(resolved.get("cold_store.target_gb").unwrap().value, ConfigValue::Integer(160));
    assert_eq!(resolved.get("cold_store.min_idle_days").unwrap().value, ConfigValue::Integer(7));
}
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p hipfire-config cold_store_defaults_resolve` → FAIL (unknown key).

- [ ] **Step 3: Implement** — add `ColdStore` variant to `ConfigCategory`; add five `field!` entries to `FIELDS` following the `memory.kv_cache` shape (`lib.rs:457-468`): legacy keys `cold_store_enabled`, `cold_store_dir`, `cold_store_watermark_gb`, `cold_store_target_gb`, `cold_store_min_idle_days`; rules `ValueRule::Bool`-equivalent (`AutoBool` NOT wanted — plain bool), `PathOrEmpty`, `Integer{min:1,max:1_000_000}`, `Integer{min:1,max:1_000_000}`, `Integer{min:0,max:36500}`; help strings one line each (e.g. "Move models idle beyond the watermark to cold_dir automatically.").

- [ ] **Step 4: Run** — `cargo test -p hipfire-config --locked` → PASS (uniqueness test validates the new entries).

- [ ] **Step 5: Commit** — `git commit -m "feat(config): [cold_store] schema section"`

---

### Task 3: New crate `hipfire-coldstore` — lock, copy+hash, manifest

**Files:**
- Create: `crates/hipfire-coldstore/Cargo.toml`, `crates/hipfire-coldstore/src/lib.rs`
- Modify: root `Cargo.toml` (workspace members list), `scripts/no-gpu-ci.sh` (add `-p hipfire-coldstore` to the cargo test line)

**Interfaces:**
- Produces:

```rust
pub struct ColdStoreCfg {
    pub cold_dir: PathBuf,
    pub watermark_gb: u64,
    pub target_gb: u64,
    pub min_idle_days: u64,
}
impl ColdStoreCfg {
    /// None when cold_store.enabled is false or cold_dir is empty.
    /// Err when enabled but target_gb >= watermark_gb or cold_dir is relative.
    pub fn from_resolved(resolved: &hipfire_config::ResolvedConfig) -> Result<Option<Self>, ColdStoreError>;
}

pub struct LockGuard(/* private File */);
pub fn lock(root: &Path) -> Result<LockGuard, ColdStoreError>; // flock LOCK_EX on <root>/.coldstore.lock

/// Streams src→dst_tmp (1 MiB buffer), hashing while copying; fsyncs; renames to dst.
/// Returns (bytes_copied, blake3_hex). Progress on stderr unless quiet.
pub fn copy_hash_rename(src: &Path, dst: &Path, tmp_suffix: &str, quiet: bool, label: &str)
    -> Result<(u64, String), ColdStoreError>;

/// Re-reads a file and returns its blake3 hex (verification pass).
pub fn hash_file(path: &Path) -> Result<String, ColdStoreError>;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ManifestEntry { pub file: String, pub size_bytes: u64, pub blake3: String,
                           pub original_path: String, pub dehydrated_unix: u64 }
pub fn manifest_upsert(cold_dir: &Path, id: &str, entry: ManifestEntry) -> Result<(), ColdStoreError>;
pub fn manifest_remove(cold_dir: &Path, id: &str) -> Result<(), ColdStoreError>;

#[derive(Debug, thiserror::Error)] pub enum ColdStoreError { /* Io{ctx,source}, HashMismatch{path,expected,actual}, SizeMismatch{..}, NasUnavailable{path}, Config(String), Busy(String) */ }
pub fn now_unix() -> u64;
```

- [ ] **Step 1: Crate scaffolding**

`crates/hipfire-coldstore/Cargo.toml`:
```toml
[package]
name = "hipfire-coldstore"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
blake3 = "1"
hipfire-config = { path = "../hipfire-config" }
libc = "0.2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
```
Add `"crates/hipfire-coldstore"` to root `Cargo.toml` members (alphabetical position); add `-p hipfire-coldstore` to the `cargo test` line in `scripts/no-gpu-ci.sh`.

- [ ] **Step 2: Write failing tests** (inline mod; hand-rolled tmpdir helper copying the `test_paths` nonce pattern from `hipfire-cli/src/main.rs:8294` — name it `test_root(label) -> PathBuf`):

```rust
#[test]
fn copy_hash_rename_round_trips_and_detects_corruption() {
    let root = test_root("copyhash");
    let src = root.join("src.bin");
    let dst = root.join("dst.bin");
    std::fs::write(&src, vec![7u8; 3 * 1024 * 1024]).unwrap();
    let (n, hex) = copy_hash_rename(&src, &dst, ".partial.test", true, "src.bin").unwrap();
    assert_eq!(n, 3 * 1024 * 1024);
    assert_eq!(hash_file(&dst).unwrap(), hex);
    assert!(!root.join("dst.bin.partial.test").exists()); // tmp cleaned up by rename
    // corruption: flip a byte, hash must differ
    let mut bytes = std::fs::read(&dst).unwrap();
    bytes[0] ^= 0xFF;
    std::fs::write(&dst, bytes).unwrap();
    assert_ne!(hash_file(&dst).unwrap(), hex);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn lock_excludes_second_holder() {
    let root = test_root("lock");
    std::fs::create_dir_all(&root).unwrap();
    let _g = lock(&root).unwrap();
    // second open file description must fail LOCK_EX|LOCK_NB
    assert!(matches!(try_lock(&root), Err(ColdStoreError::Busy(_))));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn manifest_upsert_and_remove_round_trip() {
    let root = test_root("manifest");
    std::fs::create_dir_all(&root).unwrap();
    manifest_upsert(&root, "m1", ManifestEntry { file: "m1.mq4".into(), size_bytes: 5,
        blake3: "ab".repeat(32), original_path: "/x/m1.mq4".into(), dehydrated_unix: 1 }).unwrap();
    let raw = std::fs::read_to_string(root.join("manifest.json")).unwrap();
    assert!(raw.contains("m1.mq4"));
    manifest_remove(&root, "m1").unwrap();
    let raw = std::fs::read_to_string(root.join("manifest.json")).unwrap();
    assert!(!raw.contains("m1.mq4"));
    std::fs::remove_dir_all(&root).unwrap();
}
```

(`try_lock` is a pub fn with `LOCK_EX | LOCK_NB` used by tests and by the post-load background evictor to bail instead of queueing.)

- [ ] **Step 3: Run to verify failure** — `cargo test -p hipfire-coldstore` → compile FAIL.

- [ ] **Step 4: Implement lib.rs** — key bodies:

```rust
pub fn lock(root: &Path) -> Result<LockGuard, ColdStoreError> { flock_impl(root, 0) }
pub fn try_lock(root: &Path) -> Result<LockGuard, ColdStoreError> { flock_impl(root, libc::LOCK_NB) }
fn flock_impl(root: &Path, extra: i32) -> Result<LockGuard, ColdStoreError> {
    let path = root.join(".coldstore.lock");
    let file = std::fs::OpenOptions::new().create(true).write(true).open(&path)
        .map_err(|e| ColdStoreError::Io { ctx: format!("open {}", path.display()), source: e })?;
    let rc = unsafe { libc::flock(std::os::unix::io::AsRawFd::as_raw_fd(&file), libc::LOCK_EX | extra) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Err(ColdStoreError::Busy("another cold-store operation is running".into()));
        }
        return Err(ColdStoreError::Io { ctx: "flock".into(), source: err });
    }
    Ok(LockGuard(file))
}
```

```rust
pub fn copy_hash_rename(src: &Path, dst: &Path, tmp_suffix: &str, quiet: bool, label: &str)
    -> Result<(u64, String), ColdStoreError> {
    let total = std::fs::metadata(src).map_err(io_ctx("stat src"))?.len();
    if let Some(parent) = dst.parent() { std::fs::create_dir_all(parent).map_err(io_ctx("mkdir dst parent"))?; }
    let tmp = dst.with_file_name(format!("{}{}", dst.file_name().unwrap().to_string_lossy(), tmp_suffix));
    let mut reader = std::fs::File::open(src).map_err(io_ctx("open src"))?;
    let mut writer = std::fs::File::create(&tmp).map_err(io_ctx("create tmp"))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut done: u64 = 0;
    let started = std::time::Instant::now();
    let mut last_report = std::time::Instant::now();
    loop {
        let count = std::io::Read::read(&mut reader, &mut buffer).map_err(io_ctx("read src"))?;
        if count == 0 { break; }
        std::io::Write::write_all(&mut writer, &buffer[..count]).map_err(io_ctx("write tmp"))?;
        hasher.update(&buffer[..count]);
        done += count as u64;
        if !quiet && last_report.elapsed() >= std::time::Duration::from_millis(500) {
            report_progress(label, done, total, started.elapsed());
            last_report = std::time::Instant::now();
        }
    }
    writer.sync_all().map_err(io_ctx("fsync tmp"))?;
    drop(writer);
    if done != total { let _ = std::fs::remove_file(&tmp);
        return Err(ColdStoreError::SizeMismatch { path: src.into(), expected: total, actual: done }); }
    std::fs::rename(&tmp, dst).map_err(io_ctx("rename tmp"))?;
    if !quiet { eprintln!(); }
    Ok((done, hasher.finalize().to_hex().to_string()))
}
```

`report_progress` mirrors `hipfire-cli/src/main.rs:1702` (percent, GB/GB, MB/s, ETA — prefix the `label`). `hash_file` = same read loop, hasher only. Manifest = read `manifest.json` if present (`BTreeMap<String, ManifestEntry>` under key `entries` plus `"schema_version": 1`), mutate, write via tmp+rename (`.tmp.<pid>`). `now_unix` = `SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()`. `ColdStoreCfg::from_resolved` reads the five keys, returns `Ok(None)` if `!enabled || cold_dir.is_empty()`, errors if `target_gb >= watermark_gb` or `!cold_dir.is_absolute()`.

- [ ] **Step 5: Run** — `cargo test -p hipfire-coldstore --locked` → PASS; `cargo build --workspace --locked` → PASS.

- [ ] **Step 6: Commit** — `git commit -m "feat(coldstore): new crate with lock, verified copy, manifest"`

---

### Task 4: Tier transitions — `dehydrate_model`, `hydrate_model`, `reconcile`

**Files:**
- Modify: `crates/hipfire-coldstore/src/lib.rs`
- Test: same inline mod

**Interfaces:**
- Consumes: Task 1 catalog fields, Task 3 primitives.
- Produces:

```rust
/// Move a LOCAL model to the cold store. Holds the lock for the duration.
/// Catalog is loaded fresh, mutated, and persisted (write_catalog_toml) inside the lock.
pub fn dehydrate_model(cs: &ColdStoreCfg, paths: &hipfire_config::ConfigPaths, id: &str, quiet: bool)
    -> Result<(), ColdStoreError>;

/// Bring a COLD model back to NVMe. Returns the hot path. Holds the lock.
pub fn hydrate_model(cs: &ColdStoreCfg, paths: &hipfire_config::ConfigPaths, id: &str, quiet: bool)
    -> Result<PathBuf, ColdStoreError>;

/// Startup/pre-op recovery: delete orphaned *.partial.* / *.hydrating.* in models dir and cold_dir;
/// disk wins over catalog (local file present+size>0 → tier=Local; local absent+cold present → tier=Cold;
/// both absent → leave entry, return a warning string).
pub fn reconcile(cs: &ColdStoreCfg, paths: &hipfire_config::ConfigPaths) -> Result<Vec<String>, ColdStoreError>;

/// Stamp last_load_unix on a successful load. Best-effort (errors -> warning, not failure).
pub fn note_loaded(paths: &hipfire_config::ConfigPaths, id_or_path: &str);

/// Set/unset pin. Errors if the id is unknown to the catalog AND not a file in the models dir
/// (unknown files get a catalog entry created with path filled in).
pub fn set_pinned(paths: &hipfire_config::ConfigPaths, id: &str, pinned: bool) -> Result<(), ColdStoreError>;
```

**Transition semantics (implement exactly):**

- **Locking discipline (applies to every transition):** the flock is acquired ONCE at the public entry point (`dehydrate_model`, `hydrate_model`, `dehydrate_auto`, `set_pinned`, `reconcile`). Internal shared implementations take `&LockGuard` (`fn dehydrate_model_locked(guard: &LockGuard, …)` etc.) so `dehydrate_auto` never re-acquires — flock conflicts between open file descriptions of the SAME process, so a nested acquire is a self-deadlock.
- `dehydrate_model`: lock → reconcile-lite (orphan sweep only) → load catalog; entry must be `tier == Local` with an existing local file (resolve via `catalog.model_id(id)`; if the id has no entry but `<models>/<id>` exists and `is_model_file`, create an entry with `path` set). Refuse (`Busy`) if the file appears in any `/proc/*/maps` (helper `fn is_mapped(path) -> bool`, substring match of canonicalized path per line). Then: `copy_hash_rename(local, cold_dir/<file>, ".partial.<pid>", …)` → `hash_file(cold_copy)` re-read verify equals streamed hash (else delete cold copy, error) → catalog entry: `tier=Cold, cold_path=Some, blake3=Some(hex)` → `write_catalog_toml` → `manifest_upsert` → **re-check `is_mapped` one final time** (a load may have mmapped the file during the long copy; if mapped, revert catalog to `tier=Local` and keep the local file — the cold copy stays as a valid pre-warmed copy) → `std::fs::remove_file(local)` LAST → done.
- `hydrate_model`: lock → load catalog; entry must be `tier == Cold`; `cold_path` must exist (else `NasUnavailable` mentioning the path AND whether `/mnt/nas` itself is mounted) → `copy_hash_rename(cold, local_path, ".hydrating.<pid>", …)`; streamed hash must equal recorded `blake3` (else remove local, `HashMismatch`) → catalog `tier=Local`, `last_load_unix=now` → write catalog → return hot path. Cold copy is retained on NAS (a later dehydrate of an unchanged file can skip the copy if sizes+hash match — implement this skip: if `cold_path` exists and `hash_file(cold)` equals catalog blake3, dehydrate just removes the local file).
- `reconcile`: for every catalog entry with tier fields set, compare disk truth as documented in the signature comment; sweep `*.partial.*`/`*.hydrating.*` older than nothing (any orphan — the pid suffix means a crashed run) in both dirs; persist catalog only if changed.

- [ ] **Step 1: Write failing tests** — full round trip against two tmpdirs (fake models dir + fake cold dir; "models" are 2 MiB random files; write a `models.toml` via `write_catalog_toml` first):

```rust
#[test]
fn dehydrate_hydrate_round_trip_preserves_bytes() { /* create model file, entry; dehydrate;
    assert local gone, cold exists, catalog tier=Cold w/ blake3; hydrate; assert bytes equal original,
    tier=Local, last_load_unix > 0, cold copy still present */ }

#[test]
fn hydrate_detects_corrupted_cold_copy() { /* dehydrate, flip byte in cold file, hydrate
    → Err(HashMismatch), local file NOT present, catalog still Cold */ }

#[test]
fn dehydrate_skips_copy_when_cold_copy_already_valid() { /* dehydrate, hydrate, dehydrate again
    → second dehydrate leaves cold file mtime unchanged (no rewrite) and removes local */ }

#[test]
fn reconcile_cleans_orphans_and_trusts_disk() { /* strew a .partial.999 in cold dir and a
    .hydrating.999 in models dir; set catalog tier=Cold while local file actually exists
    → reconcile deletes orphans, flips entry to Local, returns no both-absent warnings */ }

#[test]
fn reconcile_reports_both_absent() { /* tier=Cold, no cold file, no local file
    → warning mentioning the id; entry NOT deleted */ }
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p hipfire-coldstore` → FAIL.

- [ ] **Step 3: Implement** per the transition semantics above. `is_mapped` reads `/proc/<pid>/maps` for all numeric `/proc` entries, ignoring read errors (processes may vanish mid-scan).

- [ ] **Step 4: Run** — `cargo test -p hipfire-coldstore --locked` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(coldstore): dehydrate/hydrate/reconcile tier transitions"`

---

### Task 5: Eviction planner

**Files:**
- Modify: `crates/hipfire-coldstore/src/lib.rs`

**Interfaces:**
- Produces:

```rust
pub struct EvictionPlan { pub candidates: Vec<String>, pub usage_bytes: u64, pub target_bytes: u64 }

/// Pure planning: given catalog + per-model (size, mapped?) probes, pick oldest-first
/// unpinned Local models idle >= min_idle_days until projected usage <= target_gb.
/// `usage_bytes` counts ALL regular files in the models dir (not just catalog entries).
pub fn plan_eviction(
    cs: &ColdStoreCfg,
    catalog: &hipfire_config::ModelCatalog,
    usage_bytes: u64,
    size_of: &dyn Fn(&str) -> Option<u64>,
    is_mapped: &dyn Fn(&str) -> bool,
    now_unix: u64,
) -> EvictionPlan;

/// Orchestrates: reconcile → measure → plan → dehydrate each candidate (continuing past
/// per-model failures, collecting warnings). Returns (evicted ids, warnings).
pub fn dehydrate_auto(cs: &ColdStoreCfg, paths: &hipfire_config::ConfigPaths, quiet: bool)
    -> Result<(Vec<String>, Vec<String>), ColdStoreError>;
```

- [ ] **Step 1: Write failing tests** — pure `plan_eviction` tests, no filesystem:

```rust
#[test]
fn eviction_orders_by_last_load_and_respects_pins_and_idle_floor() {
    // usage 100GB, watermark 80, target 60; four 20GB models:
    //   a: last_load 10d ago            -> candidate (oldest)
    //   b: last_load 20d ago, pinned    -> excluded
    //   c: last_load 2d ago             -> excluded (min_idle_days 7)
    //   d: last_load 15d ago, mapped    -> excluded
    //   e: no entry / never loaded, file mtime 30d ago -> treated as oldest of all
    // expect candidates == [e, a] stopping when projected usage 60GB reached
}

#[test]
fn eviction_noop_under_watermark() { /* usage < watermark → empty plan */ }
```

(Models with no catalog entry: `dehydrate_auto` synthesizes `last_load_unix` from `max(mtime, atime)` of the file — the spec's migration rule — via `size_of`/catalog-building preamble; `plan_eviction` itself only sees the synthesized values.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** `dehydrate_auto` acquires the lock ONCE via `try_lock` (Busy → return immediately with a "skipped: busy" warning — this is the background-spawn no-op path), measures usage with a one-level `read_dir` of `paths.models` summing regular-file sizes (matches `local_model_paths` depth), synthesizes missing entries (`is_model_file` names only), calls `plan_eviction`, then `dehydrate_model_locked(&guard, …)` per candidate catching per-model errors into warnings.

- [ ] **Step 4: Run** — `cargo test -p hipfire-coldstore --locked` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(coldstore): watermark eviction planner + dehydrate_auto"`

---

### Task 6: CLI — `dehydrate`, `pin`, `unpin`, tier column in `list`

**Files:**
- Modify: `crates/hipfire-cli/Cargo.toml` (add `hipfire-coldstore = { path = "../hipfire-coldstore" }`)
- Modify: `crates/hipfire-cli/src/main.rs` — `Commands` enum (~83), args structs (~307), `run()` match (~586), `list_command` (~1396), `list_local_models` (~1463), new `dehydrate_command`/`pin_command`
- Test: inline mod at ~8290

**Interfaces:**
- Consumes: Task 4/5 API.
- Produces CLI surface (used by Task 7's spawn and by the user):
  - `hipfire dehydrate --auto [--quiet]` | `hipfire dehydrate <model>...`
  - `hipfire pin <model>` / `hipfire unpin <model>`
  - `hipfire list` shows `cold` / `pinned` markers; `--json` rows gain `"tier"` and `"pinned"`.

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn list_includes_cold_models_from_catalog() { /* test_paths fixture: one real file in models dir,
    one catalog entry tier=Cold with no local file → list_local_models returns both, cold row
    carries tier=Cold and the catalog size (size_bytes from cold file metadata if reachable, else 0) */ }

#[test]
fn pin_command_round_trips() { /* fixture file; pin_command(paths, "fixture", true);
    reload catalog → pinned==true; unpin → false */ }
```

- [ ] **Step 2: Run to verify failure** — `cargo test -p hipfire-cli list_includes_cold_models_from_catalog` → FAIL.

- [ ] **Step 3: Implement**

Clap additions (copy `PullArgs`/`RmArgs` shapes):
```rust
/// Move idle models to the cold store (NAS) or bring usage under the watermark.
Dehydrate(DehydrateArgs),
/// Protect a model from automatic dehydration.
Pin(PinArgs),
/// Remove dehydration protection.
Unpin(PinArgs),

#[derive(Args, Debug)]
struct DehydrateArgs {
    /// Models to dehydrate now (omit with --auto).
    models: Vec<String>,
    /// Evict coldest models until usage is under target_gb.
    #[arg(long)]
    auto: bool,
    /// Suppress progress output.
    #[arg(short, long)]
    quiet: bool,
}
#[derive(Args, Debug)]
struct PinArgs { model: String }
```

`dehydrate_command`: build `ResolvedConfig` the same way existing commands do (global + env layers via `load_global`/`load_env_layer` + `resolve`), `ColdStoreCfg::from_resolved` — `bail!("cold store is disabled (set [cold_store] enabled=true, cold_dir=...)")` on `None`. `--auto` → `dehydrate_auto`, print evicted + warnings; explicit models → `dehydrate_model` each (resolve input through `catalog.model_id` falling back to filename). `pin_command` → `set_pinned`.

`list_local_models`: after the directory scan, append catalog entries with `tier == Cold` that have no matching local file. Extend `LocalModel` struct with `tier: &'static str`-style field (`String`) and `pinned: bool`; local table line gains a suffix marker (`" [cold]"` / `" [pinned]"`); JSON serializes both fields.

- [ ] **Step 4: Run** — `cargo test -p hipfire-cli --locked` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(cli): dehydrate/pin/unpin commands, tier-aware list"`

---

### Task 7: Load-path integration — hydrate at the choke point, stamp loads, background eviction

**Files:**
- Modify: `crates/hipfire-cli/src/main.rs` — `find_model_path` (~5663), pull-recovery blocks (~1796-1808 and ~4658-4667), the three `engine.load` success sites (`run_command` ~1938, `ServeRuntime::ensure_model` ~4692, `open_bench_engine` ~6386)
- Test: inline mod

**Interfaces:**
- Consumes: `hydrate_model`, `note_loaded`, `try_lock`, `ColdStoreCfg::from_resolved`.
- Produces: transparent cold-model loads for `run`, `serve` (HTTP), `bench`.

**Semantics:**
1. `find_model_path` grows a preamble: resolve the input through the catalog (`catalog.model_id`); if the entry is `tier == Cold`:
   - cold store enabled → `hydrate_model(...)` (progress visible, NOT quiet) and return the returned hot path;
   - disabled → fall through (existing behavior finds nothing) but `eprintln!` a one-line hint: `model '<id>' is dehydrated; enable [cold_store] or run hipfire dehydrate --help`.
   `find_model_path` must NOT stamp `last_load_unix` (it is also called by `rm`/`config`/probes).
2. Both pull-recovery blocks check the catalog for a cold entry BEFORE falling back to `pull_command` (a cold local model must never trigger a network re-download).
3. After each of the three successful `engine.load` calls: `hipfire_coldstore::note_loaded(&paths.config, <resolved id or path string>);` then spawn detached background eviction:
```rust
fn spawn_background_eviction() {
    let exe = match std::env::current_exe() { Ok(exe) => exe, Err(_) => return };
    let _ = std::process::Command::new(exe)
        .args(["dehydrate", "--auto", "--quiet"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}
```
(`dehydrate --auto` itself exits immediately on `try_lock` Busy or disabled config — that is the no-op guarantee. The spawned process re-resolves config; nothing is inherited.)

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn find_model_path_hydrates_cold_model() {
    let paths = test_paths("hydrate-choke");
    // fake cold store: cold dir inside test root, 2MiB fixture, catalog entry tier=Cold
    // with correct blake3; config.toml enabling cold_store written via write_global_toml
    // → find_model_path(&paths, &bundled_registry, "fixture-model")
    //   returns Some(models/fixture-model.mq4) and the file exists with matching bytes;
    //   catalog now says tier=Local.
}

#[test]
fn find_model_path_cold_disabled_returns_none_with_hint() { /* same fixture, enabled=false
    → returns None, models dir still empty */ }
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** per semantics above. Config resolution inside `find_model_path` must be cheap: resolve once per process via `std::sync::OnceLock<Option<ColdStoreCfg>>`.

- [ ] **Step 4: Run** — `cargo test -p hipfire-cli --locked` → PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(cli): transparent hydration at find_model_path + load stamping + background eviction"`

---

### Task 8: `rm` cold-awareness + reconcile on startup

**Files:**
- Modify: `crates/hipfire-cli/src/main.rs` — `rm_command` (~1728); `dehydrate_command` and `serve` startup call `reconcile` first (warnings to stderr)

**Interfaces:**
- Consumes: `reconcile`, `manifest_remove`.
- Produces: `hipfire rm <cold-model>` deletes the NAS copy + catalog entry (with the existing `--yes`/confirmation flow, message stating it is removing a COLD copy); `rm` of a local model with a stale cold copy also removes the cold copy + manifest entry.

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn rm_removes_cold_copy_and_catalog_entry() { /* fixture cold model (as Task 7 test);
    rm_command(&paths, RmArgs { model: "fixture-model".into(), yes: true }) → Ok;
    cold file gone, manifest no longer contains id, catalog entry gone */ }
```

- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — `cargo test -p hipfire-cli --locked` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(cli): cold-aware rm + startup reconcile"`

---

### Task 9: Workspace green + docs

**Files:**
- Modify: `docs/` user docs — add `docs/cold-store.md` (usage: enabling, config keys, pin/dehydrate, what the list markers mean, recovery notes: manifest.json, reconcile behavior)
- Verify: whole workspace builds + tests + clippy clean

- [ ] **Step 1: Full gates**

```bash
cargo build --release --workspace --all-targets --locked
cargo test -p hipfire-config -p hipfire-coldstore -p hipfire-cli -p hipfire-registry -p hipfire-client -p hipfire-tui --locked
cargo clippy -p hipfire-coldstore -p hipfire-config -p hipfire-cli --locked -- -D warnings 2>/dev/null || cargo clippy -p hipfire-coldstore --locked   # clippy is advisory in CI; fix what's ours
bash scripts/ci-rustfmt-changed.sh || cargo fmt -p hipfire-coldstore -p hipfire-config -p hipfire-cli
```
Expected: build PASS, tests PASS, no new clippy warnings in touched crates.

- [ ] **Step 2: Write `docs/cold-store.md`** — ~60 lines: what it does, config block example (the five keys with the k9lin values `cold_dir = "/mnt/nas/kaden/hipfire-cold"`), command examples, safety model (verified-before-unlink, lock, reconcile), troubleshooting (NAS unmounted error, Busy, both-absent warning).

- [ ] **Step 3: Commit** — `git commit -m "docs(coldstore): user guide"`

---

### Task 10: E2E smoke on k9lin (gate before PR)

**Not a cargo test — run on the real box from the worktree.**

- [ ] **Step 1: Pre-req check** — the 0-byte stubs: `stat -c '%s %n' ~/.hipfire/models/qwen3.6-27b.mq4 ~/.hipfire/models/qwen3.6-27b-awq.mq4`. If still 0-byte: remove both files and any catalog/models.json references (they are broken regardless of this feature; the real AWQ lives at `~/hipfire-models/qwen3-27b-3.6.mq4-awq`). Record what was done.
- [ ] **Step 2: Isolated smoke** (never touches the real `~/.hipfire`): `HIPFIRE_HOME=/tmp/hipfire-smoke` with a copied `qwen3.5-0.8b.mq4` (550MB) fixture, `cold_dir=/mnt/nas/kaden/hipfire-cold-smoke`:
```bash
target/release/hipfire dehydrate qwen3.5-0.8b.mq4       # → moves to NAS
target/release/hipfire list                              # → shows [cold]
target/release/hipfire run qwen3.5-0.8b.mq4 -p "2+2=" -n 8   # → hydrates with progress, generates
target/release/hipfire list --json | grep '"tier"'       # → "local" again
blake3sum equivalents match before/after (hash_file path exercised by the run)
rm -rf /tmp/hipfire-smoke /mnt/nas/kaden/hipfire-cold-smoke
```
Expected: cold load produces tokens; second run loads instantly (no hydration line).
- [ ] **Step 3: Push branch + open PR to `beta`** — `git push origin feat/model-cold-store`; PR body links the two docs/plans files, lists the smoke evidence.

---

## Deviations from the 2026-08-01 design doc (approved evolutions)

1. Tier state lives in **ModelCatalog** (`models.toml` / legacy `models.json`), not a "models.json v3" — beta's actual local-state layer. The remote `hipfire-registry` is untouched.
2. Timestamps are `last_load_unix: u64`, not RFC3339 (no time-formatting dep in these crates).
3. Eviction triggers are: CLI post-load background spawn + `dehydrate --auto` + serve startup reconcile — there is no separate long-lived daemon hook (beta's daemon receives absolute paths and does no resolution; covering `find_model_path` covers run/serve/bench).
4. Hydrate verifies the streamed hash against the recorded blake3 (single read of NAS copy); dehydrate additionally re-reads the NAS copy before unlinking the local file (destructive step gets the paranoid check).
