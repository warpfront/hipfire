# Model Cold Store: On-Demand Rehydration for ~/.hipfire/models

**Date:** 2026-08-01
**Status:** Approved design, pre-implementation
**Target branch:** feature branch off `origin/beta` (warpfront/hipfire), PR back to `beta`

## Problem

`~/.hipfire/models` holds 402G of quant variants on k9lin's 1.8T NVMe; only a
handful are hot at any time. Past workaround (models as NAS symlinks) made
*every* load slow (~80s vs ~8s local, no page-cache reuse). We want cold quants
to live on the NAS (`/mnt/nas`, 41T local btrfs on bcache) and return to NVMe
automatically when used.

## Decisions (settled with user)

1. **Hipfire-native**: hydration lives in the model-resolution path used by the
   daemon and CLI bins (`models.json` lookup), not an external tool.
2. **Auto watermark eviction + pins**: keep models-dir usage under a cap;
   evict coldest unpinned models automatically. Pin list protects always-hot ones.
3. **Block with progress**: requesting a cold model transparently copies it
   back (progress to stderr + daemon log), then loads. No separate hydrate command.
4. **Scope**: `~/.hipfire/models` only (registry-managed). `hipfire-models/`
   and `~/.hipfire/hf/` stay manual. Single box; fleet later.
5. **Registry-as-truth**: no stub files on disk. A cold model's local file is
   simply absent; `models.json` carries its state. NAS-side manifest as backup.

## Registry schema (v2 → v3)

Each entry in `models.json` gains:

```json
{
  "tier": "local | cold",
  "cold_path": "/mnt/nas/kaden/hipfire-cold/<filename>",
  "blake3": "<hex>",
  "last_load_at": "<RFC3339>",
  "pinned": false
}
```

- `last_load_at` is stamped on every successful load. Migration initializes it
  to `max(mtime, atime)` of the file — never null — so pre-existing idle models
  order correctly for eviction without being instantly evictable en masse
  before `min_idle_days` is considered.
- `schema_version` bumps to 3; v2 files migrate in place (missing fields get
  defaults: `tier=local`, `pinned=false`).

## Architecture

New `ColdStore` module in hipfire core (exact crate/file chosen during
planning; it must be reachable from both daemon and CLI bin resolution).

**Hydrate flow** (cold model requested):

```
resolve(id) → tier == cold
  → copy cold_path → models/<name>.hydrating   (progress: bytes/total)
  → verify size_bytes + blake3
  → rename to final name → tier = local → registry write (tmp+rename)
  → normal load proceeds
```

**Dehydrate flow** (watermark eviction) — runs at daemon startup, after each
successful load, and via `hipfire dehydrate [--auto | <model>...]`:

```
usage(models dir) > watermark_gb
  → candidates: tier == local, !pinned, not mmapped by a live process,
    idle ≥ min_idle_days — ordered by oldest last_load_at
  → evict until usage ≤ target_gb:
      copy → cold_dir/<filename>.partial → fsync → verify blake3
      → rename → registry tier=cold → unlink local file (last)
```

**NAS layout**: flat `cold_dir/<filename>` plus `cold_dir/manifest.json`
(id, size_bytes, blake3, original path, timestamp per entry; rewritten
atomically on each dehydrate). The cold store must be reconstructible without
`models.json`.

**CLI surface**:
- `hipfire models list` — adds tier column
- `hipfire dehydrate [--auto | <model>...]`
- `hipfire pin <model>` / `hipfire unpin <model>`

## Safety invariants

1. A model's bytes exist in a verified state in ≥1 tier at all times. The
   source copy is unlinked only after destination verification (size + blake3).
2. All transfers write to `.partial`/`.hydrating` names, fsync, atomic rename.
3. All tier transitions serialize through `flock(~/.hipfire/.coldstore.lock)`;
   registry writes are tmp+rename.
4. NAS unavailable: hydrate fails with a clear error naming `cold_path`;
   dehydrate warns and skips. Cold-store failures never block hot-model serving.
5. Startup reconciliation: delete orphaned `.partial`/`.hydrating`; disk wins
   over registry (local file present with matching size → `tier=local`);
   both copies missing → entry marked with a load-time error, never silently dropped.
6. Eviction never touches a model mmapped by a live process, nor anything
   loaded within `min_idle_days`.
7. Disk-full mid-hydrate: remove partial, fail with clear error.

## Config (`config.toml`)

```toml
[cold_store]
enabled = false            # opt-in; flipped after smoke test passes
cold_dir = "/mnt/nas/kaden/hipfire-cold"
watermark_gb = 200
target_gb = 160
min_idle_days = 7
```

## Testing

- Unit: tier state machine over tmpdir fixtures (hydrate, dehydrate, pin,
  watermark ordering, migration v2→v3).
- Crash safety: kill -9 mid-copy in each direction; reconciliation recovers
  with no data loss and no stale partials.
- Race: load request arrives while same model is mid-eviction → eviction
  aborts cleanly, load wins.
- NAS-unreachable behavior for both flows.
- E2E smoke: `qwen3.5-0.8b.mq4` (550MB) round-trip on real hardware —
  dehydrate, cold load (hydrates + serves), verify blake3 stability.

## Rollout

1. **Pre-req fix**: `qwen3.6-27b.mq4` + `qwen3.6-27b-awq.mq4` are 0-byte
   hardlinked stubs (truncated 2026-07-30); repair or deregister before any
   sweep so the migrator can never archive garbage.
2. Land feature via ultracode workflow: fresh worktree from `origin/beta`,
   phased implementation (schema/module → loader integration → eviction/CLI →
   tests), independent adversarial verify agents gate each phase; PR to `beta`.
3. Enable on k9lin; first `hipfire dehydrate --auto` moves ~240G+ to NAS.
4. Fleet adoption deferred (out of scope).

## Non-goals

- No prefetch/warm command (loading is hydration).
- No management of `hipfire-models/` or `hf/` dirs.
- No network cold store / multi-host sharing in v1.
- No dedup/compression of cold files (btrfs zstd on cold_dir handles it).
