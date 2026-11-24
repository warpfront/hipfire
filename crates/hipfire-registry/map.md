# hipfire-registry — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Operator — strict bundled/dynamic model registry and tag → path resolution; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Strict bundled/dynamic model registry: the checked-in `registry/v1.json` is compiled in as the offline floor; dynamic loading preserves `fresh cache → network → stale cache → bundled` order; a malformed registry is rejected wholesale and model code never consumes a partially validated catalog. Owns typed `Registry`/`ModelEntry` deserialization and `resolve_tag` → path helpers. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Cache TTL is `24h` and fetch timeout is `3.5s`; stale-cache fallback is intentional — do not fail closed when bundled exists.
- Schema version `1` is checked before any model entries are trusted; version mismatch is a hard reject, not a best-effort parse.
- Every Flash-Next MQ4 tag publishes the split base `qwen3.8-flash-next-base.mq4-xts` on the shared PLE pair; no bundled entry names the removed monolithic. A verified local monolithic keeps serving those tags through the compiled-in `GRANDFATHERED_MONOLITHICS` pin (`ModelEntry::grandfathered_monolithic()`, `grandfathered_monolithic_in`); the CLI verifies its bytes and links the legacy spelling. The registry performs no filesystem migration.
- Old cached/remote registries that still carry the monolithic pin keep parsing: Flash-Next XTS normalization runs only after validation, for the exact repo/size/SHA pin and legacy/canonical filenames (`ModelEntry::is_flash_next_xts()`).
- Retired tags (`qwen3.8:flash-next-rtn-asym`, `qwen3.8:flash-next-mq6q8-pleq8`) are dropped from `registry/models.json` and `registry/v1.json` and stripped from every loaded view (`RegistryV1::strip_retired` on parse and on cache read, so an older remote payload or cache cannot re-advertise them). The compiled `RETIRED_MODELS` pins (`RetiredModel`: repo, file, sha256, size, card via `RetiredModel::entry()`, one-line `refusal()`) are the only record: a verified local copy still runs, `pull` refuses, nothing downloads. `names_registry_model` admits the pinned files so listings keep them.
- The short user-facing `qwen3.8:flash` alias targets `qwen3.8:flash-next`; the strict v1 wire schema has no hidden/deprecated entry flags.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
