# hipfire-engine — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Serve engine — above `hipfire-loader` and below `hipfire-generate`/`hipfire-daemon`, arch-free; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Architecture-neutral daemon machinery extracted from `crates/hipfire-daemon/src/main.rs` to break the `daemon → loader → daemon` cycle. Owns `ContinuousBatchScheduler` (`scheduler.rs`), `emit_gen_start`/`emit` lifecycle, prompt-frame helpers, and `redline_*` bench harnesses. Its `//!` docs in [`src/lib.rs`](src/lib.rs) state the layering contract explicitly — it must not depend on `hipfire-pflash` or any `hipfire-arch-*`.

## Gotchas

- Must remain arch-free: depending on any `hipfire-arch-*` recreates the cycle the extraction was created to break (`cargo tree` checks this).
- Weight manifests and device placement are out of scope (PR #527) — they stay in the loader.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
