# hipfire-atlas — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Observability — Kernel Atlas schema + JSONL writer (bench corpus), side-DAG; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Typed measurement schema and JSONL writer for the bench corpus (`AtlasRow` per `prefill` / `decode_ar` / `decode_dflash` phase, emitted via `--emit-atlas`; `scripts/kernel_atlas.py` does ranking, `docs/methodology/kernel-atlas*.md` defines the three-layer split). The `//!` docs in [`src/lib.rs`](src/lib.rs) explain the collection/analysis/advisor split and retain legacy stdout parsers as a migration bridge. GLOSSARY name `Atlas` is `production`.

## Gotchas

- `--emit-atlas <path>` is the canonical corpus path — do not add stdout-scraping for newly instrumented phases.
- Schema is `AtlasRow` — breaking its JSON shape breaks `kernel_atlas.py` analysis; `parse`/`render`/`suggest` are the compatibility boundary.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
