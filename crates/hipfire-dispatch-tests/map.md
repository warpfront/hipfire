# hipfire-dispatch-tests — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Observability / test — dispatch coverage tests for the kernel-select table, not part of the shipped graph; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Functional dispatch tests that verify correct kernel selection per model family × arch × quant format. Owns the family × dtype × chip matrix coverage that `hipfire-dispatch`'s own unit tests do not fully span. The `//!` docs in [`src/lib.rs`](src/lib.rs) describe the contract: one entry per family, no `DType` hand-matching in callers.

## Gotchas

- This crate is test-only — adding a runtime path dependency on it would artificially inflate the shipped graph.
- Coverage is family-parallel; a skipped row should be an explicit `#[ignore]` with arch/quant justification, not a silent missing case.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
