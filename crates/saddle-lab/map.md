# saddle-lab — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Direct-KMD research — archived harnesses and legacy benchmarks, not in the default build graph; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Archived research harnesses and one-off benchmarks relocated from `hipfire-runtime` so they remain buildable (`cargo build -p saddle-lab --examples`) without bloating `cargo build`. No library API is exported; every module is effectively an `examples/` harness. The `//!` docs in [`src/lib.rs`](src/lib.rs) state the scope explicitly — do not add new runtime-owned code here.

## Gotchas

- Excluded from `cargo build` by design — `cargo build -p saddle-lab --examples` is the correct check; adding a `[[bin]]` here pollutes the exempt list.
- Legacy harness helpers may shadow current runtime APIs — read the harness `main.rs` before reusing a utility function.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
