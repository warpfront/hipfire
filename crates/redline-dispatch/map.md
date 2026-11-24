# redline-dispatch — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Retained replay — hazard-checked HIP/public-AQL record/replay runtime, product-integrated path; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) (contrast with `redline` research transport).

## Purpose

Backend-neutral record/replay core for Redline Objective 2. Owns `Recorder`/`Resource`/`Region`/`KernelLaunch`/`Access` DAG recording, memory-order validation, deterministic logical lanes, `CompileOptions`/`ReplayMode` plan selection, and explicit `DispatchBackend` (`hip` runtime + `aql` direct-ROCr prototype). The `//!` docs in [`src/lib.rs`](src/lib.rs) include the runnable `MockBackend` example.

## Gotchas

- DAG edges are explicit `depends_on` — missing an `Access::write` → `read` edge is a silent ordering hazard, not a blocked queue.
- `ReplayMode::TokenLatency` vs `ReplayMode` variants change lane assignment; benchmarks must fix the mode or results are incomparable.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
