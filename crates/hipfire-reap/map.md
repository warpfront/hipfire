# hipfire-reap — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Utility — selective expert pruning and re-quant overlay for MoE (SP2+), side-DAG; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md). GLOSSARY name `Reap` is `research`.

## Purpose

Model-agnostic REAP: selective expert pruning + (SP2) selective re-quant overlay for MoE models. Owns `gather` (first-axis byte row-gather of kept rows for keep maps), `plan` (`ExpertPlan`), `hook` (`ReapArchHook`), `load`/`source` overlay applied at load. Spec is `docs/superpowers/specs/2026-06-11-generic-moe-reap-design.md`. The `//!` docs in [`src/lib.rs`](src/lib.rs) are the crate entry point.

## Gotchas

- Overlay is model-agnostic only at the `ExpertPlan`/`ReapArchHook` boundary — per-arch hook wiring is still required (see `src/hook.rs`).
- SP2 re-quant assumes a prior SP1 plan; chaining out of order will mis-attribute expert indices.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
