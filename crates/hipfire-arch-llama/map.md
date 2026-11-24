# hipfire-arch-llama — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 0/1 facade); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Facade for dense LLaMA / Mistral / plain Qwen3 (`arch_id` 0 dense LLaMA, 1 Mistral/plain-Qwen3). Implements the `Architecture` bring-up triple and re-exports the canonical forward path `hipfire_runtime::llama` (shared transformer infrastructure — `KvCache`, `WeightTensor`, GEMV helpers, dequantisers, sampler) under [`src/lib.rs`](src/lib.rs) so call sites can use the arch-named import consistently. The rationale is the `//!` docs in [`src/lib.rs`](src/lib.rs) — the body stays in runtime as `hipfire-arch-qwen35` depends on it for its plain-LLaMA drafter branch.

## Gotchas

- Canonical forward and `LlamaConfig`/`LayerWeights` still live in `hipfire_runtime::llama` — this crate re-exports them; editing only this facade leaves the hot loop untouched.
- PR #11 forbids touching `hipfire-arch-qwen35` to move the shared primitives — respect that boundary when adding helpers.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
