# saddle-core — map

> **Status:** `research` (vocabulary owned by
> [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md), which has a `saddle-core` row).
> **Layer:** substrate — sits below `hipfire-runtime` and every
> `hipfire-arch-*` crate; see the layering table in
> [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Target-agnostic model composition for the saddle substrate: grammar matching,
KV cache, architecture capability contract, sampling policy, and the shared
spec-decode constant. The authoritative design rationale is the `//!` docs in
[`src/lib.rs`](src/lib.rs); the layering contract is the comment block in
[`Cargo.toml`](Cargo.toml).

## Gotchas

- May depend only on `rdna-compute`, `hip-bridge`, `serde`, and `std`. Never
  `hipfire-runtime`, never `hipfire-dispatch`, never any `hipfire-arch-*` —
  a cycle here defeats the layer.
- Abstract the model, never the kernel. If a unification requires naming an
  arch-specific type or kernel, it does not belong here (see the B2 findings
  in [`src/spec.rs`](src/spec.rs)).
- `caps` is declared by the carrier, never inferred from an `arch_id` inside
  this crate.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
