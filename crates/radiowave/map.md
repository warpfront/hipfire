# radiowave — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Kernel select / policy — vendored Redline helper beside `rdna-compute`; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Policy boundary between HIP source and LLVM/AMDGPU (vendored from upstream `redline` @ `f4a0994`, plus the scoped additive `union` module, see [`VENDOR.md`](VENDOR.md)). Injects reviewed source-level lowering rules, invokes `hipcc`, inspects the emitted code object, and records reproducible build evidence. Owns `Atomics`/`Oracle`/`Partition`/`Campaign`/`Recipes` modules; the `//!` docs in [`src/lib.rs`](src/lib.rs) are authoritative. [`src/union.rs`](src/union.rs) holds the CPU-only serde identity records for the HIP/Vulkan executable union (provenance only; separate from `CompileManifest`/`PeacemakerRecord`/`OracleReceipt`).

## Gotchas

- Vendored crate — do not edit lowering rules without recording build evidence; `arch.rs`/`CodeObjectIdentity` tracks reproducibility.
- LLM/hipcc invocation is toolchain-sensitive; `toolchain.rs` / `campaign.rs` encapsulate the exact `hipcc` flags and ISA versioning.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
