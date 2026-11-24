# redline-rocr — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Retained replay / Device FFI — public ROCr/HSA ABI + PM4/AQL builders that `redline-dispatch::aql` replays through; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Low-level public ROCr/HSA runtime and AQL packet mechanics. Owns the dynamically loaded ROCr ABI (`abi.rs` — transcribed from MIT ROCm 7.2 public headers `/opt/rocm/include/hsa/{hsa.h,hsa_ext_amd.h}`, commit `c0430a5…`), resource lifetimes, architected 64-byte AQL packet encoding (`packet.rs` — `KernelMetadata`, `LaunchGeometry`), and release-ordered queue publication + PM4 builders (`pm4.rs`/`pm4_gfx10.rs`). The `//!` docs in [`src/lib.rs`](src/lib.rs) stress no dispatch-DAG policy lives here.

## Gotchas

- ABI values are from public headers — no ROCm source is vendored; re-transcribe from the same headers on upgrade, don't copy `hsaco` internals.
- 64-byte AQL packets are architected — mis-aligned or partially written packets corrupt the queue's release semantics.
- Only 64-bit little-endian hosts are supported (`compile_error!` guard in `lib.rs`).

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
