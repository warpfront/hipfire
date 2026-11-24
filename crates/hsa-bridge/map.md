# hsa-bridge — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Device FFI — sibling to `hip-bridge` on the HSA/ROCr path; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Thin Rust wrapper around `libhsa-runtime64.so` that submits AQL packets directly to a user-mode HSA queue, bypassing HIP's ~10 µs launch overhead. Scope is intentionally narrow — Phase 2 is just enough to load a single `.hsaco` kernel and compare per-dispatch latency against HIP; the full loader and multi-queue support is deferred. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Owns `hsa_init`/`hsa_shut_down` lifecycle via `HsaRuntime::load` — double-init or shutdown while queues live is undefined.
- AQL dispatch is 64-byte architected packets; manual ROCr ABI in [`src/ffi.rs`](src/ffi.rs) only supports 64-bit little-endian hosts (`compile_error!` otherwise).
- Not a drop-in for HIP — kernel launch path (`HsaQueueRaw`, `HsaKernelDispatchPacket`) bypasses `HipRuntime` stream semantics.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
