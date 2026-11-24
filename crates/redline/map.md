# redline — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Direct-KMD research — bare `libdrm` dispatch that bypasses the HIP runtime, not the serving transport; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md). GLOSSARY name `Redline` is `research`.

## Purpose

Direct-KMD GPU engine that bypasses `libamdhip64` and talks to `/dev/dri/renderD128` via `libdrm_amdgpu.so` (thin DRM userspace, only the `amdgpu` driver required). Owns `device.rs`/`memory.rs`/`queue.rs`/`dispatch.rs`/`sync.rs` — GPU init, BO alloc/VA mapping, compute queues/PM4 submission, kernel load from `.hsaco`, fences. The `//!` docs in [`src/lib.rs`](src/lib.rs) carry the HIP→Redline ioctl table. Normative contributor procedure is `docs/REDLINE.md` (`branch-implemented` vs `origin/beta`).

## Gotchas

- Not the serving transport — default remains HIP/`hip-bridge`; Redline is gated behind the `redline-dispatch` replay and env-gated trials.
- Requires only the `amdgpu` KMD, not a full ROCm install, but still needs `libdrm_amdgpu.so` available via `dlopen`.
- Parsed `.hsaco` ELF kernel descriptors are the source of truth for launch geometry — `hipModuleLoad`-style caching does not apply.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
