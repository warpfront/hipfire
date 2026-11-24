# peacemaker-npu — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** NPU, bottom layer: no workspace dependencies; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Builds AIE2P (XDNA2, Strix / Strix Halo NPU) programs without the vendor toolchain: the core VLIW encoder/scheduler (`isa`), CDO/PDI writer, DPU transaction writer, DMA/route/register model, the GEMM / IU4 / IEF15 / ring / experts kernel designs (`kernels`), and an offline functional AIE2P simulator (`sim`). See the `//!` docs in [`src/lib.rs`](src/lib.rs) and [`src/sim/mod.rs`](src/sim/mod.rs), and [`docs/npu/README.md`](../../docs/npu/README.md). Binaries: `peacemaker-npu-objdump`, `peacemaker-npu-sim`.

## Gotchas

- `hipfire-arch-qwen4` depends on this crate for the Flash-Next NPU trunk prefill route (`src/npu_trunk/`, exact gfx1151); program changes reach production there.
- `isa/gen.rs` and `isa/sched.rs` are generated (`npu-tools tblgen2rs`) from llvm-aie TableGen and carry `Apache-2.0 WITH LLVM-exception`; regenerate, do not hand-edit. See NOTICE.
- `tests/golden_bytes.rs` pins the default V9/V10 PDI/TXN bytes: a diff there is a design change, not noise.
- The sim suites (`array`, `g80`) take minutes in debug; use `--release` for iteration.
- Vendor cross-checks need `AIE2P_VENDOR_CORPUS` (not in repo) and skip without it; `isa_decode` is `#[ignore]` (needs an llvm-aie objdump).

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
