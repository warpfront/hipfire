# hipfire-quantize — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Encode — CPU-only `safetensors` → `.mq*`/`.hfq*`, no GPU deps and CI-safe; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Shared model-format helpers (`float16`, `hessian_io`, `safetensors_file`) plus offline weight packers that convert safetensors → MagnumQuant (MQ4/MQ8/MQ3/MQ6/MQ2 + Lloyd variants, FWHT-rotated) `.hfq*` artifacts and collect per-group Hessians for the RHT path. The batch variants are `hipfire-quantize-batched` (8× throughput) and `hipfire-quantize-moe` (MoE-aware). No `hip-bridge`/`rdna-compute` dependency so CI can run without a GPU.

## Gotchas

- No GPU deps by design — adding `hip-bridge` or `rdna-compute` would break the CI-safe invariant.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
