# hipfire-ds4-parent — map

> **Status:** `research` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Encode / tooling — DeepSeek V4 Flash parent-checkpoint calibration oracle (not the serving path); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

DeepSeek V4 Flash parent-checkpoint tooling (`arch_id` 9): Hessian collection, reference GEMM oracle, and calibration for the compressed-KV split. The `//!` docs in [`src/lib.rs`](src/lib.rs) header is explicit — `NOT A CALIBRATION REFERENCE (2026-08-02)` — and the end-to-end fixture is `tokens.bin` (1024 ids, sha `48b0f834…`). Used offline for quantization/caches, not during `hipfire run` serve.

## Gotchas

- Header banner says not a calibration reference — do not ship its numerical constants as production quantization tables without re-derivation.
- Only the 1024-id `tokens.bin` fixture is locked; other token streams are not spec and will drift `BlockHessianAcc` shapes.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
