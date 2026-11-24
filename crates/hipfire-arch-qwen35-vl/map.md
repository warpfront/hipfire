# hipfire-arch-qwen35-vl — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 5/6 + vision tower); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Qwen3.5-VL vision-language extension. Wraps the Qwen3.5 text decoder (`hipfire-arch-qwen35`) and owns the vision tower (patchifier + ViT), the safetensors-source VL weight load, and the VL-aware `generate_vl` path. Loaded only when `arch_id` 5/6 claims a VL model; text-side weights reuse the Qwen3.5 bundle. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- VL weights are optional overlay — a text-only Qwen3.5 checkpoint must still load and run without this crate's tower.
- Image preprocessing is resolution-sensitive; prompt-frame image token insertion must match the text decoder's expected positions.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
