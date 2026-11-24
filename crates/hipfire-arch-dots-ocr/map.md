# hipfire-arch-dots-ocr — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 8); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

dots.ocr layout-analysis VLM (Qwen2-VL family): 42-block `DotsVisionTransformer` + plain Qwen2 text decoder (`arch_id` = 8). The text-side trait impl delegates entirely to `hipfire-arch-qwen2` (text weights stored as `model.*` identical to plain Qwen2); dots.ocr-specific code is the vision tower, image preprocessing, and prompt-frame / EOS overrides. The `//!` docs in [`src/lib.rs`](src/lib.rs) document bring-up rev 4 state.

## Gotchas

- Text path reuses `hipfire-arch-qwen2` loader/forward unchanged — duplicate Qwen2 weight logic here will drift.
- Vision and text weights are split by `model_type` namespace; `dots_ocr::vision_forward` must not assume text-side KV layout.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
