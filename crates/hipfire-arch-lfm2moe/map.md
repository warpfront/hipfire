# hipfire-arch-lfm2moe — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 11); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

LFM2.5-8B-A1B hybrid (`arch_id` 11, `model_type` `lfm2_moe`): 18 double-gated LIV short-convolution mixers + 6 GQA+QK-norm attention layers, feeding a DeepSeek-style sigmoid-bias top-4 MoE (layers ≥ `num_dense_layers`) or dense SwiGLU MLP. Owns `config.rs`/`forward.rs`/`program.rs`/`spec_impl.rs` and the `Architecture` bring-up. Single-lane decode is one typed `hipfire_dispatch` step program per layer (`program.rs`, design in [`docs/design/lfm2-declarative-program.md`](../../docs/design/lfm2-declarative-program.md)); MoE layers run the shared sealed MoE executor with a biased sigmoid router. Prefill is eager/decode-shaped (sequential `decode_step` per position; `prefill_embed_step` for VL, per-lane `prefill_lane` in `batch.rs`) — no batched-prefill path exists in this crate, and `HIPFIRE_LFM2_PREFILL_BATCH` has no reader (the gfx1201 batched-prefill design in `docs/design/` is historical only). The continuous-batching decode (`forward_batch.rs`, dense only) is still its own loop. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Mixer choice comes from checkpoint `layer_types`, not a hard-coded table — hardcoding will mis-route LIV vs attention.
- Batched prefill (gfx1201, 350M dense MQ4) is a historical design proposal only — not implemented in this crate; prefill is always eager/decode-shaped. `HIPFIRE_LFM2_PREFILL_BATCH`, `HIPFIRE_LFM2_PREFILL_MAX_BATCH` and `HIPFIRE_LFM2_GFX1201_DECODE_FUSION` have no readers; do not advertise them as product.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
