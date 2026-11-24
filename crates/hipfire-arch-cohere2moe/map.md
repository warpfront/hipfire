# hipfire-arch-cohere2moe — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 12); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Cohere2-MoE (`CohereLabs/North-Mini-Code-1.0`, `arch_id` = 12): parallel-block transformer — every layer is `h += attn(layernorm(h)); h += ffn(layernorm(h))` sharing a single mean-centered `Cohere2LayerNorm`. Attention is interleaved GQA 32/4 × 128 with RoPE on `sliding_attention` (window 4096) and NoPE on `full_attention`; FFN is 1 dense layer then 128-expert `sigmoid` top-8 MoE (`norm_topk_prob=false`, no bias, no shared expert). See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Parallel block uses one `Cohere2LayerNorm` for both attn and ffn branches — a pre-norm-per-branch rewrite would be a modeling error.
- Full-attention layers are NoPE (detected via `layer_rope_theta == 0`), not RoPE — indexing by `layer_types` string alone misses the `theta` override.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
