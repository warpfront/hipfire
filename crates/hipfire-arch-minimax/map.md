# hipfire-arch-minimax — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 10); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

MiniMax-M2 MoE (`arch_id` = 10, `model_type` `minimax_m2`): pre-norm transformer with per-layer GQA + per-layer QK-norm (flat RMSNorm before reshape) + partial rotate_half RoPE, then `sigmoid(router)+bias` top-k MoE (Mixtral/DeepSeek-V3 style, SwiGLU experts, no shared expert). Forward mirrors the `hipfire-runtime::llama` shape. Verified against `modeling_minimax_m2.py`. See the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Shared MoE `rotate_half` / SiLU helpers in `hipfire-runtime::llama` are `deltanet`-gated — keep the feature flag even though MiniMax uses full rotate_half, for kernel-family parity.
- QK-norm is per-layer on the flat `n_heads*head_dim` tensor before head reshape — a per-head norm would be silently wrong.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
