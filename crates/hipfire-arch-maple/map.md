# hipfire-arch-maple — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 15); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Maple-Preview 20B-A1B (`deepgrove/maple-preview`, `model_type: maple`, `arch_id` = 15): a **natively ternary** MoE. Every linear weight in the published checkpoint is already `{-s_r, 0, +s_r}` with one bf16 scale per output row, so `MQ2G256LloydU` (qt=51) carries it losslessly — this crate packs, it does not quantize. 24 layers, hidden 2048, untied lm_head, GQA 16/4 × 128 with QK-norm on both; `layer_types` is a strict [L,L,L,G] 3:1 pattern of 18 `sliding_attention` (window 512, RoPE) and 6 `full_attention` (**NoPE**); MoE on every layer with 256 experts, top-8, `moe_intermediate_size` 512. See the `//!` docs in [`src/lib.rs`](src/lib.rs) and [`docs/design/2026-08-22-maple-preview-20b-a1b.md`](../../docs/design/2026-08-22-maple-preview-20b-a1b.md).

## Gotchas

- **QK-norm runs BEFORE RoPE.** Swapping the order still generates text, just worse text — there is no error to catch it.
- **`MQ2G256LloydU` is the unrotated sibling of MQ2-Lloyd.** Byte-identical 72 B/group layout, so every MQ2-Lloyd kernel binds — but it must *not* FWHT-rotate the activation. `run_moe_decode` rotated the gate→down intermediate for this dtype; fixed, and the test requires rotated and unrotated dtypes to disagree on both rotation flags so it cannot pass vacuously.
- **Clamped SwiGLU is asymmetric:** `silu(clamp(gate, max=7)) * clamp(up, -7, 7)` — the gate clamps only from above.
- **`intermediate_size` (4096) is a dead key** in this config; experts are 512 wide.
- **`partial_rotary_factor` 0.5** — only the first 64 of 128 head dims rotate.
- Layers are the typed `Step` program of [`src/program.rs`](src/program.rs) (#666 G17); `forward.rs` only stages scratch, positions and embeddings. The MoE is sealed (`MoeRecipe::SoftmaxRoutedNoShared` + an `MoeMq2Lloyd` declaration carrying the 7.0 clamp); the batched dense GEMM is `ProjectLadder::GroupedLloydF16`, the grouped MoE WMMA kernel driven as a single-expert case (`E=1`, zero tile ids, identity slot index, `x_row_div=1`). No new GPU kernels.
- Decode always takes the expanded down + k8 combine; the self-combining residual down is no longer selectable. `HIPFIRE_MAPLE_DOWN` and the `HIPFIRE_MAPLE_DUMP_ROUTER` probe are gone.
- Numerics regression gate is **mean KL 0.084**, not perplexity. Maple's absolute PPL (~54) is dominated by the model — the bf16 reference scores the same — so PPL would hide a real kernel regression inside its own noise.
- Research probes under `examples/` are gated behind `--features lab`; build them with `cargo build -p hipfire-arch-maple --examples --features lab`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
