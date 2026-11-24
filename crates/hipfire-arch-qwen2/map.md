# hipfire-arch-qwen2 — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 7); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Plain Qwen2 dense text decoder (GQA, RMSNorm, SwiGLU, 1-D RoPE, `attention_bias=true` on Q/K/V, tied `lm_head`). Implements `Architecture` for `arch_id` 7 with real config parser, weight loader (HFQ4G256/HFQ4G128/F16 + F16→F32 tied-head expansion), 28-layer `Qwen2State` scratch graph + F32 KV, and `forward_step` / `forward_step_greedy` validated end-to-end on gfx1151 against the shipped HF F32 reference. The `//!` docs in [`src/lib.rs`](src/lib.rs) carry the validation notes and `examples/infer_qwen2.rs` driver.

## Gotchas

- Weights use HFQ4G256/HFQ4G128 with host-side F16→F32 for tied `lm_head` — Q8 sweep is the definitive correctness lock (16/16 top-1) vs F32; 4-bit divergences at synonym positions are expected quant error, not bug.
- Per-step scratch graph is fully pre-allocated; `state.next_pos` is bumped inside `forward_step`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
