# hipfire-arch-muse-glimmer — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 14); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Muse Glimmer (`model_type: muse_glimmer`, `arch_id` 14) — 52-layer dense text tower, the 10k/arch existence proof. Owns 3:1 sliding/NoPE-full attention (GQA 32/2 × 128, window 2048, `rope_theta` 500k on sliding / 0 on full), gated attention, sandwich RMSNorm with split eps (`rms_norm_eps` 1e-5 pre, `post_norm_eps` separate), scale-less QK-norm, SiLU SwiGLU, untied `lm_head` with output multiplier and logit softcap. See the `//!` docs in [`src/glimmer.rs`](src/glimmer.rs) for the three silent-wrong pitfalls.

## Gotchas

- Split norm eps — `rms_norm_eps` vs `post_norm_eps` — collapsing them breaks full-tower fidelity.
- NoPE is `layer_rope_theta == 0`, keyed off the theta array, not just `layer_types == "full_attention"`.
- Untied `lm_head` with output multiplier and `logit_scale` softcap — omitting either clips the sampled distribution.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
