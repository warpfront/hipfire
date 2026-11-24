# hipfire-arch-gemma4 — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Arch forward — hangs off `hipfire-loader` via `Carrier` (`arch_id` 13; the arch-22 EAGLE drafter rides it); see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Gemma 4 text on one weight stack, state and layer program: the 12B dense, strictly admitted E2B/E4B, and the 26B-A4B MoE. Hybrid attention (5 sliding window-1024 layers per 1 full layer with head_dim 256/512, K=V sharing on full), sandwich RMSNorm with per-layer `layer_scalar`, attention scale 1.0 (Q pre-scaled so kernel's `1/√d` cancels), `logit_softcap` = 30, `softcap_scale` 60. Verified against HF Gemma4 F32. The `//!` docs in [`src/lib.rs`](src/lib.rs) enumerate the distinguishing features vs plain pre-norm. `bundle.rs` owns the engine session cache (`SessionState`); `mtp.rs` puts the EAGLE drafter on the generic `MtpDrafter`/`SpecTarget` seams; `emit.rs` routes thought-channel output for AR and spec.

## Gotchas

- K=V sharing on full layers — no `v_proj`; V is `k_norm` output renormed by a weight-less RMSNorm. A separate `v_proj` load is a bug.
- Period comes from `layer_types` array, not an assumed 5:1 — a computed period will mis-index the K=V layers.
- QK-norm, sliding RoPE, and full-layer proportional RoPE are family-specific; plain-Qwen2/llama helpers do not apply verbatim.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
