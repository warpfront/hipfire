# Amendment 1 to `2026-11-01-lloyd-full-twin-ab-gfx1100.md`: decode D3 float LUT

**Lifecycle:** `historical`. Amends (does not modify) the parent record.

## What changed

Decode Step 2 of the local-only final decode plan: all four Lloyd decode
kernels stage the codebook as `float LUT_SMEM[]` (converted once in the
prologue) instead of `unsigned short` + per-nibble `__half2float`
(`kernels/src/gemv_mq4g256v2.hip`, `fused_gate_up_mq4g256v2.hip`,
`fused_qkv_mq4g256v2.hip`, `fused_qkvza_mq4g256v2.hip`). Binaries
`hipfire 3e9872bd…` / `daemon 20cae9fb…`.

## Result

- Bit-exact: HE/3 greedy 512 tokens byte-identical pre/post (612/612 chars).
- Perf (same de-confounded instrument, `bench --matrix --pp 512
  --ctx 26700,107000 --tg 32`, KV q8 legacy, AR): 38.30 → **38.30** @ 26.7K,
  32.57 → **32.40** @ 107K (stdev ≤0.08; the −0.5% at depth is noise).
- The four LUT modules recompiled under new cache keys (verified in the
  identity-run log), so the zero-delta is a real measurement, not a stale
  cache: the LDS gather + per-nibble cvt is NOT the decode bottleneck.

## Disposition

D3 kept (harmless, exact, marginally less LDS pressure) but contributes 0.
The remaining Lloyd-vs-uniform decode gap (~15-17%, ctx-flat) must come from
memory traffic / launch shape: D2 (dual-row residual + SRD buffer loads +
float4 x-reuse) and D4 (K-specialization + x-hoist) are now the live levers.
Parent record's prefill numbers and identity gates stand unchanged.
