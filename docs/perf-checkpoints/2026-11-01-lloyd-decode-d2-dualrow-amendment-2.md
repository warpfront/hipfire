# Amendment 2 to `2026-11-01-lloyd-full-twin-ab-gfx1100.md`: decode D2 dual-row residual

**Lifecycle:** `historical`. Amends (does not modify) the parent record and
amendment 1.

## What changed

Decode Step 3 of the local-only final decode plan: the Lloyd residual GEMV
(`w_down` + `wo`, ~47% of dense weights) moved from the single-row
`gemv_mq4g256v2.hip` body to the dual-row `gemv_mq4g256v2_residual.hip`
body with the LUT decode folded in (`kernels/src/gemv_mq4g256v2_residual.hip`:
LUT kernargs + shared `float LUT_SMEM[16]` + `MQ4V2_W` lookup macro),
plus SRD buffer loads (`gfx12_weight_cache_policy.inc`) on both LUT
sources, and the launcher grid halved (`m.div_ceil(2)`).
Binaries `hipfire 64967124…` (first boot, symbol-name fix) →
`163bbce5…`/`64967124…` final.
A wrong first attempt at the source composition (entry-point macro name)
failed closed in the loader (`hipModuleGetFunction`) before any number
printed; no wrong output ever escaped.

## Result

- Bit-exact: HE/3 greedy 512 tokens byte-identical pre/post (612/612 chars).
- De-confounded decode (`--matrix --pp 512 --ctx 26700,107000 --tg 32`,
  KV q8 legacy, AR): 38.30 → **41.21** @ 26.7K (+7.6%), 32.40 → **34.47**
  @ 107K (+6.4%), stdev ≤0.07. Gap vs uniform (46.09/38.03, same rig):
  **−10.6% / −9.4%** (was −16.9% / −14.3%).

## Disposition

D2 is the first lever that moves decode. Remaining gap (~10%, ctx-flat)
points at the fused projections (gate_up/qkv/qkvza): D4 (K-specialization,
float4 x-hoist) is the live lever. Parent + amendment-1 numbers stand.
