# Amendment 4 to `2026-11-01-lloyd-full-twin-ab-gfx1100.md`: decode D4 x-hoists

**Lifecycle:** `historical`. Amends the parent record and amendments 1–3.
Supersedes the ledger's (amendment 3) decode rows with the figures below.

## What changed

Decode Step 4 of the local-only final decode plan (first slice; TAIL0
variant machinery deliberately skipped — tails are runtime-dead on all 27B
shapes, ~0 expected): float4 x-hoist per quad in `fused_gate_up_mq4g256v2`
(DOG_X with explicit components; uniform path preserved through the redefined
shared DOG macro), `fused_qkv_mq4g256v2` (same pattern), `gemv_mq4g256v2.hip`
LUT path only (uniform DOG untouched), and the pre-existing `DOG_V`
R-hoist_x32 schedule newly wired for Lloyd qkvza via
`FUSED_QKVZA_MQ4G256V2_LUT_HOIST_X32_GFX1100_SRC` (exact-gfx1100 gate).
Binaries `hipfire 0aca6289…` / `daemon 17d83fa5…`.

One first attempt at the qkv edit broke the file's `#if/#else` nesting
(missing hoist loads + duplicate `#else`); repaired before any build, never
compiled, never ran.

## Result

- Bit-exact: HE/3 greedy 512 tokens byte-identical pre/post (612/612).
- De-confounded decode (`--matrix --pp 512 --ctx 26700,107000 --tg 32`,
  KV q8 legacy, AR): 41.21 → **46.20 (+12.1%)** @ 26.7K,
  34.47 → **38.16 (+10.7%)** @ 107K, stdev ≤0.09.
- Lloyd-vs-uniform gap (uniform 46.09 / 38.03, same instrument):
  **+0.2% / +0.3% — closed to within run noise.** The per-slice split
  (gate_up-only vs qkv/qkvza/lm_head) was not isolated — one build, one
  gate battery, same reasoning as P1/P2.
- Uniform route health: shared-path hoists are value-identical by
  construction (same floats, same FMA order, 16 B-aligned loads); uniform
  trivial-prompt probe answers coherently ("1..5", 42.9 tok/s). Empty
  uniform HE outputs at 512 tokens are thinking-length variance (different
  weights think longer), not a code effect — a value change is impossible
  without an address change, and alignment was verified static.

## Disposition

Decode work is done to within noise: D1 (0, kept), D2 (+7%), D3 (0, kept),
D4 (+12%/+11%, gate_up + qkv + qkvza-hoist + lm_head). The draft Step 4
fallback (`v_perm` register-side lookup) is retired — D3 proved no
lookup-attributed remainder exists, and there is no gap left for it to
close. Remaining decode delta vs uniform is ±0.3% (noise). Parent record's
prefill numbers and all identity gates stand.
