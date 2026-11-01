# Amendment 3 to `2026-11-01-lloyd-full-twin-ab-gfx1100.md`: step→gain ledger

**Lifecycle:** `historical`. Consolidates the parent record and amendments
1–2 into one implementation-step → measured-speedup table. No new
measurements; all values cited from those records (same rig, same
instrument families; cross-session anchors flagged).

## Prefill (mq4l, KV q8 legacy, AR)

| Step | What landed | pp2048 | 26.7K | 107K | Evidence |
|---|---|---|---|---|---|
| X128 baseline (A4=0) | — | 1051 | 913.5 | 664.2 | 3 fresh legs each, stdev ≤0.8; reproduces draft within 0.3% |
| Shipped T3 (branch) | residual + gate_up staged-i8 twin | 1218.2 | — | — | branch checkpoint, md5-pinned binaries `226ae11d…/4ceda09f…`, same rig/instrument (cross-session anchor) |
| P1 (this session) | qkv/qkvza/partial/single-weight on existing SET/ADD; no new TU | part of +10% below (not isolated — shares the A4=1 flag with P2, no separate kill-switch by design) | — | — | binds show `full_set`/`full_add` on qkv/qkvza arms; HE/3 byte-identical |
| P2 (this session) | all four fused i8 producers; A4 route satisfies the symmetric-file gate (no qwen3.8 file carries the marker) | 1336–1338 (with P1) | 1119.5 | 762.9 | binds show 4 fused producers + gone X128/standalone-quantize; HE/3 byte-identical |
| **P1+P2 combined** | | **+9.8% over T3** (1218→1337) | **+22.5% over X128** | **+14.9% over X128** | same-session for X128 legs; cross-session for the 1218 anchor |

Amdahl read: +27.2% → +22.5% → +14.9% with depth is attention dilution
(shared KV/attention code), consistent with the twin covering only GEMM time.

## Decode (mq4l, de-confounded synthetic fill, TG32, KV q8 legacy, AR)

| Step | What landed | 26.7K fill | 107K fill | Gap vs uniform (46.09 / 38.03) |
|---|---|---|---|---|
| D1 | `is_mq4_rot` dispatch parity | 0 (bind-identical, tok/s unmoved) | 0 | — (kept: bit-exact, FA already fused via GQA path) |
| D3 | float `LUT_SMEM` in 4 GEMV kernels | 38.30 → 38.30 (+0.0%) | 32.57 → 32.40 (−0.5%, noise) | LDS gather ruled out |
| D2 | dual-row Lloyd residual + SRD loads + halved grid | 38.30 → **41.21 (+7.6%)** | 32.40 → **34.47 (+6.4%)** | **−10.6% / −9.4%** (was −16.9% / −14.3%) |

Short-ctx decode (~41 TG32 / 41.4–41.6 serve) unmoved by all steps, as
expected (A4 is prefill-only; D2/D3 help most where GEMM dominates).

## Caveats (load-bearing)

- The P1-vs-P2 split was never isolated: one flag (`HIPFIRE_LLOYD_MMQ_A4=1`)
  arms both by design (minimal surface). The +9.8% is P1+P2 jointly.
- The 1218 T3 anchor is cross-session (different binaries, same rig +
  instrument + md5 pins). Same-session deltas (X128 legs, D2/D3 legs) have
  stdev ≤0.8 against deltas of 100+ tok/s (prefill) and 2–3 tok/s on
  stdev ≤0.08 (decode).
- Uniform mq4 decode (46.09/38.03) is unaffected by construction (Lloyd-only
  kernels/routes); it was measured once per depth, not interleaved per leg.
- D4 (fused-projection TAIL0 + x-hoist) is scoped but unbuilt; the remaining
  ~10% ctx-flat gap is its target.
