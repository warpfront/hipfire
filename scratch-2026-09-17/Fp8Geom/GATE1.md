# Fp8Geom gate 1 — full oracle on 128x64w8: PASS

Cmd: `gate1.sh` (`tmp_gemm_v2_oracle --device 0 --n 512`, full run, no
`--quick`, no `TIME`). Env: `HIPFIRE_GFX12_MQ4V2_FP8_V2=1`,
`HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x64w8`, `HIPFIRE_GFX12_MQ4V2_FP8_SLABS=2`,
ordinal 2 (ROCR_VISIBLE_DEVICES=2, HOME ab2). Binary md5
359306ec5a6feda0686d5e566fd8a371. RC=0. Raw: `gate1-128x64w8.stdout`,
`gate1-128x64w8/results.json` (`v2_geometry: BM128/BN64/BK64/8w route v2`).

- CPU suite 9/9 pass; GPU cases 28/28 pass (37/37 total; the brief's "36/36"
  is off by one — every executed check passes, nothing skipped).
- Canonical N=512 hashes, all `hash == hash_repeat` (repeat bit-identity):
  - gate_up K=5120: `2adf7620c1d722e1` — EXACT pin match
  - qkv K=5120: `09726b10ba1eb1b2` — EXACT pin match
  - qkvza K=5120: `082a7b24def26f75` — EXACT pin match
  - residual K=17408: `373cb087fb519b9e` — EXACT pin match

The half tile (NB==2, no W prefetch) is bit-identical to the incumbent v2
math on all four families: same fold, same K order, same drain.
