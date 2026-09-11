# gfx1100 Q8 tile128 long-context dispatch — 2026-09-11

**Lifecycle:** `historical`

**Disposition:** measured candidate evidence; not a product baseline or admission decision.

## Question and scope

Measure selecting the existing Q8 flash-attention tile128 implementation for exact `gfx1100` when `max_seq > 8192`. The candidate changes dispatch only; tile32 remains selected through 8192 and other architectures are unchanged.

Current-beta base: `c88a1ba0dc3fb2104ab78e6cb65b3067973e41ac`.

## Fixture and method

- Radeon RX 7900 XTX, `gfx1100`, 24,560 MiB reported VRAM; HIP/ROCm 7.2; Linux 6.18.37.
- Qwen3.8-27B MQ4, MD5 `d1292b4d5bd6046693604201a6ca8074`.
- Prompt `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`, MD5 `b4d0b63cddcac872648ddf3cdd92cac2`; 21,550 rendered tokens.
- AR, Q8 KV, 128 generated tokens, native daemon path. Three fresh processes per arm; the first process retained three measured samples and the next two one each.
- Base daemon MD5 `5e21183e687279b8f878db02ffd6d70a`; candidate daemon MD5 `afd99d1f1f44f21bc120d98ce7479d85`.

## Result

| route | raw decode samples (tok/s) | median | range | delta |
|---|---|---:|---:|---:|
| beta tile32 | 38.1, 38.0, 38.0, 37.9, 37.8 | 38.0 | 37.8–38.1 | — |
| candidate tile128 | 40.3, 40.2, 39.8, 40.2, 40.1 | 40.2 | 39.8–40.3 | +5.8% |

Decoded output was inspected and showed no attractor or coherence failure.

## Route validation

- `test-kernels gfx1100`: 16 passed, 0 failed, 0 skipped.
- Redline at 8,193 Q8 context tokens: stable 754-dispatch/17-kernel capture, sequence hash `ce950b24540439c6`; all 17 AQL contracts loaded; HIP/blob/AQL logits, KV and recurrent state bit-exact; DeltaNet frame exact; report `pass=true`.

Raw local discovery artifacts: `/mnt/data4/claude-scratch/20260831-hipfire-tp2/evidence-20260911/tile/`. This path is a discovery pointer; the fixture and full sample arrays are preserved above.

## Interpretation

The result supports review of this exact-gfx1100, long-context dispatch candidate. It does not transfer to <=8K contexts, other architectures, other KV formats, models, or prompts without a new record.
