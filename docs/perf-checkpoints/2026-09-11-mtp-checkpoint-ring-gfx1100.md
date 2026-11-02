# Qwen MTP divergent-render checkpoint ring — gfx1100 — 2026-09-11

**Lifecycle:** `historical`

**Disposition:** measured candidate evidence; not a product baseline or admission decision.

## Question and scope

Measure a bounded MTP DeltaNet checkpoint ring that restores the latest resident recurrent-state checkpoint at or before a divergent rendered-history LCP, then replays only the tail.

Current-beta base: `c88a1ba0dc3fb2104ab78e6cb65b3067973e41ac`. This candidate is stacked on the MTP strict-prefix terminal repair.

## Fixture and method

- Radeon RX 7900 XTX, `gfx1100`, 24,560 MiB reported VRAM; HIP/ROCm 7.2; Linux 6.18.37.
- Qwen3.8-27B MQ4, MD5 `d1292b4d5bd6046693604201a6ca8074`; MTP sidecar MD5 `d4733b7c91e454743ea9c453ae7b67b8`.
- Q8 KV, MTP K=3, deterministic edit 6,000 characters before the end of a 19,729-token second rendered turn.
- Three fresh processes per arm. The only arm difference was checkpoint resume off/on. Default interval 2,048 tokens and cap 8.
- Candidate daemon MD5 `658aa68bf2b9dc58514805cb69c9a096`.

## Result

| route | second-turn prefill tokens | raw prefill samples | median | work ratio |
|---|---:|---|---:|---:|
| cold replay | 19,729 | 50.737, 50.738, 50.532 s | 50.737 s | 1.997 |
| checkpoint resume | 2,833 (16,896 cached) | 10.644, 10.617, 10.510 s | 10.617 s | 1.141 |

The route selected checkpoint position 16,896 for LCP 17,065 on all three candidate runs. That is 85.6% fewer second-turn prefill tokens and 79.1% lower median second-turn prefill time. Decoded previews were byte-identical within and across arms.

## Cost and lifecycle validation

- A fully populated eight-snapshot ring on an 18,436-token prompt reduced free VRAM from 5,900 to 4,748 MiB: **1,152 MiB additional VRAM** on this 27B fixture.
- First-prefill medians across the three main pairs were 52.491 s with resume off and 52.280 s on (-0.4%, noise); no ingest slowdown was measured.
- Official `serve_harness.py --mode session`: 8/8 non-empty turns, no attractor, both recall checks 3/3, retrieval gate passed, and MTP cache reuse observed.

Raw local discovery artifacts: `/mnt/data4/claude-scratch/20260831-hipfire-tp2/evidence-20260911/pr12/`. This path is a discovery pointer; the full timing arrays and fixture are preserved above.

## Interpretation

The latency gain is large but is not free: the default ring consumes 1,152 MiB for this model. Operators may reduce `HIPFIRE_CACHE_CKPT_MAX` at the cost of shallower rewind coverage. No gfx1201 performance or memory claim is made.
