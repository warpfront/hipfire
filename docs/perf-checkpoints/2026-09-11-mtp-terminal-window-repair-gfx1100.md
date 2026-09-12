# Qwen MTP strict-prefix terminal repair — gfx1100 — 2026-09-11

**Lifecycle:** `historical`

**Disposition:** measured candidate evidence; not a product baseline or admission decision.

## Question and scope

Measure restoring MTP's pre-window target snapshot and replaying only the committed strict prefix after EOS/tool termination, instead of invalidating the full conversation cache.

Current-beta base: `c88a1ba0dc3fb2104ab78e6cb65b3067973e41ac`.

## Fixture and method

- Radeon RX 7900 XTX, `gfx1100`, 24,560 MiB reported VRAM; HIP/ROCm 7.2; Linux 6.18.37.
- Qwen3.8-27B MQ4, MD5 `d1292b4d5bd6046693604201a6ca8074`; MTP sidecar MD5 `d4733b7c91e454743ea9c453ae7b67b8`.
- Q8 KV, MTP K=3, deterministic two-turn tool-result chain, second rendered prompt 41,944 tokens.
- Three fresh processes per arm. The only arm difference was `HIPFIRE_SPEC_WINDOW_ROLLBACK=0` versus `1`; checkpoint resume was disabled.
- Candidate daemon MD5 `b9d3f3c8b0c6a67ce1fefc0e49ae986a`.

## Result

| route | second-turn prefill tokens | raw prefill samples | median | work ratio |
|---|---:|---|---:|---:|
| conservative reset | 41,944 | 168.191, 172.384, 171.306 s | 171.306 s | 1.999 |
| window repair | 26 (41,918 cached) | 1.447, 0.178, 0.176 s | 0.178 s | 0.999 |

The candidate processed 99.94% fewer second-turn tokens and reduced median second-turn prefill time by 99.90%. All six decoded previews were byte-identical. Medium 10K/20K contexts were neutral because those fixtures already retained their prefix without repair.

The implementation reuses the existing MTP `trunk_snap`: an 18,436-token diagnostic reported 5,900 MiB free VRAM in both arms. First-turn prefill medians were 177.7 s off and 175.5 s on, so no steady-path slowdown was measured.

Raw local discovery artifacts: `/mnt/data4/claude-scratch/20260831-hipfire-tp2/evidence-20260911/pr11/`. This path is a discovery pointer; the full timing arrays and fixture are preserved above.

## Interpretation

This evidence supports the gfx1100 MTP lifecycle candidate. It does not transfer to other architectures or model/state shapes. The exact window-identity guard and opt-out remain part of the candidate's safety case.
