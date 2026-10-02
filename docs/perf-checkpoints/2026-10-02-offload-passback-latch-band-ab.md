# Offload pass-back: the latch re-open band is not field-identifiable — gfx1201 — 2026-10-02

**Lifecycle:** `historical`

**Disposition:** the A/B behind commit `ce8f3e5ef`'s `REOPEN_EPS` choice. It is a
**null**: the fixture does not separate the band values, so the shipped value
(`0.01`, twice the freeze band) rests on the hysteresis margin and the
deterministic host-load model in `offload_split`'s tests, not on a field win. It
is not a baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. Recorded so the null is not
re-run as if it were open.

## Context

`ce8f3e5ef` tightened the latch's re-open band from `REOPEN_EPS = 0.02` to `0.01`
and added `REOPEN_CONFIRM = 3` (three consecutive same-direction out-of-band
proposals). An earlier spread on the same fixture (commit `9c722823c`) had
reported `0.005` worst at `−1.8 % / −5.2 %` and `0.01 / 0.02 / 0.05` inside its
noise. Those two findings are confounded — the streak postdates the earlier
spread — so four daemon binaries differing only in the two constants were built
and compared.

## Fixture (measured)

- Host: 1 × Radeon RX 9070 XT `gfx1201`, 16 GB, PCIe 4.0 ×16, HIP 7.2; Ryzen 7
  7800X3D. Host load 3–6, uncontrolled, and moving through the session.
- `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`.
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, md5
  `5835c71e471849b4a72e1dc8e39695e7` (59 tokens).
- `target/release/hipfire` md5 `ee24d2671dc3480ac8949a43df71b156` held constant
  across arms (only the daemon carries the scheduler).
- Mode `memory.offload_exec=passback`, `share auto`, `--spec off`, greedy,
  `--backend noslots --workload stateless`, 128 tokens.

## Method

Arms (daemon md5, `REOPEN_EPS` / `REOPEN_CONFIRM`):

| arm | md5 | band / confirm |
|---|---|---|
| `b005c1` | `c24cbc1e74e326b42582b6ea5f57cafb` | 0.005 / 1 |
| `b005c3` | `7374ca893027f07105c1e3160e77c20f` | 0.005 / 3 |
| `b01c3` | `ca16c00b9ef3948430038a5b4f2aac09` | 0.010 / 3 |
| `b02c3` | `b6a3f167c705bc2aa2b7f5ce676d9898` | 0.020 / 3 |

Three interleaved fresh-process rounds per budget, `HIPFIRE_DAEMON_BIN` selecting
the arm so no rebuild sits between rounds; budgets 24 and 16 (8 and 16 of 32
layers spilled). One 128-token run per (arm, budget, round).

## Result (measured) — a tie

Median decode tok/s, with the 3-round range:

| arm | 8 / 32 spilled | 16 / 32 spilled |
|---|---|---|
| 0.005 / no streak | 33.9 `[33.6–34.3]` | 19.1 `[17.8–19.7]` |
| 0.005 / streak | 33.9 `[33.9–34.1]` | 18.8 `[18.1–20.2]` |
| 0.010 / streak | 34.4 `[33.8–34.6]` | 18.8 `[17.9–20.3]` |
| 0.020 / streak | 34.0 `[33.6–34.3]` | 19.3 `[19.1–19.3]` |

## Reading

- **Every arm sits inside its own run-to-run spread**, and the within-arm spread
  at 16 spilled (±6 %) is wider than any between-arm gap. The ordering also flips
  between budgets (`0.01` leads at 8 spilled, `0.02` at 16), which is what a
  noise-dominated comparison looks like. The fixture does not resolve the band.
- The earlier `0.005` penalty **did not reproduce**: `b005c1` (0.005, no streak)
  is at 33.9 / 19.1, competitive with every other arm. So the previous result is
  read as a non-reproduction, not as "the streak fixed it".
- The fixture cannot, on this host, show the band at all; per `ce8f3e5ef` the
  re-open path *is* exercised on live serving (the trace's `reopens` counter is
  nonzero), but the static 128-token bench does not move the balance point, so
  the band has nothing to act on. The band value therefore rests on the
  freeze-band margin and the `offload_split` host-load model.

## Not measured / caveats

- **Arm order was not rotated** within a round (fixed `b005c1, b005c3, b01c3,
  b02c3`), so a monotonic drift across the ~12-minute session lands on arm
  position. The result is a null regardless, but a future run should rotate order.
- Three rounds; the ±6 % within-arm spread means a `≤2` point difference would not
  be detectable at 16 spilled.
- One host, one model, one prompt, 128 tokens, host load uncontrolled.
- The greedy and sampled serve-route runs recorded alongside `ce8f3e5ef` flag a
  model-side token attractor (reproduced on the `cpu`-mode control); unrelated to
  the scheduler and not part of this A/B.
