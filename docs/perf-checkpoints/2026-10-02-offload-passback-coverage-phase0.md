# Offload pass-back Phase 0: coverage is not the deficit — gfx1201 — 2026-10-02

**Lifecycle:** `historical`

**Disposition:** the measurement [`2026-10-01-offload-passback-perf-gap.md`](../investigations/2026-10-01-offload-passback-perf-gap.md)
§ 7 names ("CPU-step wall time bucketed splittable vs refused"). It answers that
document's § 4a/§ 6 open question — whether the refused steps carry a
disproportionate share of CPU-executed wall — and it is the gate § 5 Phase 0 put
on any further scheduler tuning. It is not a baseline, not an admission, and not
a [`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim.

## The question

The 9B fixture realizes roughly half its own stated overlap ceiling (perf-gap § 2),
and the share is where the probe says it should be (§ 5). The two candidate
remaining terms were (a) **coverage** — the "32 % of steps never split" reading,
which is a *step-count* fraction and was tagged `[INFERENCE]`, "an assumption, not
a measurement" — and (b) **per-step cost** on the steps that do split (sync,
ordering, contention). § 6: "these two branches differ by most of the shortfall."
This record measures (a) by wall time.

## Fixture (measured)

- Host: 1 × Radeon RX 9070 XT `gfx1201`, 16 GB, PCIe 4.0 ×16, HIP 7.2; Ryzen 7
  7800X3D. Host load was **not** controlled; absolute rates are not comparable
  across sessions.
- `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366`
  (the registry `qwen3.5:9b`, every projection qt 13 `Mq4G256`).
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, md5
  `5835c71e471849b4a72e1dc8e39695e7` (59 tokens).
- `target/release/daemon` md5 `b35c25f032f80882d757512d0f8ee98a`;
  `target/release/hipfire` md5 `ee24d2671dc3480ac8949a43df71b156`. The daemon
  carries the tuned latch (commit `ce8f3e5ef`) — irrelevant to this measurement,
  which only reads the trace.
- Mode `memory.offload_exec=passback`, `share auto`, `--spec off`, greedy.

## Method

```
HIPFIRE_GPU_LAYER_BUDGET=<B> HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_CPU_EXEC_TRACE=1 \
hipfire bench ~/.hipfire/models/qwen3.5-9b.mq4 --spec off --runs 1 --warmups 1 \
  --max-tokens 128 --backend noslots --workload stateless \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json
```

`<B>` = layers resident, so `32 − B` spilled: **24 → 8 spilled, 16 → 16, 8 → 24**.
One fresh process per point, one run each.

`HIPFIRE_CPU_EXEC_TRACE=1` prints one line per `(m, k, rotated, residual, awq)`
shape: `split: …` for a shape that reached the co-inference seam, `cpu exec: …`
for one that never did. Each line carries the shape's `calls` and its mean
per-step wall (`d2h`, `gemv`, and `join` for a split / `h2d` for the whole-CPU
step, in ms). Wall per shape ≈ `calls × (d2h + gemv + join_or_h2d)`;
`coverage = refused_wall / (refused_wall + split_wall)`. Every shape whose line
prints is CPU-executed; resident (GPU) layers do not appear.

## Result (measured)

| spilled | decode tok/s | refused shapes | coverage (refused wall / cpu-exec wall) |
|---:|---:|---:|---:|
| 8 / 32 | 33.0 | 1 | **3.73 %** |
| 16 / 32 | 19.5 | 1 | **3.72 %** |
| 24 / 32 | 13.9 | 1 | **6.30 %** |

The refused set is a single shape at every point: **`m=32 k=4096`, the DeltaNet
beta/alpha rows** (called 1024 / 2048 / 4096 times, i.e. 8 per token). It is far
below `MIN_SPLIT_BYTES = 2 MiB` (32 × 2176 B ≈ 68 KiB), so it is refused by the
size gate, exactly as the headroom record predicted. Its share of CPU-executed
wall stays under a tenth even at the deepest spill; at 8 spilled it is ≈ 2.7 % of
the *token* wall (≈ 0.8 ms of ≈ 30 ms/token).

## Reading

- **The `[INFERENCE]` branch in perf-gap § 6 is the true one.** The "32 % of steps
  never split" is a step-count fraction that carries **3.7–6.3 % of wall**; the
  refused steps are nearly free, so the coverage term collapses toward zero, not
  toward "most of the shortfall". Phase 0's alternative — "coverage is a
  non-issue and the correct action is to stop" — is what the numbers say.
- Therefore the 9B's remaining gap is **per-step cost on the splitting steps**
  (the `1.27×`-of-`1.4×` the split record § 3 derives, plus contended-vs-solo
  bounds), i.e. the mechanism (blocking copies, same-stream ordering), not the
  scheduler and not coverage. That is out of scope for the scheduler tuning the
  `ce8f3e5ef` commit did; the only structural levers are the ones the split and
  headroom records already list as unmeasured (the two-stream contingency, a
  larger split eligible set).
- Consequence for the estimator defects recorded in design § 6.2.2 ("Known
  estimator limitations"): they are real but, per perf-gap § 5, the *share
  placement is already where the probe says it should be* on this fixture, so
  fixing them is not where the 9B's gap is either.

## Not measured / out of scope

- **One run per point, uncontrolled host load.** These are wall *fractions*
  within a run, which are far more robust to load than absolute tok/s, but a
  second session would be needed to quote any of it as a cross-session figure.
- **The 27B / 64-layer case.** § 3 of the perf-gap record found it realizes
  ~100 % of its ceiling; nothing here changes or confirms that.
- **Any other host, link, or arch.** The refused set is a property of this model's
  shape list plus `MIN_SPLIT_BYTES`, so the 68 KiB figure transfers; the wall
  fractions are this model's.
- **The split shapes' internal `d2h + gemv + join` proxy** ignores the GPU arm's
  own overrun (already folded into `join`), so split wall is a slight lower bound
  and coverage a slight upper bound. Immaterial at 4–6 %.
- **The serve / slots path, prefill, spec decode, MoE** — untouched.
