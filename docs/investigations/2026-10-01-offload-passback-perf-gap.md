# Offload pass-back: predicted ceiling vs realized gain — gfx1201 — 2026-10-01

**Lifecycle:** `historical` (per [`docs/investigations/README.md`](../README.md))

**Disposition:** a derived gap analysis over two existing perf checkpoints. It
introduces **no new measurement** — every field number below is quoted from a
record, and every derived number is marked `[DERIVED]` with its assumption
stated. It is not a baseline, not an admission, and not a
[`docs/BENCHMARKS.md`](../BENCHMARKS.md) claim. It lives here rather than in
[`docs/perf-checkpoints/`](../perf-checkpoints/) because that directory's stated
authority is "dated, fixture-bound **Measured** evidence only" and this document
measures nothing.

## The question

`memory.offload_exec=passback` was built against a pre-implementation estimate of
**up to +40 %**. The shipped mode measures **+8.5 % to +11.7 %** on the 9B
fixture. Where did the rest go, and is the mechanism or the scheduler at fault?

## Sources (both unchanged)

| record | what it supplies |
|---|---|
| [`2026-10-01-offload-passback-headroom-idle.md`](../perf-checkpoints/2026-10-01-offload-passback-headroom-idle.md) | idle-share per spill amount; the token-win formula; the two-engine contention probe; the stated +25–30 % band |
| [`2026-10-01-offload-passback-split-gfx1201.md`](../perf-checkpoints/2026-10-01-offload-passback-split-gfx1201.md) | the mode A/B/C, the offload-amount sweep, the 27B point, `offload-bench` rates, the share trajectory and per-step `split:` trace |
| `crates/hipfire-dispatch/src/offload_split.rs` | the scheduler as shipped (`offload_split.rs:76-107`, `:166-178`, `:406-417`, `:427-435`, `:450-525`, `:803-909`) |

Fixture for every field number below: RX 9070 XT `gfx1201`, PCIe 4.0 x16, Ryzen 7
7800X3D, 28 GB RAM, HIP 7.2; `qwen3.5-9b.mq4` (registry 9B, all qt 13
`Mq4G256`) unless stated; prompt `gpu_offload_probe.txt`
(md5 `5835c71e471849b4a72e1dc8e39695e7`); `--spec off --backend noslots
--workload stateless`, greedy, fresh daemon per arm. Host load 3.3–5.9 through the
arms, which depresses absolute rates (split record, identity table).

## 1. Where "+40 %" comes from

The headroom record never states +40 %. It states, for this exact fixture:

> The token-level win is `idle_share × (1 − 1/overlap_factor)` minus the extra
> sync/launch cost of the split. For 9B 8/32 (77.7 % idle, 1.4× ceiling) that is
> **≈ +25–30 % at best**; nothing here says it is reachable.
> — headroom record, "Reading"

+40 % is that same formula evaluated at the record's **extreme idle point**
(94.6 % idle, 24 of 32 spilled) with the **best measured format bound** from the
split record's `offload-bench` table (1.75×, `mq5g256v2`):
`0.946 × (1 − 1/1.75) = +40.5 %` `[DERIVED]`.

So +40 % is a **ceiling extrapolation that combines two records and neither
combination was benched**. It is not a prediction the project made and missed,
and the headroom record explicitly disclaims reachability. Stated that way the
question is narrower and answerable: *against the formula's own prediction at the
points that were actually benched, how much did the implementation realize?*

## 2. Ceiling vs realized, at the points that were benched

Ceiling = `idle_share × (1 − 1/O)` for overlap factor `O`, evaluated at the two
bounds the records carry: `O = 1.4` (headroom's stated probe bound) and
`O = 1.75` (best `offload-bench` format). Idle shares are the headroom record's
measured last-window values; realized is the split record's § 1 / § 8.1 medians.

| fixture | spilled | idle share | realized | ceiling @1.4× | ceiling @1.75× | realized / ceiling@1.4× |
|---|---:|---:|---:|---:|---:|---:|
| 9B | 4 / 32 | not measured | +8.4 % | — | — | — |
| 9B | 8 / 32 | 77.3 % | **+8.5 % – +11.7 %** | +22.1 % | +33.1 % | 38–53 % |
| 9B | 12 / 32 | 92.4 % | +13.2 % | +26.4 % | +39.6 % | 50 % |
| 9B | 16 / 32 | 89.8 % | +13.9 % | +25.7 % | +38.5 % | 54 % |
| 27B mq3-xt | 8 / 64 | 66.0 % | **+18.8 %** | +18.9 % | +28.3 % | **~100 %** |

The 8/32 realized range is the record's own honesty note: § 8.1 re-measured the
same point hours later on a differently loaded host and got +8.5 %, agreeing with
§ 1's +11.7 % on the pass-back arm to 0.3 % while the `cpu` arm moved 0.7 tok/s.
Both are the same fixture and both stand.

**The 9B realizes roughly half of its own stated ceiling at every spill amount
measured. The 27B realizes essentially all of it.** That is the finding, and it
rules out one explanation outright: a too-small spill. The 9B's ratio is flat
from 8 to 16 spilled while its idle share climbs from 77 % to 90 %, so the
deficit does not shrink as the ceiling grows. Share misplacement is also not the
9B's problem (§ 5) — though note the 27B's share trajectory was not recorded, so
that exclusion covers the 9B only.

## 3. Why: a candidate variable is per-step weight bytes

The 27B side of this comparison rests on a `[PROXY]` (see below), so read this
section as a hypothesis with a named test, not as a measured cause.

| fixture | projection bytes | source |
|---|---:|---|
| 9B `gate_up`, m=12288 k=4096, row_bytes 2176 — **measured field shape** | 26.7 MB `[DERIVED]` | split record § 4 trace |
| 27B projection — **real per-step shape not recorded** | `[PROXY]` | see below |

**The 27B row is a stand-in, not a measurement.** Split record § 8.2 records no
per-step shape for the 27B, so there is no measured per-step byte count to put
beside the 9B's. The nearest number available is `offload-bench`'s **default
synthetic matrix** — its own header reads "default matrix, `k=5120`, 192 MiB per
format" (§ 2) — whose `mq3g256v2` row is `m=96792 k=5120, 192 MiB`. That row is
a sized buffer for the same quant the 27B uses (every 27B projection is qt 49
`Mq3G256V2`, headroom record fixture identity), so it is a plausible order of
magnitude; it is **not** the 27B's real `m × row_bytes` and must not be quoted as
such.

What the 7× figure therefore rests on is a proxy plus an unmeasured assumption
(that the refused 32 % of steps carries a proportional share of wall time —
see § 6). The *direction* is corroborated independently: split record § 8.2
(L344-346) already asserts the mechanism in its own words — "The mechanism is
per-step bytes: each spilled projection is a larger vector, so the CPU arm's
serialized host time per step is longer and there is more of it to recover." This
document does not introduce that claim; it is reading the ratio table against it
and pointing at the measurement that would test it.

The mechanism is consistent with the per-step trace: the split's cost that does
*not* scale with bytes is fixed, so it amortizes.

Worked on the largest field shape, `m=12288 k=4096` at the settled share 0.360
(split record § 4):

- CPU arm `gemv` = 0.46 ms on its 64 % of rows → 64 % × 26.7 MB = 17.1 MB in
  0.46 ms = 37.2 GB/s, matching the trace's reported 35.1 GB/s `[DERIVED]`.
- The same 26.7 MB whole-weight on the CPU **alone** at the `offload-bench` rate
  for this format (44.3 GB/s) = **0.603 ms** `[DERIVED]`.
- Split step wall = `d2h` 0.03 + `max(CPU arm, GPU arm)` + `join` 0.05 = **0.54 ms**.
- CPU-only step wall = 0.03 + 0.603 + 0.05 = **0.683 ms**.
- ⇒ per-step speedup **1.27×** `[DERIVED]`, against a 1.4–1.58× bound.

Assumption: the host GEMV is linear in row count (each weight row is an
independent 2176-byte read with no reuse), so a 64 % row subset costs 64 % of the
whole. If it is not linear, this over-states the CPU-only wall and the true
per-step speedup is better than 1.27×.

So **even the biggest 9B step realizes 1.27× of a 1.4× bound**, and the join is
reported at the floor (`join=0.05ms`), i.e. the GPU arm finished first — the
overlap is working as designed on that step. The 9B's token-level deficit is
therefore **not** on its large steps. It is in the steps that never split and the
steps too small to amortize.

## 4. The two terms that remain, both measurable

### 4a. Coverage — 32 % of steps never split

The split record's own accounting: **10,752 split steps out of 15,871**
(68 %). Every refusal falls back to the whole-CPU step (`offload_split.rs:803-909`),
never to `pcie`. The refusals with a time cost are:

- weight below `MIN_SPLIT_BYTES = 2 MiB` (`offload_split.rs:82`) — the headroom
  record names the DeltaNet beta/alpha rows at ~64 KB;
- fewer than 2 alignment quanta of rows (`m < 16`, `offload_split.rs:170`);
- any step shape outside `Gemv{Raw,Prerotated}` + the two residual forms.

The headroom record's 77.3 % idle share is **time** inside CPU-executed steps and
includes all of these. The ceiling formula multiplies that whole fraction by an
overlap factor that only applies to the splittable subset. **This is the single
largest unquantified term and it is not currently measured.**

### 4b. Small steps — fixed sync dominates

From the same trace, per split step: `m=12288` costs `d2h` 0.03 / `gemv` 0.46 /
`join` 0.05 ms; `m=1024` costs 0.03 / 0.09 / 0.03 ms, i.e. **40 % of that step is
the two blocking copies**, and its CPU rate (11.8 GB/s) is latency-bound, not
bandwidth-bound. Note the D2H and join are *not* differential cost — `cpu` mode
is a host sync point too, and both modes disable hipGraph capture
(`cpu_exec.rs:367-372`, which prints under either mode). The differential is the
extra GPU launch and the serialization the ordering imposes: the GPU arm cannot
start until the D2H drains, and the join is stream-ordered after it on the
default stream. That is the "two-stream contingency" both records list as not
measured.

### 4c. The bounds are solo-rate bounds

`predicted_speedup = (r_gpu + r_cpu)/max(r_gpu, r_cpu)` assumes solo rates survive
concurrency. The contention probe measured CPU retention at **86–94 %** warm
(3 of 4 runs; the cold run's 112 % is the record's flagged artifact) while the
GPU held 102–103 % of its solo rate. Substituting 0.9 × 44.3 turns the 1.58×
`mq4g256` bound into **1.48×** `[DERIVED]`. Separately, the `offload-bench` GPU
column is a raw `memcpy_htod`, which the headroom record itself labels an **upper
bound** for a kernel that reads host-mapped weights inside a launch. Both push
the ceiling down, i.e. they account for a few points of the gap, not most of it.

## 5. Share placement is **not** the defect on this fixture

The scheduler's converged shares were **0.279–0.360** across the four covered
field shapes, against `offload-bench`'s independent **0.28–0.45** by format and
0.36–0.40 for the same `k=4096` shapes and sizes. At the 8/32 point the shape
froze at 0.360 having dipped to 0.322 once it first observed a waiting join. The
share is where the probe says it should be.

There **is** a latent defect, worth recording because it will bite elsewhere:

`next_share` (`offload_split.rs:427-435`) has no branch for "the GPU arm finished
first, and by a lot". In the CPU-straggler regime the join sits at the shape's
no-wait floor, no `r_gpu` sample is taken (`offload_split.rs:485-498`), and the
fallback branch nudges the share **up** by 0.02 per adjustment. On a host where
the GPU arm is genuinely faster per byte, the share therefore ratchets toward
`SHARE_MAX = 0.50` (`offload_split.rs:90`) with the controller learning nothing,
and a frozen shape (`FROZEN_APPLIED = 64`, `FREEZE_EPS = 0.005`,
`offload_split.rs:106-107`) never moves again. The 9B and 27B points here did not
trigger it — both had joins exceeding the floor and observable `r_gpu` — but a
faster link or a slower CPU would.

Note also that `gpu_samples` counts **waiting joins only** (it increments inside
the wait branch, `offload_split.rs:497`); it is not a step count and cannot be
read as time-to-convergence. `applied` (incremented once per `APPLY_EVERY = 4`
update, `offload_split.rs:519`) is that quantity, and it reached 68 at freeze.

## 6. What this does not explain

`[DERIVED]` sizing of the three terms against the 9B 8/32 point, against
28.2 ms/token of CPU-executed time and 3.8 ms/token recovered:

- per-step sync on the big shape: the 1.27×-vs-1.4× gap on a step that is ~a
  quarter of the token's CPU time is on the order of 1–2 ms/token;
- unsplit coverage `[INFERENCE]` — **this term is an assumption, not a
  measurement**: it assumes the 32 % unsplit steps carry wall time in
  proportion to their step count. Nothing in either record establishes that, and
  it is plausible the refused steps are *small* (the headroom record names the
  ~64 KB DeltaNet beta/alpha rows, far below `MIN_SPLIT_BYTES`), in which case
  they are nearly free and this term collapses toward zero while coverage
  becomes a non-issue. If instead they carry a *disproportionate* share of wall
  time, coverage dominates the deficit. **The two branches differ by most of the
  shortfall, and which one is true is the single most valuable thing § 7 would
  settle.** Under the proportional assumption the unrecoverable idle is
  ~8 ms/token;
- contended-vs-solo bounds: ~1–3 points of percentage.

**No term above is measured.** The coverage term is the widest unknown in the
document: it either dominates the shortfall or is nearly irrelevant, and those
two branches differ by most of the gap. This document does not close the budget;
it names the two measurements that would.

## 7. The measurement that would settle it

Not another mode A/B. One run, same fixture, same committed prompt:

```
HIPFIRE_GPU_LAYER_BUDGET=24 HIPFIRE_OFFLOAD_EXEC=passback HIPFIRE_CPU_EXEC_TRACE=1 \
hipfire bench qwen3.5:9b --spec off --runs 1 --warmups 1 --max-tokens 128 \
  --backend noslots --workload stateless \
  --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json
```

and read, from the trace, **CPU-step wall time bucketed splittable vs refused per
shape** (term 4a) alongside the existing per-shape `d2h`/`gemv`/`join` means
(term 4b). The share trajectory is already recorded and is not the question.
Running the same command under `HIPFIRE_OFFLOAD_EXEC=cpu` gives the
split-vs-whole per-shape wall directly, which turns both terms into arithmetic
instead of bounds.

## Not measured / out of scope

- Nothing here is a new measurement; no bench was run for this document.
- The 27B row is **n=1**: one round, because the second `pcie` arm wedged under
  memory pressure at 0 GB free (split record § 8.2). The ~100 % realization on
  the 27B is a single observation and the strongest claim in this document by
  exactly that much.
- The 27B's **real per-step shape was never recorded**, so § 3's per-step-bytes
  comparison uses `offload-bench`'s synthetic default matrix as a `[PROXY]`.
  Splitting the 27B would replace it with the measured `m × row_bytes` and
  settle whether the 7× step-size difference is real.
- The unsplit-step wall-time share (§ 4a, § 6) is `[INFERENCE]`: neither record
  reports per-shape wall time for the refused steps, and the deficit swings on it.
- Not modelled: prefill (never enters the seam), the serve/slots path
  (`forward_batch_slots` bypasses `execute_steps`), spec decode, MoE/routed
  paths (which never reach the split and have no arms), and non-AVX2 hosts
  (refused entirely by the split).
- Per-step CPU cost is reported as flat at 441–458 µs/step across 8→24 spilled on
  the 9B (headroom record), which is a *step count* flatness, not a per-shape one;
  this analysis needs the per-shape distribution the trace in § 7 would produce.
- The 40 % figure is not reproduced from any single record and should not be
  quoted as a target.