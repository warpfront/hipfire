# Amendment 1 — offload pass-back Phase 0 coverage — 2026-10-02

**Lifecycle:** `historical`

**Amends (does not modify):**
[`2026-10-02-offload-passback-coverage-phase0.md`](2026-10-02-offload-passback-coverage-phase0.md).
That file is unchanged. This is a new, separately dated record per
[`README.md`](README.md) ("Corrections are new, separately dated amendment files
that link to the **unchanged** original").

## 1. The call counts are power-of-two snapshots — strike "8 per token"

The original reads "called 1024 / 2048 / 4096 times … (i.e. 8 per token)". That
parenthetical does not follow from the counts, and the counts themselves are not
exact. `trace_step` prints a shape's line only when its running call count is a
**power of two** (`crates/hipfire-dispatch/src/cpu_exec.rs`, `stats.calls.is_power_of_two()`),
so `calls` in the trace is the largest power of two at or below the real count —
1024 / 2048 / 4096 are 2¹⁰/2¹¹/2¹², not the true totals, and their 1:2:4 ratio is
quantization, not a layer-count relationship. The exact tokens executed per point
were not recorded for these runs (`--runs 1 --warmups 1 --max-tokens 128`, EOS
early-stop), so **any per-token derivation is withdrawn**: strike "i.e. 8 per
token" and the "≈ 2.7 % of the *token* wall (≈ 0.8 ms of ≈ 30 ms/token)" figure
in the Reading section.

The headline coverage fractions are unaffected in substance: coverage is a ratio
of two walls, each `calls × mean`, both taken at the same per-shape power-of-two
count, so the quantization is common-mode and the 3.7 % / 3.7 % / 6.3 % reading
stands (a slight, shape-dependent bias remains and is immaterial here).

## 2. The refused shape is size-refused, not capacity-refused

The original already names the shape; making the classification explicit because
"non-split" conflates *never eligible* with *declined for capacity*. Here there is
no ambiguity. The single non-split shape is `m=32 k=4096` (`Mq4G256`, plain
`Gemv`). At the model's 2176 B/row that is `32 × 2176 = 69 632 B = 68 KiB`, far
under `MIN_SPLIT_BYTES = 2 MiB` (`crates/hipfire-dispatch/src/offload_split.rs`),
so `run_with` refuses it at the size gate on **every** step. It is not a
splittable shape the seam declined for capacity reasons; there were none of those
in these runs (every other CPU-executed shape split). The two-deficit conflation
the amendment guards against does not arise on this fixture.

## 3. Loop-closer: the largest step is already at its balanced floor

[DERIVED] The same run's own trace numbers for `m=12288 k=4096`: `share=0.428`,
`cpu=34.2 GB/s`, `gpu≥22.2 GB/s` (a lower bound), mean split step
`d2h=0.03 + gemv=0.41 + join=0.08 = 0.52 ms`, no-wait floor `0.02 ms`.

- Two-engine balance floor from those rates:
  `floor + m·row_bytes/(r_cpu + r_gpu) = 0.02 + 26.7 MB / 56.4 GB/s ≈ 0.49 ms`.
  The measured step wall (0.52 ms) is within ~5 % of it.
- Balance point `r_gpu/(r_gpu + r_cpu) ≈ 22.2/56.4 ≈ 0.39` (a lower bound, since
  `gpu≥` is). The settled share (0.428) is within 0.04 of it.

So the largest step is already at its balanced floor, and the share is where the
run's own rates put it. The residual is the sync points: `d2h + join ≈ 0.11 ms`
of the 0.52 ms step (~20 %), plus the fact that this compares the split against
`d2h+max(arm)+copy` rather than a same-run whole-CPU arm. This is consistent with
perf-gap § 5 ("share placement is not the defect") and confirms the ORIGINAL's
conclusion: the remaining 9B gap is per-step cost, not coverage and not the
share. It is labelled [DERIVED] because it uses the split's own estimator-side
rates as the oracle, not an independent measurement.

## Not changed

The method, fixture, and the 3.7 % / 3.7 % / 6.3 % headline; the note that a
second session is needed for cross-session figures.
