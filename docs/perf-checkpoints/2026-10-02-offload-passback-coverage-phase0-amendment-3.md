# Amendment 3 — offload pass-back Phase 0 coverage — 2026-10-02

**Lifecycle:** `historical`

**Amends (does not modify, and closes the chain on):**
[`2026-10-02-offload-passback-coverage-phase0.md`](2026-10-02-offload-passback-coverage-phase0.md),
[`…-amendment-1`](2026-10-02-offload-passback-coverage-phase0-amendment-1.md),
[`…-amendment-2`](2026-10-02-offload-passback-coverage-phase0-amendment-2.md).
All unchanged, per [`README.md`](README.md).

## Corrected sentence

Amendment-2's "What survives" ends "…and at most a few percent is scheduling
slack", which contradicts the table two paragraphs above it (5.0–10.2 %). The
corrected sentence:

> The step's residual is the `d2h + join` sync (0.11 ms ≈ 21 % of the 0.52 ms
> step), with a further **~5–10 %** of scheduling slack on the balance point
> (≥ ~5 % is a lower bound; ~10 % assumes the share converged — see below);
> neither is a share-placement error.

## Consolidated reading (supersedes the arithmetic in amendment-1 § 3 and amendment-2)

For `m = 12288 k = 4096` on this fixture, from the run's own trace numbers
(`share=0.428`, `r_cpu=34.2 GB/s`, `gpu≥22.2 GB/s`, mean step
`d2h 0.03 + gemv 0.41 + join 0.08 = 0.52 ms`):

1. **Coverage** — 3.7 / 3.7 / 6.3 % of the spilled-step wall across 8 / 16 / 24
   spilled, one shape (`m=32 k=4096`, 68 KiB), size-refused on every step. Not
   the deficit.
2. **Floor / slack** — the achievable floor is **≤ ~0.494 ms**: it is built from
   `gpu≥`, a *lower* bound on the GPU rate, so the floor built from it is an
   *upper* bound and the true floor is at most that. Against the achieved 0.52 ms
   the scheduling slack is therefore **≥ ~5 %**. If the share converged, the
   implied rate `share·r_cpu/(1−share) ≈ 25.6 GB/s` puts the floor at ~0.467 ms
   (≈10 % slack) — an *estimate*, not a bound, so the range is "≥5 %, ~10 % if
   converged".
3. **Sync** — `d2h + join` = 0.11 ms ≈ 21 % of the step, pure copy/sync.

Neither residual is a share-placement error, so the remaining 9B gap is per-step
cost, and the next lever is structural (the two-stream / sync-amortisation
contingency), not the band and not the estimator.

This is the final word on this checkpoint; further corrections start a new
record.
