# Amendment 2 — offload pass-back Phase 0 coverage — 2026-10-02

**Lifecycle:** `historical`

**Amends (does not modify):**
[`2026-10-02-offload-passback-coverage-phase0.md`](2026-10-02-offload-passback-coverage-phase0.md)
and its [`amendment-1`](2026-10-02-offload-passback-coverage-phase0-amendment-1.md).
Both are unchanged, per [`README.md`](README.md).

## The oracle in amendment-1 §3 has its bound backwards

Amendment-1 §3 computed the two-engine floor as
`0.02 ms + m·row_bytes/(r_cpu + r_gpu)` using the trace's `gpu≥22.2 GB/s`, and
concluded the step is "already at its balanced floor (within ~5 %)". **That
inference runs the wrong way.** `gpu≥` divides the GPU arm's bytes by the *whole*
step wall, so it is a **lower** bound on the GPU rate. A lower `r_gpu` makes the
denominator `r_cpu + r_gpu` smaller and the floor `m·rb/(r_cpu+r_gpu)` **larger** —
so `0.494 ms` is an **upper** bound on the achievable floor, and the real slack
(achieved − floor) is *larger* than the ~5 % §3 reported, not smaller.

Redo it with the bound's direction stated, on the same `m=12288 k=4096` numbers
(`share=0.428`, `r_cpu=34.2 GB/s`, `m·rb=26.7 MB`, copies = floor 0.02 ms,
achieved step wall 0.52 ms):

| r_gpu used | value | floor | slack | note |
|---|---|---:|---:|---|
| trace lower bound | 22.2 GB/s | 0.494 ms | 5.0 % | **upper** bound on the floor ⇒ **lower** bound on the slack |
| `share`-implied, `share·r_cpu/(1−share)` | 25.6 GB/s | 0.467 ms | 10.2 % | holds only if the controller converged; a bound/estimate, not a measurement |

So the honest reading is **~5–10 % of the step is scheduling slack**, not "~5 %,
at the floor". The `share`-implied rate is only valid *because* the share is
assumed converged (there `share` equals the balance point by construction), so it
is circular for confirming share placement — do not read it as independent
confirmation.

## What survives

The conclusion is unchanged and reinforced: the step's residual is
`d2h + join = 0.11 ms ≈ 21 %` of the 0.52 ms step — pure copy/sync — and at most
a few percent is scheduling slack. The remaining 9B gap is the sync points, not
the share. The 3.7 / 3.7 / 6.3 % coverage headline and the size-gate
classification are untouched.

Also withdrawn from amendment-1 §3: the sentence "So the largest step is already
at its balanced floor" (too strong — bound-dependent) and the "within 0.04 of
the balance point" comparison (it used the same lower-bound rate).

## Not changed

Method, fixture, the coverage fractions, and the note that a second session is
needed for cross-session figures.
