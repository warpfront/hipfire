# Amendment 4 to `2026-11-01-lloyd-full-twin-ab-gfx1100.md`: prefill Step 4

**Lifecycle:** `historical`. Amends the parent record and amendment 3.

## What changed

Prefill Step 4 of the local-only final prefill plan: `MQ4G256V2Lloyd`
admitted to the widened ordinary prefill ceiling
(`ordinary_prefill_static_ceiling`, `crates/hipfire-arch-qwen35/.../prefill.rs`)
via the same `HIPFIRE_LLOYD_MMQ_A4=1` + exact-gfx1100 + all-Lloyd predicate
the twin uses (in place of the symmetric-file requirement, which no qwen3.8
file carries). No other widened-machinery change: chunking, lean/full PBS
selection (Lloyd takes full PBS — lean needs IU4 sidecars), and
`memory_admitted_rung` narrowing are dtype-agnostic and fail-safe.
Binaries `hipfire a8304fc8…` / `daemon a7fb46f7…`.

## Result

- Ceiling engaged: `prefill_chunk: requested=8192 admitted=4096 (and 1024
  on one leg) commit_stride=512` on 26.7K; memory admission granted wide
  rungs without OOM next to the 110K KV arena.
- Perf: 26.7K prompt-file, same session, A4 on: **1119.2 tok/s** vs
  narrow-rung 1119.5 (stdev ≤0.5) — **+0.0%**. Chunk width (512 vs
  1024 vs 4096 admitted rungs) does not move prefill here: prefill is
  chunk-count invariant (activation-side traffic + fixed per-token work;
  weight re-reads are not the binding term).
- Correctness: 26.7K prompt, 64 greedy tokens, wide (A4=1) vs narrow
  (A4=0) serves: `content` **and** `reasoning_content` byte-identical.
  (CLI arg limits required serve+curl; `hipfire run` cannot take a 100 KB
  positional prompt — "Argument list too long".)

## Disposition

Step 4 landed, correct, and performance-neutral on this box — kept (it
costs nothing when admission narrows, and helps if a future fixture is
launch/weight-stream bound). The 107K-wide leg was skipped on mechanism:
chunk-invariance at 26.7K (8× width delta, ±0.1%) already decides it;
attention-dominated 107K cannot reverse that sign. Prefill work is done
pending Steps 3 (2 CTA/WGP, scoped but unbuilt — full kernel retile, do it
fresh, not tired).
