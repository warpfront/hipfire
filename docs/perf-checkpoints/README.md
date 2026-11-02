# Performance checkpoints

**Lifecycle:** `historical` (every file, including the newest)  
**Authority:** Dated, fixture-bound **Measured** evidence only. Never current defaults, automatic baselines, or admission decisions. Newest file ≠ current baseline.  
**Claims:** Measurement under the record’s exact fixture and method. Not shipped performance; not transferable across model, quant, GPU, prompt, route, or method without a new record.

Append-only ledger of screens, A/Bs, redline slices, and arch investigations. Measurement is not admission. Product pages may cite a checkpoint only with date, fixture, lifecycle label, and disposition.

**Active owners:** [`docs/INDEX.md`](../INDEX.md) (lifecycle/navigation) · [`docs/VALIDATION.md`](../VALIDATION.md) (validation routes) · [`docs/BENCHMARKS.md`](../BENCHMARKS.md) (current product benchmark claims, admission-gated). Protocol: [`docs/methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md).

**Preservation:** Immutable and append-only — do not modify or delete an existing checkpoint. Corrections are new, separately dated amendment files that link to the **unchanged** original; amendments never authorize mutating the original. Rejected candidates retain their rejection. Local/`/tmp` paths are discovery pointers until a durable copy exists.

**Amendment chain (read newest first):**
- `2026-10-04-mtp-9prompt-path1-sweep-amendment-1.md` amends `2026-10-04-mtp-9prompt-path1-sweep.md` (retracts the 61–63 K=1 cells, aggregate accept 0.681/0.494/0.364, short-run caveat). The parent's "no cell wins" and per-cell ratios stand qualified by it. Neither file carries an error bar: a 1.5× run-to-run swing on identical inputs (code_python K=1 @512: 41.6 vs 63.0) is uncharacterized — no ratio from that session is quotable until the spread is measured.
