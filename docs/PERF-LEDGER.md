# Perf ledger

The perf ledger is the history of every prefill and decode number the project
has published or measured on priority hardware. It lives in the `ledger` array
of [`registry/receipts-v1.json`](../registry/receipts-v1.json), next to the
`receipts` table that `hipfire fit` reads. Ledger rows are never read by
`hipfire fit`, and they are never deleted: a superseded or contradicted number
stays, with its status and caveat.

The seed is the 2026-11-15 PerfLedger audit (built against beta `a3a096297d`;
the file's `ledger_source` field says so). It holds 63 rows: 33 `raw`, 30
`prose`, 0 `contributor`.

## Row fields

| field | meaning |
|---|---|
| `id` | Stable id, for example `27B-R-01` (model, card class, number). |
| `section`, `card`, `arch` | Ledger section, card and topology as stated (for example `2× R9700 EP2`), and the GPU arch when the row names exactly one. |
| `model`, `model_sha` | Model file key and its sha256 prefix when known (`null` otherwise). |
| `route`, `metric`, `value`, `before_after` | What ran, what was measured, the number(s) as published, and the comparison arm. |
| `date`, `commit` | Measurement date as precise as the source, and the release or commit. |
| `receipt` | Path(s) or URL(s) of the file the number was read from. |
| `method` | Process and run counts, statistic. |
| `prov` | `raw`: read in a raw log/json written by the run, or a run-record table written by its reducer next to per-run raw files. `prose`: only a release note, CHANGELOG or report states it. `contributor`: measured by an external contributor on their card. |
| `prov_detail` | The audit's provenance cell, verbatim, with its caveats (⚠ marks a mismatch with raw). |
| `guard` | `true` only when the row may back a lane guard or a public claim. |
| `beta_status` | The route's status on beta when the row was recorded (`DEFAULT`, `DEFAULT†`, `OPT-IN`, `SUPERSEDED`, `HIST`, `REFUSED`). |

## Rules

- A `prose` or `contributor` row is never a guard: it cannot back a lane guard,
  a speed floor or a public claim. Quote it only with its caveat.
- A `guard` row is always `prov = raw`. A raw row whose provenance cell is
  partly prose (for example `RAW (pp); PROSE (decode)`) is not a guard.
- Every `raw` row records its receipt path or URL, a `YYYY-MM` date and a
  method.

The unit tests in `crates/hipfire-registry/src/receipts.rs`
(`ledger_rows_are_well_formed`, `raw_ledger_rows_record_a_receipt`) check the
schema and these rules; they run in CI with the workspace `--lib` tests
(`cargo test -p hipfire-registry --lib receipts`).

## Admitting a perf claim

1. Measure with the protocol in
   [`methodology/perf-benchmarking.md`](methodology/perf-benchmarking.md):
   fresh processes, the stated statistic, the exact commit and model file.
2. Keep the raw output (per-run stderr/json or the reducer's run record with
   its per-run files) where it can be cited.
3. Add one ledger row with `prov = raw`, the receipt path or URL, the date,
   commit, card, model sha, route, metric, value and method. Do not edit or
   remove older rows; mark a replaced row's `beta_status` `SUPERSEDED` in the
   same change.
4. Set `guard = true` only when the receipt is raw and complete. Only guard rows
   may be cited as lane guards or public claims; the claim language rules in
   [`BENCHMARKS.md`](BENCHMARKS.md#claim-language-fail-closed) still apply, and
   product admission stays in [`admissions.yml`](admissions.yml).
5. A number that should also drive `hipfire fit` goes in `receipts`, under that
   table's own rules (current registry tag, verbatim quote from a repo source).
