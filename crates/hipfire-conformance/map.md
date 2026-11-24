# hipfire-conformance — map

> **Status:** `research` (V2 slices B0 and B1a; card half: I5i and I2 through the daemon).
> **Layer:** 11, beside `hipfire-daemon`; currently depends on no workspace crate.

## Purpose

The conformance runner of V2-PLAN 6.4. It takes a row `{fixture, recipe_pair, input_schedule, comparison_class, budgets}`, runs two arms, and returns `{executed_receipts, per_row_metrics, first_difference, verdict}`. `classes.toml` is the class table as data: every invariant (I1..I10) with its always-exact part and, where allowed, the one named arithmetic difference a row may bound. No budgets live in it.

Verdicts: `pass`; `fail` (first differing step, component, row and owner); `not_covered` (an intended kernel never appeared in an arm's receipt); `unqualified` (bounded class, no budget). A row whose invariant is not in the table, or a bounded row on an always-exact invariant, is refused before it runs.

## Gotchas

- State is compared as host bytes ([`src/digest.rs`](src/digest.rs)): fixed parts digested whole, row streams digested per row, in the layout order `SessionState::snapshot_parts` publishes. Filling a `HostState` from a card's `StateLayout` is the second half.
- A bounded row that does have a budget still reports `unqualified` here: logit-level bounded comparison needs the card half.
- Receipts come only from `HIPFIRE_MODULE_LOAD_AUDIT` lines; a malformed audit line is an error, never a shorter receipt. The `hip` and `embedded` kinds carry JSON provenance and land in `Receipt::provenance`.
- Card rows ([`src/card.rs`](src/card.rs), `--card ROWS.json [REPORT.json]`) drive the real `daemon` binary over JSON lines with `HIPFIRE_EMIT_TOKEN_IDS=1`; a request's committed token ids are one row stream, so a fail names the request and the first differing token. The runner stamps `attempt_id` and answers every `commit_ready` with `commit` (without it the daemon ends the request `aborted client_cancelled`).
- `c1_cn` (I5i): a CN burst that never has every request live at once is an arm error ("burst serialized"), not a pass; a family without lanes (LLaMA, Gemma4 on beta) cannot run this row. For Qwen4 fn-lanes every request carries `min_tokens` so a lone C1 request enters the lane driver at the same width instead of the AR singleton.
- `restore` (I2): conversation turn 1, then turn 2 whose history holds turn 1's reply. Default reference `cold` (a new daemon prefills the whole history, `cached_tokens == 0`) against candidate `live` (turn 2 on the daemon that served turn 1, `cached_tokens > 0`); `evicted` puts an unrelated request between the turns so turn 2 must come from a session snapshot. A `live`/`evicted` arm with no reuse, or a `cold` arm with reuse, is an arm error, not a pass. `cold` prefills in a different partition from `live`: the engine's canonical tile stream is what makes them exact (V2-PLAN 6.4 I1/I2); the legacy loops are not.
- A request that streams text but no `committed` ids is an arm error: the route does not emit ids (the Qwen3.5 VMM batch route on beta), and empty streams would compare equal.
- Card plants change one arm's wire input (seed, a lane's prompt, a stale turn-1 history); they prove the comparison fails, not that an engine-internal fault would reach the wire.
- Not here yet: gfx1100 rows, archive packaging, bounded (logit) card rows.
- `hipfire-conformance --self-test` plants a swapped KV row, a stale restore and a wrong seed in a host fake.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
