# GDN state-precision proof collector

Developer-only, Python 3 stdlib only. `collect.py` launches the real, hash-verified
binaries behind the gates in `release-0.4.3/gdn-state-precision/PLAN.md` §5 (G0–G5),
one fresh process per arm, and persists argv/env/identity/rc/timestamps and the parsed
observations. It downloads nothing, generates no teacher or reference, starts no second
runtime, pins no route that is not on the command line, and never infers a PASS: every
verdict field names the rule it applied.

```sh
python3 tools/gdn_state_precision/collect.py --stage smoke|quality|restore|memory|perf|sampled|greedy \
    --binary-dir ABS --model ABS --wt2-ref ABS --code-ref ABS \
    --modes q8,fp32[,q8_ef] --runs 3 --max-seq N --out ABS
```

Optional: `--dry-run` (print the exact commands/env, execute nothing, no hold needed),
`--prompt-file`, `--long-prompt-file`, `--graph 0,1,unset`, `--analyze` (CPU-only; recompute
quality/perf statistics from an existing `--out`, newest run or `--run-id`), `--prehash`
(CPU-only; fill `<out>/hash-cache.json` with MD5+SHA256 of every input, so tens of GiB are
not hashed inside a hold), `--run-id`, `--child-timeout-secs` (default 1700; timeouts are
recorded), `--sampled-trials` (default and floor 2000), `--bench-max-seq` (also pass
`--max-seq N` to `hipfire bench`; the plan commands pass only `--ctx N`), `--keep-going`
(default: stop at the first failed child and list the rest as `not_run`), `--oracle-only`
(smoke only; see below), `--only`, `--max-chunks`, `--greedy-tokens` (see "Selective runs and
the greedy kill experiment"). `--runs` defaults to 3 for quality/restore/perf/sampled and 1
for smoke/memory/greedy.

## Refusals (exit 2, nothing launched)

Relative paths; missing model/refs/prompts/binary dir; unknown or repeated modes
(`q8`, `fp32`, `q8_ef` only); `quality`/`perf` without `q8` in `--modes` (the paired
baseline); a missing/ambiguous/unlisted/non-executable/hash-mismatched binary; a
non-empty `<out>/<stage>/<run-id>`; and, for every non-dry-run stage, a missing or invalid
fabric hold: `FABRIC_JOB=pid:starttime:cards` whose wrapper must be an ancestor of the
collector with a matching start time, and (except `--oracle-only`) the wrapper chain must
hold `hostmem-flashnext.lock` — the same check as `session_cache_hw.rs::hold_refusal`.
`q8_ef` in `restore` or `sampled` additionally requires its test executable in the manifest.

## Binary manifest

`<binary-dir>/BINARIES.sha256`, lines `<sha256>  <name>`. Names resolve by base name
(`qwen4_kld`, `hipfire`, `daemon`, `gdn_state_precision_oracle`, `greedy_mtp_identity_hw`,
`session_cache_hw`, `sampled_mtp_distribution_hw`, `q8_ef_product_routes_hw`), with or
without a cargo `-<hex hash>` suffix; exactly one match is required. Hashes are verified
before anything runs. `hipfire bench` finds its daemon through `HIPFIRE_DAEMON_BIN`, which
the collector sets to the manifest's hashed `daemon` (isolated `HIPFIRE_HOME` would
otherwise hide it).

## Child environment

Every `HIPFIRE_*` is unset except `HIPFIRE_DEVICES`, `HIPFIRE_LOCK_DIR`,
`HIPFIRE_FABRIC_ADMISSION`; then set: `HIPFIRE_QWEN4_NPU_TRUNK=off`, `HIPFIRE_PLE_FORMAT=q4`,
`HIPFIRE_NORMALIZE_PROMPT=0`, `HIPFIRE_KERNEL_CACHE=<out>/kcache`, per-child isolated
`HOME`, `HIPFIRE_HOME`, `HIPFIRE_LOCK_DIR`, the arm's mode and graph axis. Mode:
`HIPFIRE_STATE_QUANT=<mode>` for `qwen4_kld` and `hipfire bench`;
`HIPFIRE_GDN_TEST_STATE_QUANT=<mode>` for the hw test executables (never both).
Graph axis (`--graph`): `HIPFIRE_GRAPH` for qwen4_kld/bench; `HIPFIRE_REPLAY_BACKEND`
(`0`→`hip` eager, `1`→`auto` retained replay; recorded as the replay backend, never as
hipGraph) for the identity/session/sampled/EF tests; `--graph` on the oracle's argv;
`unset` sets nothing. Defaults: smoke 0,1; quality 0; restore 0,1; memory/perf/sampled unset; greedy 1.
Arms interleave per run: even run = modes as given, odd run = reversed.

`--child-env KEY=VALUE` (repeatable, `KEY` = `HIPFIRE_[A-Z0-9_]+`) adds a pair to every
child after the fixed knobs, for settings a child needs that the scrub would otherwise
drop, e.g. `--child-env HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=14` for the R9700 expert
placement of a Flash-Next `qwen4_kld` run. It is recorded in each child's `env.json` and
`identity.json` and in the receipt. Keys the collector sets itself are refused. A parent
`HIPFIRE_*` that the scrub would drop (other than the `HIPFIRE_TORCH_*` host tool paths)
refuses the run before any child starts: pass it with `--child-env` or unset it.

## Output layout

`<out>/<stage>/<run-id>/` holds `command.txt`, `identity.json` (MD5+SHA256 of every binary
used, model, `.ple-q4`/`.ple-q8` sidecars, refs, prompts; host; `FABRIC_JOB`/`FABRIC_CARDS`),
`children.json`, `analysis.json` (quality/perf/greedy), `receipt.json` (observed data only) and
`report.md`. Each child lives in `run<i>/g<graph>/<mode>/<label>/` with `command.txt`,
`env.json` (full child env), `identity.json`, `stdout.txt`, `stderr.txt`, `rc.txt`,
`start.txt`, `end.txt`, `observations.json` (plus `timeout.txt`, `memory_samples.jsonl`,
`out.kldseq`, test output dirs). Nothing is deleted. Exit nonzero if any child failed,
timed out, was incomplete (e.g. zero tests ran), or was not run.

## Stage → gate map

| stage | gate | what runs |
|---|---|---|
| `smoke` | G0 (+ identity seasons) | per `--graph`: `gdn_state_precision_oracle --formats <modes> --graph G --max-rows 32 --all-keeps --json F` (exit 0 match / 1 mismatch / 2 setup, recorded verbatim), then per mode `greedy_mtp_identity_hw --ignored --test-threads=1 --exact greedy_mtp_matches_ar_on_flash_next_seasons` (`summary.txt` collected). `q8_ef` skips the equality-asserting identity test (PLAN G2). `--oracle-only`: only the oracle runs; no Flash-Next load, no model hashing, guard needs the wrapper ancestor but not the host lock (P0 ≤600 s). |
| `quality` | G1 | per run/mode: `qwen4_kld eval … --max-chunks 24` on code24 and wt2 × {`--decode`, `--append-rows 256`, all-row prefill (no route flag)}; `.kldseq` kept. Analysis: per corpus/route paired per-chunk delta (candidate − q8), mean, 95% percentile bootstrap CI (10000 resamples, fixed seed recorded), each arm's mean/p99 KLD, NLL, top1, nonfinite counts, the rule `lo<=0 -> PASS; lo>0 -> PASS iff hi<=+0.010` applied and printed, code24-decode gap closure of `q8_ef` vs (`q8`,`fp32`) against the 0.75 target, and byte-identical-`.kldseq` determinism across runs. |
| `restore` | G2 | per mode: one process per `session_cache_hw` test (`--ignored --test-threads=1 --exact <test>` for each name in `SESSION_CACHE_TESTS`; every test loads Flash-Next and two loads in one process exhaust host memory, so any session_cache_hw argv without `--exact` is refused, and a binary whose `--list --ignored` differs from the tuple is refused before GPU work) and `greedy_mtp_identity_hw --ignored --test-threads=1 greedy_mtp_matches_ar_on_flash_next` (p1 + seasons). `q8_ef` runs `q8_ef_product_routes_hw --ignored --nocapture --test-threads=1` instead (refuses if absent). Collects summary.txt, identity JSON, decoded `<arm>.txt` paths. |
| `memory` | G3 | per mode one `hipfire bench M --backend noslots --workload stateless --spec mtp --prompt-file P --max-tokens 32 --runs 1 --warmups 0 --ctx N --json`; daemon-log lines (`qwen4 state:`, GDN format, R, placement, partition_digest, max_seq, card_cap), refusals verbatim, and `MemAvailable` + `amd-smi metric --mem-usage --json` (or rocm-smi) sampled at 1 s (before/after/peak). |
| `perf` | G4 | per run, interleaved per mode: the four PLAN G4 commands (spec off, spec mtp, `--ttft` long prompt, `--matrix --pp 8192 --tg 32`), `--runs 1 --warmups 1 --ctx N --json`. Medians per mode and the loss vs q8 against the 5% policy line (an observation, not a verdict). Matrix numbers are labeled synthetic. |
| `sampled` | G5 | per mode `sampled_mtp_distribution_hw --ignored --nocapture --test-threads=1 --exact <case>` with `HIPFIRE_SAMPLED_MTP_TRIALS=2000`, `HIPFIRE_SAMPLED_MTP_OUT=<child>/sampled`: `sampled_mtp_matches_ar_distribution_on_flash_next` for q8/fp32, `sampled_mtp_q8ef_matches_verify_distribution_on_flash_next` for q8_ef. **q8_ef fails closed**: a missing exe/case, 0 tests run or only `ignored`, nonzero rc, mode `q8_ef` absent from the receipts, or a receipt lacking the coverage counters below yields `g5_result` FAIL/INCOMPLETE and a nonzero exit, never PASS. |
| `greedy` | P0 kill experiment | per run/graph/mode, then `--spec off` and `--spec mtp`: `hipfire run M --temp 0 --spec S --max-tokens N --max-seq MAXSEQ --json PROMPT` (see below). |

q8_ef sampled coverage contract (read from any JSON under the sampled output dir with a top-level
`{"coverage": {…}}`, each counter > 0): `reject_keep0, reject_partial, full_accept,
post_reset_window, restore_transaction, restore_session, restore_prefix, restore_disk,
ring_half_0, ring_half_1`, plus `graph_capture, graph_replay` when `--graph 1`. The EF tests and
this receipt arrive with Track B; until then the q8_ef cases cannot pass.

## Selective runs and the greedy kill experiment

- `--only LABEL[,LABEL...]` (any stage): after planning, keep only the children whose label is
  listed, in planned order. A label that matches no planned child (e.g. `ef-product-routes`
  without `q8_ef`) or an empty selection is refused before launch. Labels: smoke `oracle`,
  `identity-seasons`; quality `<code24|wt2>-<decode|append256|whole>`; restore `session-cache`,
  `identity`, `ef-product-routes`; memory `memory-mtp-ctx`; perf `stateless-spec-off`,
  `stateless-spec-mtp`, `ttft-long`, `matrix-pp8192-tg32`; sampled `sampled-ar`, `sampled-law`;
  greedy `greedy-spec-off`, `greedy-spec-mtp`. The filter is recorded in `receipt.json`
  (`only`), `report.md` and the dry-run output. `--analyze` of a filtered run works: quality
  pairs whatever (graph, corpus, route) cells exist and lists the rest under `missing_cells` /
  `errors`; perf reports the labels that exist.
- `--max-chunks N` (quality only, default 24, must be ≥ 2): the `qwen4_kld eval --max-chunks`
  value for every quality child; recorded in `receipt.json`. Paired analysis is unchanged, so
  chunk counts must match between arms.
- `--stage greedy` (P0 kill experiment): needs `--prompt-file` (UTF-8 text, trailing newlines
  stripped; it is passed as the single trailing argument of `hipfire run`, so it must not start
  with `-` and must be shorter than 131072 bytes, the Linux single-argument limit) and
  `--max-seq N`; `--wt2-ref`/`--code-ref` are not needed. `--greedy-tokens N` (default 64,
  1..4096) is `--max-tokens`; `--graph` defaults to `1`, `--runs` to 1. Per run/graph/mode, two
  children `greedy-spec-off` and `greedy-spec-mtp` (`hipfire run M --temp 0 --spec S
  --max-tokens N --max-seq MAXSEQ --json PROMPT`, `HIPFIRE_STATE_QUANT=<mode>`,
  `HIPFIRE_DAEMON_BIN`, memory sampled). Binaries: `hipfire` + `daemon`. A child is `ok` only
  with rc 0 and a non-empty string `content` in the one-line JSON on stdout. `analysis.json`
  per (run, graph, mode) compares spec-off vs spec-mtp text (equal, first differing character
  offset, lengths, token counts); per mode `mtp_equals_ar`. `fp32_mtp_equals_ar` is the **GATE**
  (Track A F32 tripwire); `q8_mtp_equals_ar` and the q8-vs-fp32 content comparison per spec are
  **DIAGNOSTIC**. `all_loaded` is true iff every child is `ok` (the explicit `--max-seq`
  admission held for every arm), and each child's `max_seq`/`card_cap`/`admission` daemon log
  lines are listed. A missing pair or failed child is an analysis error and makes the verdict
  `null`, never a pass. `--analyze --stage greedy` recomputes it from a run dir.

## Hold sizing (PLAN §6)

The collector is run *inside* the hold: `fabric-run.sh … --hostmem-lock -- python3
tools/gdn_state_precision/collect.py …`. It verifies the hold but declares nothing; the
caller sizes it.

- **P0-Halo, ≤600 s** (G0 + short kill experiment): `--stage smoke --oracle-only` fits;
  anything that loads Flash-Next, hashes the model (run `--prehash` outside the hold first),
  or runs full-corpus KLD does not — use P1.
- **P1-Halo**: quality + restore + sampled + perf. Prior four-arm quality took ~16 min;
  both-corpus decode × 3 runs × modes, identity/restore and sampled (2000 trials/arm, ≥3 fresh
  runs) need P1, not chained P0 renewals. Size `--max-secs` ≥ children × `--child-timeout-secs`
  worst case or lower the timeout so the cap is never silently exceeded. Split quality from
  perf if timing must be stationary; nothing else on the card during timings.
- Memory/host figures come from the manifest/PLE/owners plus runtime headroom (PLAN §6),
  not arbitrary small `--hostmem-gb`; the session-test Halo precedent is `--mem-gb 120`
  and is not a guarantee a wide-ring EF fits.
- Per-discrete-arch P0/P1 follow PLAN §6; a single failing prerequisite stops downstream runs
  (default fail-fast).

## Tests

```sh
python3 tools/gdn_state_precision/collect.py --help
python3 -m unittest tools/gdn_state_precision/test_collect.py -v
```
