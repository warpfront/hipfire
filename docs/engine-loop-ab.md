# Engine-loop A/B parity harness

`scripts/engine_loop_ab.py` is the parity gate for the G12 engine loop
(HIPF-468). It runs **one daemon binary as two processes**, one with
`HIPFIRE_ENGINE_LOOP=legacy` and one with `HIPFIRE_ENGINE_LOOP=engine`,
drives the same scenario matrix through both and diffs what they produce.

The harness also works on a binary that does not know the flag yet: both arms
then run the legacy loop and must come out all-equal. That run proves the
harness itself reports no false diffs. Repeat it after any harness change.

## Run

Every run loads a model, so hold the halo GPU lock:

```sh
flock ~/.halo-gpu.lock python3 scripts/engine_loop_ab.py \
    --daemon target/release/daemon \
    --route qwen4-ar.json \
    --out ~/mesh-validation/<KEY> [--perf] [--conc | --conc-only] [--batch | --batch-only]
```

Exit code `0` means every gate passed (see Gates). `1` means a gate failed, and
`2` means a harness or daemon failure (load error, daemon exit, event timeout).
`--self-test` checks the diff rules and gates without a GPU. `--conc` also runs
the route's concurrency matrix (see Concurrency), `--conc-only` runs only that;
both need the `hipfire` CLI (`--hipfire`, default: next to `--daemon`).
`--batch` also runs the batch-path cache scenario (see Batch-path cache
scenario), `--batch-only` runs only that; `--conc-only --batch` runs both
without the A/B scenarios. Outputs in `--out`:

- `<route>.json` (`<route>.conc.json`, `.batch.json` or `.conc-batch.json`
  without the A/B scenarios): the receipt. It holds the verdict, the
  per-scenario status and notes, the sampled, loop, isolation, G-spec, G-perf,
  concurrency and batch results, the daemon (and CLI) md5, the fixture sha256,
  the harness sha256 (plus git head and a dirty flag when run from a checkout),
  the host state and every arm's raw results.
- `<route>.<arm>.jsonl` / `.stderr`: each daemon's raw stdout and stderr
  (`<route>.batch.<arm>.*` for the batch scenario).
- `<route>.conc.<loop>-<backend>.{json,log,wire.jsonl}`: each concurrency arm's
  bench JSON, bench stderr (with the daemon's stderr) and the tapped daemon
  stdout.
- `EVIDENCE.md`: one row per run, with the harness sha256, daemon md5, fixture
  sha256, exact command, receipt path, result and disposition.

Both arms run with `HIPFIRE_EMIT_TOKEN_IDS=1` (except the perf arm), so every
request also records its committed token ids and positions; tokens are then
compared on ids as well as text. A family that emits no ids is compared on text
only.

## Route file

```json
{
  "name": "qwen4-mtp",
  "model": "~/.hipfire/models/<fixture>",
  "load": {"max_seq": 8192, "mtp_mode": "on"},
  "env": {"HIPFIRE_PLE_FORMAT": "q4"},
  "max_tokens": 64,
  "perf_decode_tokens": 256,
  "spec_noise": true,
  "expect_engine": true
}
```

`load` is passed verbatim as the daemon `load` params. Spec variants are their
own route files: Qwen4 native MTP via `"mtp_mode": "on"`, a DFlash drafter via
`"draft": "<path>"`. `env` is applied to both arms. Never put
`HIPFIRE_ENGINE_LOOP` in it, because the harness sets that per arm.

- `spec_noise` (default false): compare the speculation counters loosely (see
  below). Set it only for a route whose legacy counters are nondeterministic
  (Qwen4 native MTP, HIPF-474).
- `expect_engine` (default false): every engine-arm done envelope must carry
  `"loop": "engine"`. Set it on every route on the engine allowlist, so that
  an engine arm that silently fell back to legacy fails.
- `conc`: the concurrency matrix, see Concurrency.
- `batch_load` (default `{"continuous_batch_size": 4, "serve_vmm_batch": {}}`):
  merged over `load` for the batch-path cache scenario.

## Scenarios

| Scenario | Requests | Compared |
|---|---|---|
| `load` | the `loaded` envelope | non-timing fields |
| `cold` | first greedy request on a fresh load | token events, done fields |
| `multi_turn` | 4 turns, each resending the history | token events, done fields incl. `cached_tokens` |
| `interleaved` | two 2-turn conversations sent A1 B1 A2 B2: each turn 2 must reuse its own conversation's prefix with the other's state live in between | token events, done fields incl. `cached_tokens` |
| `abort_prefill` | long prompt with the abort queued right behind it, then the same prompt again | abort result and the follow-up incl. `cached_tokens`. The abort lands at the loop's first stdin poll, which on some families comes after decode tokens |
| `abort_decode` | abort after the 8th token or reasoning event, then the same prompt again | `finish_reason` must be `aborted` on both arms; the common prefix of the aborted stream; the follow-up exactly |
| `eos_budget` | probe the natural length `n` on the legacy arm, then rerun both arms with `max_tokens` `n` and `n+1` (the engine arm reuses the legacy `n`, because its own probe does not count EOS) | token events, done fields |
| `seeded_repro` | the same seed twice at T=0.8 | each arm must reproduce itself; `cached_tokens`/`prefill_tokens` exact |
| `seeded_sweep` | `--seeds` seeds at T=0.8 | the engine must give as many distinct outputs as legacy (it honours the seed); TVD of the first-8-token mix `<= --tvd-max` (0.35); `cached_tokens`/`prefill_tokens` exact |
| `isolation` | the `cold` prompt again, after every scenario above | cross-arm like `cold`; within each arm its tokens and ids must equal `cold`'s (state-leak check) |

The aborted stream's length is not compared, because the abort lands
asynchronously and a run may emit a few more tokens before it sees it. Done
fields are compared except timing keys (`*_ms`, `*_s`, `*tok_s`, `*timings`,
..., at any nesting depth) and `loop`.

**Scenario status** is `equal`, `intended` (only whitelisted diffs),
`unexercised` or `diff`. Only `diff` fails a scenario. `unexercised` means the
scenario did not test what its name says: the `eos_budget` probe hit its
budget before EOS, a `multi_turn` turn after the first reused no prefix
(`cached_tokens` 0), the `abort_prefill` abort landed after decode tokens, or
the `abort_decode` abort landed past half the budget (all judged on the legacy
arm); or, judged on the engine arm, an `interleaved` turn 2 or an abort
follow-up reused no prefix. The legacy loop keeps one live session record and
drops it on an abort (Qwen4 AR on halo: `interleaved` turn 2 and both abort
follow-ups reuse 0, HIPF-472), so these measure the engine's N records.
The receipt verdict is then `pass-partial` (exit 0) and the EVIDENCE disposition names the unexercised scenarios; a gate that relies on that scenario
needs another route or fixture. Baseline: on Gemma 4 26B-A4B, `multi_turn`,
`abort_prefill` and `abort_decode` are all `unexercised` (HIPF-475).

**Speculation counters** (`tau`, `cycles`, `rounds`, `accepted`,
`mtp_windows`, `floor_ar_windows`, `ar_windows`, `mtp_retired`, `ngram_mod_*`,
`dflash_profit_ratio`, `dflash_retired`, `dflash_retire_cycle`) are compared
exactly by default. With `"spec_noise": true` a differing value is a
`spec-stat` note and only G-spec judges it. On Qwen4 native MTP these counters
differ between two **legacy** processes even when the tokens are identical,
with or without the AR floor (HIPF-474). A counter present on one arm only is
a diff in both modes. [INFERENCE] The DFlash `dflash_*` fields come from the
wall-clock profit guard and may need `spec_noise` on DFlash routes.

## Intended diffs (whitelist)

These are reported as `intended` and do not fail the run:

1. **Per-request RNG.** The engine seeds a per-request RNG instead of the
   global `reset_cpu_sampler_rng`. Sampled output is therefore never compared
   bitwise across arms: only `seeded_repro` (within an arm) and the
   `seeded_sweep` distribution are checked.
2. **EOS not counted in `tokens`.** Engine `tokens` = legacy `tokens - 1` when
   legacy finished with `stop`.
3. **Stop on the last budget token** is `finish_reason: "stop"` on the engine,
   following the `qwen_dflash_hit_length_cap` semantics. Legacy `length` vs
   engine `stop` is allowed in `eos_budget`.
4. **More prefix reuse where legacy drops its record.** In `interleaved`
   turn 2 and the abort follow-ups, engine `cached_tokens` above legacy's is
   allowed when `cached_tokens + prefill_tokens` (the prompt) is equal and the
   tokens are equal (M1 gate: N live records, a reusable session after an
   abort). Less reuse than legacy is a diff.
5. **Gemma 4 at the cache cap** (not a harness scenario; judged by hand,
   HIPF-546). A request whose `max_tokens` exceeds the context left is clamped
   to `max_seq - prompt - 1` on the engine and ends `length` 2 tokens before
   legacy (whose last picks are emitted without running); the ids are a prefix
   of legacy's. A prompt of `max_seq - 1` tokens is refused on the engine.
6. **Committed positions.** Every engine route stamps a `committed` token with
   its conversation position; legacy Gemma 4 EAGLE counts from the first
   generated token. On a Gemma 4 EAGLE route (`load` names a `drafter`), equal
   token ids at different positions are `intended` (HIPF-471); on every other
   route a position difference is a diff (HIPF-552).

Every other difference is a diff.

## Gates

A run passes when all of these hold:

- **G-tok:** no scenario is `diff`.
- **Sampled:** both arms reproduce `seeded_repro`, and `seeded_sweep` passes.
- **Loop:** every done envelope carries `loop` once an arm emits it at all;
  the value is `legacy` or the arm's own. With `expect_engine`, every
  engine-arm done envelope is `engine`. A pre-engine binary emits no `loop`
  and passes.
- **Isolation:** within each arm, `isolation` gives `cold`'s tokens and ids
  when it prefilled cold or restored the whole prompt. A partial prefix reuse
  is not token-exact with a cold prefill (see Concurrency), so it is
  `unexercised`.
- **G-spec:** for a route whose legacy arm reports `tau`, the engine reports
  `tau` on the same requests and its mean over every greedy request is at
  least legacy tau − 1 %. On Qwen4 MTP the legacy-vs-legacy spread already
  exceeds 1 % (HIPF-474), so a G-spec failure there must be compared against
  a legacy-vs-legacy run of the same route before it counts.
- **G-perf** (`--perf`): fresh loads, interleaved legacy/engine,
  `--perf-reps` (3) per arm. The run takes the median prefill and decode
  tok/s. Both deltas must be within ±2 %. Each rep records the loadavg,
  foreign `/dev/kfd` holders and the current gfxclk. Take perf numbers only
  under the lock with no foreign `cargo`/`rustc`, and treat a single delta
  under ~5 % on halo as noise until it repeats.
- Both daemons exit 0.
- With `--conc`: every concurrency gate below.

## Concurrency (`--conc`)

The route's `conc` block runs `hipfire bench --concurrency` once per arm, the
daemon behind a wire tap (`daemon-tap.sh`, `HIPFIRE_DAEMON_BIN`): the bench
JSON gives each request's wire times, the tapped stdout its tokens, ids and
done envelope. Before each arm the harness waits (at most 120 s) until no
hipfire GPU reservation lock is held, because the previous arm's daemon can
still hold it after its bench returned.

```json
"conc": {
  "k": "1,2,4",
  "workloads": ["stateless", "multiturn"],
  "runs": 3,
  "arms": [["legacy", "noslots"], ["engine", "batch"], ["legacy", "batch"]],
  "ref": ["legacy", "noslots"],
  "target": ["legacy", "batch"],
  "tol_pct": 2.0,
  "args": ["--num-prompts", "16", "--max-tokens", "128", "--max-seq", "16384", "--spec", "off"]
}
```

- `arms`: `[loop, backend]` pairs; `backend` is the bench `--backend`
  (`noslots`, `batch` = the `serve.vmm_batch` route, fn-lanes on Qwen4, or
  `slots`). Each arm is one bench process, `--workload both` unless
  `workloads` names one. Defaults: legacy and engine on `noslots`.
- `args`: extra bench arguments, e.g. `--spec mtp` for a spec route.
  `--runs`, `--concurrency`, `--backend`, `--workload` and `-j` are set by the
  harness.
- `ref` (default legacy-noslots): the serialized legacy loop, the token
  reference at every k.
- `target`: the arm an engine arm must match at k>1 (Qwen4: legacy fn-lanes;
  Gemma4, which has no batched route: legacy-noslots). Omitted, no
  throughput/TTFT gate.

Per cell (workload, k) the receipt reports the median over runs of output
tok/s (Σ tokens / sample wall), TTFT p50/p99 from `t_send` (a client-side queue
wait counts, so a serialized loop shows it) and TPOT p50/p99. Overlap-rejected
samples are measured too. Gates, on every engine arm (legacy arms are reference
rows; legacy fn-lanes are not row-exact by default):

- every request ends in `done`; `loop` stamps as in the Loop gate
- **C1 == CN:** each request at k>1 has the tokens and ids of the same
  request (round, conversation, turn) at k=1. Legacy rows report their count
  as `reference_c1_ne_cn`: the legacy loop keeps one session record, so at k>1
  a multi-turn follow-up prefills cold, and a cold prefill is not token-exact
  with a reused prefix (Qwen4 AR mq4-xts, halo: 75 of 96 follow-ups at k=2/4
  differ from k=1, every stateless request and first turn is equal, HIPF-472).
  An engine that keeps each conversation's prefix passes it.
- **G-tok:** tokens and ids equal the `ref` arm's request with the same
  `cached_tokens`, at the same k first, else at another k. A request whose
  reuse no `ref` request matched is listed (`ref_reuse_unmatched`) and marks
  the run `pass-partial`.
- **cache:** every multi-turn follow-up reuses at least the `ref` arm's
  `cached_tokens`, and at k>1 at least the same request at k=1
  (`cache_below_c1`; legacy-noslots reuses nothing at k>1, so only C1 sees a
  lost lane record, Loki HIPF-586 S3). A follow-up exempt from the C1 check is
  one whose lane record the engine logged as `[engine] lane record <tag> not
  continued` (`refused`). The harness maps a tag to its request through the
  wire: tag N is the N-th `gen_start` (the engine takes the tag right before
  it), and the refused request is the next turn of that conversation. The
  mapping needs every request on the wire batched (a singleton request takes
  no tag); otherwise the tags are listed as `refusals_unmapped` and exempt
  nothing. No follow-up with a prefix hit on the engine marks the run
  `pass-partial`
- **G-spec:** mean tau per cell at least the `ref` cell's − 1 %
- **target** (k>1): output tok/s at least the target's − `tol_pct`, TTFT p50
  and p99 at most the target's + `tol_pct`

An arm whose bench exits non-zero or yields no cells fails the run. A
`noslots` engine arm holds requests client-side, so it cannot overlap them.
On Qwen4 the engine's N-request scheduler serves the `batch` backend
(`serve.vmm_batch`, HIPF-505): the engine arm is `["engine", "batch"]`.

## Batch-path cache scenario (`--batch`)

Plain `generate` requests run the singleton route, which keeps one session
record, so the A/B scenarios cannot show what the N-request scheduler's lane
records keep. `--batch` loads one more daemon per arm with `load` +
`batch_load` and sends greedy chat turns as `hipfire bench --backend batch`
does (`serve_continuous_batch`, neutral penalties, thinking off). Two twin
pairs of conversations (identical messages): P (abort in the prompt) and D
(abort in decode).

| step | requests, sent together | abort |
|---|---|---|
| 1 | turn 0 of P, P~, D, D~ | – |
| 2 | turn 1 of P (a ~3k-token user turn), P~, D (counting to 100, 4× budget), D~ | P on its `gen_start` (admitted, so it lands in the prompt); D after 8 token events |
| 3 | turn 1 of P and D again (`P1r`, `D1r`) | – |

Gates, on the engine arm (the legacy arm is a reference row: legacy fn-lanes
keep no records):

- every request ends in `done`; P1 and D1 `aborted`; `loop` is `engine` with
  `expect_engine`
- the twins' turn 0 are token-exact
- **reuse:** P~1, D~1 (concurrent follow-ups) and the retries P1r, D1r
  (after the aborts) reuse at least their conversation's committed turn 0:
  `cached_tokens` >= its prompt plus every committed id (a follow-up that
  reuses the whole turn reports exactly that). The receipt lists
  `reuse[turn] = [cached, committed turn 0]`.
- a retry with its control's reuse (P1r vs P~1, D1r vs D~1, identical
  prompts) has the control's tokens and ids; with a different reuse the pair is
  only noted

The notes also list the requests whose lane record the engine logged as not
continued (`refused`, tag N = the N-th `gen_start` on the daemon's stdout), so
a reuse failure caused by a known refusal (e.g. a retired MTP lane) is
attributable. Such a failure still fails the gate.

Unexercised (`pass-partial`): an abort that did not land, P1's abort after
decode tokens, D1's past half its budget, or a finished request that did not
run batched (`continuous_batch.executed`).

## Coverage gaps

- Gemma 4 EAGLE has no concurrency arm: `hipfire bench` cannot pass a
  drafter (the drafter itself is a HIPF-522 fixture since HIPF-542).
