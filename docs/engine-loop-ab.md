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
    --out ~/mesh-validation/<KEY> [--perf]
```

Exit code `0` means every gate passed (see Gates). `1` means a gate failed, and
`2` means a harness or daemon failure (load error, daemon exit, event timeout).
`--self-test` checks the diff rules and gates without a GPU. Outputs in `--out`:

- `<route>.json`: the receipt. It holds the verdict, the per-scenario status
  and notes, the sampled, loop, G-spec and G-perf results, the daemon md5, the
  fixture sha256, the harness sha256 (plus git head and a dirty flag when run
  from a checkout), the host state and every arm's raw results.
- `<route>.<arm>.jsonl` / `.stderr`: each daemon's raw stdout and stderr.
- `EVIDENCE.md`: one row per run, with the harness sha256, daemon md5, fixture
  sha256, exact command, receipt path, result and disposition.

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

## Scenarios

| Scenario | Requests | Compared |
|---|---|---|
| `load` | the `loaded` envelope | non-timing fields |
| `cold` | first greedy request on a fresh load | token events, done fields |
| `multi_turn` | 4 turns, each resending the history | token events, done fields incl. `cached_tokens` |
| `abort_prefill` | long prompt with the abort queued right behind it, then the same prompt again | abort result and the follow-up incl. `cached_tokens`. The abort lands at the loop's first stdin poll, which on some families comes after decode tokens |
| `abort_decode` | abort after the 8th token or reasoning event, then the same prompt again | `finish_reason` must be `aborted` on both arms; the common prefix of the aborted stream; the follow-up exactly |
| `eos_budget` | probe the natural length `n` on the legacy arm, then rerun both arms with `max_tokens` `n` and `n+1` (the engine arm reuses the legacy `n`, because its own probe does not count EOS) | token events, done fields |
| `seeded_repro` | the same seed twice at T=0.8 | each arm must reproduce itself; `cached_tokens`/`prefill_tokens` exact |
| `seeded_sweep` | `--seeds` seeds at T=0.8 | the engine must give as many distinct outputs as legacy (it honours the seed); TVD of the first-8-token mix `<= --tvd-max` (0.35); `cached_tokens`/`prefill_tokens` exact |

The aborted stream's length is not compared, because the abort lands
asynchronously and a run may emit a few more tokens before it sees it. Done
fields are compared except timing keys (`*_ms`, `*_s`, `*tok_s`, `*timings`,
..., at any nesting depth) and `loop`.

**Scenario status** is `equal`, `intended` (only whitelisted diffs),
`unexercised` or `diff`. Only `diff` fails a scenario. `unexercised` means the
scenario did not test what its name says, judged on the legacy arm: the
`eos_budget` probe hit its budget before EOS, a `multi_turn` turn after the
first reused no prefix (`cached_tokens` 0), the `abort_prefill` abort landed
after decode tokens, or the `abort_decode` abort landed past half the budget.
It is listed in the EVIDENCE summary; a gate that relies on that scenario
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

Every other difference is a diff.

## Gates

A run passes when all of these hold:

- **G-tok:** no scenario is `diff`.
- **Sampled:** both arms reproduce `seeded_repro`, and `seeded_sweep` passes.
- **Loop:** every done envelope carries `loop` once an arm emits it at all;
  the value is `legacy` or the arm's own. With `expect_engine`, every
  engine-arm done envelope is `engine`. A pre-engine binary emits no `loop`
  and passes.
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

## Coverage gaps

- No Gemma 4 spec route: no Gemma 4 EAGLE drafter artifact exists
  (fixture-missing; the drafter is also gated off by default).
