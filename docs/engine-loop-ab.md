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

Exit code `0` means every scenario is equal or is an intended diff. `1` means a
real diff, and `2` means a harness or daemon failure (load error, daemon exit,
event timeout). Outputs in `--out`:

- `<route>.json`: the receipt. It holds the verdict, the per-scenario status
  and notes, the sampled checks, tau, perf medians, the daemon md5, the fixture
  sha256, the head SHA, the host state and every arm's raw results.
- `<route>.<arm>.jsonl` / `.stderr`: each daemon's raw stdout and stderr.
- `EVIDENCE.md`: one row per run, with the head, daemon md5, fixture sha256,
  exact command, receipt path, result and disposition.

## Route file

```json
{
  "name": "qwen4-mtp",
  "model": "~/.hipfire/models/<fixture>",
  "load": {"max_seq": 8192, "mtp_mode": "on"},
  "env": {"HIPFIRE_PLE_FORMAT": "q4"},
  "max_tokens": 64,
  "perf_decode_tokens": 256
}
```

`load` is passed verbatim as the daemon `load` params. Spec variants are their
own route files: Qwen4 native MTP via `"mtp_mode": "on"`, a DFlash drafter via
`"draft": "<path>"`. `env` is applied to both arms. Never put
`HIPFIRE_ENGINE_LOOP` in it, because the harness sets that per arm.

## Scenarios

| Scenario | Requests | Compared |
|---|---|---|
| `load` | the `loaded` envelope | non-timing fields |
| `cold` | first greedy request on a fresh load | token events, done fields |
| `multi_turn` | 4 turns, each resending the history | token events, done fields incl. `cached_tokens` |
| `abort_prefill` | long prompt with the abort queued right behind it, then a fresh request | abort result and the follow-up. The abort lands at the loop's first stdin poll, which on some families comes after decode tokens (Gemma 4: ~60) |
| `abort_decode` | abort after the 8th token or reasoning event, then the same prompt again | `finish_reason` must be `aborted`; the common prefix of the aborted stream; the follow-up exactly |
| `eos_budget` | probe the natural length `n` on the legacy arm, then rerun both arms with `max_tokens` `n` and `n+1` (the engine arm reuses the legacy `n`, because its own probe does not count EOS) | token events, done fields |
| `seeded_repro` | the same seed twice at T=0.8 | each arm must reproduce itself |
| `seeded_sweep` | `--seeds` seeds at T=0.8 | distribution only: TVD of the first-8-token mix `<= --tvd-max` (0.35) |

The aborted stream's length is not compared, because the abort lands
asynchronously and a run may emit a few more tokens before it sees it. Done
fields are compared except timing keys (`*_ms`, `*_s`, `*tok_s`, ...) and
`loop`. The `loop` field is checked per arm: it must be absent (pre-engine
binary), `legacy`, or the arm's own value (`legacy` for a route off the engine
allowlist).

Speculation bookkeeping (`tau`, `cycles`, `accepted`, `mtp_windows`,
`floor_ar_windows`, `ar_windows`, `mtp_retired`, `ngram_mod_*`) is listed as
`spec-stat` notes and judged only by G-spec. On Qwen4 native MTP these
counters differ between two **legacy** processes even when the tokens are
identical, with or without the AR floor (HIPF-468 baseline). An exact compare
would fail every run.

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

- **G-tok:** every greedy scenario is `equal` or `intended`.
- **G-spec:** for a route that reports `tau`, the mean engine tau over every
  greedy request must be at least legacy tau − 1 %. On Qwen4 MTP the
  legacy-vs-legacy spread already exceeds 1 % (see the HIPF-468 receipts), so
  a G-spec failure there must be compared against a legacy-vs-legacy run of
  the same route before it counts.
- **G-perf** (`--perf`): fresh loads, interleaved legacy/engine,
  `--perf-reps` (3) per arm. The run takes the median prefill and decode
  tok/s. Both deltas must be within ±2 %. Each rep records the loadavg,
  foreign `/dev/kfd` holders and the current gfxclk. Take perf numbers only
  under the lock with no foreign `cargo`/`rustc`, and treat a single delta
  under ~5 % on halo as noise until it repeats.
