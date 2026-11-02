# MTP parity fixture and ratchet pre-registration (R9700 x2, Flash-Next) — 2026-11-01

**Lifecycle:** `historical`

**Disposition:** registration record. It pins the fixture bytes and the ratchet
rule *before* any hipfire baseline exists on that fixture, and records the
run-to-run spread the rule's constants come from. It is not a baseline, not an
admission, and not a throughput claim. No card job produced this record; every
number below is read from an existing receipt under the fixture that receipt
used, and none of them is a baseline for `benchmarks/mtp-parity/ratchet.json`.

## Fixture (frozen as `benchmarks/mtp-parity/fixture.json`, id `mtp-parity-v1`)

| prompt | class | path | bytes | md5 |
|---|---|---|---:|---|
| story | story | `benchmarks/prompts/fiction_lighthouse.txt` | 134 | `d3768bd6479ce9aae08c9e0152b39e06` |
| code | code | `benchmarks/prompts/humaneval_0_has_close_elements.txt` | 348 | `5333a1f70d884060807676347c0edb93` |
| prose_long | prose_long | `benchmarks/mtp-parity/prompts/prose_long.txt` | 148 | `39fb57f301d387e83a191769a2fef23d` |
| agentic | agentic | `benchmarks/mtp-parity/prompts/agentic_frozen_chain.json` | 7715 | `dedb861a5aa5dcff554b9c3ec608bf20` |
| math | math | `benchmarks/mtp-parity/prompts/math_reason.txt` | 239 | `2adc75296df0877a1c38aa53511a7d3d` |
| multilingual | multilingual | `benchmarks/mtp-parity/prompts/multilingual_zh.txt` | 117 | `44476c7d649b4ed4626f4fcca6342009` |

Request: chat completions, streamed, `max_tokens` 256, thinking off, no nonce
line, session and prefix caches off. Modes: `t1` = temperature 1.0, top_p 0.95,
top_k 20; `greedy` = temperature 0. Cells: story and code at C1/C4/C8, the four
breadth classes at C1/C8, each in both modes (28 cells), 3 fresh-process reps.
`agentic` is a frozen multi-turn tool transcript (three fixed assistant tool calls,
their results, and a closing user instruction; the model generates the final prose
answer, which quotes the tool output). A first version ended on the last tool result
and Radiance answered with one more 28-token tool call (21 reached trials per 3
reps), so the closing user turn was added and cells with a median completion under
128 tokens are invalid, so both engines prefill the
same context; it is the copy-lane case. Texts of `prose_long`, `math`,
`multilingual`, `story` and `code` are byte-identical to the other engine's
`diverse.json` entries (md5 compared).

## Definitions

`a_d = accepted_d / reached_d`, `reached_d` = windows that offered >= d draft rows
and accepted the first d-1 (MtpTrain's `a_d`; the same convention as the Radiance
1.11.0 `/stats draft_positions` counters, whose `reached` at depth d equals the
`accepted` at depth d-1). `yield` = committed tokens per lane-window. `window_ms`
= wall ms of one engine step.

## Pre-registered rule (`ratchet.json` `rule`)

`z` 3.0, `abs_floor` 0.02, `yield_floor_rel` 0.03, `window_floor_rel` 0.05,
`min_reps` 3, `min_trials_per_depth` 50, `max_cv_register` 0.15.

- `a_d` fails if `new < base - max(abs_floor, z*sqrt(p_b(1-p_b)/n_b + p_n(1-p_n)/n_n), z*sd_rep_base)`.
- `yield` fails if `new < base*(1 - max(yield_floor_rel, z*cv_rep_base))`.
- `window_ms` fails if `new > base*(1 + max(window_floor_rel, z*cv_rep_base))`.
- A cell registers only from >= 3 fresh-process reps with yield cv <= 0.15.
- A depth with fewer than 50 reached trials is INSUFFICIENT and fails closed.
- A pass needs every gated cell of every class (no partial pass).
- Loosening a baseline, widening the rule, or un-gating a cell needs a
  `RATCHET-RAISE:` line in the commit message (`check-mtp-parity-ratchet.py guard`).

## Where the constants come from (measured spread, other fixtures)

| source | what | spread |
|---|---|---|
| Radiance 1.11.0 stock, b1 (`mgmtp/parity-mtp/radiance/b1-T1-512tok`, 3 reps x 6 cells, max_tokens 512, nonce line, T=1) | yield per user-step, step ms, per-depth rate across reps | yield cv 0.000-0.116 (code highest: short outputs), step_ms cv 0.003-0.055 (range up to 0.100 on code-C1), per-depth `a_d` sd 0.000-0.049 |
| hipfire PP2 mq3 sampled singleton, `r9700x2-par/l2/runs/l2-c1-on{1,2,3}` (6 categories, 20 requests each) | window-weighted tau and tokens/window across fresh servers | tau sd 0.028-0.113 (range 0.056-0.216); yield cv 0.007-0.028 |
| hipfire EP2 lanes, `mgmtp/epserve/runs/e8-bench-on{1,2,3}`, `g3-bench-on{1,2}` (greedy, 32 prompts x 256) | out tok/s across fresh servers at equal config | range 0.1-3.2% at C1-C4, C8; one 6.0% outlier (C6, n=2) |
| hipfire EP2 mq4 C8 AR step, FLOOR kREF1 (n=4 and n=3 alternating) | `[fn-lane-step]` ms | range 0.4% |

Reading: z=3 over the binomial SE covers the within-cell sampling noise (C1
cells carry about 100-200 reached trials per depth, C8 cells about 1,000-4,000);
the floors sit at the smallest observed cv; noisier cells register wider
tolerances from their own reps, so C1 cells are weak detectors and C8 cells are
the sharp ones.

## Existing hipfire per-position evidence (not on this fixture; context only)

Singleton EP2 mq4, `HIPFIRE_MTP_TRACE=1`, sampled T=0.7, K adaptive up to 7, n-gram
takeover on (`mgmtp/c1/runs/c1t`, bin3 = beta 614563ae30 + D1/A1; five timed code
requests, 722 windows, 719 native, 3 n-gram), parsed with
`check-mtp-parity-ratchet.py extract-trace --requests 3:8`:

| depth | offered | reached | accepted | a_d |
|---:|---:|---:|---:|---:|
| 1 | 716 | 716 | 620 | 0.866 |
| 2 | 696 | 609 | 492 | 0.808 |
| 3 | 597 | 447 | 387 | 0.866 |

Committed tokens per window 4.003. `reached` counts only offered drafts: the depth
chooser and margin prune make deeper depths a selected subset, so `a_2`, `a_3`
here are conditional on the drafter having been confident.

## What does not exist yet

- Any hipfire per-position count on this fixture.
- Any EP2/PP2 *lane* per-position count on any prompt: the lanes' `done` envelope
  carries no `tau`/`spec_cycles` (only the single-card vmm lane path attaches
  them, `batch.rs` `attach_vmm_spec_evidence`), and the lane step's
  `OpCounts{drafts_generated, accepted}` is an aggregate. The EP2 lane bench
  `tau` column therefore cannot be a lane statistic [INFERENCE: it matches the one
  unbatched request per sample, 31 of 32 batched]: in the OFF arm (AR lanes, no
  windows) it still reads 3.4-4.3.
- A baseline for any cell. `ratchet.json` cells are `baseline: null`; `check`
  fails closed (exit 3) until a registration campaign runs.
