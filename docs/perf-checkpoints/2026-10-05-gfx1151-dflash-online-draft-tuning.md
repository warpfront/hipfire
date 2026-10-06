# Online DFlash draft tuning (developer-only) on gfx1151

Lifecycle: **historical**. Fixture-bound measurement of the developer-only
`HIPFIRE_DFLASH_ONLINE_TUNE=on` route (`crates/hipfire-runtime/src/dflash_online.rs`).
It is not a default and not an admission. Disposition: research; off unless the env var is set.

## What it is

Chain DFlash, greedy, batched LM-head path. The draft's top-16 per block row is
taken on device (re-grid top-K) and re-ranked on the host by a per-request
linear model trained online from tokens the target already verified.

Features describe the chain the draft cannot see, because block rows are drafted in parallel:
- n-gram continuations (orders 1 to 3) of the proposed chain over the prompt plus the emitted stream;
- the longest suffix match, with a long-match continuation injected into slot 15;
- stutter (the candidate equals the chain token one or two back);
- the candidate equals the previous row's top-1 or the next row's top-1/top-2;
- a rank prior;
- a depth-scaled logit gap.

Weights are kept per row-depth bucket and carried across requests. Text statistics reset per request.

Verify is unchanged, so every emitted token is the target's greedy argmax. Greedy text on the
tuned route differs from shipping DFlash only where batched-verify numerics flip a near-tie, the same
class by which shipping DFlash already differs from AR. On `code_parser_rust`, 700 tokens:
- shipping DFlash run twice: identical;
- stats mode vs shipping: identical;
- tuned vs shipping: first difference at byte 1140 of 2421;
- shipping DFlash vs AR: first difference at byte 54.

## Fixture

- gfx1151 (Strix Halo), HIP 7.2, `~/.hipfire/rocm-merged`.
- Target `qwen3.5-9b.mq4`, MD5 `296092bf1e6a45d78c1acf815eb93366`.
- Draft `qwen35-9b-dflash-mq4.hfq`, MD5 `590f35403cd7f1d634945233234a12b7`.
- Prompts: `benchmarks/prompts/online_tune/*.txt`, 10 long-session prompts with greedy output capped at 1536 tokens:

| prompt | md5 |
|---|---|
| code_edit_typehints | 6a1a7bda04ad… |
| code_inventory | bb5c51ff87b0… |
| code_parser_rust | 0857efaa3d32… |
| code_router_js | 10b2b69a0c41… |
| mixed_tcp | 804738bdde4e… |
| prose_bread | c8b98bfef421… |
| prose_letter | 0ba3a438a580… |
| prose_lighthouses | d8624c675e5e… |
| prose_oped | 6754fd08ccce… |
| prose_snowstorm | 22abbae6b1d8… |

## Method

- **Replay (deterministic).** `examples/dflash_online_replay.rs` runs over argmax-session dumps (`HIPFIRE_DFLASH_ONLINE_TUNE=stats`, `HIPFIRE_DFLASH_ONLINE_DUMP`). The text and block starts are fixed, so the ratio of tuned τ to argmax τ isolates proposal quality. `--carry-loo` replays each session after the other 9 (weights carried, never trained on the scored session).
- **Online.** Paired `hipfire bench --spec dflash --runs 1 --warmups 0 --backend noslots --workload stateless`. Each pair is a fresh process per arm, in alternating order. A = shipping DFlash (tuner unset), B = tuned.

## Results

Replay over 10 sessions (prompt-seeded):

| config | τ ratio |
|---|---|
| leave-one-out with weights carried | ×1.062 |
| cold, one request per daemon | ×1.054 |
| first 32 cycles, carried | ×1.058 |
| first 32 cycles, cold | ×1.018 |

Carried, per session:

| session | τ ratio |
|---|---|
| code_parser_rust | ×1.122 |
| mixed_tcp | ×1.092 |
| code_inventory | ×1.089 |
| code_router_js | ×1.077 |
| code_edit_typehints | ×1.065 |
| prose | ×1.023 to ×1.049 |

Online, 9 prompts × 2 repetitions = 18 pairs (36 fresh processes). Build `5685eccb7`, before prompt seeding and carry; daemon MD5 `890545289ddd6cdbd997c2321b19ce3f`.

| subset | τ | tokens/window | decode tok/s |
|---|---|---|---|
| all | ×1.049 | ×1.034 | **×1.034** (11 of 18 pairs win) |
| code (n=6) | | ×1.070 | ×1.070 |
| mixed (n=2) | | ×1.029 | ×1.032 |
| prose (n=10) | | ×1.015 | ×1.014 |

Online, 4 pairs, build `c46f2f200` (prompt seeding and carry):

| subset | τ | decode tok/s |
|---|---|---|
| all | ×1.048 | ×1.035 |
| code_edit_typehints | | 122.4 → 128.5 |
| code_inventory | | 76.8 → 80.9 |

Cost:
- **Host:** re-rank plus training take 96 + 18 µs per cycle.
- **Shipping top-K kernel:** the one-block-per-row top-K costs 1349 µs per cycle at 15×248320, K=16. It ate the gain: draft phase 22.4 → 24.9 ms against shipping argmax.
- **Re-grid top-K:** 63 µs. It is byte-identical on gfx1151 under `test_select_regrid` with the arch gate bypassed (210 launches compared, ties, NaN and −inf included).

Prose DFlash on this pair still trails AR (river prompt: DFlash ~33 vs AR 47.3 tok/s). The tuner does not change that conclusion.

## Negative results (replay unless noted)

| change | result |
|---|---|
| rank-16 LoRA over the draft hidden, within a request | online essay τ 1.10 → 0.94; in-sample after 4 epochs 1.88 (memorizes) |
| same LoRA, carried across requests at a low rate | +0.6 pp (×1.062 → ×1.068), at ~1 ms/cycle host cost and −0.5 pp cold; not kept |
| per-token bias | neutral |
| target's stale verify argmax past the rejection | right 14% of the time at row 0, and 1.6% when argmax was wrong |
| training on all rows instead of chain-reachable rows | negative |
| override gate, neighbour softmax probabilities, previous-block view, gap interactions, shift features, shared-plus-bucket weights | ×1.045 to ×1.054, against ×1.055 |
| 4-gram, history window or `MAX_OCC` changes, lr / acc0 / margin / inject thresholds | within ±0.3 pp |

Mismatched-draft case not measured: Qwen3.8-27B MQ4XTS stops after 2 tokens on these prompts with the
tuner unset (pre-existing).
