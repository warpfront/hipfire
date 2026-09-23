# ATT lane Slice 1 final gate

Verdict: **KILL**. The short/long hybrid is byte-exact across all 921 oracle cases, but it misses the required pp8192 serving gain in both pinned-daemon pairs. Pair 1 is -3.153% and pair 2 is -0.312% versus the incumbent; both are below the required +1.5%. No source commit was made.

## Pinned artifacts

| Arm | Daemon MD5 | Attention HSACO SHA-256 |
|---|---|---|
| incumbent | `bdf6849f8838d4b7b729fa29022aa00b` | `0ed79d32134add3e2217a472e2c2a69b4303d3f3450dac4e715c462ab6d8812c` |
| hybrid candidate | `d8c8385b4bf00c3b710e428f08313bec` | wide object `33a493cc2519504a4a672fea61a609ae41bae642144921fbf989c991049ee9a6` |

The immutable daemon copies are `bin/daemon-base` and `bin/daemon-candidate`. The rejected source diff is `hybrid-killed.patch` (SHA-256 `3a7595305f42d0da212d137bc766bd588ee745237de747d22ebd9c02912cda9a`).

## Candidate route

The candidate retains the incumbent narrow Q128/KT64 packet kernel below KV extent 1024 and selects a Q384/KT48 packet kernel at or above 1024. The narrow entry uses block 256 and 49,408 B dynamic LDS. The wide entry uses block 768 and 37,120 B dynamic LDS. The wide direct entry compiles with 216 live VGPRs, 29 SGPRs, and no scratch or spills; its runtime occupancy is 24 resident waves/CU.

## Exactness

`hybrid-oracle.log` records `cases=921 failed=0 input_diffs(Q/K/V/P)=[0,0,0,0]` in 2.449 seconds after `hybrid-oracle-warm.log`. This covers direct, split-1, and split-8 paths across both sides of the route threshold and compares device-resident output bytes against the incumbent implementation.

## Standalone body timing

The pinned standalone wide-body experiment uses the same device-resident Q/K/V/position fixture, five warmups, and 25 post-warmup HIP-event samples per point. Candidate deltas below are versus the midpoint of the incumbent A1/A2 medians; negative is faster.

| Context | Candidate delta |
|---:|---:|
| 96 | +12.563% |
| 257 | +8.163% |
| 1536 | -5.624% |
| 3072 | -8.987% |
| 6144 | -11.520% |
| 12288 | -12.063% |
| 32768 | -10.399% |

The standalone sum improves 10.361%. The hybrid route was introduced specifically to retain the incumbent short path while using the wide body only for long extents. Full samples are in `pinned-body-timing.jsonl`.

## Pinned-daemon matrix

Command: `hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`.

| Pair | Shape | Incumbent | Candidate | Candidate delta |
|---:|---|---:|---:|---:|
| 1 | pp512 | 2363.9 | 2290.1 | -3.122% |
| 1 | pp2048 | 2511.8 | 2441.3 | -2.807% |
| 1 | pp8192 | 2452.0 | 2374.7 | **-3.153%** |
| 1 | pp32768 | 2084.0 | 2014.1 | -3.354% |
| 1 | decode tg64@128 | 31.3187 | 31.2567 | -0.198% |
| 2 | pp512 | 2282.6 | 2287.9 | +0.232% |
| 2 | pp2048 | 2429.5 | 2429.9 | +0.016% |
| 2 | pp8192 | 2374.1 | 2366.7 | **-0.312%** |
| 2 | pp32768 | 2034.1 | 2006.3 | -1.367% |
| 2 | decode tg64@128 | 31.3259 | 31.2784 | -0.152% |

Raw evidence: `matrix-p1-base.json`, `matrix-p1-candidate.json`, `matrix-p2-candidate.json`, and `matrix-p2-base.json`.

## Pinned TTFT

Command: `hipfire bench qwen3.8:27b-mq4-xt --ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --json`.

| Metric | Incumbent | Candidate | Candidate delta |
|---|---:|---:|---:|
| median TTFT ms | 2486.4436 | 2512.7687 | +1.059% |
| median prefill tok/s | 2376.4880 | 2351.5910 | -1.048% |

Raw evidence: `ttft-pinned-base.json` and `ttft-pinned-candidate.json`.

## In-daemon rocprof pp8192 attribution

Each arm received an untraced one-run warmup followed by one traced run with zero traced warmups. The table sums device timestamps for the 32 full-attention layer dispatches in the pp8192 pass.

| Arm | Symbol | Median/dispatch | Sum |
|---|---|---:|---:|
| incumbent | `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` | 12422.0075 us | 397.130265 ms |
| candidate | `attention_fp8_e4m3_fa2_gqa_packet_q384_gfx1201` | 13752.9885 us | 439.812244 ms |

The candidate attention-kernel sum regresses **10.748%**. Decode retains the narrow symbol; its small 32-dispatch sum is 0.809462 ms to 0.922335 ms (+13.944%), with negligible absolute weight. Trace CSVs are `rocprof-pinned-base/trace-pinned-base_kernel_trace.csv` and `rocprof-candidate/trace-candidate_kernel_trace.csv`.

## Supporting attribution

The incumbent instruction-level ATT attribution and provenance are in `att_summary.md`; raw ATT data are under `att/`. Static instruction and compiler-resource evidence are in `static-object-census.json`, `baseline-resources.log`, and `q384-resources.log`.
