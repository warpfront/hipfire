# Q64 packet attention: KILL

The Q64 packet body is exact and meets its compile-resource target, but it is a decisive daemon regression. The production kernel and launcher are restored to `5729b526f`; Step 2 was not attempted because Step 1 failed both serving gates.

## Candidate

The candidate used block 128, four waves owning 16 query-head rows each, and two uniform D128 output-slice workgroups per Q64 tile. Each wave accumulated eight output fragments. K/V fill retained KT64: 128 threads covered the 1,024 paired-K fragments, while four waves covered the eight V producer roles in two transpose rounds. Q was preconverted once so both output-slice workgroups read an immutable scratch image. Dynamic LDS was 41,216 B.

This shape necessarily duplicated QK and online softmax across the two output-slice workgroups. The reduced live accumulator did not compensate for that duplicated work or for splitting a former Q128 workgroup into four Q64/D128 workgroups.

## Resources and census

The gfx1201 resource screen passed:

| Entry | VGPR | SGPR | Scratch/spills | Compiler waves/SIMD |
|---|---:|---:|---:|---:|
| Direct | 164 | 29 | 0 | 9 |
| Partial | 167 | 32 | 0 | 9 |

The complete-direct-symbol census (one recognized ISA line is one packet) was:

| Arm | Instructions | WMMA | FP8 pack | Global | LDS | Wait | Barrier | Branch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline Q128 | 1,906 | 20 | 8 | 94 | 147 | 294 | 6 | 39 |
| Q64/D128 slice | 1,880 | 12 | 4 | 63 | 263 | 365 | 12 | 50 |

The candidate row requires two output-slice workgroups for each Q64 tile and two Q64 tiles to cover the baseline's Q128 ownership. Thus the slightly smaller per-symbol body is not a same-work reduction. Raw receipts are in `scratch-2026-09-20/AttnSmall/step1/resources.log` and `census.json`.

## Exact oracle

The 921-case device oracle passed bit-for-bit for direct, split-1 record/merge, and split-8 record/merge paths. All Q/K/V/position input-diff and guard counters were zero:

```text
SUMMARY cases=921 failed=0 input_diffs(Q/K/V/P)=[0, 0, 0, 0] elapsed_s=7.222
```

Receipt: `scratch-2026-09-20/AttnSmall/step1/oracle.log`.

## In-daemon attribution

A warmed 5,909-token daemon-protocol replay was traced with rocprof. Both arms issued 192 packet-attention dispatches.

| Arm | Packet sum | Q preconvert | Packet + preconvert |
|---|---:|---:|---:|
| Baseline | 138.629208 ms | fused | 138.629208 ms |
| Q64/D128 | 317.030848 ms | 5.958671 ms | 322.989519 ms |
| Delta | +128.690% | — | **+132.991%** |

The request-level prefill moved from 2,003.7 ms / 2,949.1 tok/s to 2,172.0 ms / 2,720.5 tok/s. Trace receipts are in `scratch-2026-09-20/AttnSmall/step1/rocprof-base/` and `rocprof-candidate/`.

## Paired daemon matrix

The required matrix used three measured runs and one warmup in alternating baseline/candidate then candidate/baseline order.

| Pair | pp512 delta | pp8192 delta | Decode delta |
|---|---:|---:|---:|
| 1 | -1.825% | **-11.904%** | -0.150% |
| 2 | -0.886% | **-11.043%** | +0.143% |

Both pp8192 pairs fail the required at-least 1.5% gain. Raw JSON receipts are the four `matrix-p*.json` files in the step directory.

## Paired client TTFT

The required 5,909-token client TTFT used eight measured runs and two warmups in the same alternating order.

| Pair | Baseline median | Candidate median | TTFT delta | Prefill-rate delta |
|---|---:|---:|---:|---:|
| 1 | 2,089.357 ms | 2,236.934 ms | **+7.063%** | -6.598% |
| 2 | 2,067.107 ms | 2,306.267 ms | **+11.570%** | -10.369% |

Both pairs exceed the maximum 1% TTFT regression. Raw JSON receipts are the four `ttft-p*.json` files in the step directory.

## Disposition

**KILL Step 1.** Exactness and compiler occupancy are insufficient: real packet time more than doubles, pp8192 loses about 11%, and TTFT regresses 7–12%. Step 2's KT32x2 ring could overlap producer fill, but cannot remove the output-slice duplication of QK and online softmax that dominates this result, so the strict sequence stops before Step 2.
