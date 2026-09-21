# Fold-free iu4 prefill experiment

## Result

**Do not ship this candidate.** The numeric contract is valid on the iu4 arm and the corrected standalone two-shape aggregate improves by 7.14%, but the daemon inverts it: paired pp8192 is **-0.185%**, pp512 is **-0.457%**, and 5,909-token TTFT regresses by **15.532 ms (+0.594%)**. Decode stays within its neutral band at -0.211%, and the five-prompt decoded-text battery is clean. The implementation remains explicit-env-only; the default path is unchanged.

## Quality gates

All scores use the exact 24-chunk prefill KLD evaluator on card-C unless noted.

| artifact / compute route / activation route | c2 | c24 | result |
|---|---:|---:|---|
| `sym-pow2h-a035.hfq`, iu4, existing K128 A scales | 0.071897 | **0.091338** | pass (`<= 0.10`) |
| `sym-a035.qat-r5s100.hfq`, iu4, existing K128 A scales | 0.064864 | 0.081199 | shipped control |
| `sym-pow2h-a035.hfq`, iu4, one A scale per token | 0.046252 | 0.055408 | pass |
| `sym-pow2-a035.hfq`, iu4, one A scale per token | 0.058773 | **0.071194** | pass |
| `sym-pow2-a035.hfq`, fp8v2 forced with row-global screening env | not run | **0.071194** | fail hard fp8 budget (`> 0.05`) |
| pow2h row-max-normalized derivative, iu4, one A scale per token | 0.098325 | 0.117126 | fail |
| pow2h MSE-selected row-common derivative, iu4, one A scale per token | not run | 0.114720 | fail |
| pow2h integer-mantissa derivative, iu4 | 0.121697 | not run | fail early |

The full-pow2 plus row-global-activation route is the clean numeric contract for shift folding. The pre-enlargement candidate produced byte-identical c2 and c24 score binaries to that route. After the tile enlargement and metadata fix, the rebuilt candidate again scored c2 = 0.058773 and its score binary was byte-identical to the baseline route.

The requested full-pow2 fp8v2 cell was run with `HIPFIRE_IU4_PREFILL=0`, row-global screening enabled, and fused producer quantization disabled. It scored 0.071194, so it does not clear the hard fp8v2 c24 budget of 0.05. Row-global A changes the int4 activation sidecar used by iu4; fp8 activation packing is already row-wide and does not consume that sidecar. The full-pow2 grid is therefore quality-valid only for the iu4 arm unless the fp8 arm keeps its current grid/artifact or receives QAT.

## Weight screens

A row-max common-scale requantization changed 5,821,518,243 of 25,621,954,560 codes (22.72%) with weight relative RMS error 0.085655. It failed c24 at 0.117126 even after row-global activation quantization.

An MSE-selected row-common scale changed 5,640,816,401 codes with relative RMS error 0.090104 and still failed c24 at 0.114720. Removing the half-step mantissa changed 7,577,507,991 codes with relative RMS error 0.12933 and failed c2 at 0.121697. A fully common weight scale is therefore not quality-admissible without retraining.

For the full-pow2 artifact, all 4,152,320 rows have exponent span at most seven. Span counts are `{0: 1,988,637, 1: 2,145,437, 2: 17,488, 3: 651, 4: 90, 5: 9, 6: 7, 7: 1}`. Adjacent K128 exponent changes are overwhelmingly zero or one, but their direction differs by row, so a wave cannot use one uniform shift direction.

## Kernel

The experimental route is selected only by `HIPFIRE_IU4_SHIFT_FOLDFREE=1`; the normal dispatch is unchanged. `HIPFIRE_IU4_GLOBAL_A_SCREEN=1` selects the one-scale-per-token activation quantizer, with fused producer quantization disabled for screening.

The kernel:

- keeps one integer accumulator set;
- uses a 256-token by 128-row workgroup tile with eight waves;
- retains a 64-by-64 tile per wave;
- shifts live accumulators between adjacent K128 exponent domains;
- applies one final floating-point scale after the K loop;
- uses 30,720 bytes LDS.

Resource compile for the full-set entry point: 184 VGPR, 44 SGPR, zero scratch, zero spills. Runtime occupancy reports two blocks/CU with 30,720-byte LDS.

A first 128-token/four-wave geometry reached 206.538 TOPS on gate/up versus 225.916 TOPS baseline and was discarded. Enlarging the token tile restored active waves and made gate/up positive. A metadata publisher bug in the first enlarged measurements let threads 128-255 read rows outside the 128-row tile; final measurements below are from the corrected kernel, where only the first 128 threads publish row metadata.

## Corrected standalone gate

Card-C, B=8192, five warmups plus 20 HIP-event samples, median of positions 9 and 10:

| shape | baseline | shift-fold | delta |
|---|---:|---:|---:|
| gate/up, M=34816 K=5120 | 12,910.419 us, 226.219 TOPS | 11,675.602 us, 250.144 TOPS | **+10.58%** |
| down, M=5120 K=17408 | 5,148.720 us, 283.622 TOPS | 5,179.470 us, 281.938 TOPS | **-0.59%** |
| two-shape aggregate | 18,059.139 us, 242.584 TOPS | 16,855.072 us, 259.914 TOPS | **+7.14%** |

For these two GEMMs across 64 layers, the standalone timing corresponds to 141.087 us/token baseline and 131.680 us/token candidate. This is not a daemon profile and must not be presented as total production GEMM time.

## Card-C decode gate

Same full-pow2 artifact and one-scale-per-token iu4 route, fresh daemon per arm, fp8 KV, graph enabled, pp512/ctx128/tg128 matrix, one warmup and three measured runs:

| arm | pp512 median | tg128@128 median | decode samples |
|---|---:|---:|---|
| control (`HIPFIRE_IU4_SHIFT_FOLDFREE=0`) | 2222.7 tok/s | 36.59365 tok/s | 36.60088, 36.59365, 36.56283 |
| candidate (`HIPFIRE_IU4_SHIFT_FOLDFREE=1`) | 2221.3 tok/s | 36.51656 tok/s | 36.52859, 36.51656, 36.50954 |

Decode delta is **-0.211%**, inside the required 1% neutral band. The pp512 row from this decode-first screen is -0.063%.

## Paired daemon prefill and TTFT

Card-C ABBA order, fresh daemon per arm, same full-pow2 artifact, fp8 KV, graph enabled, one warmup and three measured matrix runs:

| pair | pp512 control | pp512 candidate | delta | pp8192 control | pp8192 candidate | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2207.3 | 2195.9 | -0.516% | 2259.4 | 2256.5 | -0.128% |
| 2 | 2212.5 | 2203.7 | -0.398% | 2268.0 | 2262.5 | -0.243% |

Mean paired delta: **-0.457% at pp512** and **-0.185% at pp8192**. The predicted standalone uplift does not convert in the daemon.

The 5,909-token TTFT pair used eight measured runs after two warmups:

| arm | median TTFT | effective prefill |
|---|---:|---:|
| control | 2614.154 ms | 2260.389 tok/s |
| candidate | 2629.686 ms | 2247.038 tok/s |

Candidate delta is **+15.532 ms (+0.594%) latency** and -0.591% effective prefill throughput.

## In-daemon iu4 GEMM attribution

A kernel trace over two complete 8,192-token prefills (16,384 tokens total, replay chunk 512) summed the `full_set` and `full_add` iu4 families:

| arm | full-set total | full-add total | family us/token |
|---|---:|---:|---:|
| control | 3.096541 s | 1.630147 s | 288.494 |
| candidate | 3.138665 s | 1.574574 s | 287.673 |

The candidate saves only **0.821 us/token (0.285%)** in the complete profiled family. Its full-set work regresses while full-add improves, explaining why the standalone gate/up result is not representative of the production mix. These profiler numbers use a 512-row replay and are family device time, not wall-clock prefill time.

## Serve battery

The candidate completed the five built-in genres at a 512-token cap: **5/5 finish=stop**, runaway=0, empty=0, attractor=0, retrieval_miss=0. I read all decoded responses: executable sorted-list merge code, a correct 210-mile train calculation, a correct explanation of axial-tilt seasons, a coherent lighthouse/time-anomaly vignette, and five relevant refactoring practices.

## Final scope

QKVZA and QKV standalone extensions were intentionally not attempted after the paired pp8192 and TTFT verdict failed. The one-accumulator experiment is retained behind its explicit environment flag for research only.

## Reproduction pointers

- `pow2h-iu4-c24.log`: initial admissibility gate, 0.091338.
- `pow2-global-a-iu4-c24.log`: full-pow2 row-global route, 0.071194.
- `rownorm-opt-global-a-c24.log`: row-common MSE screen, 0.114720.
- `bench-baseline-corrected-pair.txt` and `bench-shiftfold-corrected-gate.txt`: gate/up pair.
- `bench-baseline-corrected-down-pair.txt` and `bench-shiftfold-corrected-down.txt`: down pair.
- `pow2-global-a-fp8v2-c24.log`: forced fp8v2 full-pow2 cell, 0.071194.
- `tg128-control-cardc.jsonl` and `tg128-candidate-cardc.jsonl`: paired decode-first daemon gate.
- `prefill-pair{1,2}-{control,candidate}.jsonl`: two ABBA daemon prefill pairs.
- `ttft-{control,candidate}-cardc.jsonl`: 5,909-token TTFT pair.
- `battery-candidate-cardc.json`: clean decoded-text battery.
- `prof-{control,candidate}/iu4free_results.db`: in-daemon kernel traces.
