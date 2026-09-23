# Q384 / KT48 packet attention diagnosis

Verdict: **KILL**. The wide body is real and exact, but it does not pass the required serving gate. A bounded tail-geometry fix improved the measured 5,909-token attention sum, yet pp8192 remained slower in both paired matrix runs. No source commit is made.

## Captures

Card-A (`GPU-9eb7aeda51c88ffd`) and base `5729b526f` were used throughout. Both daemons were warmed before measurement. The in-daemon ATT captures replay the exact five protocol records from the 5,909-token TTFT request under:

```text
rocprofv3 --att --att-target-cu 0 \
  --kernel-include-regex 'attention_fp8_e4m3_fa2_gqa_packet' -- daemon
```

Decoded inputs:

- baseline: `scratch-2026-09-20/Q384/att-base/stats_ui_output_agent_41068_dispatch_588.csv`
- all-wide candidate: `scratch-2026-09-20/Q384/att-candidate/stats_ui_output_agent_13037_dispatch_588.csv`

The same accounting convention was applied to baseline PC interval `[0x9788, 0xb058]` and candidate interval `[0x9794, 0xaf1c]`: each instruction's `Latency` belongs to one class, decoder `Idle` is separate, and the denominator is latency plus idle.

| ATT class | Baseline samples | Baseline share | Q384 samples | Q384 share | Share delta |
|---|---:|---:|---:|---:|---:|
| Pure global-load wait | 69,146 | 14.51% | 16,560 | 4.37% | -10.14 pp |
| Pure LDS wait | 40,444 | 8.49% | 22,612 | 5.96% | -2.53 pp |
| Mixed global-load + LDS wait | 62,499 | 13.11% | 51,889 | 13.68% | +0.56 pp |
| Barrier signal/wait | 51,196 | 10.74% | 121,093 | 31.92% | **+21.18 pp** |
| WMMA issue | 30,509 | 6.40% | 13,654 | 3.60% | -2.80 pp |
| Non-WMMA VALU | 58,394 | 12.25% | 33,866 | 8.93% | -3.33 pp |
| LDS instruction issue | 9,181 | 1.93% | 1,799 | 0.47% | -1.45 pp |
| Global-memory instruction issue | 5,232 | 1.10% | 1,877 | 0.49% | -0.60 pp |
| SALU/control/fence | 10,816 | 2.27% | 5,164 | 1.36% | -0.91 pp |
| Decoder idle | 139,149 | 29.20% | 110,852 | 29.22% | +0.02 pp |
| **Total accounted** | **476,566** | **100%** | **379,366** | **100%** | |

The class that grows is barrier latency. The target-CU sample observed 26 baseline tile-wave iterations and 18 Q384 tile-wave iterations; barrier latency per observed tile-wave rose from 1,969.08 to 6,727.39 samples, **+241.65%**. Q384's 24-wave workgroup makes six loader waves publish data for eighteen consumer waves at each workgroup barrier; the narrow body synchronizes only eight waves.

## PC `0xab8c`

Baseline `0xab8c` is `s_wait_loadcnt_dscnt 0x301`. The immediately preceding instructions are two `ds_load_b128` operations from the current subtile's K8 LDS plane and four `global_load_b64` operations from the current row's Q scratch; the next instruction is the first QK WMMA. It maps to the current recurrence's `kv = *kp`, `qf0/1 = qd0/1[kg]`, and first WMMA sequence in `fa2_stageb_packet_body`, not previous-tile PV/O rescaling, next-tile K fill, or the XOR-16 exchange.

The all-wide analogues are `0xaa28` and `0xaa64`. The dominant wait itself rose from 156.31 to 201.99 samples/hit (+29.22%). Summing the two mixed waits feeding the four QK WMMAs gives 157.53 samples/inner iteration at baseline versus 268.65 for Q384, **+70.53%, not 2x**. Its loop share stays close (13.11% to 13.68%); it is secondary to the barrier expansion.

## In-situ occupancy and tail quantization

Rocprof dispatch columns and dynamic-LDS arithmetic show:

| Body | Block | Waves/WG | Dynamic LDS | VGPR | LDS-limited WGs/CU | Resident waves/CU |
|---|---:|---:|---:|---:|---:|---:|
| Q128 / KT64 | 256 | 8 | 49,408 B | 240 | `floor(65,536/49,408)=1` | 8 |
| Q384 / KT48 | 768 | 24 | 37,120 B | 216 | `floor(65,536/37,120)=1` | 24 |

The baseline object also reports a six-wave/SIMD VGPR ceiling, but the actual daemon launch is LDS/workgroup limited to one 8-wave WG/CU. Q384 is likewise one WG/CU, now occupying 24 waves.

Attention-S1 dispatches one causal chunk at a time. For each full 512-query chunk, baseline launches `24 x 4 = 96` WGs while Q384 launches `8 x 4 = 32`; Q384 therefore exposes work to only 32 of 64 CUs. For the 277-query remainder of the 5,909-token request, Q128 has `ceil(277/21.33) x 4 = 52` valid WGs (81.25% maximum CU coverage), while Q384 has `ceil(277/64) x 4 = 20` (31.25%), leaving 44 CUs without a tail WG. This is the causal-tail mechanism.

The traced 5,909-token attention sums confirm it:

| Segment | Baseline | All-wide Q384 | Delta |
|---|---:|---:|---:|
| Eleven full chunks, 176 layer dispatches | 126.625269 ms | 113.066073 ms | **-10.708%** |
| 277-query tail, 16 layer dispatches | 11.071461 ms | 17.015972 ms | **+53.692%** |
| Total | 137.696730 ms | 130.082045 ms | **-5.530%** |

## Tail-route prototype

The bounded fix kept Q384/KT48 for full 512-query chunks and routed partial attention-S1 chunks to the exact Q128/KT64 symbol. The 921-case device oracle passed byte-identically:

```text
SUMMARY cases=921 failed=0 input_diffs(Q/K/V/P)=[0, 0, 0, 0]
```

Its traced attention sum was 125.611734 ms, **-8.777%** versus baseline and -3.437% versus all-wide Q384. The full-chunk portion remained 113.280144 ms; the narrow tail was 12.331590 ms.

## Required serving gate

Command: `hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`.

| Pair | Shape | Baseline | Tail-route candidate | Candidate delta |
|---:|---|---:|---:|---:|
| 1 | pp512 tok/s | 2914.6 | 2888.2 | -0.906% |
| 1 | pp8192 tok/s | 2967.4 | 2923.0 | **-1.496%** |
| 1 | decode tok/s | 36.4607 | 36.3939 | -0.183% |
| 2 | pp512 tok/s | 2873.2 | 2869.8 | -0.118% |
| 2 | pp8192 tok/s | 2922.1 | 2909.1 | **-0.445%** |
| 2 | decode tok/s | 36.3357 | 36.3547 | +0.052% |

Both pp8192 pairs miss the required +1.5% gate.

TTFT command: `hipfire bench qwen3.8:27b-mq4-xt --ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --json`.

| Pair | Baseline TTFT | Candidate TTFT | TTFT delta | Prefill-rate delta |
|---:|---:|---:|---:|---:|
| 1 | 2073.219 ms | 2076.316 ms | +0.149% | -0.149% |
| 2 | 2090.978 ms | 2077.034 ms | -0.667% | +0.671% |

TTFT passes the at-most-1% regression bound, but it cannot rescue the failed pp8192 gate. The wide body and narrow-tail routing are therefore not committed.

## Named-barrier follow-up

Verdict: **KILL**. HIP 7.15 / clang 23 does lower gfx12 scalar named-barrier builtins, and the generated Q384 object meets the static resource gate, but the resulting synchronization is not correct on gfx1201. The candidate was removed after the 921-case oracle failed; no daemon profiling or serving benchmark was run.

### Toolchain and ISA screen

A typed `__amdgpu_named_workgroup_barrier_t` in LDS is required. With that type, `__builtin_amdgcn_s_barrier_init`, `__builtin_amdgcn_s_barrier_join`, `__builtin_amdgcn_s_barrier_signal_var`, and explicit `__builtin_amdgcn_s_barrier_wait` calls lower to:

```text
s_barrier_init m0                  # IDs 1, 2, 3
s_barrier_join m0                  # IDs 1, 2, 3
s_barrier_signal 1/2/3
s_barrier_wait 1/2/3
```

The candidate metadata is `num_named_barrier=3`, 216 VGPR, zero SGPR/VGPR spills, and zero private segment. The complete compiler output is `scratch-2026-09-20/Q384/named-barrier-isa.s`.

The original wide K loop has three full-workgroup `s_barrier_signal -1` / `s_barrier_wait -1` pairs per KT48 iteration. The group-scoped candidate replaces the K-loop pairs with three independent ID streams, one per eight-wave group; only the one-time initialization rendezvous remains a full-workgroup barrier.

### Correctness result

Each eight-wave group received a disjoint K/V/scale LDS plane and staged the same KT16 subtile. Six named-barrier lifecycle variants were exercised: participant counts 8 and 256, join per subtile and persistent join, implicit/default wait and explicit waits on IDs 1/2/3. Every named-barrier variant produced the same result:

```text
SUMMARY cases=921 failed=15 input_diffs(Q/K/V/P)=[0, 0, 0, 0]
```

All 15 failures are direct `batch=512` wide cases in modes 0/1/2, with roughly 0.48M to 3.14M output differences. All 906 narrow/non-direct cases pass. The mode-3 wide cases pass only because masking removes the affected values. A grouped-LDS control using a full-workgroup barrier passes 920/921, while the original Q384/tail prototype passes 921/921, isolating the failure to the named synchronization rather than the wide arithmetic or most of the grouped LDS layout.

The trial matrix and raw-output artifact references are in `scratch-2026-09-20/Q384/named-barrier-oracle.txt`. Because correctness fails before the performance gates, ATT barrier share, in-daemon attention sum, paired matrix rows, TTFT pairs, and battery runs are intentionally absent.
