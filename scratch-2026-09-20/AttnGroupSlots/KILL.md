# KILL: gfx1201 Q384 group-private LDS slots

## Named barriers are not available

The Q384 prototype assigned named barrier IDs 1, 2, and 3 to three independent eight-wave groups, each with a private 16 KiB K/V slot. This is architecturally unsupported on gfx1201: LLVM's GFX12 execution-synchronization specification lists barrier IDs `[1,16]` (named barrier objects) as GFX12.5-only; gfx1201 has only the ordinary workgroup barrier (`-1`) available to shaders.

The compiler nevertheless accepted `__builtin_amdgcn_s_barrier_init/join/signal/wait` and emitted three named-barrier resources. At runtime those waits did not synchronize the group-private LDS slots. The exact 921-case oracle failed all 15 nonempty batch-512 direct cases. In repeated `batch=512, ctx=1, mode=0` runs, mismatched outputs moved among every one of the 8 workgroups and all 3 groups (770,064, 97,680, and 596,352 elements on successive variants/runs; maximum absolute error 42.191406), proving a nondeterministic barrier race rather than a fixed row/head/column mapping bug. Correcting the expected count from 256 threads to 8 waves and adding load/store/DS waits did not repair it. Do not retry hardware named barriers on gfx1201.

## Exact software replacement

The measured candidate replaced each named object with one generation-tagged LDS arrival counter, a lane-zero LDS atomic increment, lane-zero `ds_load_b32` polling with `s_sleep 1`, and workgroup/local release-acquire fences. The Q384 body retained three private 16 KiB K/V slots, three 128-byte scale headers, and the incumbent Q128 path for non-prefill launches. The runtime selected Q384 only for 512-row prefill chunks.

The device oracle passed all **921/921** cases with `input_diffs=[0,0,0,0]`. The compiled direct entry uses 237 VGPRs and 31 SGPRs, has zero scratch and spills, contains no named-barrier resource, and launches with 49,552 B dynamic LDS. The rejected patch is `software-group-slots.patch` (SHA-256 `6cbfc7fdf2fc8ecaefc2089be3571aec04625e2111d1c8613cced673c954b685`); its compiler assembly is `software.s` (SHA-256 `184fe9c67deafed26ae1139456eeefbd438c2ddc37aa9fe40a8068755c9a1af1`).

## ATT and real-daemon kernel timing

The decoded candidate ATT file is `att-candidate-replay/stats_ui_output_agent_43138_dispatch_588.csv` (SHA-256 `1549ca4dbdd6eaa0eeed408888309d8a323a981b609ca23ffe6f5f1504a8bebf`). Over the KT32 steady-state PC interval `[54700,61328]`, `sum(Latency) + sum(Idle)` is 153,411 samples. The four complete software group-sync intervals contribute 30,094 latency samples, or **19.62%** of that denominator. Including idle attributed to those intervals would make them 29.82%. This is diagnostic rather than an acceptance gate, but it is worse than the incumbent hardware-barrier share of 10.81% and does not approach the requested 5%.

A paired rocprof kernel trace of the real 5,909-token daemon request showed that the Q384 prefill kernel itself did improve: 176 prefill attention dispatches fell from **128.837760 ms** to **112.101639 ms**, a **12.99% reduction**. The 16 decode dispatches remained on the unchanged Q128 entry; their one-shot sum moved from 11.178912 ms to 12.615336 ms and is scheduling noise, not a Q384 decode path.

## Required paired acceptance

The exact matrix command was run in alternating order (baseline/candidate, then candidate/baseline), with three measured runs and one warmup:

| Pair | pp512 rate delta | pp8192 rate delta | Decode rate delta |
|---|---:|---:|---:|
| 1 | -0.543% | **-1.377%** | -0.097% |
| 2 | +0.333% | **-0.640%** | **-21.112%** |

The candidate regressed pp8192 in both pairs, exceeded the allowed 1% row regression in pair 1, and the second pair's decode measurement was not within 1%. The unchanged decode code path makes the latter a noisy run, but it still cannot supply passing evidence.

The required eight-run/two-warmup TTFT command was also alternated:

| Pair | Baseline TTFT | Candidate TTFT | TTFT latency delta | Prefill-rate delta |
|---|---:|---:|---:|---:|
| 1 | 1,675.602 ms | 1,706.031 ms | **+1.816%** | -1.784% |
| 2 | 1,725.743 ms | 1,670.450 ms | -3.204% | +3.310% |

The two-arm arithmetic means are 1,700.673 ms baseline and 1,688.240 ms candidate, only a **0.731% TTFT reduction**. That misses the required at-least-1.5% gain, and pair 1 individually regressed.

## Final disposition

**KILL.** The group-private Q384 kernel is exact and its own real-daemon attention sum is 12.99% better, but the improvement does not transfer reliably to end-to-end serving. It fails the TTFT gain threshold and matrix non-regression gate. No KLD or five-run battery is warranted because those are shipment gates after performance acceptance. The source and runtime routing are restored to base; no candidate code is shipped.
