# Packet attention ATT and the Q384 inversion

## Capture

This is an in-daemon capture of the incumbent packet kernel at source revision `c5f56bd86603c993ba498b5e3a3f5e64edfd59ae` on card-A (`GPU-9eb7aeda51c88ffd`). The source is `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`, SHA-256 `bc7b00bbd68600804f8540105db57a21bb5458147404bdd7949fc6520e95f3c2`. The loaded cache object was `attention_fp8_e4m3_fa2_gqa_packet_gfx1201.fcab03e8e867d730.hsaco`, SHA-256 `96a82bcf94b61df4c2515c71487e9aaeb56bc4b5751e5462c3347e68d169aaa9`.

The GPU and daemon were first warmed with one untraced client-TTFT request. The traced request used the 5,909-token `benchmarks/prompts/ttft_5900.txt`, fp8 KV, the admitted 8192-token chunk, GDN on, and `HIPFIRE_GRAPH=1`. `HIPFIRE_DAEMON_BIN` pointed to a wrapper around:

```text
rocprofv3 --att --att-target-cu 0 \
  --kernel-include-regex 'attention_fp8_e4m3_fa2_gqa_packet' -- daemon
```

The one-shot benchmark and a normal `hipfire serve` plus one curl both terminated the profiler before its decoder flushed. `slice0-daemon-stdin.bin` therefore records the exact five daemon protocol records from the one-curl serve request; replaying those records into the same daemon gave it a normal EOF and retained the same one-request workload. The decoded result is dispatch 588, agent 22763, code-object id 14:

```text
slice0-att-replay/stats_ui_output_agent_22763_dispatch_588.csv
```

The decoded CSV SHA-256 is `4d5cdb2236942f9908666fb3a82cf58e3321706412096e943c70c56843849372`. The request's daemon receipt reports 5,909 prefill tokens, 2,061.9 ms model prefill, 2,865.7 tok/s, and 27.4 decode tok/s. ATT timing is diagnostic and is not used as an acceptance timing.

## KT64 steady-state boundary

The steady packet-tile loop is the same 1,020-row ISA interval as the earlier packet capture: PC `[0x9788, 0xb058]` (decimal `[38792, 45144]`). It starts at the tile-latch load wait/barrier and ends at the branch returning from the fourth softmax/PV recurrence. The target-CU sample observed 28 tile-wave iterations, 104 four-subtile recurrence iterations (rather than 112 because of the causal tail), and 3,328 WMMA issues.

The accounting convention matches `kx/IU4ATT/att_summary.md`: each instruction's `Latency` belongs to exactly one mutually exclusive class, decoder `Idle` remains unattributed, and the denominator is `sum(Latency) + sum(Idle)`. A mixed `s_wait_loadcnt_dscnt` is not guessed into either pure wait class.

| Class | Samples | KT64-loop share |
|---|---:|---:|
| Pure global-load wait | 66,205 | 13.51% |
| Pure LDS wait | 39,035 | 7.96% |
| Mixed global-load + LDS wait | 66,200 | 13.51% |
| Barrier signal/wait | 52,970 | 10.81% |
| WMMA issue | 31,137 | 6.35% |
| Non-WMMA VALU | 57,243 | 11.68% |
| LDS instruction issue | 12,805 | 2.61% |
| Global-memory instruction issue | 5,792 | 1.18% |
| SALU/control/fence | 11,444 | 2.33% |
| Decoder `Idle`, unattributed | 147,300 | 30.05% |
| **Total accounted wave time** | **490,131** | **100.00%** |

Pure plus mixed counter waits occupy **34.98%** of accounted loop time. Barrier latency adds **10.81%**, while WMMA issue is only **6.35%**. Idle is the largest single row at **30.05%**.

## Explicit issue stalls

The loop contains 191,995 explicit `Stall` samples, 56.00% of instruction latency and 39.17% of the latency-plus-idle denominator.

| Class | Stall samples | Stall share |
|---|---:|---:|
| Pure global-load wait | 65,009 | 33.86% |
| Pure LDS wait | 35,107 | 18.29% |
| Mixed global-load + LDS wait | 65,340 | 34.03% |
| Barrier signal/wait | 0 | 0.00% |
| WMMA issue | 4,513 | 2.35% |
| Non-WMMA VALU | 14,588 | 7.60% |
| LDS instruction issue | 3,733 | 1.94% |
| Global-memory instruction issue | 1,628 | 0.85% |
| SALU/control/fence | 2,077 | 1.08% |

Barrier waiting is represented in decoded `Latency`, not `Stall`; looking only at `Stall` would erase its 10.81% contribution.

The dominant instruction is PC `0xab8c`, `s_wait_loadcnt_dscnt 0x301`: 64,765 stall samples, **33.73%** of all explicit stalls and **13.21%** of accounted loop time. It is the QK/softmax recurrence's mixed dependency wait, not the K/V publication barrier. The leading pure-load waits are the K/V fills: PC `0x9c04` contributes 7,980 stalls, PC `0x9bc4` 6,589, and PC `0x9878` 6,502.

## Dynamic footprint and resources

The loop emitted 62,535 dynamic instruction hits in the target-CU sample.

| Measure | Value |
|---|---:|
| Instructions / observed tile-wave | 2,233.393 |
| Instructions / query-head row / observed KT64 | 139.587 |
| Instructions / query-head row / key | 2.18105 |
| WMMA / observed tile-wave | 118.857 |
| Global-memory issues / observed tile-wave | 75.857 |
| LDS issues / observed tile-wave | 196.000 |

These are real causal-tail averages, not the static interior path. The exact full-interior census remains 2,798 packets on each of six common waves and 2,820 on each of two header-copy waves, or **2,803.5 weighted packets/wave/KT64**. That is **16 packets above** the assignment's 2,787.5 snapshot; no class is silently adjusted to force the old total. The decoded executable has 1,906 instruction rows for the complete direct symbol, matching the pinned compile receipt.

The direct and partial entries compile with 237 live VGPRs, 29/30 SGPRs, no scratch or spills, and six compiler-reported waves/SIMD. Executable allocation rounds the direct entry to 240 VGPRs. Launch geometry is block 256 with 49,408 B dynamic LDS; the occupancy query admits one block/CU, eight resident waves/CU, two waves/SIMD.

## Why the Q384 hybrid inverted

The killed Q384 hybrid is exact across all 921 oracle cases. Its wide body changed to 768 threads, 24 waves, KT48 and 37,120 B LDS, with 216 live VGPRs and no spills. The standalone body sum improved **10.361%**, but the real pp8192 daemon trace moved the 32 full-attention dispatches from 397.130265 ms to 439.812244 ms: **+10.748%**. Its two pinned pp8192 serving pairs were **-3.153%** and **-0.312%**, and median TTFT regressed 1.059%. The variant is therefore already **KILL** and is not repeated here.

Its preserved static object census reports baseline direct/partial counts of 1,906/1,574 instructions and hybrid-candidate counts of 1,924/1,610. The direct class counts move from 20 WMMA, 147 LDS issues, 94 global issues, 294 waits, six barriers and 39 branches to 20, 142, 89, 271, six and 35 respectively; that small static reduction did not predict the daemon regression.

The ATT result explains why the standalone result was not transferable. This kernel is not dominated by counted arithmetic or WMMA throughput: only 6.35% of accounted loop time is WMMA, while counter waits, barrier latency and unattributed idle total **75.84%**. The single recurrence wait at `0xab8c` alone costs twice the entire WMMA share. Q384 changed ownership, workgroup size, grid granularity and resident-wave shape; it did not establish that the recurrence's global/LDS dependency latency fell in the daemon. A hot, repeated standalone body can reward its wider workgroup and lower launch/body overhead, while omitting the daemon's real run partition, causal tails, producer-created cache state and interleaving with the other layer kernels. Packet savings or higher nominal resident waves are therefore not a time model for this workload.

The measured daemon attention sum supplies the decisive missing fact: under the actual producer inputs and scheduling context, the wide kernel itself took 10.748% longer. Without a decoded candidate ATT trace it would be speculative to assign that regression to one particular cache or arbiter mechanism; the exact conclusion is narrower and sufficient: the body rig optimized a throughput proxy, whereas the daemon is latency/feed dominated at the mixed load/LDS wait and idle boundaries. This is the third observed rig/daemon inversion and is why every later slice is gated first on the 921-case oracle and then on an in-daemon attention-kernel sum before paired serving acceptance.

## Common-gauge QK replay + BF16 V

The second experimental slice retained the incumbent Q128/KT64 launch shape but changed the packet body to two QK passes. The first pass computes only the row maximum, followed by one XOR-16 reduction over the whole scan. The replay pass computes the fixed-gauge probabilities and denominator while scalar-coalesced code reads dequantize V directly into fragment-major BF16 LDS. PV uses BF16 WMMA with FP32 output accumulators. This removes online `m/l/wmax/rho` state and per-subtile output rescaling at the cost of replaying QK and reducing the denominator only once at the end.

The rejected source is preserved as `common-gauge-killed.patch` (SHA-256 `15a2a4f09d7ee268a7a44f5af246381bf66b09abfc19b569d3c719ac53a8229b`). The compiled packet is `common-gauge.hsaco` (SHA-256 `7f7ec578ac9843f1c41f667a4274269c60b6cc1f3da6faa5762dfe7f933b8fb4`).

### Resources and static census

The direct candidate compiles at 182 VGPRs and 29 SGPRs with no scratch or spills; the partial entry is 230 VGPRs and 34 SGPRs, also with no scratch or spills. Rocprof reports an actual direct allocation of 184 VGPRs. Launch geometry is unchanged at block 256 and 49,408 B dynamic LDS, so the occupancy query remains one block/CU, eight resident waves/CU and two waves/SIMD. The direct-symbol assembly census drops from 1,905 recognized instruction lines and 12,216 code bytes to 1,812 lines and 10,752 bytes, reductions of 4.88% and 11.98%.

### Oracle and KLD

The exact 921-case device oracle completed with all Q/K/V/P input-diff and guard counters zero, but 694 cases differed byte-for-byte from the incumbent. That is expected for this deliberately inexact BF16-probability/BF16-PV experiment, but it means the exact oracle alone cannot admit it. The full log is `common-gauge-oracle.log` (SHA-256 `6ccadc6680bf917b60696eed6bb9007eaf521c529801a9811ced55cd322f62ad`).

KLD then produced:

| Arm | Chunks | KLD | Gate |
|---|---:|---:|---:|
| IU4 | 1 | 0.045375 | diagnostic |
| IU4 | 24 | 0.080163 | PASS, at most 0.10 |
| FP8v2 (`HIPFIRE_IU4_PREFILL=0`) | 24 | 0.050018 | **FAIL**, at most 0.05 |

### Standalone packet timing

The pinned A/candidate/A body rig used five warmups and 25 samples at every context. Positive latency delta means the candidate is slower.

| Context | Baseline midpoint, us | Candidate, us | Candidate latency delta |
|---|---:|---:|---:|
| 96 | 53.201 | 131.162 | +146.540% |
| 257 | 75.801 | 234.323 | +209.129% |
| 1,536 | 298.684 | 1,245.498 | +316.995% |
| 3,072 | 582.948 | 2,580.717 | +342.701% |
| 6,144 | 1,139.436 | 5,260.035 | +361.635% |
| 12,288 | 2,252.592 | 10,646.032 | +372.613% |
| 32,768 | 6,926.979 | 35,889.111 | +418.106% |
| **Sum** | **11,329.641** | **55,986.878** | **+394.163%** |

The stable raw run is `pinned-body-timing.jsonl` (SHA-256 `3f0285273e8cd46c3f386edd5c73878d2baadac845d8cce4422e8f1fc44d3b6a`). Replaying QK and scalar direct-to-fragment V staging outweighed the eliminated online rescale path.

### In-daemon attention sum

Warm untraced requests preceded each rocprof trace. Filtering the real TTFT request to `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` and the prefill grid `(6144,4,16)` gives 32 full-attention dispatches in each arm:

| Arm | Prefill sum | Median dispatch | Decode sum |
|---|---:|---:|---:|
| Baseline | 398.996827 ms | 12,493.170 us | 0.742088 ms |
| Common gauge | 1,534.503167 ms | 48,282.488 us | 2.293755 ms |
| Delta | **+284.590%** | | **+209.095%** |

The trace CSVs are `rocprof-base/trace-base_kernel_trace.csv` and `rocprof-candidate/trace-candidate_kernel_trace.csv`. The packet itself is already a decisive KILL before serving noise is considered.

### Paired daemon matrix and TTFT

The exact required matrix was run in alternating order, baseline/candidate then candidate/baseline, with three measured runs and one warmup:

| Pair | pp512 delta | pp8192 delta | Decode delta |
|---|---:|---:|---:|
| 1 | -0.602% | **-14.724%** | -0.055% |
| 2 | -0.517% | **-14.702%** | +0.008% |

Both pp8192 pairs fail the required at-least-1.5% improvement. Decode is effectively unchanged.

The required eight-run/two-warmup TTFT pairs were also alternated:

| Pair | Baseline TTFT | Candidate TTFT | TTFT delta | Prefill-rate delta |
|---|---:|---:|---:|---:|
| 1 | 2,096.960 ms | 2,430.305 ms | **+15.897%** | -13.716% |
| 2 | 2,111.905 ms | 2,430.542 ms | **+15.088%** | -13.110% |

Both TTFT pairs fail the at-most-1% regression gate. The eight raw matrix and TTFT JSON files are retained beside this report.

## Final disposition

**KILL both slices.** The Q384/KT48 hybrid was already rejected by the daemon despite its standalone win, and the common-gauge/BF16-V slice fails the FP8v2 KLD boundary, standalone body timing, in-daemon attention sum, both pp8192 pairs and both TTFT pairs. There is no winning evidence to integrate, so Slice 3 is intentionally empty. The kernel source was restored to the `c5f56bd86603c993ba498b5e3a3f5e64edfd59ae` baseline; no candidate commit is created.
