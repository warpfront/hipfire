# Slice 2 — INVALID UNPINNED DAEMON COMPARISON

This report is retained only as invalidated evidence. Both CLI binaries resolved the same unpinned `target/release/daemon`, so its serving rows did not compare the baseline and candidate compute daemons. The corrected daemon-pinned comparison passed gate G and is recorded in `ACCEPT.md`.

## Kernel kill

Ordinal 2; one untimed post-JIT launch per arm, discarded preflight, then 5 warmups / 20 measured events per A/B/A arm.

| shape | baseline A1 (us) | candidate B (us) | baseline A2 (us) | baseline mean (us) | speedup | A drift |
|---|---:|---:|---:|---:|---:|---:|
| gate/up M4096 K5120 N34816 | 7101.385 | 6856.243 | 7112.504 | 7106.945 | +3.657% | 0.156% |
| residual M4096 K17408 N5120 | 3174.478 | 3030.876 | 3168.878 | 3171.678 | +4.646% | 0.177% |

Short kill: PASS (both anchors >=2%, both A brackets <1%). Evidence: `kernel-aba-stable.log`.

## ISA / resources

`llvm-objdump` full-set steady K128: 32 `v_wmma_i32_16x16x32_iu4`; 2 barrier signal/wait pairs. Pair 1: `0xE994/0xE9E4`; pair 2: `0xEC98/0xEC9C`; prologue: `0xE180/0xE184`. LOAD waits per steady K128: `loadcnt 0` before pair 1; `loadcnt 3,2,1,0` before pair 2.

Full SET/ADD: 202 VGPR, 40 SGPR, 0 VGPR spills, 0 SGPR spills, 0 scratch bytes/lane, 7 waves/SIMD. Dynamic LDS 20,480 bytes. Runtime SET/ADD occupancy 3 blocks/CU.

Evidence: `objdump.txt`, `resource.log`, `kernel-aba-stable.log`.

## Serving route proof

One-pass `rocprofv3 --kernel-trace --runtime-trace` of the candidate bench recorded PID 3345076. Kernel correlation 4038 was `gemm_mq4g256v2_residual_mmq_iu4_full_set`, grid 20480 x 4, block 256, 208 allocated VGPR, and its matching `hipModuleLaunchKernel` argument was `sharedMemBytes=20480`.

The trace code-object table maps IU4 kernel IDs 21/22 to code object 4 and URI cache hash `62be8413199d7e5c`. The routed cache bundle SHA-256 is `0fcc4abcad69d799256a8fdb5d8642283d890b2f5c37761d60c84a7fd6ffd50f`. Its extracted device object SHA-256 is `58e7114e9e564aa1d45eac59771ca502660551c98f3179c50e3eadddf00e715d`; disassembly has the same two steady K128 barrier pairs at `0xE994/0xE9E4` and `0xEC98/0xEC9C`.

Evidence: `rocprof-route-full.log`, `rocprof-candidate/routed-daemon-full_kernel_trace.csv`, `rocprof-candidate/routed-daemon-full_hip_api_trace.csv`, `rocprof-candidate/routed-daemon-full_results.db`, `route-cache-device.hsaco`, `route-cache-objdump.txt`.

## Exactness

Device-resident shipped-vs-candidate comparison on the same resident weights, Xq, and Y0: 0 full-output bit mismatches, 0 repeat mismatches, 0 canary mismatches, 0 CPU-reference mismatches for 9/9: real gate SET/ADD, real down SET/ADD, M48 tail SET, N80 tail SET, K256 SET/seeded ADD, M100/N100 K512 SET.

Evidence: `oracle-full.log`.

## Serving

Binary MD5: baseline `19c832084440f7142181a54ea1607c8f`; candidate `a338e40e625a0ab21536048288046ad6`.

Throwaway pp8192 warm decode: baseline 36.516, candidate 36.483. The earlier candidate row with decode 28.988 was discarded as invalid slow-DPM evidence.

### Valid pair 1

| row | baseline | candidate | delta |
|---|---:|---:|---:|
| pp512 | 2345.30 | 2343.40 | -0.081% |
| pp2048 | 2419.20 | 2422.70 | +0.145% |
| pp8192 | 2323.00 | 2324.00 | +0.043% |
| pp32768 | 1937.20 | 1936.20 | -0.052% |
| tg64@ctx128 | 36.4954 | 36.4424 | -0.145% |

### Valid pair 2

| row | baseline | candidate | delta |
|---|---:|---:|---:|
| pp512 | 2344.40 | 2348.20 | +0.162% |
| pp2048 | 2422.60 | 2424.30 | +0.070% |
| pp8192 | 2324.80 | 2329.10 | +0.185% |
| pp32768 | 1936.60 | 1938.60 | +0.103% |
| tg64@ctx128 | 36.4774 | 36.5045 | +0.074% |

### Mean of valid pairs

| row | baseline | candidate | delta |
|---|---:|---:|---:|
| pp512 | 2344.85 | 2345.80 | +0.041% |
| pp2048 | 2420.90 | 2423.50 | +0.107% |
| pp8192 | 2323.90 | 2326.55 | +0.114% |
| pp32768 | 1936.90 | 1937.40 | +0.026% |
| tg64@ctx128 | 36.4864 | 36.4735 | -0.036% |

Both valid decode columns are >=36.4 and each baseline/candidate decode pair differs by <1%. Serving result: FAIL; pp8192 required >=+1.5%, measured +0.043% and +0.185% (+0.114% mean).

Evidence: `warm-baseline.log`, `warm-candidate.log`, `valid-B1.log`, `valid-C1.log`, `valid-B2.log`, `valid-C2.log`.

## Outcome

INVALIDATED. Superseded by `ACCEPT.md`.
