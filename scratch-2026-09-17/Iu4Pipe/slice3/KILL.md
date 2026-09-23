# Slice 3 — KILL

## Kill experiment

One nonterminal K128 transition used real `__builtin_amdgcn_s_barrier_signal(-1)` / `__builtin_amdgcn_s_barrier_wait(-1)` lowering. The next raw q+2 global loads were issued between the split barrier signal and wait while the final eight current-slab WMMAs executed.

Compile: 1.08 s.

## ISA

`gemm_mq4g256v2_residual_mmq_iu4_full_set` retained 32 IU4 WMMAs and two barrier pairs per K128. In the split transition:

- saved-bank LDS reads: `0xEBE0..0xEBF0`
- publication LDS stores: `0xEC1C`, `0xEC3C`
- publication completion: `s_wait_dscnt 0` at `0xEC44`
- split signal: `s_barrier_signal -1` at `0xEC48`
- q+2 raw global loads: `0xED40..0xED80`
- useful current-slab WMMAs after q+2 issue: eight at `0xED90..0xEDCC`
- split wait: `s_barrier_wait 0xffff` at `0xEDF4`
- next-stage first WMMA: `0xEE14`
- load-counter-zero or mixed-zero waits from q+2 issue through the next-stage first WMMA: 0
- DS-only waits in that protected interval: `s_wait_dscnt 3` at `0xED8C`, `s_wait_dscnt 2` at `0xEDA0`, `s_wait_dscnt 1` at `0xEE10`

Evidence: `kill-objdump.txt`.

## Resources

- VGPR: 202
- SGPR: 40 full SET/ADD, 43 wrapper
- scratch: 0 B/lane
- VGPR spills: 0
- SGPR spills: 0
- compiler occupancy: 7 waves/SIMD
- launch LDS: 20,480 B/block
- runtime occupancy: 3 blocks/CU

Evidence: `kill-resource.log`, `kill-kernel-aba.log`.

## Incremental kernel A/B/A

Baseline A was accepted Slice 2. Candidate B was the one-transition split experiment. Each arm used 5 warmups and 20 measured HIP events after one untimed build launch.

- gate-up M4096/K5120/N34816: A1 6840.510 us, B 6815.049 us, A2 6846.829 us, A mean 6843.670 us, candidate +0.420%
- residual M4096/K17408/N5120: A1 3011.690 us, B 2962.890 us, A2 3020.491 us, A mean 3016.091 us, candidate +1.796%

Evidence: `kill-kernel-aba.log`.

## Decision

KILL. The required incremental kernel gain was at least 2%; the gate-up anchor gained 0.420% and the residual anchor gained 1.796%. The experiment was abandoned before extension, exactness-oracle, serving-benchmark, or commit gates.
