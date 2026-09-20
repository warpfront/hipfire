# Slice 1 KILL — gfx1201 packet FA KT32×2 ring

The exact fp8 feed-only KT32×2 ring is rejected at the mandatory resource screen. No packet-kernel or launch-LDS change is admitted.

## Candidate screened

A minimal ring was compiled from base `5729b526f50f114955df3303a0e994ce882283fc` in worktree `wt-faring` on branch `gfx1201-fa-ring`.

The candidate used the frozen 41,216-byte map:

- slot0 K/V: `0..8191` / `8192..16383`
- slot1 K/V: `16384..24575` / `24576..32767`
- wave-private V transpose scratch: `32768..40959`
- slot0 SK/SV: `40960..41023` / `41024..41087`
- slot1 SK/SV: `41088..41151` / `41152..41215`

It preserved the two KT16 recurrences per KT32 stage and used a prime barrier followed by one ready/retire workgroup barrier per KT32 stage (two barrier pairs per normalized KT64, excluding prime/drain). Private V transpose scratch used `__builtin_amdgcn_wave_barrier()` between same-wave DS writes and reads. The first resource screen intentionally used eager V fill; the bounded two-row future-V overlap was not attempted after the hard resource gate failed.

Compile command:

```text
/opt/rocm/core/bin/hipcc -O3 --offload-arch=gfx1201 --cuda-device-only -S -gline-tables-only -Rpass-analysis=kernel-resource-usage -DHIPFIRE_FA2_FP8=1 -DHIPFIRE_FA2_KMODE=8 -DHIPFIRE_FA2_PACKET=1 kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip -o scratch-ring.s
```

Compile wall time was 1.25 s. Candidate source SHA-256 was `5b105cdc522d726e414598b69132ed3af3ab3523d93a3bad4dcbec38c93b4f40`; emitted assembly SHA-256 was `499cbb9178c51993bac26eb9c05880cddc225a636fda42facaac4889c6d1a959` (17,746 lines). The candidate source and assembly were removed after the kill decision.

## Resource result

| Entry | VGPR | SGPR | scratch B/lane | VGPR spill | SGPR spill | compiler waves/SIMD | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| direct packet | 256 | 31 | 432 | 152 | 0 | 5 | FAIL |
| packet partial | 249 | 32 | 0 | 0 | 0 | 5 | FAIL |

Required: at most 240 VGPR, zero scratch/spills, six compiler waves/SIMD. Both entries exceed the VGPR ceiling and lose the required six-wave compiler occupancy; the direct entry additionally spills 152 VGPRs and allocates 432 bytes scratch per lane. This is an immediate Slice 1 kill under the plan.

## Remaining contract gates

The hard compile gate stops the experiment before device execution. Therefore:

- 921-case bit-identical oracle: not run
- normalized packet delta/census: not run; the rejected source retained the required 128 fp8 WMMAs per KT64 and structural two ready/retire barrier pairs per KT64
- in-daemon attention A/B/A: not run
- pinned matrix pairs and decode rows: not run
- TTFT pairs: not run
- both-route c24: not run

No Card-A workload was launched. No daemon was started or modified. The production packet source and Rust launch LDS remain exactly at the base revision; there is no candidate commit to merge. Slice 2 is a separate decision and was not started.
