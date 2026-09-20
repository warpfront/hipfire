# KILL.md — gfx11-parity item 2: double-buffered A slabs in the K16 iu4 GEMM

Status: KILLED on perf (exactness held). Reverted by 010e4233f.
Commit killed: e521443a9
Date: 2026-09-20. Card: hipx gfx1100. Model: qwen3.8-27b.mq4-xt, kv-mode q8.

## What was tried

Port of the gfx1201 two-slot protocol (c2eb81fcd) to
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`: the Xq
activation plane got a second LDS slot (even halves slot 0, odd halves
slot 1; each fill targets the slot freed two halves ago, no
readers-done barrier before its stores). Weight tile stayed
single-buffered. 4 -> 3 barriers per K group; odd-half prefetch loads
issue above the even half's WMMAs. Host `shared_mem` 30720 -> 39936 B
to match. Compile screen (--genco -O3, gfx1100/gfx1151): main entries
190 -> 186 VGPR, 0 spills.

## Exactness: PASS (not the cause)

KLD c2 (`eval_hipfire`, wt2 ref, kv q8, prefill scoring, max-chunks 2):
slice-mean KLD 0.055645, NLL/PPL identical to 6dp, output .bin
md5-identical to base. The two-slot HSACO compiled and ran live in the
eval (fresh hash in the kernel cache). Killed on perf, not exactness.

## Perf: LOSS, reproduced (the cause)

`hipfire bench --matrix --pp 512,2048,8192 --ctx 128 --tg 128
--spec off --runs 3 --warmups 1 --kv-mode q8`, HIPFIRE_GRAPH=1,
medians, tok/s:

| arm                | pp512  | pp2048 | pp8192 | tg128 |
|--------------------|--------|--------|--------|-------|
| single-slot (p1)   | 1752.3 | 1911.2 | 1773.5 | 49.00 |
| single-slot (p2)   | 1749.6 | 1910.8 | 1773.9 | 49.00 |
| two-slot (leg 1)   | 1685.9 | 1878.8 | 1744.0 | 49.07 |
| two-slot (leg 2)   | 1681.9 | 1872.8 | 1739.5 | 49.07 |

Two-slot vs single-slot means: pp512 **-3.8%**, pp2048 **-1.8%**,
pp8192 **-1.8%**, decode +0.1%. The two legs reproduce within 0.3%,
ruling out thermal drift (back-to-back). Run-to-run spread observed
elsewhere is ~+-1%; -1.8% to -3.8% in the same direction at all three
lengths is a real loss.

## Why

LDS 30720 -> 39936 B drops GEMM occupancy 2 -> 1 WG/CU on gfx11's
64 KB budget. The K16 kernel at ~186-190 VGPR is occupancy-sensitive;
one saved barrier per K group plus prefetch/compute adjacency does not
compensate for halved resident workgroups. (On gfx1201 the same trade
won because 3 WG/CU survived at 20480 B.)

## Do not retry without

A staging design that keeps 2 WG/CU (<= 32768 B total), e.g. halving
the weight tile residency or a finer-grained (non full-tile) double
buffer. Re-measure; the bar is pp8192 >= +1.5% in both pairs.
