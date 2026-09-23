# KILL — Unit B: iu4 GEMM 4th-block LDS trim (Raster3, ordinal 4, 2026-09-19)

Variant: W single-buffer (W1 aliases W0; A stays double-buffered;
DS/SZ ping-pong unchanged) + retire barrier before each publish
(4 barriers/half-block vs 2). Layout 20480 → 16384 B; host `lds_bytes`,
oracle/probe LDS updated atomically. Reverted after the screen.

## Exactness — bit-identical oracle PASS

`tmp_iu4_gfx12_oracle` on qwen3.8-27b.mq4-xt (LDS 16384): all 9 cases
`cpu_bitwise=OK (mism=0)`, `repeat_identical=OK` — gate/set, gate/add,
down/set, down/add, m48tail, n80cols, k256 set/add, m100n100.
`iu4-oracle-wsingle.log`. The slowdown below is barrier cost, not a bug.

## Timing — slower, no occupancy gain, KILL

`kx3_foldfree_probe` rig, both shapes, 4 interleaved rounds
(`kx3-interleaved.log`), ordinal 4:

| shape | base 20480 (4 runs) | W-single 16384 (4 runs) | Δ |
|---|---|---|---|
| gate_up (M4096/K5120/N34816) | 6591.7 / 6601.9 / 6612.9 / 6620.4 us | 6781.1 / 6792.7 / 6800.7 / 6804.4 us | **+2.8%** |
| residual (M4096/K17408/N5120) | 2988.2 / 2979.9 / 2989.8 / 2994.5 us | 3071.1 / 3079.9 / 3080.5 / 3088.4 us | **+3.1%** |

Occupancy (`OCC=1` oracle + probe `RESOURCE_OCC`): **3 blocks/CU both
before and after**. LDS was never the limiter — 16384 B admits 4/CU by
LDS math (4×16384 = 65536) but the kernel stays VGPR/wave-capped at 3,
so the 4 KB buys nothing while the two extra retire barriers cost ~3%.

Alternatives considered, not implemented: A-single (identical barrier
math, same cost); SZ fp16 pack (saves 1 KB only, adds fold VALU);
dead-stage aliasing (only ~2 KB dead at publish points — insufficient
for a 4 KB slab without hot-loop address changes). None can reach
≤16384 B for free, and occupancy would still cap at 3.

**Killed — 20480 B layout kept, all changes reverted.**
