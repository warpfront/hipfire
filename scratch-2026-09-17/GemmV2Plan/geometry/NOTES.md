# Geometry: balanced-aspect v2 variants (gfx1201, 2026-09-18)

Branch `gfx12-fp8-v2-tile`, worktree `wt-gemmv2`. Host selector
`HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM` (`256x64` default, `128x128`, `64x256`,
`128x64` 4-wave control) via one `fp8_v2_geom()` tuple feeding SRC symbol,
block, dynamic LDS and grid in all four launch arms. Kernel TU delta is
preprocessor asserts + comments only (default-geometry codegen untouched).

## Staging-traffic model (correction to the ticket premise)

Per WG staged bytes = `(BM+BN)*K` (A: BMx128 + W: BNx128 per 128-K half,
K/128 halves; S/sz add ~3-6%). Launch total = WGs x (BM+BN) x K,
WGs = ceil(M/BN) x ceil(N/BM), i.e. ~MNK x (1/BM + 1/BN).

| variant | traffic ratio vs 256x64 |
|---|---|
| 128x128/8w | 0.80x |
| 64x256/8w | 1.00x (no staging saving) |
| 128x64/4w control | 1.20x |

The shared-facts estimate (~0.53 GB for 128x128) forgot token-tiling on the
A side (272 row-tiles x 4 token-tiles = 1088 WGs, not 272). Gate_up N=512:
1.783 / 1.426 / 1.783 / 2.139 GB. Only 128x128 reduces staging work (-20%).

## Metadata (from hsaco radiowave.json, current blobs)

All wave32, 0 VGPR/SGPR spill, 0 private scratch, 2 WG/CU everywhere.
VGPR: 256x64: 192/196/196/192; 128x128: 201 x4; 64x256: 204 x4;
128x64w4: 199 x4 (gate_up/qkv/qkvza/residual order within each row).
Gate ceiling is 256: pass (199-204 over the 192 target is allocator
headroom, same story as K2's 192-196). LDS: 24576/19968/25344/14848.

Default 256x64 reproduces K2 pins bit-exactly (2adf7620c1d722e1,
09726b10ba1eb1b2, 082a7b24def26f75, 373cb087fb519b9e). All three variants
bit-identical to the default on all 8 cases: fold order unchanged, no
bound re-derivation needed.

## Sweeps (36 cases each, vs V/oracle-sweep-v2 pins)

- sweep-b128x128: 36/36 pass, 36/36 bit-identical.
- sweep-b64x256: 36/36 pass, 36/36 bit-identical.
- sweep-b128x64w4: 36/36 pass, 36/36 bit-identical.

## TIME (bench-canonical, canonical N=512, same session, 11 timed reps)

| fam | incumbent s2bt8 | v2 256x64 | 128x128 | 64x256 | 128x64w4 | best/inc |
|---|---|---|---|---|---|---|
| gate_up | 1985.7 | 2012.5 | 1720.7 | 2221.3 | 2367.4 | 0.867 |
| qkv | 814.9 | 842.5 | 709.8 | 850.8 | 997.3 | 0.871 |
| qkvza | 884.9 | 971.7 | 836.1 | 999.8 | 1153.9 | 0.945 |
| residual | 1010.7 | 1262.5 | 901.6 | 1062.6 | 1463.8 | 0.892 |

L2-traffic estimate alongside (GB/staging ratio): gate_up
1.783/1.426(0.80)/1.783(1.00)/2.139(1.20); qkv 0.734/0.587/0.734/0.881;
qkvza 0.845/0.676/0.852/1.014; residual 0.891/0.713/0.891/1.070.
128x128 beats its 0.80 traffic ratio on residual (0.892 vs 0.80 comes with
grid/occupancy effects; still far from admission).

ADMISSION VERDICT: FAIL — best 128x128 rows 0.867/0.871/0.945/0.892,
all above the 0.60 cap. 64x256 is at/above parity (1.04-1.13x),
confirming the traffic model (1.00x staging). Per stop rules: STOP after
the bounded sweep. No KLD / interleave / decode / serve phase (gated on
admission). Bench hashes equal oracle hashes cross-harness; det=true all.
Winner reported even though it misses: 128x128x64/8w.
