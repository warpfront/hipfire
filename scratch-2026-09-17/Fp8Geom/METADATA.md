# Fp8Geom metadata — 128x64x64/8w half tile (NB==2)

TU: `kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip` (worktree HEAD + NB/X2 changes).
Recipe (== runtime flags: hipcc `--offload-arch=gfx1201 -O3`, default sched profile):
`hipcc --offload-arch=gfx1201 -O3 -c -DHIPFIRE_FP8_V2_TILE=1 -DHIPFIRE_FP8_V2_BM=<BM>
-DHIPFIRE_FP8_V2_BN=<BN> -DHIPFIRE_FP8_V2_BK=64 -DHIPFIRE_FP8_V2_WAVES=<W>
[-DHIPFIRE_FP8_RESIDUAL=1|-DHIPFIRE_FP8_QKVZA=1|-DHIPFIRE_FP8_QKV=1]
-DHIPFIRE_FP8_GATEUP_KERNEL=<sym> TU -o out.o -Rpass-analysis=kernel-resource-usage`
Logs: `meta/chk3_*.res.log` (post-X2), `meta/chk_*.res.log` (pre-X2), `meta/chk2_*` (mask probe, reverted).

## VGPR (used; -Rpass-analysis; 0 VGPR spill + 0 SGPR spill + 0 scratch on all)

| family   | 128x128x64/8w (control) | 128x64x64/8w (new) | delta |
|----------|------------------------:|-------------------:|------:|
| gate_up  | 189 (occ 8)             | 120 (occ 12)       | -69   |
| residual | 198 (occ 7)             | 119 (occ 12)       | -79   |
| qkv      | 211 (occ 7)             | 126 (occ 10)       | -85   |
| qkvza    | 213 (occ 7)             | 128 (occ 10)       | -85   |

Control == pre-change baseline TU (`meta/baseline.hip` gate_up 189): NB==4 path
is codegen-neutral. New geometry is NB==2 (32x32 wave tile, T+P 32+32 acc VGPR)
+ W-prefetch dropped on NB==2 only (saves Wpk 2 regs; W stages post-compute
from globals, identical values; A prefetch unchanged).

Allocated (granule 8): 120 / 120 / 128 / 128 — all ≤ 128. Gate 2 PASS.
Waves/SIMD under the §resource 512-file model: 4 / 4 / 4 / 4 → 4 SIMDs/WGP × 4
= 16 waves = two 8-wave WGs per WGP. (Compiler-pass occupancy column above uses
a different model; reported verbatim.)

LDS (host vlds = max(stage, WAVES*2048)): stage 14848 B, passed 16384 B.
Two WGs = 32768 B ≤ 64 KiB pool, ≤ 32 KiB/WG conservative budget.
Single accumulator bank kept (T only; P is the per-half partial, folded in
place — same fold as 128x128, per-128-K, unchanged order).

## Rejected without GPU time
- 128x128x32/8w: keeps the full 64x32 wave tile (128 acc VGPRs) → ~189 VGPR,
  no occupancy gain. Fails gate 2 by construction.
- 64x128x64/8w: same acc count as 128x64w8 but doubles HBM-streaming W traffic
  (N/BM 8 vs 4 token tiles; W is the streaming operand, A is L2-resident at
  N=512). Worse traffic story; kept as fallback only.
- OOB-mask packing (5 arrays → 1 bitmask): 0 VGPR saved on all fams (peak is
  in-loop live state, masks overlap) — reverted, schedule untouched.
