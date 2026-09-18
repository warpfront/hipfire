# iu4port gate 5b — interleaved end-to-end bench OFF/ON/OFF/ON (ordinal 1)
# Binaries: wt-iu4 target/release/hipfire + target/release/daemon, both rebuilt
# with the new kernel (verified: `grep -c wave_row0` = 5 on daemon; engine JIT
# of gemm_mq4g256v2_residual_mmq_iu4_gfx12 from the new source during ON runs).
# Env: HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1, IU4 flag 0/1/0/1, kv q8, spec off,
# runs 3 warmups 1. Raw JSON+stderr: bench/{off1,on1,off2,on2}.{json,stderr}.
# NOTE: first ON1 attempt ran under a stale daemon (old kernel, 0.82x) and was
# quarantined to bench/on1-stale-daemon.* — it is NOT part of this table.

## Medians (tok/s) with raw samples [r1, r2, r3]
| row | OFF1 | ON1 | OFF2 | ON2 |
|---|---|---|---|---|
| pp512 | 1479.0 [1476.1, 1480.9, 1479.0] | 1631.0 [1629.3, 1635.1, 1631.0] | 1476.1 [1476.1, 1472.4, 1481.2] | 1623.8 [1621.5, 1623.8, 1624.2] |
| pp2048 | 1448.3 [1447.8, 1448.3, 1448.4] | 1599.4 [1599.4, 1596.8, 1599.5] | 1446.8 [1446.0, 1446.8, 1447.4] | 1590.5 [1590.7, 1590.3, 1590.5] |
| pp8192 | 1342.1 [1343.2, 1342.1, 1340.6] | 1471.8 [1472.2, 1471.8, 1470.2] | 1341.3 [1342.4, 1341.3, 1341.3] | 1467.0 [1467.5, 1466.5, 1467.0] |
| pp32768 | 1033.5 [1033.9, 1033.5, 1033.3] | 1110.9 [1112.6, 1110.6, 1110.9] | 1033.9 [1033.9, 1034.4, 1033.9] | 1110.2 [1110.2, 1110.5, 1110.2] |
| tg64@128 decode | 36.496 [36.503, 36.496, 36.490] | 36.490 [36.479, 36.490, 36.492] | 36.475 [36.502, 36.467, 36.475] | 36.487 [36.482, 36.502, 36.487] |

## Ratios (ON median / matched OFF median)
| row | ON1/OFF1 | ON2/OFF2 |
|---|---|---|
| pp512 | 1.103 | 1.100 |
| pp2048 | 1.104 | 1.099 |
| pp8192 | 1.097 | 1.094 |
| pp32768 | 1.075 | 1.074 |
| decode | 1.000 | 1.000 |

OFF reproduces its bands in both interleaves (1476-1479 / 1446-1448 / 1341-1342 /
1033-1034, all inside 1470-1485/1439-1458/1334-1352/1030-1039). ON gains are
+7.4-10.4% on every prefill row in both interleaves — real (>1.5%), same
direction, no bad rows. Decode flat at 36.48-36.50 (≥36.4 clean): iu4 does not
touch the decode path, as designed (prefill-only admission).
