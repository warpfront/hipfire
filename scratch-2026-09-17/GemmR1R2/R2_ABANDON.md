# R2 abandonment receipt (GemmR1R2, ordinal 0)

R2 (cross-half depth-two ring, 4-phase unrolled, plan section 4.2/C2) was
implemented in full in K (see r2_attempt.diff) and fails three resource gates
at compile time, before any timing could admit it:

| Family | VGPR (R2 cap / hard cap) | VGPR spills (gate 0) |
|---|---|---|
| gate_up | 256 (212 / 240) | 105 |
| residual | 256 (224 / 240) | 100 |
| qkv | 256 (232 / 240) | 103 |
| qkvza | 256 (233 / 240) | 100 |

- `.text` proxy (hip_fatbin section): 31576 vs R1 25048 = 1.261x (gate <=1.25x).
- Remediation tried: flattened 1D ring arrays (r2flat.hip) -> still 256 VGPR +
  109 spills. Structural: the 4-phase straight-line body explodes allocator
  pressure vs the loop version; the plan section 4.4 budgets were engineering
  estimates, not measured allocations.
- WMMA operand feed itself stays register-resident (ds_load -> VGPR -> wmma),
  but ~100 spilled VGPRs add scratch traffic exactly where the window was to
  be recovered ("excess bookkeeping that consumes the recovered window fails
  admission", plan section 4.4).

Per the plan: R2 abandoned, accepted C1 stands. No production change from this
attempt (K reverted to C1 commit 007caae7a); receipts: r2_attempt.diff,
gu_r2.s / *_r2.res.log, gu_r2f.s, r2flat.hip in this directory.
