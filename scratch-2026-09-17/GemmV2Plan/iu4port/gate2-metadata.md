# iu4port gate 2 — compile/ISA metadata (v2 staged tile, 128x128x64/8w)
//# Source: tmp_iu4_gfx12_oracle.6b0ed2b72335273b.radiowave.json (.hip/.hsaco same stem)
# Session: 2026-09-18, ordinal 1. Receipts: gate2-occ.log (OCC=1 run).

| check | value | gate | verdict |
|---|---|---|---|
| wavefront | 32 | wave32 | PASS |
| VGPR (all 3 symbols) | 239 | ≤ 256 | PASS |
| SGPR (base/add/set) | 40/38/38 | — (info) | — |
| VGPR spill | 0 | 0 | PASS |
| SGPR spill | 0 | 0 | PASS |
| private scratch | 0 | 0 | PASS |
| LDS (launch contract) | 12288 B (12 KiB) | ≤ 32 KiB | PASS |
| max active blocks/CU (block 256, LDS 12288) | 3 (set + add) | ≥ 2 WG/CU | PASS |

Notes:
- VGPR 239 exceeds the plan's ≤192 planning target but clears the hard 256
  ceiling with 0 spill; occupancy (measured, not modeled) gives 3 WG/CU.
- LDS 12288 = A 5120 + W 5120 + DS 1024 + SZ 1024; store slots (10240 B)
  overlap the dead compute planes.
