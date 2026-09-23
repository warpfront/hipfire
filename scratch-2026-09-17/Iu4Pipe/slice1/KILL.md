# Slice 1 — KILL

- Source baseline: `99c44a96b`.
- Resources (`full_set`/`full_add`): 202 VGPR, 40 SGPR, 0 scratch bytes/lane, 0 SGPR/VGPR spills, 7 waves/SIMD; runtime occupancy 3 blocks/CU at 19,456 B.
- `llvm-objdump` steady K128: 32 WMMAs and four barrier signal/wait pairs at `0xE970/0xE9C0`, `0xEC20/0xEC30`, `0xEC74/0xEC78`, and `0xEF60/0xEF64`. Counter waits before those publications are respectively `s_wait_loadcnt_dscnt 0x0`, `s_wait_loadcnt 0x0`, `s_wait_loadcnt_dscnt 0x0`, and `s_wait_loadcnt_dscnt 0x0`. Prologue pair: `0xE180/0xE184`.
- Raw-load assertion: raw loads are assembly lines 6774–6777; the first two eight-WMMA bundles are lines 6782–6804; no mask use or LOAD wait occurs in either bundle. `s_wait_loadcnt 0x0` is line 6806 after the second bundle.
- Corrected A/B/A: each arm post-JIT warmed, gate preflight discarded, then 5 warmups + 20 measured events per arm, ordinal 2:
  - gate/up M4096 K5120 N34816: A1 7100.194 us, B 7116.314 us, A2 7096.374 us, baseline mean 7098.284 us, candidate -0.253%, A-bracket drift 0.054%.
  - residual M4096 K17408 N5120: A1 3154.093 us, B 3139.353 us, A2 3156.053 us, baseline mean 3155.073 us, candidate +0.501%, A-bracket drift 0.062%.
- Kill result: FAIL. Neither anchor reached the required +2%; gate/up regressed.
- Oracle: not run after short timing kill.
- Paired serving rows: not run after short timing kill.
- Outcome: KILL, no commit.

Evidence: `resource.log`, `iu4_combined-hip-amdgcn-amd-amdhsa-gfx1201.s`, `slice1.objdump`, `kernel-aba-corrected.log`.
