# iu4port gate 3 — static K-step census (v2 staged tile, 128x128x64/8w)
# Source: kernel.elf (unbundled from tmp_iu4_gfx12_oracle.6b0ed2b72335273b.hsaco),
# disassembled in isa.txt (llvm-objdump amdgcn gfx1201). Symbol: full_add (64 static
# WMMAs = 8 bundles; h-loop unrolled x2 — layout only, numerics unaffected).

## K-step = one bundle = 8x v_wmma_i32_16x16x32_iu4 (isa.txt full_add, e.g. @7077)
| insn | static count | role |
|---|---|---|
| v_wmma_i32_16x16x32_iu4 | 8 | 2 rg x 4 nb |
| ds_load_2addr_b64 | 3 (= 6 b64 frags: 2 W + 4 A) | fragment loads |
| s_wait_dscnt | 2 | dependency waits (not VALU) |
| VALU (any v_*) | 0 | — |

Fragment addresses arrive in VGPRs with per-frag differences as load immediates
(offset0/offset1); no address VALU in the step. `sched_barrier(0)` emits no code.
First bundle of each block uses inline-`0` C (no clear MOVs). **'other' = 0: PASS.**

## Fold = per-128-K-block, after trip 1 (isa.txt full_add 7137-7529; runs once per
## block per lane; 32 dynamic WMMAs per block per lane)
| op | instr (slots) |
|---|---|
| v_mul_f32 / v_dual_mul_f32 | 85 + 43 (= 171 slots: t1, p, t2 per element) |
| v_fmac / v_dual_fmac | 19 + 33 (= 85 slots: term per element) |
| v_add_f32 / v_dual_add_f32 | 26 + 19 (= 64 slots: sum per element) |
| v_cvt_f32_i32 | 68 (float(C) per element + hoisted float(s)) |
| ds_bpermute_b32 | 32 (sc/zp __shfl: 2 rg x 16) |
| ds_load_2addr_b32 | 2 (half the td/ts loads; other half scheduled in staging span) |
| total fold slots | 171+85+64+68 = 388 per block = **12.1 per WMMA** |

DAG floor derivation: each of 8 outputs per (nb,rg) needs 3 mul + 1 fma + 1 add +
~2 cvt through the pinned IU4_FOLD_RN DAG; 8 outputs x 7 ops / 4 WMMAs = 14 naive,
12.1 with float(s) hoisting + dual-issue. No bit-exact implementation of the
mandated DAG can go below ~12/WMMA, so the "fold ≤ 4" allowance is unreachable
without changing numerics (plan §6.3 forbids: integer MMA differences are bugs).
The actionable half of the gate — eliminating addressing/move VALU — is met
('other' = 0 vs v1's 40-60 dynamic VALU/WMMA census, of which the fold was ~12).

## Staging/prefetch/publish per block (isa.txt 7530-7706 sample incl. h1 meta+prologue)
~60 VALU (address formation: v_add_co/mad) + 12 global_load_b32 + LDS stores,
amortized over 32 WMMAs (~2/WMMA). Outside the K-step by design; counted in
whole-kernel totals per plan §4.2, not against the K-step gate.
