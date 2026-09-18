# iu4frag — fragment-order LDS layout (gates 1–4; STOP at gate 4)

Branch `gfx12-iu4-k32` @ `af16ec4bd` + this change. Only file changed:
`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip`.
Env: ordinal 1 (`ROCR_VISIBLE_DEVICES=1`, ab1 home/cache). All GPU runs same
session (2026-09-18 ~08:00–08:30 UTC), gfx1201, model
`/home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`.

## Lane→byte mapping implemented

Per 64-K slab, 16 fragments × 256 B. Fragment `(g8, sb)` at
`slab + (g8*2+sb)*256`; lane `L = tid & 31` reads `+ L*8` (= row
`g8*16+(L&15)`, K-half `L>>4`, K bytes `sb*16+(L>>4)*8`, low nibble first —
the header's K32 operand mapping, unchanged, only placement permuted).
Consumer groups: A `g8 = tok_half*4+nb`; W `g8 = wt_pair*2+rg`; `sb` unroll
const; all hoisted at kernel top (`frA[4][2]`, `frW[2][2]`), K-step indexes
by unroll consts only.
Producer write for slab row `R = r*64+(tid>>2)`, quad `q = tid&3`:
frag `(R>>4)*2+(q>>1)`, lane `(R&15)+16*(q&1)` — verified bijection
(512 pairs → 512 slots, every fragment slot written exactly once per
publish; same bytes as the old row map). Global-side addresses, OOB
clamp/zero-select semantics, DS/SZ planes, fold DAG, barriers, store:
untouched. LDS 19456 → 16384 B (host still passes 19456, superset, same
occupancy; `_Static_assert(IU4_LDS_BYTES <= 19456)` guards geometry drift).

Write-side bank enumeration (32 banks × 4 B): write dword addr ≡
`slab/4 + 2*(R&15)` (mod 32) — depends only on `R&15`. Each wave's 32 writes
cover 8 banks × 4 hits = **4-way write conflict** (accepted: byte written
once, read 4× (A) / 2× (W)). Read side: 64 consecutive dwords = each bank
exactly twice = the b64 width floor (matches probe k_J pattern).

## Gate 1 — oracle: PASS 9/9 (`oracle-new2.log`)

`cpu_bitwise=OK (mism=0)` + `repeat_identical=OK` on all 9
(gate/set, gate/add, down/set, down/add, m48tail, n80cols, k256/set,
k256/add, m100n100). `GFX12-IU4 ORACLE PASS`. JIT entry
`tmp_iu4_gfx12_oracle.b836d576fc809f14` carries the fragment-order source
(markers verified; `.hip` identity matches post-cleanup worktree).

## Gate 2 — metadata: PASS (one +1 VGPR, no occupancy effect)

Oracle-JIT radiowave (apples-to-apples, same pipeline):
BASE `c1777e9d` → VGPR 182 / SGPR 40 / spill 0/0 / scratch 0;
NEW `b836d576` → VGPR **183** / SGPR 40 / spill 0/0 / scratch 0.
OCC=1: 3 WG/CU at both `IU4_LDS` 19456 (host default) and 16384.
(+1 VGPR survives a hoist cleanup (`lane8`, `R`/`q` temps); it is live-range
noise with zero occupancy/register-pressure consequence: 183 ≤ 256.)

## Gate 3 — ISA K-loop census: PASS

Steady-state trip (32 WMMAs), oracle-JIT disasm (`isa-base2.txt` vs
`isa-new.txt`, `kloop-new.txt`):

| trip | frag reads (dual b64) | other LDS | globals | int VALU | WMMAs |
|---|---|---|---|---|---|
| BASE | 12× `ds_load_2addr_b64` (= 24 b64) | 20× b32 (DS/SZ/fold) + stores | 8× b64 + 2× b32 | 10 add_co + 1 mad | 32 |
| NEW | 5× `2addr_b64` + 7× `2addr_stride64_b64` (= 24 b64) | same 20× b32 + dual-b64 stores | same | 10 add_co + **0 mad** | 32 |

Fragment reads are all b64-class dual-issue with immediate offsets off
loop-invariant bases (`v160 = s26+v146` once per trip = the existing W0↔W1
re-point; `v164` = base copy). Zero per-fragment address VALU; whole-body
integer VALU 146 → 142. K-step = ds_reads + WMMAs (+ fold VALU at half
boundary). Fold DAG source-unchanged.

## Gate 4 — TIME N=512, same session: FAIL (null result, target missed)

| case | BASE (08:0x) | NEW r1 | NEW r2 (`time-new-r2.log`) | ratio r2/BASE |
|---|---|---|---|---|
| gate set | 484.2 | 497.4 | 504.8 | 1.042 |
| gate add | 501.7 | 499.0 | 508.3 | 1.013 |
| down add | 501.3 | 524.4 | 517.8 | 1.033 |
| qkvza set | 482.1 | 495.8 | 497.7 | 1.032 |
| qkv set | 398.6 | 414.3 | 416.8 | 1.046 |

NEW spread ≤2% across 3 runs (r1, unlogged rerun 501/511/525/496/414, r2);
BASE→NEW +1–5%, i.e. no win — the probe's 4.7× fed-loop gain (K: 266–285
TOPS) does not transfer to the kernel (~180 TOPS both). Target gate-set ≤
350 µs missed (~505). Cache-`.hip` identity verified: TIME executed the
fragment-order kernel (entry `b836d576`, post-cleanup markers present).
Card healthy (44–48 °C on ordinal 1, `rocm-rerun.txt`).

Per gate order: STOP. Gate 5 (KLD pins, interleaved bench, decode, serve)
not run. Incumbent s2bt8 refs (iu4stage, prior session): gate_up fused
1679.3, residual 850.1 — not re-measured (out of scope after stop).

## Question for Main

The layout change is proven correct (bit-identical) and the ISA gate holds,
but the kernel is not moved by conflict-free fragment reads — the limiter
is elsewhere (WMMA issue + fold VALU interleaving at ~180 TOPS vs probe-A
register-resident 345 TOPS is the remaining gap; staging bytes are trivial
at ~0.35 TB/s). Options: (a) leave branch as-is (null recorded here);
(b) profile the real limiter; (c) try the `ds_read_b128` two-k-step variant
(bigger register pressure, may not help if reads aren't the bound).
No second layout improvised — awaiting direction.

## Files

- `oracle-base.log`, `oracle-new.log`, `oracle-new2.log` (gate 1)
- `time-base.log`, `time-new.log`, `time-new-r2.log` (gate 4)
- `rocm-rerun.txt` (ordinal-1 thermals)
- `isa-base2.txt`, `isa-new.txt` (oracle-JIT disasms), `kloop-new.txt`,
  `kloop-compare.txt` (gate 3)
- `build-base.log`, `build-new.log`, `build-new2.log`
- (excluded from commit: `tu-*.hip/.o`, `isa-*.dev.o` — rebuildable)
