# Stage B1a receipt — Q0 f16 fill port onto fragment order (exactness)

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-fa2a`, branch `gfx1201-fa2-fragorder`.
Base: `e555cc08f` (merge of `gfx1201-fp8-kv`). Plan §14 (commit `98d6987a5`).
Device gates ran from isolated worktree `/home/kaden/ClaudeCode/warpfront/wt-b1a`
@ `e555cc08f` + kernel patch only (no sibling noise).
GPU: ordinal 1 (`HOME=ab1`, `ROCR_VISIBLE_DEVICES=1`).

## Change (one file: `kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`)

KMODE=8 K fill (was L223-277) and V fill (was L371-428) wrote the deleted
`Kdw`/`Vdw`/`fa2_swiz` planes. Ported onto `fa2_koff`/`fa2_voff` per §14.2,
keeping slot decomposition and `f16(f32(scale)*decode_e4m3(code))` byte for
byte; only the destination address changed. Merge note (was L377-380) deleted.
LDS stays 65536 (f16 planes). No new symbols. KMODE=0/3 untouched.

## G0 compile/resource gate — PASS

Recipe: `hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress
-Rpass-analysis=kernel-resource-usage` (HIP 7.15). 0 warnings all TUs.
Objects: `fa2_k0.o`, `fa2_k3.o` (k3 = KMODE3 + turbo_common.h prepend),
`fa2_k8.o`, plus FP8-guard combos (K0+FP8, K8+FP8) — all compile, 0 warnings.

- KMODE=0 object BIT-IDENTICAL to base (same-filename `cmp`, whole object).
- KMODE=3 object BIT-IDENTICAL to base. (Preprocessed `-E -P` sources also
  identical; stronger than the `.text` gate.)
- KMODE=8 (did not compile on base): direct 235 VGPR / SGPR 34, partial 236 /
  SGPR 36, preconvert 13, merge 17; 0 spill V/S, 0 scratch, 6 waves bodies,
  16 waves preconvert/merge everywhere. Gate ≤240 VGPR: PASS.

## Host bijection (width-1 ported addresses) — PASS

Independent python check of the exact shipped index math: K slots (k 0..63,
b 0..7, w 0..7, 4-wide at dd) cover all 16384 halves exactly once, fragment
base j ∈ {0,4} (never crosses an 8-half boundary); V slots (m 0..31, b, w, c,
both keys) cover all 16384 halves exactly once. Matches §13 width-1 proof.

## Fill packet census (static, gfx1201 ISA, direct K8 kernel) — RECORD

Method: `save-temps` ISA; loop structure verified by branches/loads (K fill:
t-looped 4 × w-looped 8 around one static body with 1 `global_load_b32`;
V fill: t-unrolled ×2 copies, each w-looped 8 with 2 static `global_load_b32`).
Per-lane dynamic = trips × static body packets (static census convention:
all packets, not cycles; predicated OOR arms counted).

- K Q0-decode fill ≈ 4 × (64 + 8 × 207) ≈ **6880 packets/lane**.
- V Q0-decode fill ≈ 2 × (60 + 8 × ~370) ≈ **6040 packets/lane**.
## G1 KMODE=0/3 exactness — PASS (ordinal 1, wt-b1a release binaries)

Binaries built in wt-b1a @ e555cc08f + kernel patch only
(`eval_hipfire`, `tmp_fa2a_g2` [G2's untracked throwaway, copied in, never
committed], `fa2_fp8_q0_compare_gfx1201`, `tmp_fa2_fp8_screen` [base version],
`tmp_b1a_kout` [bb064224f screen restored under a scratch name for the
G2 `--kernel-out` path the merged screen dropped]).
Env: `HOME=ab1 ROCR_VISIBLE_DEVICES=1 HIPFIRE_LLOYD_GFX12=1
`
`HIPFIRE_GFX12_FA2_PREFILL=1 HIPFIRE_GFX12_FA2_FP8=0`.

- Screen `--kernel-out` raw-O md5 **`1f7a1183a751de7555066bcf5a1ca11c`** ✓
  (`kern.O.f32.O.f32`, 9,437,184 B) and gate `8c0287afddd1a3aef7f43496c3127c23` ✓
  (`screen/run.log`). Both match G2.1 exactly.
- Oracle 7/7 cksums match G2.2 exactly (`shapes/run.log`):
  00002f8fbf83e1fd, 00005f1b6d7166e1, 00017cb6701d7d37, 0002faa14e5a12e5,
  0005f53d7121c8da, 000bec6d809ddedf, 0017da4c7061f932.
- Harness 27/27 files `cmp`-identical vs `E/Fa2A/g2on/` (`harness/run.log`).
  The run then faults at the first split case with the DOCUMENTED PRE-EXISTING
  bench-only partial fault (same kernel `attention_q8_0_fa2_gqa_partial_gfx1201`,
  same panic site `tmp_fa2a_g2.rs:273`, same HipError 700 signature as G2.5 on
  OFF/af16ec4bd). That kernel's object is bit-identical base-vs-patch (G0),
  so the fault cannot be a B1a regression — fault parity, G2-splits still BLOCKED.

## G2 KMODE=8 Q0 exactness — PASS (ordinal 1)

- `fa2_fp8_q0_compare_gfx1201` (same native K/V, FA2-Q0 vs tiled f32 ref):
  max_abs=1.546264e-3 mean_abs=8.215926e-5 tail1pct=2.192105e-4
  max_rel=4.055091e-1 nonfinite=0, deterministic across reruns
  (`compare/compare.log`). Format-only error, as expected.
- 1-chunk WT2 eval (fp8 route): slice-mean **0.031193**, NLL 2.202592,
  PPL 9.0484 — all three reproduce `flip_fa21_receipt` exactly (`eval/fa1.log`).
- 24-chunk WT2 eval: slice-mean **0.048413**, NLL 1.855469, PPL 6.3947
  (`eval/fa24.log`), `.kldseq` md5 **`76c1087d2d1e38c2d7f38ceb71f94303`** —
  bit-for-bit vs the pre-merge Q0 receipt (`E/Fp8Kv/b/b2-fa224.kldseq`).
  §14.2 exactness gate: PASS ("close" would be fail; it is identical).

