# Change B — scale broadcast via LDS (no __shfl)

## Change
Fold gather: per (nb, rg) the 8 (sc, zp) pairs are read straight from the
staged SZ plane with narrow `ds_read` (`sz_lds[jbase]`, `jbase =
(wave_row0 + rg*16 + 8*k_grp + j) * 8`) instead of 2 LDS reads + 16
`__shfl` (bpermute) from the owning lanes. Exact: SZ content identical, the
shuffle source lane's value IS `SZ[row]`; IU4_FOLD_RN DAG and loop order
untouched. Host (gemm.rs LDS 19456, oracle) unchanged — pure .hip edit.

Bank-clean on gfx1201: 8 consecutive rows x 2 words = 16 distinct banks
(`(2R+w) mod 32`), identical addresses across a k_grp half (LDS broadcast);
the two halves cover all 32 banks exactly once.

## Census VALU/WMMA (static whole-kernel, gfx1201, rolled h-loop)

| op | A (tu-changeA3.o) | A+B (tu-changeAB.o) | delta |
|---|---|---|---|
| v_wmma_i32_16x16x32_iu4 | 128 | 128 | 0 (compute untouched) |
| ds_bpermute_b32 | 143 | 15 | -128 (all fold shfls gone; 15 = prelude shfl_xor) |
| ds_load (b32+b64) | 328 | 384 | +56 (16 SZ reads/fold x unrolled copies) |
| ds_store | 184 | 184 | 0 |
| v_mul (f32+dual) | 433+247=680 | 425+245=670 | ~0 (fold DAG whole) |
| v_fmac/fmac | 160+110=270 | 152+112=264 | ~0 |
| v_cvt_f32_i32 | 272 | 272 | 0 |
| v_add (f32+dual) | 280+100=380 | 288+92=380 | 0 (jbase math dual-packs) |
| global_load/store | 205/258 | 205/258 | 0 (staging untouched) |

Dynamic per 128-K block per lane: 32 WMMAs; fold 32 bpermutes -> 0,
+28 narrow ds_reads (16 SZ + 12... exact: 32 SZ reads replace 32 shfls and
4 SZ base reads). v2 reference (gate3, unrolled static): 256 WMMA,
271 bpermute, 400 ds_load, fold 388 slots = 12.1/WMMA.

## Gates
1. Oracle 9/9 PASS, cpu_bitwise=OK mism=0, repeat_identical=OK (B-oracle.log).
2. Metadata (JIT 2e97be91cd66ae43): VGPR 212 (all 3 symbols) <= 256 |
   spill 0/0 | scratch 0 | LDS 19456 B (unchanged) | WG/CU 3 (OCC, unchanged).
3. TIME N=512, same session x4 runs (B-time.log + 3 runs below):
   gate/set 657 (653/660/659/657), gate/add 669, down/add 638 (636/637/640/641),
   qkvza/set 636 (634/636/639/638), qkv/set 538 (537/538/538/540).
   vs A alone (680/685/660/662/561): gate -3.4%, res -3.3%.
   vs v2 same-session (~681/~702): gate -3.5%, res -9.1%.
   Incumbent s2bt8 same-session: gate_up fused 1679.3, residual 850.1.

## TIME table (us/call medians; ratios vs v2 / vs incumbent per-matrix)
| case | v2 | A alone | A+B | A+B / v2 | A+B / s2bt8 |
|---|---|---|---|---|---|
| gate set (17408,5120,512) | ~681 | 680 | 657 | 0.96 | 0.78 (2x657/1679) |
| gate add | ~740* | 685 | 669 | — | — |
| residual add (5120,17408,512) | ~702 | 660 | 638 | 0.91 | 0.75 (638/850) |
| qkvza set (16480,5120,512) | ~709* | 662 | 636 | — | — |
| qkv set (14336,5120,512) | ~604* | 561 | 538 | — | — |
(* cross-session oracle refs from the v2 baseline run; gate/residual
same-session via the attrib launcher.)

Gate-4 trigger (>= 20% on gate_up): NOT met (gate_up -3.5%). KLD pins,
interleaved bench, decode, serve battery NOT run per the gate order.
