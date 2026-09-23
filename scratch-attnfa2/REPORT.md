# gfx11 FA2 q8 prefill: Q-residency and codes-in-LDS — KILL on both cards

**Verdict: KILL both levers on gfx1100 and gfx1151.** Every exact variant reproduced the production output bit for bit, but none was faster than production beyond run-to-run noise on either card. Most were slower. No production code changed, so there is no kill switch, trace or WT2. The ablations show where the FA2 time actually goes: the per-tile K/V fill, not the Q reloads.

## Method

- Base: `855573abe` (official mq4-lloyd, V2B). The FA2 kernel is unchanged from `ef52914a4`.
- `fa2_var.hip` is a macro-parameterised twin of `kernels/src/attention_q8_0_fa2_gqa.gfx11.hip` (Q8 K, Q16 tile, KT32). V0 is the production body. Every variant was compiled with the runtime JIT flags (`--genco -O3 --no-offload-compress`), built locally and copied to hipx as binaries only. The sources reached hipx through git.
- `fa2_bench.cpp` runs the production kernel (`prod_<arch>`, built from the real source with `KT=32 Q16=1`) and each variant on the same non-periodic inputs:
  - Q: hashed f16 in [-2, 2).
  - K/V: Q8_0 with log-spread scales from 1/512 to 1/8, 1/64 of blocks with 4× outliers, and full-range codes.
- Correctness: the output buffer was poisoned with NaN before each run. Every output bit was then compared against production, along with a non-finite count.
- Timing: variants run interleaved per rep; the reported figure is the median of HIP event times.
- Schedules:
  - `halo`: 16 launches of B512, one per chunk start 0…7680, i.e. one layer of a pp8192 pass.
  - `xtx`: one whole-chunk B8192 launch.
  - `odd`: B100@37, B77@300, B512@1000, B64@0, B1024@7168 (partial and tail tiles).
- Lease granted by Stacker. Pre- and post-check `rocm-smi` showed only gpusentry. No model load and no hipx compile.

## Variants

Only the exact variants are shipping candidates; the two ablations deliberately produce wrong output and exist only to measure attribution.

- **Q-residency** (exact):
  - `qres2`/`qres4`: the first N dim-chunks are kept resident as replicated half16.
  - `qresx4`/`qresx8`: the first N chunks are kept resident non-replicated at 4 VGPR each and rebuilt per chunk with `permlanex16`.
  - `qpipe1`/`qpipe2`/`qpipe1w`: the Q load is lookahead-pipelined by 1 or 2 chunks, or carried across tiles.
  - Combinations with V pipelining or V register prefetch: `vpipe`, `vpf`, `qp1vp`, `qp1vpf`, `qresx4qp1`.
- **Codes-in-LDS** (exact):
  - `codesk`: K int8 codes plus f16 scales, XOR-swizzled 16-B chunks.
  - `codesv`: V codes transposed per dim with a key-slot swizzle.
  - `codeskv`: both. LDS drops from 32,768 B to 17,408 B.
  - Fragments are dequantized at use with `v_perm` bias tricks and `v_pk_*_f16`. This is exactly `RNE(scale·code)`, the same as production's f32 product cast to f16.
  - Also tried combined with qpipe and without the helper waves (`nohelp*`: 192 threads).
- **Ablations** (inexact, attribution only):
  - `abl_noq`: Q is synthesized in registers, so no Q loads.
  - `abl_nofill`: K/V are filled only for tile 0.

## Results

### Median ms per schedule (speedup vs production)

Mismatches = 0 and non-finite = 0 for every exact variant, on both cards and all schedules.

| Variant | VGPR (occ) | gfx1151 `halo` | gfx1100 `xtx` |
|---|---|---:|---:|
| prod | 218 (6) | 50.537 (1.000) | 20.846 (1.000) |
| v0 (twin) | 218 (6) | 50.100 (1.009) | 21.186 (0.984) |
| v0u (QK loop fully unrolled) | 233 (6) | 49.982 (1.011) | 20.967 (0.994) |
| **abl_noq** (Q-load ceiling) | 224 (6) | 48.888 (**1.034**) | 18.584 (**1.122**) |
| **abl_nofill** (fill ceiling) | 207 (7) | 38.230 (**1.322**) | 14.723 (**1.416**) |
| qres2 | 235 (6) | 53.002 (0.954) | 20.942 (0.995) |
| qres4 | 251 (5) | 59.029 (0.856) | 23.831 (0.875) |
| qresx4 | 235 (6) | 51.307 (0.985) | 20.996 (0.993) |
| qresx8 | 251 (5) | 58.339 (0.866) | 22.879 (0.911) |
| qpipe1 | 228 (6) | 52.469 (0.963) | 21.710 (0.960) |
| qpipe2 | 236 (6) | 54.368 (0.930) | 22.245 (0.937) |
| qpipe1w | 228 (6) | 52.832 (0.957) | 21.567 (0.967) |
| vpipe | 247 (5) | 61.567 (0.821) | 23.818 (0.875) |
| vpf | 252 (5) | 60.826 (0.831) | 23.952 (0.870) |
| qp1vp | 239 (6) | 52.815 (0.957) | 20.846 (1.000) |
| qresx4qp1 | 245 (5) | 59.856 (0.844) | 22.561 (0.924) |
| codesk | 220 (6) | 54.404 (0.929) | 22.366 (0.932) |
| codesv | 244 (5) | 67.390 (0.750) | 26.247 (0.794) |
| codeskv | 251 (5) | 71.370 (0.708) | 28.482 (0.732) |
| nohelp | 235 (6) | 56.639 (0.892) | 21.810 (0.956) |
| nohelpcodeskv | 235 (6) | 66.536 (0.760) | 26.026 (0.801) |

`odd` on gfx1100 (tails): prod 6.089 ms, v0 6.011, v0u 5.954, qpipe1 5.975, codeskv 7.736. All exact.

Scale check: 50.5 ms/layer × 16 layers ≈ 808 ms/pass on Halo, and 20.8 × 16 ≈ 334 ms on XTX. Both agree with the traced ~830 ms and ~340 ms.

### Why they lose

- **Q-residency.** Even free Q (the `abl_noq` ceiling) buys only **3.4% on Halo** and **12% on XTX**. The f16 Q is an L2-resident 16-B load per 8 dims, shared across both KT32 subtiles (qshare2), so there is little left to remove.
  - Keeping any real amount of Q resident pushes VGPRs from 218 to 235–251. The half16 must be replicated across lane pairs for the gfx11 WMMA B operand, and at ≥241 VGPRs occupancy falls from 6 to 5 waves/SIMD. That costs more than the saved loads.
  - The cheapest forms (non-replicated plus `permlanex16`, or a one-chunk lookahead) break even on XTX and lose 1.5–4% on Halo.
- **Codes-in-LDS.** Staging codes shrinks LDS, but dequantization moves from once per tile, shared by all 8 waves, to once per fragment use in each of the 6 compute waves. That is 6× for K; V is dequantized on every PV chunk. The ALU cost and VGPR pressure (V: 244–251, 5 waves) outweigh the smaller fill. The cost is −7% for K alone and −25…−30% for K+V.
  - The halved LDS does not buy residency, because VGPRs already cap residency at 2 WGs/CU.
- **Where the time is.** `abl_nofill` shows the per-tile K/V global→dequant→LDS fill plus its barrier costs **24% of FA2 on Halo** and **29% on XTX**. That is the pool worth attacking. Options:
  - an async/double-buffered fill;
  - a larger query tile, to amortize each KV fill over more queries;
  - V register prefetch that stays under 241 VGPRs. `vpf` hit 252 VGPRs in this form and lost.

## Files

- `fa2_var.hip`, `fa2_bench.cpp`, `build.sh`, `run_screen.sh`: the screen.
- `results/gfx1151-halo.txt`, `results/gfx1100-odd-xtx.txt`: raw harness output.
- `results/resources.txt`: per-variant VGPR, spill, scratch and occupancy on both archs.
