# gfx11 IU4 symmetric-fold report

## Verdict

**SHIP behind the default-off typed knob `kernel.gfx11_iu4_symfold`.** On the symmetric MQ4V2 artifact, the port improves exact TTFT-5909 by **4.42% on gfx1100** and **4.98% on gfx1151**, passes the WT2 quality budget, improves both c24 references relative to the same-build OFF arm, has zero spills, and completes 5/5 decoded battery prompts on both cards.

The implementation is admitted only when all of these hold:

- architecture is exactly `gfx1100` or `gfx1151`;
- the loaded artifact carries `mq4v2_symmetric`;
- typed config `kernel.gfx11_iu4_symfold` is true;
- developer override `HIPFIRE_IU4_SYMFOLD` is not `0`;
- the launch is a full/interior gridspec launch. The guarded tail intentionally remains on the production asymmetric kernel.

The default is false. Existing shared entry points and the production schedules, LDS layouts, block shapes, grid ordering, and tail dispatch are unchanged.

## Mechanism and port

The gfx1201 reference is not a separate algorithmic source file. Its source constant prepends `#define IU4_SYMMETRIC_FOLD 1` and selects renamed entry symbols from the same IU4 source. The relevant implementation sites are additionally guarded by `__gfx1201__`. Before this work, gfx11 parsed and carried the `mq4v2_symmetric` marker but had no dispatch consumer and no corresponding gfx11 symbols.

The gfx11 port reproduces the gfx1201 contract:

1. packed weight dwords are rebased exactly once while staging to LDS with `q ^= 0x88888888`;
2. WMMA consumes the rebased nibbles as signed weights and the existing activation nibbles as signed values;
3. the zero-point term is removed only after that rebias, leaving the scale-only fold;
4. both the ordinary 8-wave family and gfx1100 LF16 16-wave family have private, add-only symfold symbols.

The production gfx1100 row-LF16 SET proof was:

- base symbol `gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100`;
- candidate symbol `gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_gfx1100_symfold`;
- grid `(136,46)` for M=17408, N=5909;
- block `(32,16)`, dynamic LDS 30,720 bytes;
- VGPR 96 -> 90. `hipOccupancyMaxActiveBlocksPerMultiprocessor(fn, 512, 30720)` reported two blocks/CU for **both** base and symfold, i.e. a 32-wave resource ceiling for wave32.

That 32-wave figure is not an occupancy improvement caused by the candidate: the base entry reports the same value. It is the HIP API's resource-admissible ceiling, not a runtime residency counter. The LDS arithmetic is internally consistent (`2 * 30,720 = 61,440` bytes, leaving 4,096 bytes of a 64 KiB budget), and `hipFuncGetAttributes` reports zero static LDS, but this campaign did not instrument achieved resident blocks. No performance claim depends on 16 versus 32 achieved waves.

The initial standalone gfx1100 harness launch exposed and then fixed a harness-only geometry error: the row wrapper requires `(M/128,N/128)`, while the Halo column wrapper uses `(N/128,M/128)`. The production dispatcher already had the correct row/column distinction.

## ISA evidence

Disassembly was generated for both gfx1100 and gfx1151 before timing. The candidate contains the expected loader `v_xor_b32 ... 0x88888888` and signed-weight WMMA encoding (`neg_lo:[1,1,0]`, versus `[0,1,0]` in the unsigned production entry).

| family | static asymmetric instructions removed per 32-WMMA outer body | reduction | VGPR | SGPR | spills/private segment |
|---|---:|---:|---:|---:|---:|
| ordinary | 393 (983 -> 590) | 39.98% | 189 -> 182 | 22 -> 21 | 0 |
| LF16 | 96 (281 -> 185) | 34.16% | 96 -> 90 | 26 -> 26 | 0 |

The census is a static outer-body instruction count, not the earlier logical dynamic VALU estimate; it is deliberately reported as such.

## Numerical correctness

The scalar-f32 oracle used non-power-of-two scales and activation factors. Both cards produced the same comparison:

| measurement | production | symfold |
|---|---:|---:|
| RMSE vs scalar f32 | 1.23549768e-7 | 1.23669586e-7 |
| max absolute error vs scalar f32 | 9.53674316e-7 | 9.53674316e-7 |
| symfold vs production RMSE | - | 1.27275613e-8 |
| symfold vs production max absolute | - | 1.1920929e-7 |
| differing outputs | - | 3,276 / 16,384 |

The paths are algebraically equivalent under `zp = -8*scale`, but not bit-identical in f32. Production computes an unsigned-WMMA accumulator and combines the separately rounded `scale*d*acc` and `zp*d*s` terms with an FMA before adding to the running sum. Symfold forms the exactly related signed int32 accumulator in WMMA and applies one rounded `scale*d*acc_signed` product. The integer identity is exact; the different f32 operation tree explains the one-ULP-scale output differences. The scalar oracle, paired c24 results, and decoded batteries rule out the invalid missing-rebias failure mode.

## Exact-shape standalone ABBA

Each cell used four fresh processes in forward/reverse/reverse/forward arm order, production gridspec interior plus the production guarded tail, and the actual card-specific family. Values below are the means of the four process medians. Each process used two warmups and nine timed launches per arm.

| GPU | shape | N | production us | symfold us | paired ratio |
|---|---|---:|---:|---:|---:|
| gfx1151 | gate/up SET, M=17408 K=5120 | 5909 | 21,238.480 | 20,370.188 | 1.042626x |
| gfx1151 | gate/up SET, M=17408 K=5120 | 8192 | 29,848.137 | 28,863.002 | 1.034131x |
| gfx1151 | down ADD, M=5120 K=17408 | 5909 | 22,408.192 | 20,608.417 | 1.087332x |
| gfx1151 | down ADD, M=5120 K=17408 | 8192 | 30,364.042 | 28,452.550 | 1.067182x |
| gfx1100 | gate/up SET, M=17408 K=5120 | 5909 | 8,240.017 | 7,566.223 | 1.089053x |
| gfx1100 | gate/up SET, M=17408 K=5120 | 8192 | 10,952.472 | 10,009.164 | 1.094244x |
| gfx1100 | down ADD, M=5120 K=17408 | 5909 | 8,748.797 | 8,064.099 | 1.084907x |
| gfx1100 | down ADD, M=5120 K=17408 | 8192 | 11,532.895 | 10,401.268 | 1.108797x |

The larger Halo down gain is mechanistically consistent: down executes 136 K128 steps per output block versus 40 for gate/up, so it pays the removed per-K128 asymmetric descale work 3.4 times as often.

### Order-sensitivity caveat

The gfx1100 standalone harness was strongly first-arm sensitive despite very tight within-process medians. Per-process ratios were:

- gate/up N=5909: 1.140655, 1.039477, 1.038579, 1.142664;
- gate/up N=8192: 1.178919, 1.015673, 1.020194, 1.174596;
- down N=5909: 1.144716, 1.041520, 1.030084, 1.130394;
- down N=8192: 1.196470, 1.030715, 1.026002, 1.194292.

The forward/reverse pairing cancels which arm receives the colder first slot, but the standalone ratios are supporting evidence only. The graph-on end-to-end TTFT result is the ship metric.

## Weighted-pass arithmetic

At N=5909, using the traced gate/up and down budgets:

- gfx1151: `4194.6 - 4194.6/1.042626 = 171.5 ms` saved in gate/up and `2092.0 - 2092.0/1.087332 = 168.0 ms` in down, for **339.5 ms** total. This is about **3.20%** of the 10,611 ms traced pass budget before considering other unchanged work.
- gfx1100: `1288.3 - 1288.3/1.089053 = 105.4 ms` saved in gate/up and `679.2 - 679.2/1.084907 = 53.1 ms` in down, for **158.5 ms** total, or **5.49%** of the 2,889.6 ms measured GEMM pass.

The measured end-to-end results below are consistent in direction and magnitude.

## End-to-end TTFT-5909

Command contract: graph-on production dispatch, `--spec off`, exact prompt MD5 `ed720348b81a19fab64d4783c75c1ae3`, two warmups, and fresh-process A1/B1/B2/A2. Halo used eight timed samples per process; gfx1100 used the revised four-sample protocol after two warmups.

| GPU | OFF A1 ms | OFF A2 ms | ON B1 ms | ON B2 ms | paired OFF ms | paired ON ms | speedup | ON tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gfx1100 | 3286.027 | 3311.822 | 3154.849 | 3163.478 | 3298.924 | 3159.164 | **1.044240x** | **1870.43** |
| gfx1151 | 8947.753 | 9004.573 | 8551.539 | 8548.750 | 8976.163 | 8550.145 | **1.049826x** | **691.10** |

This is +4.42% on gfx1100 and +4.98% on gfx1151. Neither card regresses.

## c24 quality and provenance

All recorded runs scored 24/24 chunks and 24,552 tokens. The same-build paired gfx1100 result is:

| GPU | reference | OFF KLD | ON KLD | delta |
|---|---|---:|---:|---:|
| gfx1100 | WT2 | 0.088665 | 0.086942 | -0.001723 |
| gfx1100 | agentic | 0.256882 | 0.253436 | -0.003446 |
| gfx1151 | WT2 | not rerun; predicted/entailed 0.088665 | 0.086942 | predicted -0.001723 |
| gfx1151 | agentic | not rerun; predicted/entailed 0.256882 | 0.253436 | predicted -0.003446 |

The WT2 ON result is below the required 0.10 budget. ON is quality-positive on both references relative to its own build's OFF arm.

The absolute OFF result is digit-for-digit identical to `Gfx11A4Cand`'s independently built baseline, resolving the apparent shift from earlier campaign absolutes as build provenance rather than a symfold regression. The ON results are also digit-for-digit identical across gfx1100 and gfx1151. Before the optional Halo OFF confirmation, the explicit prediction was 0.088665 / 0.256882; it was not spent because the same-build OFF pair plus cross-architecture identity of both ON results entail that row under the shared LF16 source/accumulation path. The missing Halo OFF cell is therefore explicit, not a failed or omitted measurement.

## Decoded batteries

A manually started typed-config server was used for each arm, and `serve_harness.py --no-spawn` connected to it. This avoids the stock harness's isolated config replacing the typed experimental knob.

| GPU | arm | decoded prompts | runaway | empty | attractor | retrieval miss |
|---|---|---:|---:|---:|---:|---:|
| gfx1100 | OFF | 5/5 | 0 | 0 | 0 | 0 |
| gfx1100 | ON | 5/5 | 0 | 0 | 0 | 0 |
| gfx1151 | OFF | 5/5 | 0 | 0 | 0 | 0 |
| gfx1151 | ON | 5/5 | 0 | 0 | 0 | 0 |

All twenty responses were readable and terminated normally.

## Provenance and validation

- branch base: `7a34a572a` plus cherry-picked gfx1100 LF16 commit `e296c2bb9`;
- gridspec remains default-on; LF16 remains default-off and gfx1100-gated; symfold is new and default-off;
- model SHA-256: `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`;
- model stat tuple: `14987185152 1789995527 5398606`;
- prompt MD5: `ed720348b81a19fab64d4783c75c1ae3`;
- remote `cargo build --release` passed; remote `cargo build --release --example eval_hipfire` passed;
- both device runs asserted `gcnArchName` (`gfx1100`, `gfx1151`);
- zero spill/private-segment evidence came from both ISA and HIP function-resource inspection.

Primary artifacts are `exact-gfx1100.log`, `exact-gfx1151.log`, `ttft-gfx*.json`, `c24pair-gfx1100-*.stderr`, `c24-gfx1151-*.stderr`, `battery-gfx*-*.console`, the scalar harness, and the ISA census in this directory.
