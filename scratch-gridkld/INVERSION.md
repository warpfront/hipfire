# Shift-fold daemon inversion

## Verdict

**KILL the daemon candidate and do not extend it to the family.** The inversion is class (2): the standalone rig and daemon do different work. There is no hidden scheduling win to recover.

Two route errors obscured that result before the corrected gate:

1. The first profiler replay omitted the common and arm flags from its `configure` record. Once the daemon installs `ProcessConfig`, ambient variables do not override that snapshot. Both nominal arms therefore ran the same incumbent IU4 symfold kernel.
2. Embedding the reported common flags still leaves `kernel.gfx12_fp8_stream=true`, its gfx1201 default. With both IU4 producer fusions disabled, the FP8 stream wins the dispatch ladder before standalone IU4. Those approximately 2,260 tok/s results are FP8v2 results, not shift-fold results.

The honest row-global shift-fold configuration is:

```text
kernel.iu4_prefill=true
kernel.gfx12_silu_quant_fused=false
kernel.gfx12_producer_quant_fused=false
kernel.gfx12_fp8_stream=false
developer.iu4_global_a_screen="1"
developer.iu4_shift_foldfree="0" or "1"
```

On that route the actual daemon IU4 GEMMs do not get faster: set is **130.822 -> 132.276 us/token (+1.112%)**, add is **61.414 -> 61.209 us/token (-0.334%)**, and their sum is **192.235 -> 193.485 us/token (+0.650%, slower)**. Independent card-C profiling agrees on the conclusion: **193.094 -> 193.708 us/token (+0.318%, slower)**.

Corrected card-E pp8192 ABBA is neutral/noise (**+0.126% candidate throughput**), while corrected client TTFT C-X-X-C is a clear failure: **1,849.956 -> 1,883.053 ms, +33.097 ms / +1.789% latency**. The candidate fails the daemon gate even before the separate artifact-quality decision.

The independent configuration audit is `/home/kaden/ClaudeCode/warpfront/wt-inv/scratch-inv/INVERSION.md` at commit `9547a6872`. This report owns the route and performance diagnosis; the audit owns the command/configuration-chain review.

## Provenance

- Candidate source/binary worktree: `/home/kaden/ClaudeCode/warpfront/wt-iu4free`, commit `463731a6f`
- Daemon md5: `e0659c8e702f6f905888f83dcf10eb19`
- Artifact: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq`
- Artifact sha256: `febf70570354ccf8867ae2c7330da0dd80b4263e3ce2603f724abd62eccaf7c8`
- Card-C: `GPU-085289909a86cc63`; card-E: `GPU-05f92432f2312a0e`
- Kernel profile rows below cover two pp8192 calls, so total kernel nanoseconds are divided by 16,384 tokens.
- C is symfold (`developer.iu4_shift_foldfree="0"`); X is shift-fold (`"1"`). All other process configuration is identical within each pair.

## Route truth table

Assumptions for this table: gfx1201, MQ4v2 Lloyd weights, IU4 enabled, eager execution, admitted batch and K. `S` is `kernel.gfx12_silu_quant_fused`, `P` is `kernel.gfx12_producer_quant_fused`, `A` is `developer.iu4_global_a_screen`, and `F` is `kernel.gfx12_fp8_stream`.

`P` controls gate/up, qkv, qkvza, and the other non-down prepared producers. `S` controls down. Prepared IU4 is checked first, FP8 stream second, and standalone quantization last. `A` is consulted only by the standalone fallback, so it has no effect on prepared-IU4 or FP8 routes.

| S | P | A | F | Non-down projections | Down projection |
|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 0 | standalone IU4, `quantize_int4_mmq_ds128` | standalone IU4, `quantize_int4_mmq_ds128` |
| 0 | 0 | 1 | 0 | standalone IU4, `quantize_int4_mmq_row` | standalone IU4, `quantize_int4_mmq_row` |
| 0 | 0 | 0 | 1 | FP8v2 stream | FP8v2 stream |
| 0 | 0 | 1 | 1 | FP8v2 stream | FP8v2 stream |
| 0 | 1 | 0 | 0 | prepared IU4 | standalone IU4, `quantize_int4_mmq_ds128` |
| 0 | 1 | 1 | 0 | prepared IU4 | standalone IU4, `quantize_int4_mmq_row` |
| 0 | 1 | 0 | 1 | prepared IU4 | FP8v2 stream |
| 0 | 1 | 1 | 1 | prepared IU4 | FP8v2 stream |
| 1 | 0 | 0 | 0 | standalone IU4, `quantize_int4_mmq_ds128` | prepared IU4 |
| 1 | 0 | 1 | 0 | standalone IU4, `quantize_int4_mmq_row` | prepared IU4 |
| 1 | 0 | 0 | 1 | FP8v2 stream | prepared IU4 |
| 1 | 0 | 1 | 1 | FP8v2 stream | prepared IU4 |
| 1 | 1 | 0 | 0 | prepared IU4 | prepared IU4 |
| 1 | 1 | 1 | 0 | prepared IU4 | prepared IU4 |
| 1 | 1 | 0 | 1 | prepared IU4 | prepared IU4 |
| 1 | 1 | 1 | 1 | prepared IU4 | prepared IU4 |

Source anchors:

- Gate/up route order: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6980-7036`
- Down route order: `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:7326-7377`
- Producer admission: `crates/rdna-compute/src/dispatch.rs:3040-3069`
- FP8-stream admission: `crates/rdna-compute/src/dispatch.rs:3086-3101`
- Row-global selection only in standalone quantization: `crates/rdna-compute/src/scratch.rs:1513-1519`
- gfx1201 defaults S=1, P=1, F=1: `crates/rdna-compute/src/feature_flags.rs:652-659`
- Stable config names shadow developer spellings: `crates/hipfire-config/src/lib.rs:3295-3304`
- Installed process config is the runtime source of truth: `crates/hipfire-config/src/lib.rs:3433-3445`

### Orientation rule

A shipped-ish IU4 arm is near **1,671 ms TTFT / 3,535 pp tok/s** on the 5,909-token prompt and around **3,580 pp8192 tok/s** on the shipped artifact. FP8v2 is near **2,610 ms TTFT / 2,260 pp8192 tok/s**. A supposed IU4 control in the FP8 band is misconfigured and must not be used for a kernel verdict.

The forced standalone row-global route pays about 21 us/token for its separate row quantizer, so its corrected pp8192 band is approximately 3,270-3,290 tok/s on card-E. Its symbols, not a throughput guess, prove the route: 512 `quantize_int4_mmq_row` calls plus IU4 `full_set`/`full_add` symbols and no FP8v2 GEMM symbols.

## How the earlier profile compared the wrong kernels

The nominal exact-common-flag profile kept F at its gfx1201 default of one. Both arms dispatched FP8v2 symbols; changing the shift-fold flag could not affect those kernels. Steady wall was 429.384 -> 431.205 us/token (+0.424% latency), while all kernel time was 426.622 -> 428.194 us/token (+0.368%). The full profile is included to make the route failure explicit.

| Kernel | Calls C/X | C us/tok | X us/tok | Delta us/tok | Delta |
|---|---:|---:|---:|---:|---:|
| `gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` | 128/128 | 162.466 | 163.069 | +0.603 | +0.371% |
| `gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold` | 256/256 | 103.499 | 103.941 | +0.442 | +0.427% |
| `gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` | 96/96 | 55.311 | 55.567 | +0.257 | +0.464% |
| `fused_silu_mul_mq_rotate_awq_mq4v2_fp8_gfx12` | 128/128 | 24.940 | 24.934 | -0.005 | -0.021% |
| `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` | 32/32 | 23.480 | 23.590 | +0.110 | +0.469% |
| `gemm_qkv_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` | 32/32 | 16.357 | 16.421 | +0.064 | +0.391% |
| `fused_rmsnorm_mq_rotate_awq_mq4v2_fp8_gfx12` | 256/256 | 10.642 | 10.659 | +0.017 | +0.158% |
| `gdn_chunk_scan` | 1536/1536 | 8.907 | 8.939 | +0.031 | +0.351% |
| `gated_norm_mq_rotate_awq_mq4v2_fp8_gfx12` | 96/96 | 6.933 | 6.946 | +0.013 | +0.194% |
| `gdn_chunk_prep` | 96/96 | 5.299 | 5.302 | +0.003 | +0.062% |
| `deinterleave_q_rmsnorm_f32_batched` | 32/32 | 2.708 | 2.708 | +0.001 | +0.025% |
| `sigmoid_mul_rotate_x_mq_awq_mq4v2_fp8_gfx12` | 32/32 | 2.157 | 2.158 | +0.000 | +0.020% |
| `gdn_chunk_kkt_solve` | 1536/1536 | 2.038 | 2.042 | +0.004 | +0.210% |
| `rope_partial_halfsplit_batched_f32` | 32/32 | 0.715 | 0.716 | +0.001 | +0.135% |
| `rmsnorm_f32` | 34/34 | 0.429 | 0.430 | +0.002 | +0.411% |
| `kv_cache_write_fp8_e4m3_batched` | 64/64 | 0.359 | 0.361 | +0.001 | +0.364% |
| `__amd_rocclr_fillBufferUnAligned` | 1824/1824 | 0.190 | 0.210 | +0.020 | +10.415% |
| `gemv_mq4g256v2_multirow_r2` | 2/2 | 0.131 | 0.132 | +0.001 | +0.596% |
| `__amd_rocclr_copyBuffer` | 180/180 | 0.030 | 0.030 | +0.000 | +0.981% |
| `embedding_q8_batched` | 2/2 | 0.030 | 0.037 | +0.007 | +22.007% |
| `mq_rotate_x` | 2/2 | 0.000 | 0.000 | +0.000 | +0.862% |

## Honest IU4 full-kernel profile

This pair sets F=0, emits the row quantizer, and changes all IU4 consumers from `_symfold` to `_shiftfold`. Sorted by control cost:

| Kernel | Calls C/X | C us/tok | X us/tok | Delta us/tok | Delta |
|---|---:|---:|---:|---:|---:|
| `gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold` | 736/736 | 130.822 | 132.276 | +1.455 | +1.112% |
| `gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold` | 256/256 | 61.414 | 61.209 | -0.205 | -0.334% |
| `attention_fp8_e4m3_fa2_gqa_packet_gfx1201` | 32/32 | 23.744 | 22.720 | -1.024 | -4.312% |
| `fused_silu_mul_mq_rotate_awq` | 128/128 | 22.032 | 22.200 | +0.168 | +0.760% |
| `quantize_int4_mmq_row` | 512/512 | 21.071 | 20.881 | -0.190 | -0.902% |
| `gdn_chunk_scan` | 1536/1536 | 9.003 | 8.455 | -0.548 | -6.085% |
| `fused_rmsnorm_mq_rotate_awq` | 256/256 | 8.693 | 8.799 | +0.106 | +1.215% |
| `gated_norm_f32` | 96/96 | 5.864 | 5.874 | +0.010 | +0.173% |
| `gdn_chunk_prep` | 96/96 | 5.269 | 5.353 | +0.084 | +1.603% |
| `rotate_x_mq_awq` | 128/128 | 4.912 | 4.911 | -0.001 | -0.022% |
| `deinterleave_q_rmsnorm_f32_batched` | 32/32 | 2.726 | 2.792 | +0.066 | +2.410% |
| `gdn_chunk_kkt_solve` | 1536/1536 | 2.058 | 1.949 | -0.109 | -5.284% |
| `sigmoid_mul_f32` | 32/32 | 1.904 | 1.905 | +0.000 | +0.006% |
| `rope_partial_halfsplit_batched_f32` | 32/32 | 0.720 | 0.744 | +0.024 | +3.330% |
| `rmsnorm_f32` | 34/34 | 0.434 | 0.417 | -0.017 | -3.915% |
| `kv_cache_write_fp8_e4m3_batched` | 64/64 | 0.376 | 0.403 | +0.026 | +6.974% |
| `__amd_rocclr_fillBufferUnAligned` | 2016/2016 | 0.220 | 0.215 | -0.005 | -2.142% |
| `gemv_mq4g256v2_multirow_r2` | 2/2 | 0.133 | 0.134 | +0.001 | +0.448% |
| `__amd_rocclr_copyBuffer` | 180/180 | 0.031 | 0.030 | -0.001 | -1.850% |
| `embedding_q8_batched` | 2/2 | 0.030 | 0.030 | -0.000 | -0.211% |
| `mq_rotate_x` | 2/2 | 0.000 | 0.000 | -0.000 | -5.983% |

All-kernel sum is 301.457 -> 301.298 us/token (-0.053%). That tiny apparent gain comes from unrelated attention/GDN run noise offsetting the **+1.250 us/token IU4 GEMM regression**. It is not evidence that saved GEMM time moved into another subsystem: there is no saved GEMM time on the daemon shapes.

### IU4 shape split

The critical shape difference is visible after splitting the aggregated set kernel by launch grid. `M` here is the output dimension recovered from `GridX / 2`; the profile contains two pp8192 calls.

| Kernel | M | raw GridX | GridY C/X | Calls C/X | C us/tok | X us/tok | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| set | 128 | 256 | 64/32 | 192/192 | 0.695 | 1.328 | +91.03% |
| set | 1,024 | 2,048 | 64/32 | 64/64 | 1.370 | 1.385 | +1.166% |
| set | 6,144 | 12,288 | 64/32 | 96/96 | 12.072 | 11.856 | -1.783% |
| set | 10,240 | 20,480 | 64/32 | 96/96 | 20.068 | 19.743 | -1.616% |
| set | 12,288 | 24,576 | 64/32 | 32/32 | 8.017 | 7.872 | -1.801% |
| set | 17,408 | 34,816 | 64/32 | 256/256 | 88.601 | 90.091 | +1.682% |
| add | 5,120 | 10,240 | 64/32 | 256/256 | 61.414 | 61.209 | -0.334% |

The dominant daemon gate and up operations are separate M=17,408 set calls. The standalone rig joined them as one M=34,816 gate/up launch, where shift-fold measured +10.58%; it also tested down as set, while the daemon uses M=5,120 add. Therefore the rig did not reproduce either important daemon call shape.

The candidate launch also doubles the N tile from 128 to 256 (`crates/rdna-compute/src/gemm.rs:19758-19766`) and uses 30,720 bytes LDS instead of 20,480 (`:19762`), reducing residency from four to two blocks/CU. [INFERENCE] That trade becomes unfavorable after gate/up is split into the smaller M=17,408 launches. Independent exact-shape card-C timings support this: M=17,408 set is **5,239.8 -> 5,462.2 us (+4.24%)** and M=5,120 add is **5,240.7 -> 5,328.4 us (+1.67%)**.

For the 5,909-token TTFT prompt the 256-row candidate tile also covers 6,144 rows, while the 128-row control covers 6,016. That extra padding is consistent with the stronger short-prompt regression.

## Corrected daemon gates

### pp8192

Card-E fresh-daemon ABBA, using the second warmed request from each replay:

| Pair | C tok/s | X tok/s | X delta |
|---|---:|---:|---:|
| 1 | 3,291.1 | 3,288.2 | -0.088% |
| 2 | 3,272.5 | 3,283.6 | +0.339% |
| paired mean | 3,281.8 | 3,285.9 | **+0.126%** |

Independent card-C C-S-C-S medians measured 2,511.378 -> 2,522.030 ms, **+0.424% latency / -0.422% throughput**. The opposite small signs across cards classify pp8192 as neutral/noise, not a converted standalone win.

### Client TTFT, 5,909-token prompt

Card-E C-X-X-C. Each process used two warmups and eight timed client-stream measurements with prompt md5 `ed720348b81a19fab64d4783c75c1ae3`.

| Pair | C median ms | X median ms | X latency delta |
|---|---:|---:|---:|
| 1 | 1,843.537 | 1,883.149 | +39.612 ms / +2.149% |
| 2 | 1,856.374 | 1,882.957 | +26.583 ms / +1.432% |
| paired mean | 1,849.956 | 1,883.053 | **+33.097 ms / +1.789%** |

The corresponding CLI prefill-rate proxy medians average 3,194.170 -> 3,137.992 tok/s, **-1.759%**. Independent card-C direct 5,909-token prefill walls agree: 1,845.568 -> 1,874.373 ms, **+28.805 ms / +1.560%**. A separate card-C client TTFT C-X-C-X was noisier but had the same sign: pooled eight-sample medians were 1,840.104 -> 1,859.549 ms, **+19.445 ms / +1.057%**.

## Evidence files

- Honest replays: `scratch-gridkld/inversion-replay-control-forceiu4.jsonl`, `scratch-gridkld/inversion-replay-candidate-forceiu4.jsonl`
- Honest profile databases and CSVs: `scratch-gridkld/inversion-prof-control-forceiu4/`, `scratch-gridkld/inversion-prof-candidate-forceiu4/`
- Wrong-route replays: `scratch-gridkld/inversion-replay-control-active.jsonl`, `scratch-gridkld/inversion-replay-candidate-active.jsonl`
- Wrong-route profile databases and CSVs: `scratch-gridkld/inversion-prof-control-active/`, `scratch-gridkld/inversion-prof-candidate-active/`
- Card-E pp outputs: `scratch-gridkld/inversion-pp-A1-control.jsonl`, `inversion-pp-B1-candidate.jsonl`, `inversion-pp-B2-candidate.jsonl`, `inversion-pp-A2-control.jsonl`

No source change is proposed. The corrected measurements close the performance question: shift-fold does not convert its synthetic +7.14% aggregate into daemon IU4 prefill, and it fails TTFT.