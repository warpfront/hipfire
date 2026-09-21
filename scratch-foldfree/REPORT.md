# gfx1201 fold-free FP8 prefill GEMM

## Scope

The four MQ4G256V2 FP8-v2 prefill families now have fold-free gfx1201 kernels: gate/up, residual, qkvza, and qkv. The 256x128 workgroup is eight wave32s, each wave owns a 64x64 output tile, and all 16 accumulator vectors remain live across K. The existing producer is row-major and does not provide efficient lane-linear scalar fragments, so A is cooperatively staged in two 128-row phases. Packed W expands once per K64 tile through the exponent-fold LUT. The 18,944-byte dynamic-LDS footprint admits three blocks/CU.

The HFQ loader validates `mq4v2.pow2scale`, derives one F32 `2^e_row` sidecar per output row from real qt44 headers, and associates it with the uploaded weight pointer. Nonzero pow2scale selects the new kernels; `HIPFIRE_FP8_FOLDFREE=0` restores the incumbent symmetric-fold route. Non-pow2 artifacts are unchanged.

## Fixture and hardware

- Artifact: `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-pow2h-a035.hfq`
- Artifact SHA-256: `febf70570354ccf8867ae2c7330da0dd80b4263e3ce2603f724abd62eccaf7c8`
- Metadata: `mq4v2.symmetric=1`, `mq4v2.pow2scale=2`
- Quality reference: `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`
- Kernel/exactness/standalone throughput: Card C, `GPU-085289909a86cc63`
- KLD, daemon pairs, battery, and rocprof: Card B, `GPU-e475645fe0200397`, HOME `/home/kaden/.hipfire-homes/ab1`
- Daemon gates: fp8 KV, `HIPFIRE_GRAPH=1`, `HIPFIRE_IU4_PREFILL=0`, fresh daemon per arm; ABBA matrix order.

## Compiler resources

All four selector builds used `hipcc --offload-arch=gfx1201 -O3 -Rpass-analysis=kernel-resource-usage`.

| family | VGPR | SGPR | spills/scratch | compiler occupancy | dynamic LDS | measured blocks/CU |
|---|---:|---:|---:|---:|---:|---:|
| gate/up | 185 | 36 | 0 | 8 waves/SIMD | 18,944 B | 3 |
| residual | 185 | 36 | 0 | 8 waves/SIMD | 18,944 B | 3 |
| qkvza | 185 | 61 | 0 | 8 waves/SIMD | 18,944 B | 3 |
| qkv | 185 | 41 | 0 | 8 waves/SIMD | 18,944 B | 3 |

## Real-slab numerical oracle

`scratch-foldfree/bench_foldfree.hip` reads the actual pow2-half qt44 slabs, independently decodes and applies the two-stage E4M3 rounding/exponent-fold contract on CPU, and compares 128 output rows at N=256. Residual starts from 0.125 to cover the add epilogue.

| family | K | relative RMS | max absolute | nonfinite |
|---|---:|---:|---:|---:|
| gate/up | 5,120 | 0 | 0 | 0 |
| residual | 17,408 | 0 | 0 | 0 |
| qkvza | 5,120 | 0 | 0 | 0 |
| qkv | 5,120 | 0 | 0 | 0 |

Gate: relative RMS <= 1e-6, no nonfinite. Result: PASS for all four families.

## Standalone full-shape throughput

Card C, N=8192, five warmups, median of 20 HIP-event timings.

| family | M | K | median us | TFLOP/s |
|---|---:|---:|---:|---:|
| gate/up | 34,816 | 5,120 | 19,884.857 | 146.874 |
| residual | 5,120 | 17,408 | 9,227.285 | 158.258 |
| qkvza | 16,480 | 5,120 | 9,044.764 | 152.845 |
| qkv | 14,336 | 5,120 | 7,751.771 | 155.138 |
| aggregate | — | — | 45,908.677 | 151.734 |

The prior standalone aggregate was 167.924 TFLOP/s; the new aggregate is 9.64% lower. This is recorded rather than hidden; the in-daemon pp8192 result below is positive.

## Quality

`eval_hipfire`, prefill scoring, fp8/q8 KV, all 24 WT2 chunks (24,552 scored tokens):

| route | KLD | mean NLL | PPL | gate vs 0.057072 |
|---|---:|---:|---:|---:|
| fold-free | **0.056627** | 1.883571 | 6.5769 | -0.000445, PASS |

The required tolerance is +/-0.001 around 0.057072.

## Paired daemon matrix

`hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`.

| pair | pp512 OFF | pp512 ON | delta | pp8192 OFF | pp8192 ON | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2263.7 | 2154.3 | -4.83% | 2333.5 | 2345.6 | +0.52% |
| 2 | 2220.3 | 2151.8 | -3.09% | 2297.9 | 2341.7 | +1.91% |

Mean paired pp8192 delta: **+1.21%**. The pp512 loss is retained in the evidence.

## TTFT pair

5909-token fixture (`benchmarks/prompts/ttft_5900.txt`), eight measured runs after two warmups:

| arm | median TTFT | effective prefill |
|---|---:|---:|
| OFF | 2610.252 ms | 2263.767 tok/s |
| ON | 2606.674 ms | 2266.874 tok/s |

Delta: **-3.578 ms (-0.137%) TTFT**, +0.137% effective prefill throughput.

## Serve battery

`scripts/serve_harness.py`, recipe:nothink, fp8/vmm, speculation off, five built-in genres: **5/5 PASS**, finish=stop for every turn, runaway=0, empty=0, attractor=0, retrieval_miss=0. Evidence: `scratch-foldfree/battery.json`.

## In-daemon rocprof

The captured daemon protocol was reduced to `configure, ping, diag, load, diag, bench_prefill(8192), bench_prefill(8192)` and replayed directly under rocprofv3 so daemon EOF flushes the trace. Normalizer: 2 x 8192 = 16,384 tokens.

| GEMM family | calls | us/token |
|---|---:|---:|
| gate/up fold-free | 128 | 155.450 |
| residual fold-free | 256 | 106.557 |
| qkvza fold-free | 96 | 56.760 |
| qkv fold-free | 32 | 16.138 |
| **GEMM total** | — | **334.906** |

Compared with the requested 340.5 us/token reference: **-5.594 us/token (-1.64%)**. Evidence: `scratch-foldfree/prof-foldfree/bench_kernel_stats.csv`.

## Build and commits

Scoped release builds passed for `hipfire-daemon`, `hipfire-cli`, and the `hipfire-runtime` `eval_hipfire` example. Earlier scoped `cargo check -p rdna-compute` and `cargo check -p hipfire-arch-qwen35` also passed (pre-existing warnings only).

- `20b743e0c` gate/up kernel and shared body
- `f08a1f0c9` residual selector
- `4409c7c92` qkvza selector
- `cea320ca9` qkv selector
- `9ee38ac79` metadata, loader sidecars, dispatch, and launch integration
