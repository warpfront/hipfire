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

## Fragment-order direct-A follow-up

### Producer layout and CPU proof

The fused FP8 producers were given fragment-order variants using

```text
lane = (row & 15) + 16 * ((k >> 3) & 1)
offset = (((row >> 4) * (K >> 4) + (k >> 4)) * 32 + lane) * 8 + (k & 7)
```

Only the FP8 byte plane is padded to a complete 256-row GEMM tile. The decoded
half-sum and row-scale planes retain their true-N extents. A complete CPU
permutation/inverse check covered every byte at the pp8192 gate shape and a
non-aligned tail:

| N | padded N | K | bytes | permutation | padded tail |
|---:|---:|---:|---:|---|---|
| 8,192 | 8,192 | 5,120 | 41,943,040 | exact bijection | exact zero |
| 8,195 | 8,448 | 5,120 | 43,253,760 | exact bijection | exact zero |

All ten producer source variants (RMSNorm, SwiGLU, rotate, sigmoid-gated
rotate, gated norm, and their AWQ twins) compiled for gfx1201.

A gfx1201 device smoke check packed `N=259, Npad=512, K=512` through the
row-major and fragment-order helper variants. Untiling the fragment byte plane
matched every row-major byte, the padded tail was all zero, and half sums plus
row scales matched (`byte_mismatch=0`, `tail_nonzero=0`,
`metadata_mismatch=0`).

### ISA gate

The final direct-A form uses four wave-uniform scalar fragment bases and one
unsigned lane offset. Parenthesizing the complete 32-bit offset before pointer
addition was necessary for SADDR selection. An empty compiler memory barrier
after the sixteen source loads was also necessary to prevent the scheduler
from stranding one four-load group above W expansion.

The resulting gate/up ISA has exactly one contiguous clause per K64:

```text
s_clause 0xf
global_load_b64 ..., v180, s[18:19]
global_load_b64 ..., v180, s[18:19] offset:256
...
global_load_b64 ..., v180, s[24:25] offset:768
```

There are sixteen scalar-address `global_load_b64` instructions in that
clause. The following 64-WMMA drain contains no `s_wait_loadcnt`; its waits are
only `s_wait_dscnt` for W LDS reads. The A LDS plane is absent. Dynamic LDS is
9,728 B (9,216 B W plus 512 B LUT).

| family | SGPR | VGPR | scratch/spill | compiler waves/SIMD | measured blocks/CU |
|---|---:|---:|---:|---:|---:|
| gate/up | 34 | 206 | 0 / 0 | 7 | 3 |
| residual | 34 | 205 | 0 / 0 | 7 | 3 |

The LDS reduction did not reach four blocks/CU because VGPR allocation remains
the limiting resource.

### Real-slab exactness

Card C, real pow2-half qt44 slabs, N=256:

| family | K | relative RMS | max absolute | nonfinite |
|---|---:|---:|---:|---:|
| gate/up | 5,120 | 0 | 0 | 0 |
| residual | 17,408 | 0 | 0 | 0 |

### Standalone decision gate

Card C, N=8192, five warmups, median of 20 HIP-event timings:

| family | median us | TFLOP/s | shipped median us | time delta |
|---|---:|---:|---:|---:|
| gate/up | 16,849.316 | 173.335 | 18,059.408 | -6.70% |
| residual | 9,470.285 | 154.197 | 8,029.042 | +17.95% |
| combined | 26,319.601 | 166.449 | 26,088.450 | +0.89% |

The direct-A candidate improves gate/up but loses materially on residual and
misses the shipped combined baseline by 0.88% throughput. It therefore fails
the required first-two-family win. QKV/QKVZA, daemon pairs, KLD, TTFT, battery,
and profiling were intentionally not run. The implementation is retained only
behind explicit `HIPFIRE_FP8_FRAGMENT_ORDER=1`; the default fold-free route
remains the shipped staged-A implementation.

## Fragment-order W preshuffle follow-up

Commits:

- `a7a200de3` `gfx1201: preshuffled fragment-order W for fold-free fp8 GEMM`
- `fca0979ed` `gfx1201: extend fold-free W preshuffle to qkvza`
- `8bf488a1b` `gfx1201: extend fold-free W preshuffle to qkv`
- `ba8c5ac6b` `gfx1201: apply W preshuffle to symmetric-fold fp8 GEMM`

The CPU proof remains bit exact: bijection, round trip, dequantization, and
padded-tail checks all pass for 19 logical rows in 32 storage rows. The
permutation replaces the uploaded MQ4V2 byte buffer without changing its
length, so prepared-weight VRAM delta is exactly 0 bytes.

On card C, the fold-free gate/up and residual pair measured 14,795.144 us
(197.401 TFLOP/s) and 8,826.476 us (165.444 TFLOP/s), respectively:
23,621.620 us combined and 185.46 aggregate TFLOP/s. This reproduces the
pre-reboot 186.099 TFLOP/s result within 0.34%. QKVZA and QKV real-slab
checks at N=256 were exact (`rel_rms=0`, `max_abs=0`, no nonfinite values).
Their warmed standalone results were 7,007.517 us (197.280 TFLOP/s) and
5,978.076 us (201.167 TFLOP/s).

The pow2-half artifact remains a quality dead end: its 24-chunk KLD is
0.056627, above the 0.05 shipping limit. The symmetric artifact
`qwen3.8-27b.mq4v2.xt.sym-a035.qat-r5s100.hfq` measured KLD 0.045510,
mean NLL 1.858547, PPL 6.4144 over 24,552 tokens; a two-chunk candidate and
row-major control were byte-identical (MD5
`d4b51612337125932d5e338299f83ca7`).

The symmetric standalone candidate measured 17,866.916 us gate/up
(163.463 TFLOP/s) and 7,944.252 us residual (183.817 TFLOP/s), for
25,811.168 us and **169.728 aggregate TFLOP/s**, 1.07% above the shipped
167.924 TFLOP/s baseline.

### Symmetric daemon gate

Card B paired ABBA rows, with IU4 disabled:

| pair | pp512 OFF | pp512 ON | pp8192 OFF | pp8192 ON | tg128 OFF | tg128 ON |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2258.1 | 2304.9 | 2334.0 | 2379.7 | 36.681 | 27.718 |
| 2 | 2213.5 | 2308.0 | 2297.3 | 2377.5 | 36.572 | 27.680 |

Mean pp8192 increased by 62.95 tokens/s, or 2.72%. TTFT changed from
2612.190 ms OFF to 2535.006 ms ON, a 77.183 ms (2.95%) improvement.
With the normal flag-free IU4 route, the single measured TTFT row was
1624.381 ms (3637.696 effective prefill tokens/s).

The required battery gate failed decisively: the preshuffled symmetric route
scored **0/5**, producing only repeated exclamation marks for every prompt,
while the row-major control remained coherent. Synthetic production-shape
scalar gate/up and residual checks matched the CPU reference to relative RMS
1.334e-6 and 2.557e-6, so the failure is in integration rather than the
standalone permutation formula. The candidate is therefore not shippable and
profiling was not used to override the correctness gate. Symmetric
preshuffling is retained only as an explicit experiment with
`HIPFIRE_FP8_WPRESHUFFLE=1`; the default current-artifact route remains the
row-major shipped implementation.
