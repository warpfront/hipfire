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

## Fragment-order W preshuffle follow-up — final decision

Commits:

- `a7a200de3` `gfx1201: preshuffled fragment-order W for fold-free fp8 GEMM`
- `fca0979ed` `gfx1201: extend fold-free W preshuffle to qkvza`
- `8bf488a1b` `gfx1201: extend fold-free W preshuffle to qkv`
- `ba8c5ac6b` `gfx1201: apply W preshuffle to symmetric-fold fp8 GEMM`
- `f43e0f24d` `gfx1201: coalesced K64 fragment order for symmetric W preshuffle`

The CPU proof remains bit exact: bijection, round trip, dequantization, and
padded-tail checks all pass for 19 logical rows in 32 storage rows. The
permutation replaces the uploaded MQ4V2 byte buffer without changing its
length, so prepared-weight VRAM delta is exactly 0 bytes.

The fold-free result remains 185.46 aggregate TFLOP/s after reboot, within
0.34% of the earlier 186.099 TFLOP/s result and 10.8% above the shipped
symmetric-fold baseline. It is not a shipping candidate: the required pow2
grid's 24-chunk KLD is 0.056627, above the hard 0.05 limit. The current
symmetric artifact remains inside the limit at KLD 0.045510, mean NLL
1.858547, and PPL 6.4144 over 24,552 tokens. A two-chunk candidate and
row-major control were byte-identical (MD5
`d4b51612337125932d5e338299f83ca7`).

### Final coalesced-K64 standalone and exactness

Card C, N=8192, five in-process warmups, median of 20 HIP-event timings:

| family | median us | TFLOP/s |
|---|---:|---:|
| gate/up | 17,833.392 | 163.770 |
| residual | 8,063.789 | 181.092 |
| combined | 25,897.181 | 169.164 |

The combined result is **0.74% above** the shipped 167.924 TFLOP/s baseline
and 0.33% below the earlier pre-remap 169.728 TFLOP/s result. It misses the
1.5% standalone shipping bar.

Final-mapping exactness was checked against the row-major symmetric kernel
using real layer-0 slabs from
`qwen3.8-27b.mq4v2.xt.sym-a035.qat-r5s100.hfq`, N=256, and 128 output rows
per projection:

- gate/up: 65,536 float outputs were byte-identical, MD5
  `4971f449cc6f63a0eae70a4c6a780811`;
- residual: 32,768 float outputs were byte-identical, MD5
  `90fcff8968871a4cb2051a548c9bc127`;
- neither path produced a non-finite value;
- the independent production-shape scalar gate/up check had relative RMS
  `1.33396503e-6` and max absolute error `4.48226929e-5`;
- the prior QKVZA and QKV real-slab checks remain exact (`rel_rms=0`,
  `max_abs=0`, no non-finite values).

### Mandatory final decode gate

The coalesced-K64 candidate was built in release mode and run first on card B
with IU4 disabled, pp512/ctx128/tg128, three measured runs, and one warmup.
The saved evidence is `symwp-final-decode.json`.

- pp512 median: 2312.5 tokens/s;
- tg128 median: **31.1583 tokens/s**;
- tg128 samples: 31.2260, 31.1583, 31.1046 tokens/s;
- validated shipped baseline: 36.68 tokens/s;
- candidate delta: **-15.05%**.

This is far outside the required 1% decode-neutral band. The mandated
early-stop therefore fired. Final-remap pp8192, TTFT, KLD, long-context,
battery, profiling, and IU4 reruns were intentionally not performed.

### Archival pre-remap daemon evidence

These numbers predate the final coalesced-K64 remap and are retained only to
explain the decision history; they are not final-candidate measurements.

| pair | pp512 OFF | pp512 ON | pp8192 OFF | pp8192 ON | tg128 OFF | tg128 ON |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2258.1 | 2304.9 | 2334.0 | 2379.7 | 36.681 | 27.718 |
| 2 | 2213.5 | 2308.0 | 2297.3 | 2377.5 | 36.572 | 27.680 |

The pre-remap mean pp8192 gain was 62.95 tokens/s, or 2.72%. Its TTFT changed
from 2612.190 ms OFF to 2535.006 ms ON, a 77.183 ms (2.95%) improvement.
The flag-free IU4 row measured 1624.381 ms (3637.696 effective prefill
tokens/s).

After repairing all known layout-sensitive small/tail routes, the pre-remap
battery produced five coherent responses. The verbatim outputs and metrics
are preserved in `symwp-battery-fixed.json`: three ended normally and two
coherent answers reached the harness's 128-token cap, so the harness counted
3/5 rather than the required 5/5. This archival battery is not evidence for
the final mapping. A higher-cap final battery was skipped because the decode
gate had already required termination.

Flag-free IU4 loading remains row-major and therefore unchanged. Explicitly
combining IU4 prefill with W preshuffle routes the affected MLP prefill work
through FP8 rather than an IU4-preshuffled implementation. That interaction
does not alter the current shipped/default behavior, but it would be a
shipping-contract blocker if W preshuffle became a default.

### Verdict

**KILL.** The only quality-eligible symmetric candidate gains just 0.74% in
the standalone aggregate, below the 1.5% bar, and loses 15.05% on tg128,
violating the no-row-more-than-1%-slower rule. The pre-remap pp8192 and TTFT
wins cannot rescue a different final mapping. Keep the default current
symmetric artifact on the row-major shipped implementation; do not ship or
enable the W-preshuffle experiment by default.

## 2026-09-21 shape-correct fold-free + W-preshuffle re-screen

The earlier 185.46--186.10 TFLOP/s headline used aggregate rig shapes that
the daemon does not execute, so it is not used as a predictor here. This
re-screen used the audited daemon shapes directly on card C
(`GPU-085289909a86cc63`), with `N=8192`, five in-process warmups, and the
median of 20 HIP-event timings:

- MLP gate and up independently: `M=17408, K=5120`, `full_set`,
  128 calls/pass in total;
- MLP down: `M=5120, K=17408`, `full_add`, 64 calls/pass;
- linear-attention QKV: `M=10240, K=5120`, 48 calls/pass;
- linear-attention Z: `M=6144, K=5120`, 48 calls/pass.

The pair uses the direct-A fragment-order kernel and its original K16 W
layout. There is no separate current "pair without fragment-order A" binary:
the fold-free W-preshuffle kernels consume fragment-order W directly.
Accordingly, the requested pair and pair-plus-fragment-A entries are the same
physical configuration rather than two measurements. The folded
W-preshuffle-only arm uses the newer coalesced K64 mapping.

### Standalone matrix at daemon shapes

| configuration | shape | median us | us/token | TFLOP/s | blocks/CU | dynamic LDS | batch tile |
|---|---|---:|---:|---:|---:|---:|---:|
| shipped symfold | gate/up | 9,618.273 | 1.174106 | 151.824 | 3 | 18,944 B | 128 |
| shipped symfold | down-add | 9,184.495 | 1.121154 | 158.995 | 3 | 18,944 B | 128 |
| shipped symfold | LA QKV | 5,759.710 | 0.703090 | 149.138 | 3 | 18,944 B | 128 |
| shipped symfold | LA Z | 3,479.301 | 0.424719 | 148.132 | 3 | 18,944 B | 128 |
| fold-free only | gate/up | 9,498.373 | 1.159469 | 153.741 | 3 | 18,944 B | 256 |
| fold-free only | down-add | 9,339.951 | 1.140131 | 156.349 | 3 | 18,944 B | 256 |
| fold-free only | LA QKV | 5,857.575 | 0.715036 | 146.647 | 3 | 18,944 B | 256 |
| fold-free only | LA Z | 3,622.589 | 0.442211 | 142.273 | 3 | 18,944 B | 256 |
| folded W preshuffle only | gate/up | 9,423.194 | 1.150292 | 154.968 | 3 | 18,944 B | 128 |
| folded W preshuffle only | down-add | 8,970.631 | 1.095048 | 162.786 | 3 | 18,944 B | 128 |
| folded W preshuffle only | LA QKV | 5,507.388 | 0.672289 | 155.971 | 3 | 18,944 B | 128 |
| folded W preshuffle only | LA Z | 3,352.525 | 0.409244 | 153.734 | 3 | 18,944 B | 128 |
| fold-free + W preshuffle + fragment A | gate/up | 7,318.166 | 0.893331 | 199.543 | 3 | 9,728 B | 256 |
| fold-free + W preshuffle + fragment A | down-add | 6,691.704 | 0.816858 | 218.224 | 4 | 9,728 B | 256 |
| fold-free + W preshuffle + fragment A | LA QKV | 4,500.976 | 0.549436 | 190.846 | 4 | 9,728 B | 256 |
| fold-free + W preshuffle + fragment A | LA Z | 2,765.811 | 0.337623 | 186.345 | 4 | 9,728 B | 256 |

The pair reduces time by 23.91%, 27.14%, 21.85%, and 20.51% respectively
against the shipped control. Weighting `us/token` by the audited calls/pass
gives:

| configuration | weighted GEMM us/token | weighted TFLOP/s |
|---|---:|---:|
| shipped symfold | 276.174 | 153.087 |
| fold-free only | 276.928 | 152.670 |
| folded W preshuffle only | 269.234 | 157.033 |
| fold-free + W preshuffle + fragment A | **209.204** | **202.093** |

Thus the shape-correct standalone pair clears the 2% admission gate:
**-24.249% weighted GEMM time**, equivalently **+32.012% weighted
throughput**. Fragment-order preshuffle round-tripped exactly, and each
pair kernel matched its independently decoded reference with relative RMS
zero and no non-finite output. Raw evidence is
`scratch-foldfree/shape-screen-card-c.txt`; compiler evidence is
`scratch-foldfree/shape-screen-compile.log`.

The batch tile changes from 128 rows in the control to 256 rows in the pair.
For the 5,909-token TTFT fixture this implies 6,016 control rows versus 6,144
candidate rows, 128 extra padded rows. This geometric penalty is kept
separate from the kernel result.

### Minimal route integration and dispatch proof

The loader now chooses the W layout explicitly. The existing non-pow2
symfold path retains its coalesced K64 condition and mapping unchanged.
Pow2-marked weights may use the old K16 fragment layout only for MLP
gate/up/down, and only when all three developer settings are route-proven:
`developer.fp8_foldfree=1`, `developer.fp8_wpreshuffle=1`, and
`developer.fp8_fragment_order=1`, with `kernel.iu4_prefill=false`.
All settings remain default-off. A scoped release build of
`hipfire-daemon` passed.

Both arms were replayed under `rocprofv3` with the settings embedded in the
daemon `configure` object. The route-proving MLP signatures were:

| arm/family | kernel symbol | VGPR | SGPR | spills | dynamic LDS | blocks/CU | Grid Y |
|---|---|---:|---:|---:|---:|---:|---:|
| control gate/up | `gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` | 186 | 32 | 0 | 18,944 B | 3 | 64 |
| control down/output residual | `gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold` | 191 | 30 | 0 | 18,944 B | 3 | 64 |
| candidate gate/up | `gemm_gate_up_mq4g256v2_wmma_fp8_v2_foldfree_frag_wp_gfx1201` | 193 | 29 | 0 | 9,728 B | 3 | 32 |
| candidate MLP down | `gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_frag_wp_gfx1201` | 191 | 30 | 0 | 9,728 B | 4 | 32 |

The candidate profile contains 128 gate/up calls and 128 MLP-down
`frag_wp` calls. QKVZA, QKV, and the output residual deliberately remain
the row-major fold-free `frag` symbols, proving the admission is limited to
the three intended MLP tensors. Neither trace contains an IU4 kernel.
Evidence:
`scratch-foldfree/prof-pair-control-route/pair-control_kernel_{stats,trace}.csv`
and
`scratch-foldfree/prof-pair-candidate-route/pair-candidate_kernel_{stats,trace}.csv`.
The profiler reports zero dynamic LDS on this device, so the LDS values above
come from the actual launch request; VGPR/SGPR/spill counts come from the
generated code-object metadata, and Grid Y from the trace.

### Mandatory tg128 gate

The daemon ignores ambient feature variables, so each arm was installed via
`hipfire config set` under card C's HOME before starting a fresh benchmark
process. Both arms pinned the FP8 route:
`kernel.iu4_prefill=false`, `kernel.gfx12_fp8_stream=true`,
`kernel.gfx12_silu_quant_fused=true`, and
`kernel.gfx12_producer_quant_fused=true`. The branch does not register
`kernel.attn_qresident`, so no unrecognized setting was injected. Control
pp512 was 2,224.4 tokens/s, inside the FP8 orientation band.

Command:
`hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512 --ctx 128 --tg 128
--spec off --runs 3 --warmups 1 --kv-mode fp8 --json`.

| arm | pp512 median | tg128 median | tg128 samples |
|---|---:|---:|---|
| shipped symfold control | 2,224.4 tok/s | **36.6198 tok/s** | 36.6327, 36.6198, 36.6146 |
| fold-free + W preshuffle | 2,721.7 tok/s | **31.0424 tok/s** | 31.1003, 31.0424, 31.0065 |

The pair gains 22.36% at pp512 but loses **15.23%** at tg128. This is far
outside the required 1% decode-neutral band and reproduces the decisive
decode regression seen in the folded preshuffle experiment. Evidence:
`scratch-foldfree/pair-decode-{control,candidate}.json`.

The mandatory early-stop fired. Two ABBA pp512/pp8192 pairs, the 5,909-token
TTFT pair, a new KLD run, long-context, and the five-prompt battery were not
performed because none can rescue a 15.23% decode loss. The current pow2h
artifact independently remains over the hard FP8-v2 quality budget
(`c24=0.055408` versus `0.05`); training run 8 must close that gap before any
future performance re-gate could become shippable. Even its measured
2,721.7-token/s pp512 row remains below the roughly 3,600-token/s IU4
performance arm.

### Shape-correct verdict

**KILL.** The fold-free + W-preshuffle pair is a real, large prefill GEMM
win at the daemon's actual shapes, but it is not a shippable route: its
tg128 throughput regresses by 15.23%, and the required pow2h artifact also
misses the hard quality budget. Keep the integration behind its three
existing default-off developer settings; do not enable it by default.

## 2026-09-21 fragment-W decode consumer conversion

### Symbol-diff diagnosis

A 40-token short generation and the existing pp8192 replay were traced with
the candidate settings embedded in the daemon `configure` message. The
pp8192 MLP set contained only
`gemm_gate_up_mq4g256v2_wmma_fp8_v2_foldfree_frag_wp_gfx1201` and
`gemm_mq4g256v2_residual_wmma_fp8_v2_foldfree_frag_wp_gfx1201`. Before this
change, the short trace added two decode-only MLP consumers:

| consumer | calls | mean time |
|---|---:|---:|
| `fused_gate_up_mq4g256v2_wp` | 2,560 | 203.019 us |
| `gemv_mq4g256v2_residual_wp` | 2,560 | 94.235 us |

Those symbols consumed the newer coalesced-K64 mapping even though the
resident MLP image held the older K16 fragment mapping. This accounts for
the repeated decode failure signature without implicating the converted
prefill kernels. Evidence:
`scratch-foldfree/prof-decode-candidate-pre/decode-candidate-pre_kernel_{stats,trace}.csv`
and
`scratch-foldfree/prof-pair-candidate-route/pair-candidate_kernel_{stats,trace}.csv`.

### Conversion and exactness

Commit `a743ed2ae` replaces the weight-layout boolean in dispatch metadata
with an explicit `RowMajor`, `CoalescedSymfold`, or `Fragment` layout. The
loader derives that value directly from the active artifact and flag set,
so the old K16 fragment mapping is an explicit consequence of the fold-free,
W-preshuffle, and fragment-order settings. The pre-existing non-pow2
coalesced-K64 route remains distinct.

The MLP scalar fused gate/up and residual GEMV consumers now have fragment
readers and distinct symbols:

- `fused_gate_up_mq4g256v2_frag_wp`;
- `gemv_mq4g256v2_residual_frag_wp`;
- `gemv_mq4g256v2_frag_wp` for the plain GEMV route.

Attention QKV/QKVZA, attention output projection, lm_head/output,
embeddings, routers/MoE, and TP column slices remain row-major. In
particular, the concurrent `gemv_mq4g256v2_residual` symbol in the short
trace is attention output projection, not an unconverted MLP down
consumer.

The focused proof in `scratch-foldfree/frag_wp_decode_proof.hip` checked
every byte of the K16 permutation for both daemon MLP shapes:

| shape | bytes | bijection | byte round-trip | dequant values |
|---|---:|---:|---:|---:|
| K=5,120 | 87,040 | exact | exact | bit-exact |
| K=17,408 | 295,936 | exact | exact | bit-exact |

GPU-reader comparison against an independently decoded reference gave
relative RMS `1.23214881e-06` and max absolute `2.67028809e-05` for plain,
fused-gate, and fused-up at K=5,120, and relative RMS `3.46142711e-06`
with max absolute `0.00013923645` for residual at K=17,408. The rig passed.
The resident allocation remains size-preserving: **0 bytes VRAM delta** and
no second weight copy. The scoped release build and
`cargo check -p rdna-compute -p hipfire-arch-qwen35` passed.

### Post-conversion route proof and final tg128 gate

The post-conversion short trace contains 2,560 calls each to
`fused_gate_up_mq4g256v2_frag_wp` and
`gemv_mq4g256v2_residual_frag_wp`. Representative resource signatures are
0 B LDS / 96 VGPR / Grid Y 1 for fused gate/up and 0 B LDS / 88 VGPR /
Grid Y 1 for residual. The short generation produced coherent reasoning
through its 40-token cap. No coalesced `_wp` MLP decode symbol remains.
Evidence:
`scratch-foldfree/prof-decode-candidate-post/decode-candidate-post_kernel_{stats,trace}.csv`.

The same mandatory command and same-card control as above were used after
a fresh release build and warm run. The exact candidate settings were
installed under card C's HOME, including the four string-valued developer
settings:

| arm | pp512 median | pp512 samples | tg128 median | tg128 samples |
|---|---:|---|---:|---|
| shipped symfold control | 2,224.4 tok/s | 2,222.9, 2,226.1, 2,224.4 | 36.6198 tok/s | 36.6327, 36.6198, 36.6146 |
| fragment-W decode conversion | **2,732.4 tok/s** | 2,724.7, 2,732.4, 2,739.3 | **27.1542 tok/s** | 27.2267, 27.1542, 27.0728 |

The final candidate gains **22.838%** at pp512 but loses **25.848%** at
tg128. It is also 12.525% slower at tg128 than the already-failing
pre-conversion route. The trace identifies no remaining unconverted MLP
consumer by symbol. Instead, the correctly converted decode kernels
themselves dominate the profile: fused gate/up averages 242.505 us
(+19.45% versus the former wrong-layout reader) and residual averages
126.814 us (+34.57%). Thus the required fragment resident layout is
fundamentally hostile to these scalar decode readers in their current
access geometry.

Evidence is `scratch-foldfree/frag-wp-decode-gate.json`. The mandatory
early-stop fired. Two pp512/pp8192 ABBA pairs, final pp8192, the 5,909-token
TTFT pair, new KLD, long-context, and the five-prompt battery were not run;
none can rescue a 25.848% decode loss. The pp8192 delta and actual candidate
TTFT are therefore intentionally unreported rather than inferred.

### Product position and final verdict

Even if the proven 67 us/token prefill saving survived a future redesign,
it projects to roughly **2,214 ms** for the 5,909-token fixture, still well
behind the IU4 arm at roughly **1,640 ms**. Its only product role would be
the quality arm. The shipped FP8-v2 artifact has c24 `0.045510` versus IU4
at `0.078594`, but the pow2h artifact required here remains at **0.055408**,
above the hard FP8-v2 budget of `0.050`. QAT run 8 must close the remaining
`0.005408`; the best possible future state is ready pending run 8, never
ship now.

**KILL.** The consumer conversion is exact, complete for the MLP resident
image, default-off, and uses no extra VRAM, but the pair fails the blocking
decode gate by 25.848%. Keep commit `a743ed2ae` as diagnostic implementation
and evidence only; do not enable the fragment-W pair.
