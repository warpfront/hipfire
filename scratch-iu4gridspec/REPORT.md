# gfx11 IU4 partial-N grid specialization

## Result

The default-off `kernel.gfx11_iu4_gridspec` experiment splits a partial-N grid into:

1. complete 128-column tiles using the existing `FULL=true` production entry, and
2. one guarded tail-column launch using the existing `FULL=false` body through a tiny wrapper that preserves the original N stride.

The split is bit-identical to the shipped route in every tested process. It materially improves N=5909, especially on gfx1151, and leaves exact N=8192 on the source-identical shipped route. It does **not** pass the advancement bar because gfx1100 gate/up improves only **1.0803x**, below the required 1.15x. gfx1151 gate/up improves **1.4672x**. Therefore daemon matrix, TTFT, c24, and serve-battery gates were not run.

Chosen split-launch gate/up headline:

- gfx1100: **103.400 -> 111.699 TOPS, 1.0803x, 40.82% -> 44.10% of 253.296 TOPS**. This remains 50.281 TOPS below the 161.98-TOPS target.
- gfx1151: **32.690 -> 47.963 TOPS, 1.4672x, 29.80% -> 43.73% of 109.690 TOPS**. This remains 4.217 TOPS below the 52.18-TOPS target.

## Source mechanism

At base `34ccae74b`, `crates/rdna-compute/src/gemm.rs:19779-19810` sets `full` only when both M and N are multiples of 128. N=5909 therefore selects the generic entry for all 47 column tiles; N=8192 selects a `FULL=true` entry. On gfx1151 full SET additionally selects the 16-wave LF16 column entry.

The generic/full differences are in `kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip`:

- `:259`: the generic body has a workgroup bounds exit.
- `:271` and `:127-163`: generic A loading clamps every row with `min(row0+i, M-1)` even though both production M values are exact multiples of 128; the full body uses the row directly.
- `:273-289`: every K128 half fills 2,304 32-bit X words. Generic code checks `col0 + l/18 < N` per word and writes zero when false; full code performs the direct load.
- `:301-340`: full epilogue constructs one uniform tile-relative buffer resource and uses 32-bit byte offsets.
- `:341-362`: generic epilogue performs a compound row/column bounds test and forms a 64-bit pointer for every valid output.
- `:628-653`: gfx1151 LF16 and ordinary column-adjacent full entries are distinct and therefore mechanically separable.

Static cost per 128x128 output workgroup, excluding compiler folding:

| Shape | K128 halves | checked X-fill decisions | checked X-fill decisions/lane | generic output bounds/pointers | full epilogue |
|---|---:|---:|---:|---:|---|
| gate/up, K=5120 | 40 | 92,160 | 360 | 16,384 compound bounds; up to 16,384 64-bit pointers | one uniform descriptor; 16,384 32-bit offsets; no bounds |
| down, K=17408 | 136 | 313,344 | 1,224 | 16,384 compound bounds; up to 16,384 64-bit pointers | one uniform descriptor; 16,384 32-bit offsets; no bounds |

At N=5909, the chosen split removes these checks from 6,256 gate/up interior workgroups and 1,840 down interior workgroups. That removes 576,552,960 checked X-word decisions in either shape, plus 102,498,304 gate/up or 30,146,560 down per-output bounds tests. The required guarded work remains on the 136 gate/up or 40 down tail workgroups.

These are source-level operation counts, not a cycle attribution. The controlled throughput deltas below measure their aggregate effect together with the full epilogue and A-loader specialization.

## Implementation and correctness

The host route is at `crates/rdna-compute/src/gemm.rs:19779-19936`. The wrapper entries are in `kernels/src/gemm_mq4g256v2_residual_mmq_iu4_gridspec.gfx11.hip:10-48`.

For N=5909:

- interior column tiles are `[0,46)`, launched with the existing full production symbol and column-adjacent grid when eager;
- the tail wrapper computes `col_tile = N/128 + blockIdx.y`, so it addresses tile 46 while retaining the original N=5909 Xq K-block stride;
- the tail uses the ordinary 8-wave generic body, which zero-fills X columns 21..127 and guards stores;
- both launches use the same stream and write disjoint output columns, so SET has no overlap and ADD reads/adds each residual exactly once;
- M-partial shapes retain the shipped generic path;
- exact N=8192 does not set `gridspec`, so module, symbol, grid, and launch count remain shipped.

A pointer-rebased tail was rejected: Xq is laid out as `[K128][N]`, so passing tail N=21 would incorrectly change every K-block stride from 5909 to 21.

The alternative single-launch entry uses a uniform workgroup branch on `col_tile < N/128` and instantiates `gemm_iu4_body<true>` for interior blocks and `<false>` for the tail. It uses all eight ordinary waves; it is a diagnostic, not the selected host route.

### Loader/addressing and active-lane audit

- Interior grid counts are exactly 46x136 gate/up and 46x40 down; tail counts are 136 and 40. Their sum equals the shipped 47-column grid, with no duplicate or missing tile.
- Xq addressing retains the original N stride for both launches. The wrapper changes only the logical column tile.
- The A payload loader remains the shipped 32-word-per-row copy. Header loading remains two 32-bit header lanes per row followed by four LDS replications; the experiment adds no redundant header fetch.
- Ordinary entries launch 8x32 active lanes. gfx1151 LF16 interior launches 16x32 active lanes. The guarded tail necessarily has only 21/128 globally valid X columns, but all lanes participate in zero-fill, compute, and guarded epilogue exactly as in the shipped generic tail.
- Dynamic LDS remains 30,720 B. No new accumulator phase, barrier, or register-resident staging was added.

## Measurement protocol

Each cell used four fresh processes in forward/reverse/reverse/forward order. Each arm used two warmups and nine event-timed samples; the process median is reported and the table value is the arithmetic mean of four process medians. Every process asserted `gfx1100` or `gfx1151` from `gcnArchName` rather than trusting the device index.

The artifact hash was checked before each measurement window:

`de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`

Every pre/post `rocm-smi --showpids` check was empty except the persistent zero-VRAM `gpusentry`. One 5.68-second sibling hipcc compile started immediately after the gfx1100 branch-window announce. The intervening precheck/hash took 29.24 seconds, so the compile ended more than 23 seconds before the 7.23-second GPU runner began; no sample overlapped it.

Across the split and branch windows there were 64 fresh-process bitwise comparisons. Every comparison reported zero mismatches. The additional gfx1151 ordinary-split route-control window also reported zero mismatches in all 16 processes.

The timed ADD windows initialized residuals to zero. That proves bitwise route parity but would not detect an accidental SET in place of ADD. A follow-up changed the parity prefill to nonzero `0.25f`. Its first attempt exposed a harness defect: the shipped arm had been omitted before comparison, so all outputs mismatched; that attempt is invalid and was discarded. After restoring both launches and recompiling, fresh split smokes on gfx1100 and gfx1151 each reported **0 / 30,254,080 mismatches** with the nonzero residual. This was a harness defect, not a kernel defect.

## Chosen split-launch results

TOPS values are four-process means. Percent is relative to the architecture's measured register-only ceiling.

| Arch | Shape | N | shipped TOPS | split TOPS | ratio | shipped % | split % |
|---|---|---:|---:|---:|---:|---:|---:|
| gfx1100 | gate/up | 5909 | 103.400 | **111.699** | **1.0803x** | 40.82% | **44.10%** |
| gfx1100 | gate/up | 8192 | 118.076 | 118.251 | 1.0015x | 46.62% | 46.68% |
| gfx1100 | down | 5909 | 101.032 | **116.077** | **1.1489x** | 39.89% | **45.83%** |
| gfx1100 | down | 8192 | 122.397 | 122.187 | 0.9983x | 48.32% | 48.24% |
| gfx1151 | gate/up | 5909 | 32.690 | **47.963** | **1.4672x** | 29.80% | **43.73%** |
| gfx1151 | gate/up | 8192 | 48.014 | 47.983 | 0.9994x | 43.77% | 43.74% |
| gfx1151 | down | 5909 | 32.772 | **46.865** | **1.4301x** | 29.88% | **42.73%** |
| gfx1151 | down | 8192 | 47.840 | 47.696 | 0.9970x | 43.61% | 43.48% |

N=8192 is a source-identical control: both arms launch the same symbol through the same host path. Its 0.9970x..1.0015x range is the observed repeatability band, not a candidate-path regression.

## Single-launch per-workgroup branch results

| Arch | Shape | N | shipped TOPS | branch TOPS | ratio | branch % ceiling |
|---|---|---:|---:|---:|---:|---:|
| gfx1100 | gate/up | 5909 | 103.232 | **110.915** | **1.0744x** | 43.79% |
| gfx1100 | gate/up | 8192 | 118.224 | 118.225 | 1.0000x | 46.67% |
| gfx1100 | down | 5909 | 102.136 | **113.049** | **1.1068x** | 44.63% |
| gfx1100 | down | 8192 | 122.385 | 122.025 | 0.9971x | 48.17% |
| gfx1151 | gate/up | 5909 | 32.583 | **45.587** | **1.3991x** | 41.56% |
| gfx1151 | gate/up | 8192 | 48.013 | 47.889 | 0.9974x | 43.66% |
| gfx1151 | down | 5909 | 32.566 | **44.268** | **1.3593x** | 40.36% |
| gfx1151 | down | 8192 | 47.766 | 47.602 | 0.9966x | 43.40% |

The one-branch form recovers most of the specialization benefit but is consistently slower than separate entries. The split is 0.7% faster on gfx1100 gate/up, 2.7% faster on gfx1100 down, 5.2% faster on gfx1151 gate/up, and 5.9% faster on gfx1151 down. The likely mechanism is merged-body codegen/instruction footprint rather than branch frequency: the branch is uniform and executes only once per workgroup, while each selected body is compile-time specialized.

Gate/up N=5909 raw four-process TOPS vectors:

- gfx1100 split: shipped `[100.384, 106.959, 106.619, 99.638]`; split `[117.088, 106.835, 106.450, 116.423]`.
- gfx1100 branch: shipped `[99.750, 106.814, 106.459, 99.906]`; branch `[114.507, 106.461, 106.664, 116.028]`.
- gfx1151 split: shipped `[32.880, 32.341, 33.027, 32.512]`; split `[48.170, 47.456, 48.179, 48.048]`.
- gfx1151 branch: shipped `[32.417, 32.572, 32.618, 32.725]`; branch `[45.747, 45.558, 45.560, 45.481]`.

## gfx1151 LF16 separation

A third control used the same two-launch split but forced the ordinary 8-wave full SET entry for the 46 interior tiles. Gate/up N=5909 measured:

- shipped: **32.623 TOPS**;
- ordinary split: **48.161 TOPS, 1.4763x, 43.91% of ceiling**;
- production-route LF16 split from the main table: **47.963 TOPS, 1.4672x, 43.73% of ceiling**.

Thus the LF16 route change is separable and contributes no measured gain here: ordinary split is 1.0041x LF16 split, inside the sub-percent control band. The large gfx1151 recovery comes from limiting the generic body to the tail, not from switching the interior to LF16. The selected implementation nevertheless preserves the existing production full-tile routing rule; the two full entries are statistically tied in this partial-N experiment, and preserving the incumbent symbol is the cleaner cutover.

## Gate decision

The advancement condition was >=1.15x on gate/up N=5909 on both cards with no exact-N route regression.

- gfx1100 split: **1.0803x — fail**.
- gfx1151 split: **1.4672x — pass**.
- exact N=8192 remains source-identical and measures within the 0.3% control band.

Because the gfx1100 gate failed, no end-to-end daemon matrix, exact-5909 TTFT, c24 WT2/agentic, or campaign serve battery was run. There is therefore no decoded campaign text to report.

## Build and artifacts

- `cargo build --release`: passed after the final host dispatch change.
- HIP harness compiled for both `gfx1100` and `gfx1151`.
- `bench_gridspec.hip`: exact-shape bitwise parity and split/branch/ordinary-split timing harness.
- `run_exact.sh`: fresh-process forward/reverse/reverse/forward runner.
- `exact-gfx1100.log`: chosen split gfx1100 window.
- `exact-split-gfx1151.log`: chosen split gfx1151 window.
- `exact-branch-gfx1100.log`, `exact-branch-gfx1151.log`: single-launch branch windows.
- `exact-ordinary-gfx1151.log`: ordinary-entry split route-control window.
- `residual-smoke-gfx1100.log`, `residual-smoke-gfx1151.log`: corrected nonzero-residual ADD parity smokes.
