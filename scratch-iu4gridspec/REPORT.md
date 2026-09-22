# gfx11 IU4 partial-N grid specialization

## Result

The default-off `kernel.gfx11_iu4_gridspec` experiment splits a partial-N grid into:

1. complete 128-column tiles using the existing `FULL=true` production entry, and
2. one guarded tail-column launch using the existing `FULL=false` body through a tiny wrapper that preserves the original N stride.

This is a bit-exact win. At the exact 5,909-token production prompt shape, graph-on daemon TTFT prefill improves **1,616.6 -> 1,651.4 tok/s (+2.2%) on gfx1100** and **519.8 -> 651.1 tok/s (+25.2%) on gfx1151** in fresh-process ABBA runs. The split is bit-identical to the shipped route in every tested comparison. The knob remains default-off; integration recommendation is left to Main.

N=8192 is an exact multiple of 128, so it already takes the unchecked route and cannot gain. The lever's entire value is at partial N, which is every real prompt. Consequently pp8192 is the wrong headline for this lever: it measures the one route production prompts do not take. TTFT-5909 is the honest end-to-end measure. On gfx1151 the shipped TTFT prefill is only 520 tok/s while the synthetic exact-N pp8192 result is 645-678 tok/s, so the tracked exact-N benchmark had hidden a roughly 25% real-prompt penalty.

Microbenchmark headline, with `down` beside gate/up:

- gfx1100 gate/up: **103.400 -> 111.699 TOPS, 1.0803x**; down: **101.032 -> 116.077 TOPS, 1.1489x**.
- gfx1151 gate/up: **32.690 -> 47.963 TOPS, 1.4672x**; down: **32.772 -> 46.865 TOPS, 1.4301x**.

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

The pinned artifact was identified by full SHA-256 once for the final session and by unchanged `stat` identity thereafter:

- SHA-256: `de8ee8256033c3690b0f1a2aff14e77cc88fff490e118648b04a833a3f2969b5`
- `stat -c '%s %Y %i'`: `14987185152 1789995527 5398606`

Earlier microbenchmark windows also rehashed the artifact; the later stat-only protocol avoids evicting 15 GB of page cache and prevents shared-memory contention on gfx1151. Every pre/post `rocm-smi --showpids` check was empty except the persistent zero-VRAM `gpusentry`. One 5.68-second sibling hipcc compile started immediately after the gfx1100 branch-window announce. The intervening precheck/hash took 29.24 seconds, so the compile ended more than 23 seconds before the 7.23-second GPU runner began; no sample overlapped it.

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

## Graph-on daemon gates

The daemon gates used typed `experimental.graph.forward=true` and `experimental.graph.ar=true`. Timing used fresh-process ABBA order `A1=off, B1=on, B2=on, A2=off`; every JSON result asserted the reported architecture. `--spec off` disabled MTP so the missing MTP head could not pollute admission or timing.

### Exact 5,909-token TTFT

Command:

`hipfire bench <model> --ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --spec off --json`

Every arm asserted 5,909 prompt tokens and prompt MD5 `ed720348b81a19fab64d4783c75c1ae3`. Values below are each fresh process's eight-sample median.

| Arch | A1 off ms / tok/s | B1 on ms / tok/s | B2 on ms / tok/s | A2 off ms / tok/s | ABBA off -> on | gain |
|---|---:|---:|---:|---:|---:|---:|
| gfx1100 | 3,638.8 / 1,623.9 | 3,577.9 / 1,651.5 | 3,578.3 / 1,651.3 | 3,671.7 / 1,609.3 | 1,616.6 -> **1,651.4 tok/s** | **+2.2%** |
| gfx1151 | 11,358.8 / 520.2 | 9,070.3 / 651.5 | 9,081.7 / 650.7 | 11,377.1 / 519.4 | 519.8 -> **651.1 tok/s** | **+25.2%** |

The gfx1151 mean TTFT falls from 11,368.0 ms to 9,076.0 ms, a 20.2% time reduction and 25.2% throughput gain. The corresponding gfx1100 time reduction is 2.1%.

### Exact-N matrix controls

Command:

`hipfire bench <model> --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --json`

Both prefill lengths are multiples of 128 and therefore source-identical controls. Values are each process's three-sample median in tok/s.

| Arch | arm | pp512 | pp8192 | decode ctx128/tg128 |
|---|---|---:|---:|---:|
| gfx1100 | A1 off | 1,640.6 | 1,652.2 | 49.110 |
| gfx1100 | B1 on | 1,621.7 | 1,638.3 | 49.040 |
| gfx1100 | B2 on | 1,613.5 | 1,631.8 | 49.119 |
| gfx1100 | A2 off | 1,609.8 | 1,629.6 | 48.964 |
| gfx1151 | A1 off | 721.5 | 677.5 | 14.902 |
| gfx1151 | B1 on | 701.3 | 650.4 | 14.904 |
| gfx1151 | B2 on | 692.3 | 646.2 | 14.900 |
| gfx1151 | A2 off | 687.6 | 645.4 | 14.899 |

ABBA on/off ratios are 0.9953/0.9964/1.0009 on gfx1100 and 0.9890/0.9801/1.0001 on gfx1151 for pp512/pp8192/decode respectively. The monotone prefill drift is bracketed by ABBA; there is no exact-N or decode gain, as required by the source-identical route.

## Deterministic quality and graph-capture correctness

The gfx1100 c24 evaluator scored 24 chunks / 24,552 tokens for both references with graph forced off by `eval_hipfire`, as designed. Although an OFF/ON pair was run before the deterministic-gate protocol was narrowed, each pair is byte-identical:

| reference | slice-mean KLD | mean NLL | PPL | OFF/ON `kldseq` SHA-256 |
|---|---:|---:|---:|---|
| WT2 | 0.088665 | 1.904655 | 6.7171 | `922975d2499ed84f93870b37878c9af5b74cbea2f6415046cd8337cc8ec1c033` |
| agentic | 0.256882 | 2.041904 | 7.7053 | `ed80e3b804db94d488c4ef47a8a772a1f8057f42ad652551e3a4018f9cf285db` |

`cmp` succeeded for each OFF/ON pair. The gfx1151 c24 repeat was deliberately omitted: the standalone kernel was already bit-exact on that card, making logits and deterministic KLD arch-independent.

The graph-on gfx1100 campaign command was exactly:

`python3 scripts/serve_harness.py --mode battery --model <model> --thinking off`

There was no `--max-tokens` override. It completed 5/5 turns with `runaway=0`, `empty=0`, `attractor=0`, and `retrieval_miss=0`. The decoded text printed by the harness is reproduced verbatim below; the harness intentionally prints a 96-character `repr` preview:

```text
[code] '```python\ndef merge_sorted(a, b):\n    """Merge two already-sorted lists into one sorted li'
[reason] 'Step 1: Distance for the first part  \n\\(60 \\text{ mph} \\times 2.5 \\text{ hours} = 150 \\tex'
[factual] "The seasons on Earth are caused by the tilt of Earth's axis relative to its orbital plane "
[prose] 'Elias climbed the gale-battered rocks at dawn, expecting only kelp and broken shells.  \nIn'
[instruct] '1. Use clear, descriptive names for variables, functions, and modules.\n2. Keep functions s'
```

The five responses finished normally (`finish=stop`) with 227, 171, 213, 298, and 195 generated tokens. This one battery is the capture-sensitive check that deterministic standalone parity cannot replace. A second gfx1151 battery was deliberately omitted because graph-capture semantics and the split's disjoint-buffer ordering are arch-independent, while the card-specific performance path was covered by TTFT ABBA.

## Production-route profiler proof

A manual `hipfire serve` under rocprofiler, with graceful daemon shutdown so the CSV flushed, traced one exact 5,909-token gfx1151 request. The trace maps `Agent 2` to `gfx1151` and records:

- 272 `gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151` interior dispatches;
- 128 `gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151` interior dispatches;
- exactly 400 `gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec` tail dispatches.

Thus every one of the 400 eligible interior launches is paired with exactly one tail launch. Sample adjacent trace rows show a full interior dispatch immediately followed by `tail_gridspec`; the tail resource census is 200 VGPR, 0 scratch, 32x8 workgroup. This proves that the typed knob reaches the intended production split under graph-on serving rather than merely selecting it in the standalone harness.

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

## Verdict and gate scoping

The original microbenchmark advancement condition was at least 1.15x on gate/up N=5909 on both cards. The chosen split measured 1.0803x on gfx1100 and 1.4672x on gfx1151, so it did not satisfy that initial two-card threshold. Main explicitly overrode the stop condition to obtain the production evidence above.

The resulting verdict is: **bit-exact win, default-off, worth +25.2% exact-5909 TTFT throughput on gfx1151 and +2.2% on gfx1100**. Exact N=8192 remains source-identical and within the control/noise band. The strongest standalone effects are gate/up 1.4672x and down 1.4301x on gfx1151; both major matmul directions benefit rather than merely gate/up.

Gate scoping was deliberate:

- timing claims use fresh-process ABBA; deterministic checks do not need thermal pairing;
- c24 was run on gfx1100 for WT2 and agentic, with byte-identical OFF/ON outputs; gfx1151 c24 was omitted because bit-exact output makes deterministic KLD arch-independent;
- one graph-on five-turn campaign battery checked readable capture output; gfx1151 battery was omitted because graph ordering correctness is arch-independent;
- both cards still received graph-on matrix and exact-5909 TTFT ABBA, and both standalone gate/up and down N=5909 ratios are reported.

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
- `e2e-gfx{1100,1151}-matrix-{A1,B1,B2,A2}.json`: graph-on exact-N matrix ABBA.
- `e2e-gfx{1100,1151}-ttft-{A1,B1,B2,A2}.json`: graph-on exact-5909 TTFT ABBA.
- `quality-gfx1100-{wt2,ag}-{off,on}.stderr` and `.kldseq`: c24 metrics and byte-parity outputs.
- `battery-gfx1100.console`: exact-command graph-on five-turn decoded campaign evidence.
- `rocprof-grid-gfx1151/daemon_{agent_info,kernel_stats,kernel_trace}.csv`: manual-serve production split proof.
- `model-sha-quality-gfx1100.txt` and `model-stat-session.txt`: artifact identity evidence.
