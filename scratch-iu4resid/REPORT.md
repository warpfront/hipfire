# gfx11 IU4 partial-N residual: accounting and serialization result

## Verdict

KILL the separate-stream tail candidate.

With grid specialization and the gfx1100 LF16 route both active, the measured gate/up result is 124.614 TOPS at N=5909, or 49.20% of the 253.296 TOPS ceiling. The N=8192 control is 50.98%. The residual is therefore about 1.8 ceiling points, not the roughly 7 points in the original framing. Most of the apparent residual was the LF16 entry not running on gfx1100 before the shape change.

Explicitly allowing the N=5909 tail to overlap the interior changed gfx1100 gate/up from 124.165 to 124.243 TOPS, 1.00063x. This misses the required 1.10x gate unambiguously. No TTFT, matrix, quality battery, or production dispatch change was attempted after that microbenchmark gate failed. The typed `kernel.gfx11_iu4_resid` setting remains default-off, and the shipped dispatch path is unchanged.

## Exact source-level difference: N=5909 versus N=8192

The following describes eager gfx1100 gate/up with grid specialization and LF16 enabled, M=17408 and K=5120.

### Host dispatch and grid

- `crates/rdna-compute/src/gemm.rs:19779-19787` classifies N=8192 as a full grid and N=5909 as a partial-N grid-specialization case. Both have full M tiles.
- `crates/rdna-compute/src/gemm.rs:19791-19827` selects the eager column-adjacent mapping and the 16-wave LF16 full-tile entry. The full-tile workgroup is `[32,16,1]`.
- `crates/rdna-compute/src/gemm.rs:19870-19883` computes 136 row tiles. N=5909 has `floor(5909/128)=46` unchecked interior column tiles; N=8192 has 64 column tiles.
- `crates/rdna-compute/src/gemm.rs:19884-19922` launches N=5909 as:
  - interior grid `[46,136,1]`, 6,256 workgroups, LF16 `[32,16,1]`;
  - guarded tail grid `[136,1,1]`, 136 workgroups, generic `[32,8,1]`.
- `crates/rdna-compute/src/gemm.rs:19923-19947` launches N=8192 once as `[64,136,1]`, 8,704 workgroups, LF16 `[32,16,1]`.
- The N=5909 split therefore has 6,392 total workgroups, not 6,392 interior workgroups. There are 47 logical N tiles including the tail. N=8192 has 64 full N tiles.
- For the eager column wrapper, `blockIdx.x` is the column tile and `blockIdx.y` is the row tile (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:636-656`). The tail wrapper instead maps `blockIdx.x` to the row tile and fixes the final column tile (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4_gridspec.gfx11.hip:11-23`). This records the index mapping; hardware workgroup issue order is not assumed.

### Predicates, loads, and epilogue

- The LF16 interior/full body has no M/N boundary predicate. Its Xq loads are direct (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:569-597`) and its output stores are direct (`:616-630`). SET versus ADD is a compile-time `IS_ADD` specialization, so there is no runtime epilogue mode branch.
- The N=5909 tail calls `gemm_iu4_body<false>` (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4_gridspec.gfx11.hip:21-23`). It retains:
  - one uniform row/column workgroup guard (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:265-267`);
  - per-Xq-word N guards and zero fill (`:281-300`);
  - per-output M/N guards and the runtime SET/ADD branch (`:349-370`).
- At N=5909 the tail begins at column 5,888, so 21 of its 128 columns are useful and 107 are padding. Interior and tail output ranges are disjoint. M=17408 and M=5120 are both multiples of 128, so the tail row clamp in `load_iu4_tile<false>` (`kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:135-170`) never redirects a live row.
- Each activation block is 72 bytes (`kernels/src/block_i4_128_quant.hip:31-39`). The K-plane stride is therefore 425,448 bytes at N=5909 versus 589,824 bytes at N=8192. The latter is page-aligned; the former has offsets 8, 8, 40, 104, 232, and 3,560 modulo 16, 32, 64, 128, 256, and 4,096 bytes respectively. This is an enumerated address-layout difference, not a measured attribution.
- `crates/rdna-compute/src/gemm.rs:19828-19845` compiles partial-N through the grid-specialized translation unit and exact N through the base translation unit. The selected LF16 entry body is otherwise the same source.

### Manual-serve trace evidence

The shipped-grid manual `hipfire serve` rocprof trace is:

`/home/kaden/hipfire-iu4gridspec/scratch-iu4gridspec/rocprof-grid-gfx1151/daemon_kernel_trace.csv`

Representative rows 508-509 show the actual split dispatch:

- interior `...full_set_lf16_col...`: Queue 2, Stream 0, start 209388231906239, end 209388252659159, workgroup `[32,16,1]`, global grid `[1472,2176,1]` = `[46*32,136*16,1]`, 96 VGPR;
- tail `...tail_gridspec`: Queue 2, Stream 0, start 209388252670440, end 209388253923227, workgroup `[32,8,1]`, global grid `[4352,8,1]` = `[136*32,1*8,1]`, 200 VGPR.

The tail starts 11.281 microseconds after the interior ends. Thus the baseline launches do serialize on the same stream. The measured negative below shows that removing that serialization does not materially improve gate/up throughput; it does not claim that the baseline already overlaps them.

## Launch-cohort accounting

Use the requested model of 96 CUs and two resident workgroups per CU: 192 workgroups per full residency cohort.

| Work | Blocks | Exact blocks / 192 | Charged cohorts | Empty slots in final cohort |
|---|---:|---:|---:|---:|
| N=5909 combined | 6,392 | 33.292 | 34 | 136 |
| N=5909 interior | 6,256 | 32.583 | 33 | 80 |
| N=5909 tail | 136 | 0.708 | 1 | 56 |
| N=8192 full | 8,704 | 45.333 | 46 | 128 |

Because the interior and tail are separate kernels, N=5909 is 33 interior cohorts plus one tail cohort. That is still 34 cohorts, the same count as `ceil(6392/192)`; the split does not add a second extra cohort. The tail's final cohort is only 70.83% filled, but it is one of 34 cohorts. On total launch capacity, N=5909 fills 97.917% while N=8192 fills 98.551%. Their fill-efficiency ratio is 0.993566, only a 0.643% relative disadvantage (about 0.33 ceiling points at the N=8192 efficiency).

N=5909 also reports useful work for only 5,909 of 6,016 computed columns: `5909/6016 = 0.982214`, a 1.779% relative loss (about 0.91 ceiling points). Applying both padding and this simple cohort-fill model to the 50.98% N=8192 result predicts about 49.75%, while the measured result is 49.20%. About 0.55 ceiling points remain unexplained. The tail-cohort waste cannot account for the full remaining gap. No causal attribution is made for the remainder.

## Separate-stream experiment

### Candidate and safety audit

`scratch-iu4resid/bench_resid.hip` mirrors the shipped split route and adds a nonblocking side stream for the guarded tail:

1. record a dependency event after input preparation on the main stream;
2. make the side stream wait for it;
3. issue the unchecked interior on the main stream and the guarded tail on the side stream;
4. record tail completion and make the main stream wait before timing completes or any consumer runs.

The exact-N control calls the same full-grid function in both arms. gfx1100 uses LF16 SET and ADD as the active shape route does. gfx1151 uses LF16 SET and its shipped ordinary ADD route.

Boundary and correctness audit:

- Xq allocation is exactly `(K/128)*N` blocks; no padded host allocation masks an OOB load.
- The guarded tail zero-fills every Xq word beyond N before the shared-memory barrier.
- Output guards reject all 107 padded columns.
- Full M shapes make every launched row tile valid.
- The interior writes columns `[0,5888)` and the tail writes `[5888,5909)`, so concurrent stores cannot race.
- ADD arms initialize every output to nonzero `0.25f`, exercising actual residual accumulation rather than SET-equivalent zero data.
- The fork event orders both launches after producer initialization; the join event orders all consumers after both writers.

Across both cards, both shapes, both N values, and four fresh-process F/R/R/F runs per cell, all 32 comparisons were bit-identical: 0 mismatches across 2,541,338,624 compared float outputs. Bit identity means this scheduling-only experiment cannot move c24. Since it failed the throughput gate and the default production route did not change, a separate c24 run would add no information.

### Paired results

Each table cell is the arithmetic mean of four fresh-process medians. Each process used two warmups and nine event-timed launches per arm. The raw logs are `exact-gfx1100.log` and `exact-gfx1151.log`.

| Card | Shape | N | serialized TOPS | concurrent TOPS | ratio |
|---|---|---:|---:|---:|---:|
| gfx1100 | gate/up SET | 5909 | 124.165 | 124.243 | **1.00063x** |
| gfx1100 | gate/up SET | 8192 control | 130.591 | 131.091 | 1.00383x |
| gfx1100 | down ADD | 5909 | 115.195 | 117.814 | 1.02274x |
| gfx1100 | down ADD | 8192 control | 121.910 | 121.872 | 0.99969x |
| gfx1151 | gate/up SET | 5909 | 49.067 | 49.555 | 1.00996x |
| gfx1151 | gate/up SET | 8192 control | 48.551 | 48.604 | 1.00109x |
| gfx1151 | down ADD | 5909 | 46.170 | 46.129 | 0.99911x |
| gfx1151 | down ADD | 8192 control | 47.534 | 47.619 | 1.00180x |

The gfx1100 measurements have a pronounced run-order effect: whichever arm runs second is faster. F/R/R/F fresh-process pairing cancels it in the aggregate. The exact-N controls, where both arms execute identical code, also expose the remaining measurement scale.

The decision bar was at least 1.10x on gfx1100 gate/up N=5909 with no regression on either card. The measured 1.00063x fails the primary bar by two orders of magnitude in incremental gain. The gfx1100 down-only 1.02274x does not rescue the candidate. The default-off experiment stops here.

## Reproduction artifacts

- `bench_resid.hip`: standalone exact-shape correctness and timing harness.
- `run_exact.sh`: arch-asserted F/R/R/F runner.
- `exact-gfx1100.log`: raw gfx1100 output.
- `exact-gfx1151.log`: raw gfx1151 output.
- `trace-serialization.csv`: raw manual-serve rocprof rows proving same-stream serialization.

Both binaries were compiled with ROCm 7.15.26333 using `hipcc -O3` and the exact `--offload-arch=gfx1100` / `gfx1151` target. Successful logs identify `AMD Radeon RX 7900 XTX` as gfx1100 and `AMD Radeon 8060S Graphics` as gfx1151. GPU process state was checked empty before and after each leased window, apart from the zero-VRAM `gpusentry` process.
