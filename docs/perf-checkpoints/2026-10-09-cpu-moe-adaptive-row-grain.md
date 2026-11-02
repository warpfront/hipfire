# CPU MoE adaptive Rayon row grain — 2026-10-09

**Lifecycle:** historical. **Disposition:** fixture-bound measured evidence for a scheduling candidate, not a product admission, portable speedup or optimal-grain claim.

Implementation commit: `6c1f26a13071747fe1b2fb1cc9ec625f56f3dd80` (`cpu: adapt shared GEMV row blocks to Rayon parallelism`). Binaries were built before that commit, from the identical adaptive source bytes recorded below. Reference arms deliberately rebuilt temporary fixed-64 variants; none of those constants is retained in the committed implementation.

## Scope and research

Only `crates/hipfire-cpu/src/gemv.rs::run_shared_jobs` and its immediately necessary helper/tests changed. The outer expert iterator, SIMD kernels, quantized dot arithmetic, accumulation order, prepared activation sums and row-major output layout remain unchanged. No pool resizing, CPU-model detection, tuning table, environment knob, calibration, dispatch change or hot-row-loop allocation/synchronization was introduced.

Two independent scouts investigated repository workloads and Rayon source/documentation. No reusable adaptive grain convention was found in the repository. The production caller supplies gate/up `(m,k)=(1024,2048)` and down `(2048,512)` for this fixture. Each job owns a row-major output segment containing `m*n` values; `n` is its activation-row count. Empty jobs contribute no outer work. Multi-row jobs can differ in `n` and quant tier.

Rayon 1.12.0 / rayon-core 1.13.0 are the existing dependencies. An indexed iterator item is not necessarily one Rayon task: the indexed bridge recursively splits and then folds a contiguous run of items serially. Its splitter uses the current pool size and can replenish its split budget after work is stolen. Explicit blocks bound the smallest divisible row unit while leaving Rayon in charge of task splitting and stealing.

Primary sources:
- [IndexedParallelIterator grain controls](https://docs.rs/rayon/1.12.0/rayon/iter/trait.IndexedParallelIterator.html#method.with_min_len)
- [Splitter, LengthSplitter and indexed bridge source](https://docs.rs/rayon/1.12.0/src/rayon/iter/plumbing/mod.rs.html)
- [Mutable chunk producer](https://docs.rs/rayon/1.12.0/src/rayon/slice/chunks.rs.html)
- [Current pool worker count](https://docs.rs/rayon-core/1.13.0/rayon_core/fn.current_num_threads.html)

`with_min_len` changes leaf splitting, but retains one iterator item per row. Explicit blocks retain the measured campaign's serial consecutive-row loop. Targeting approximately one block per worker pool-wide risks inadequate balance across heterogeneous jobs; targeting a full pool per expert ignores outer parallelism. Neither is assumed optimal.

## Implemented policy

Let `T = rayon::current_num_threads()`, `J = number of nonempty job output chunks`, `m = output rows/job`, and `n = activation rows for this job`.

1. Return when `J == 0`.
2. Compute `P = ceil(T/J)` once for the batch.
3. A single-worker pool uses one whole-job block (`G=m`). Otherwise:
   `G = max(1, floor(sqrt(ceil(ceil(m/P)/n))))`.
4. Iterate `par_chunks_mut(G*n)`, then serial `chunks_mut(n)` within each block. The weight row is `block*G + within_block`.

The square root is a policy tradeoff between subdivision overhead and indivisible work, not an optimality proof or a fixed blocks-per-thread target. More active outer jobs allow coarser grain; more workers with few jobs require finer grain. More activation rows make each weight row heavier, so their blocks are smaller. For illustration, eight active jobs on sixteen workers with `n=1` yield 22 GU rows/block and 32 DN rows/block, rather than a machine-specific constant.

The helper uses overflow-safe ceiling division and integer square root, without forming `m*J` or `T*J`. Positive arguments imply `1 <= G <= m`; therefore the new chunk multiplication cannot exceed the existing valid `m*n` output span. Non-divisible row counts remain whole activation-row slices in the final block. The prepared branch uses `continue` inside the serial row loop, not an early return that would omit remaining rows.

The policy does not model instruction cost, memory bandwidth, quant-tier cost or concurrent workloads. Those are portability limitations, not additional optimization scope.

## Fixture and method

- CPU: Ryzen 7 7800X3D, 16 logical workers in the default pool.
- GPU: RX 9070 XT, gfx1201; HIP 7.2.53211.
- Target: `qwen3.6:35b-a3b`, local `qwen3.6-35b-a3b.mq4p`, MD5 `f9b5b13eb24ecbe4b39f3a3f4a297645`.
- Prompt: `benchmarks/prompts/merge_sort_thinking_off.txt`, MD5 `253c7ac50857fe6d0e10fb0d2c5e35c0`.
- All performance measurement was delegated to `hipfire-benchmark-runner` using the existing `.codeinsight+research/cpu-exec-debug-run.sh`. No replacement harness or concurrent benchmarks.
- Every arm rebuilt `cargo build --release --bin hipfire --bin daemon`, then verified source and executable MD5s.
- Script invocation: `bash .codeinsight+research/cpu-exec-debug-run.sh`; its generation command uses `hipfire run --spec off --max-tokens 256 --kv-mode q8 --kv-backend vmm -j`.
- Fixture environment: `HIPFIRE_OFFLOAD_EXEC=cpu`, `HIPFIRE_MOE_EXPERT_BUDGET=24`, `HIPFIRE_GRAPH=0`, `HIPFIRE_REPLAY_BACKEND=hip`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`, thinking off. Trace is enabled only for the diagnostic arm.
- Each invocation runs a fresh trace process and a fresh timing-disabled control. Script GPU access is flock-serialized. Scoped `max_seq=4096` was restored and verified as `200000`.
- Authoritative control `tok_s` on this arch-6 AR route includes prompt prefill and decode, but excludes model load/spawn and terminal handshake. Trace-enabled tok/s is not used for performance selection.

## Default-pool comparison

| Arm | Control samples (tok/s, execution order) | Count | Median tok/s |
|---|---|---:|---:|
| Original per-row, earlier historical campaign | 46.6, 46.0, 48.0 | 3 | 46.60 |
| Fresh fixed-64 reference | 50.4, 49.4, 50.4 | 3 | 50.40 |
| Adaptive, first batch | 48.6, 50.5, 49.5 | 3 | 49.50 |
| Fixed-64 repeat | 49.4, 50.2, 50.1 | 3 | 50.10 |
| Adaptive repeat, completed portion | 50.3, 50.5 | 2 | 50.40 |
| Fixed-64 pooled | First and repeat fixed batches | 6 | 50.25 |
| Adaptive pooled | First and completed repeat adaptive batches | 5 | 50.30 |

The default-pool adaptive result matches the fixed reference within observed variation. This does not prove equivalence or a portable speedup. The early adaptive point estimate was lower, but the repeat did not reproduce that direction. Sampled completion lengths differ, changing prefill amortization; ranges overlap. The historical original is context, not a contemporaneous control for this adaptive comparison.

Diagnostic GU/DN/total below are milliseconds per splice: median of per-invocation medians over all trace splice lines. They are not throughput results.

| Batch | GU ms | DN ms | Total ms |
|---|---:|---:|---:|
| Original historical campaign | 0.330 | 0.180 | 0.780 |
| Fresh fixed-64 reference | 0.285 | 0.150 | 0.700 |
| Adaptive first batch | 0.270 | 0.140 | 0.670 |
| Fixed-64 repeat | 0.250 | 0.130 | 0.660 |
| Adaptive repeat, first completed invocation | 0.305 | 0.160 | 0.740 |
| Adaptive repeat, second completed invocation | 0.270 | 0.140 | 0.680 |

The CPU scheduling benefit remains present, but trace variability does not establish one exact optimum. No other bottleneck was investigated.

## Worker-count validation and stopping disposition

The same unmodified script also ran fixed-64 cells with inherited `RAYON_NUM_THREADS`; these are benchmark-only overrides, not production configuration changes.

| Fixed reference workers | Control samples (tok/s) | Median tok/s |
|---|---|---:|
| 4 | 42.8, 43.0, 42.3 | 42.8 |
| 8 | 49.6, 49.4, 48.0 | 49.4 |

The user prioritized immediate pushable commits. The runner finished the active script, verified configuration restoration, and stopped: the remaining adaptive repeat and adaptive 4/8-worker performance cells were cancelled. There is no paired adaptive performance claim for those worker counts. Correctness across those pool sizes was independently exercised below. No benchmark remained in flight.

## Correctness and MTP

- `cargo test -p hipfire-cpu --lib`: 35 passed, 0 failed.
- `cargo test --release -p hipfire-cpu --lib`: 35 passed, 0 failed, including after final adaptive restoration.
- `cargo test --release -p hipfire-dispatch moe_splice_tests -- --nocapture`: 4 passed, 0 failed, including graded/uniform multi-row grouping.
- Expanded shared-job test uses `ThreadPoolBuilder::install` with 1/2/4/8 workers, scalar and runtime-SIMD modes, mixed quant tiers, output rows 1/7/65/257, 1/4/9 job fixtures, activation counts 0..4, and a 257-job heap-fallback fixture. Every computed element is checked with `f32::to_bits()` against independent per-row GEMV; extra output tails remain untouched. Empty batches and zero rows are covered.
- Grain-budget invariants exercise `usize::MAX` without allocating enormous fixtures.
- The runner inspected generated text: completed generations stopped coherently, with no hangs.

A scout verified the existing CPU-deferred MTP gates: the loader declines `auto` to AR or errors on forced MTP; `mtp_shared_verify_accept_rollback_inner` also refuses when `has_cpu_deferred_experts()` is true. No opt-in bypass exists. The positive CPU-offload oracle bypasses the loader but not the runtime refusal. The user's conditional ungated smoke therefore did not run. Neither guard was changed.

Multi-row CPU arithmetic is verified; end-to-end MTP performance is not. It requires restoration of that separate route and a suitable MTP measurement route; this AR-only script is not MTP evidence.

## Binary/source provenance

| Arm | gemv.rs MD5 | CLI MD5 | Daemon MD5 |
|---|---|---|---|
| Original historical baseline | `08940df9c7a5578f82e0f72f870702fb` | `687dce9fff1d840113f1e7f4dbfee0d2` | `7e6909c7141461eeeec250578ab5e81e` |
| Fresh fixed-64 | `77d365b835ffe136b5500b10bd4b2b42` | `beab76a8d8d3ede051cb859b205ce6fd` | `9c7c126263f2f5cd0cab2799ff58a945` |
| Fixed-64 repeat/thread cells | `9ec377aceca91e36abc7b1be2fe212ef` | `aca4a2aeda868b0b9c3c110ed6ab1bcf` | `f06f388aee0ccf4a9a6c4aa8b9cf50d3` |
| Adaptive first/repeat | `588c5d778345d0af3bee7fafcf3738d2` | `c068a289d2e3ad8ff83cb6f3354e4075` | `9595741f965d6aa8266fe78428b0c162` |

Exact local receipts (discovery pointers; the tables above preserve the measured samples/provenance):
- `.codeinsight+research/cpu-exec-debug-fixed64ref-20261009-225856.summary.json`
- `.codeinsight+research/cpu-exec-debug-adaptive-20261009-230611.summary.json`
- `.codeinsight+research/cpu-exec-debug-fixed64repeat-threads-20261009-232233.summary.json`
- `.codeinsight+research/cpu-exec-debug-adaptive-repeat2-20261009-232639.summary.json`

Default raw invocation stamps: fixed first `225610/225738/225856`; adaptive first `230332/230455/230611`; fixed repeat `231206/231333/231452`; adaptive completed repeat `232514/232639`. Each has `cpu-exec-debug-20261009-<stamp>.json`, `.control.json`, and `.trace.log` under the research directory. Fixed worker-count stamps: 4 workers `231616/231734/231849`; 8 workers `232004/232118/232233`.

## Upstream assessment

Retain the simple adaptive policy as a provisionally justified scheduling candidate: exact arithmetic is preserved and measured default-pool throughput approaches the fixed reference without an observed material regression. The worker/job/activation-aware strategy is architecture-independent in inputs, not cross-platform performance-validated. Heterogeneous CPUs, quant tiers, shapes and concurrent workloads still require fixture-bound evidence before stronger claims. No additional SIMD, GPU, activation-cache, pool-sizing or MTP restoration work is included.
