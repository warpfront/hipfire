# CPU MoE compute-primary timings and MQ4 chain experiments — 2026-10-10

**Lifecycle:** historical. **Disposition:** retain the previously measured MQ3/MQ6 prefix decoder; reject and revert both MQ4 accumulator experiments. Fixture-bound evidence, not product admission, a portable speedup, or proof of a hardware limit.

This follows [the immutable prefix-decoder checkpoint](2026-10-10-cpu-moe-prefix-broadcast.md) and [adaptive row grain](2026-10-09-cpu-moe-adaptive-row-grain.md). The user subsequently allowed small CPU arithmetic differences and made **absolute traced computation time** the primary criterion; throughput is secondary. No scheduling, transfer, quantization-format or activation-quantization change was attempted here.

## Method and fixture

All performance runs and log analysis were delegated to `hipfire-benchmark-runner`. The existing `.codeinsight+research/cpu-exec-debug-run.sh` was unchanged; no replacement harness, new timing instrumentation or concurrent benchmarks. Each invocation runs a diagnostic process and a separate timing-disabled control.

- Ryzen 7 7800X3D; default 16 logical workers, `RAYON_NUM_THREADS` unset.
- RX 9070 XT, gfx1201, HIP 7.2.
- `qwen3.6:35b-a3b`, local `qwen3.6-35b-a3b.mq4p`; model MD5 `f9b5b13eb24ecbe4b39f3a3f4a297645`.
- `benchmarks/prompts/merge_sort_thinking_off.txt`; MD5 `253c7ac50857fe6d0e10fb0d2c5e35c0`.
- `HIPFIRE_OFFLOAD_EXEC=cpu`, `HIPFIRE_MOE_EXPERT_BUDGET=24`, `HIPFIRE_GRAPH=0`, `HIPFIRE_REPLAY_BACKEND=hip`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`; thinking off.
- `hipfire run --spec off --max-tokens 256 --kv-mode q8 --kv-backend vmm -j` with that prompt. Scoped `max_seq=4096`, restored to `200000`, verified by the runner.
- All invocations reported 16/16 spilled layers CPU-covered and no uncovered quant work left on the GPU.

For this record, retain printed lines with `calls >= 16` in **every arm**. The print gate is sparse, not an exhaustive trace or a controlled per-format microbenchmark. Compute total is calculated **per line** as `gu + middle + dn + combine`, then reduced to a per-invocation median. It excludes copies/transfers. Total splice includes copies and is contextual. Across invocations, report medians of those medians; a compute-total median is not the sum of independently reduced component medians.

The original twelve decoder logs were reprocessed without rerunning inference. This common warm-call filter differs from the preceding checkpoint's all-printed-lines medians; do not mix the two tables. The nine MQ4 invocations ran four chains, restored single chain, then two chains, three serial fresh-process invocations each.

Trace fields are printed to two decimal places in milliseconds. Four rounded components give a worst-case approximately ±0.020 ms rounding uncertainty in their sum; fractional medians do not recover sub-print-resolution measurements. Workload variation and scheduling noise add uncertainty beyond rounding.

## Retained MQ3/MQ6 decoder: compute-only before/after

Six original and six prefix-decoder invocations, unchanged adaptive Rayon policy. All values below are milliseconds; throughput is from the separate trace-disabled controls.

| Decoder | GU | DN | Middle | Combine | Compute total | Total splice | Control tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original, n=6 | 0.260 | 0.140 | 0.020 | 0.010 | 0.4300 | 0.6575 | 48.5 |
| Prefix broadcast, n=6 | 0.240 | 0.125 | 0.030 | 0.010 | 0.4025 | 0.6350 | 50.8 |

Runner-computed changes: **GU −20 µs, DN −15 µs, compute total −27.5 µs (−6.4%)**, total splice −22.5 µs. The previously measured control change was +2.3 tok/s (+4.7%). The middle's 0.02→0.03 ms median is at the print resolution and is not evidence of a new middle-stage computational regression.

Original per-run compute medians: `0.430, 0.410, 0.430, 0.435, 0.400, 0.430`. Prefix medians: `0.430, 0.405, 0.395, 0.410, 0.400, 0.390`. Per-run ranges overlap, collectively 0.32–0.55 ms; this is a pooled directional improvement, not separation of every invocation or every splice. The B-C-B-C design and arithmetic-preserving integer-decoding change support retaining it, with the preceding checkpoint's fixture and completion-length caveats intact.

[Durable warm-filter receipt](data-2026-10-10-cpu-moe-prefix-compute.json) contains per-run component/compute medians, ranges and counts. The unchanged preceding [generation/provenance receipt](data-2026-10-10-cpu-moe-prefix-broadcast.json) preserves original control and diagnostic outputs, hashes and invocation identities.

## Rejected MQ4 arithmetic candidates

Only `mq4_group_dot`'s code-dot accumulator assignments and final reduction changed during these experiments. Nibble decoding, metadata, activation preparation, cached/uncached Σx quartet order, affine correction and inter-group row accumulation were unchanged.

- Four chains: one FMA chain for each of the four eight-code fragments per 32-code block; balanced final reduction.
- Two chains: fragments 0/2 share one chain, fragments 1/3 another; one final vector addition.
- Both cached and uncached entry points used the same candidate dot reduction.

Warmed per-invocation medians, milliseconds. `n` is the number of eligible printed trace lines. Control tok/s is secondary; the final two-chain control emitted 160 tokens, the other eight emitted 166. Diagnostic and control text can differ and selected quant mixtures are not fixed.

| Stamp, 20261010 | Chains | GU | DN | Middle | Combine | Compute | Compute range | Total splice | n | Control tok/s |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 042743 | 4 | 0.220 | 0.120 | 0.030 | 0.010 | 0.380 | 0.34–0.42 | 0.610 | 21 | 51.0 |
| 042915 | 4 | 0.230 | 0.120 | 0.030 | 0.010 | 0.390 | 0.35–0.42 | 0.620 | 22 | 50.9 |
| 043043 | 4 | 0.260 | 0.130 | 0.020 | 0.010 | 0.420 | 0.37–0.47 | 0.650 | 21 | 52.1 |
| 043407 | 1 | 0.230 | 0.120 | 0.030 | 0.010 | 0.390 | 0.34–0.43 | 0.615 | 22 | 51.2 |
| 043537 | 1 | 0.255 | 0.135 | 0.020 | 0.010 | 0.420 | 0.32–0.53 | 0.655 | 20 | 51.2 |
| 043701 | 1 | 0.235 | 0.120 | 0.030 | 0.010 | 0.395 | 0.32–0.46 | 0.620 | 22 | 51.9 |
| 044026 | 2 | 0.240 | 0.130 | 0.030 | 0.010 | 0.415 | 0.35–0.47 | 0.650 | 22 | 51.1 |
| 044158 | 2 | 0.240 | 0.120 | 0.030 | 0.010 | 0.400 | 0.32–0.46 | 0.625 | 22 | 50.0 |
| 044325 | 2 | 0.225 | 0.120 | 0.030 | 0.010 | 0.390 | 0.35–0.45 | 0.620 | 22 | 49.7 |

Runner-computed median-of-medians compute totals: single **0.395 ms**, four **0.390 ms**, two **0.400 ms**. These ±5 µs differences are unresolved at the trace resolution and within variation. Neither candidate demonstrates a repeatable absolute compute improvement. **Both reverted; original MQ4 arithmetic retained.**

Generated assembly (`objdump -Cd --disassemble='hipfire_cpu::simd::x86::mq4g256_row_dot_cached' target/release/daemon`) confirmed four FMA chains with three vector additions, but the fully unrolled function spilled the outer scalar row accumulator. The two-chain function had two FMA chains, one vector addition, used registers through `ymm10`, retained the row accumulator in `xmm0`, and had no stack spills. Cleaner code generation alone was not sufficient. Spill overhead is a hypothesis, not an isolated measured cause.

### Binary provenance

All nine CLI executables byte-matched: MD5 `852e62e276c6677cb1e7e12c28e48f27`. CPU GEMV runs in the daemon; daemon MD5s identify the measured arms:

| Arm | Daemon MD5 |
|---|---|
| Four chains | `27105a4e0f7fb12226f38648d2ea431e` |
| Restored single chain | `34fad9e1adaa0e6c204228363e4ea483` |
| Two chains | `4e8d31ac1d101fb5042d16f6389d42df` |

[Durable nine-invocation receipt](data-2026-10-10-cpu-moe-mq4-chain-comparison.json) preserves complete decoded control/diagnostic content, counts, finish reasons, rates, component/compute distributions, hashes and source artifact paths. Local source receipts were `.codeinsight+research/cpu-mq4-chain-comparison-20261010.summary.json` and `.codeinsight+research/cpu-gemv-prefix-compute-20261010.summary.json`. Existing checkpoint files and prior receipts are unchanged.

## Numerical acceptance and correctness

The user waived historical bit identity for candidate CPU arithmetic, not format/layout correctness. No loose numerical tolerance was introduced into production, and no reassociated candidate remains.

A scout inspected the requested local `~/Documents/vibecoding/llama.cpp` checkout:

- `tests/test-quantize-fns.cpp` defines dot error limits 0.02 and 0.04 for low-bit formats, calculated as `fabsf(result - dot_ref) / test_size` against original float inputs. This is **absolute error divided by vector length**, not a relative percentage; it includes quantization error. Its float products are accumulated in a `double` reference sum. These values are not a portable hipfire rounding budget.
- Q4_K x86 and scalar implementations use different architecture-specific reductions (`ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml/src/ggml-cpu/quants.c`). Source establishes no universal cross-backend bit-identity contract; llama.cpp inference was not run.
- Row/column ownership in `ggml-cpu.c` does not itself establish same-backend nondeterminism. Historical MQ4-Lloyd GPU PPL equality is not a CPU qt13 arithmetic gate.

A removed throwaway correctness probe compared four-chain MQ4 against the single-chain outputs on 520 awkward-metadata outputs, K=512/2048: maximum absolute difference `3.8146973e-6`, normalized by maximum reference-output magnitude `1.0896696e-7`. This is observed on that fixture, not an acceptance bound for arbitrary inputs.

Each MQ4 candidate passed 37 release CPU tests, four MoE splice tests and both ignored Qwen3.5 GPU/CPU parity tests. The real MQ4 GPU-oracle worst normalized errors were `4.317e-7` for four chains and `4.797e-7` for two chains; two-chain maximum absolute error was `1.043e-6`, including the real 2B AWQ case. The oracle normalization is maximum absolute difference over `max(max_abs_cpu_reference, 1e-6)`, not a per-element relative error.

After restoring the final retained implementation:

- `cargo test -p hipfire-cpu --lib`: **37 passed**.
- `cargo test --release -p hipfire-cpu --lib`: **37 passed**.
- `cargo test --release -p hipfire-dispatch moe_splice_tests -- --nocapture`: **4 passed**.
- `cargo test --release -p hipfire-arch-qwen35 --test gpu_gemv_parity -- --ignored --nocapture`: **2 passed**.
- `cargo build --release --bin hipfire --bin daemon`: successful.

The additional permanent `mq4_cached_and_uncached_agree_on_awkward_inputs` regression verifies bit-identical cached/uncached output on non-power-of-two metadata and activations, four activation rows, K=256/512/2048/12288. Existing all-format, shared-job and multi-row splice coverage remains. Some synthetic HFQ3 GPU cases are explicitly skipped by the existing oracle on this architecture; CPU format coverage is not skipped. End-to-end CPU-deferred MTP remains guarded/unavailable, unchanged; multi-row CPU execution is covered by existing tests, not a claimed MTP runtime pass.

Actual script-driven CLI/daemon inference ran for every measured arm. Inspected outputs implement coherent recursive merge sort with terminating base case, two-way merge and remainder append; sampled receipts finish with `stop`. Text variants and completion-length variation prevent treating these invocations as a byte-identical routed-workload experiment.

## Bottleneck interpretation and next computation target

GU is larger than DN in the sampled compute sections. **[INFERENCE]** Integer decoding overhead is a contributing computational cost: removing MQ3/MQ6 scalar stitching produced a pooled compute reduction without changing floating-point operations. The experiments do not establish whether the entire GEMV is compute-bound or bandwidth-bound, nor that the CPU is near its hardware limit. `perf` was unavailable; `perf_event_paranoid=2` alone is not evidence that user-space counters would be forbidden.

The model inventory is 51 MQ6, 77 MQ4 and 128 MQ3-Lloyd experts per layer, but this is not the selected routing mix. The trace's `quant=` identifies the first GU expert, not all GU/DN weights. No per-format bandwidth or latency is inferred from that label.

The next focused computation experiment should target MQ3-Lloyd's decode/codebook/FMA sequence, starting with generated instruction and dependency analysis rather than another Rayon change. **[INFERENCE]** Its larger inventory makes it a more plausible target than spending more on an unresolved MQ4 chain change; actual selected-format attribution is required before claiming it dominates. Reassociation is numerically feasible on the tested inputs, but must earn a repeatable GU/DN compute-time reduction before retention. Nothing here remeasures or disputes the user's observation that CPU execution is slower than GPU bus reads.
