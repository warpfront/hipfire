# CPU MoE GPU dependency wait — gfx1201, 2026-10-10

**Lifecycle:** historical. **Disposition:** retain the optimized baseline; reject `HIPFIRE_HIP_WAIT=spin` as a performance lever on this fixture. No runtime, dispatch, CPU arithmetic, synchronization, or default-policy changes. This is fixture-bound investigation evidence, not admission or a portable hardware limit.

## Finding

The approximately 0.13 ms traced `d2h_wait` is not a redundant production barrier. Production enqueues four pinned D2H downloads on the producer stream and waits once for their completion. The diagnostic adds a separate producer drain before those downloads. Removing that diagnostic drain cannot improve timing-disabled inference.

A matched production-path GPU/API timeline shows that most of the completion wait spans unfinished GPU producers and their serialized inter-dispatch gaps. The post-copy host-return tail is small. GPU wait is therefore not a cheap synchronization-removal opportunity, but faster producers, fewer dispatch boundaries, or correctly ordered overlap could still shorten it.

## Fixture and method

All benchmarking and timing analysis used `hipfire-benchmark-runner`; two read-only scouts traced the GPU producer sequence and HIP bridge semantics. The existing `.codeinsight+research/cpu-exec-debug-run.sh` was unchanged. Each invocation launches one diagnostic process and a separate timing-disabled control.

- Source: `feature/qwen35-moe-offload`, `82a06e95d2b6a61aa150af766b22866acc74275b`, clean when measured.
- CLI MD5: `653bcaee005273dd4df8f353d95c8c52`; daemon MD5: `23a7dccec849572bc68078ff25eab235`. Frozen pair: `.codeinsight+research/cpu-depbench-20261010-104403/`.
- RX 9070 XT, gfx1201; HIP/ROCm `7.2.53211`; Ryzen 7 7800X3D, default Rayon, `RAYON_NUM_THREADS` unset.
- Registry tag `qwen3.6:35b-a3b`, resolved local file `/home/avery/.hipfire/models/qwen3.6-35b-a3b.mq4p`; MD5 `f9b5b13eb24ecbe4b39f3a3f4a297645`. This graded MQ4p fixture is **not** the canonical DFlash-A3B fixture in AGENTS.md.
- Prompt: `benchmarks/prompts/merge_sort_thinking_off.txt`; MD5 `253c7ac50857fe6d0e10fb0d2c5e35c0`; thinking off.
- `HIPFIRE_OFFLOAD_EXEC=cpu`, expert budget 24, graph off, replay backend HIP, GPU 0, q8/vmm KV, scoped `max_seq=4096`, `max_tokens=256`, DPM warmup 10 s. Sixteen CPU-deferred layers; no host-mapped expert steps left on the GPU.
- Serial GPU lock `/tmp/hipfire-gpu.lock`; `GPU_MAX_HW_QUEUES` unset, automatic RDNA4 cap observed. Script restored `memory.max_seq=200000`; no wait-mode setting persisted.

Three initial unset-policy pairs were followed by spin / unset / spin / spin pairs. This supplies three fresh-process spin controls and four unset controls, including one interleaved reference, not a fully balanced ABBA campaign. The same prompt produced different sampled token streams and completion lengths; these are matched-fixture measurements, not byte-identical routed workloads.

[Durable receipt](data-2026-10-10-cpu-moe-gpu-dependency-wait.json) retains full diagnostic/control generation JSON, identities, final shape lines, sixteen matched timeline rows, busy/idle splits, and the rejected-policy result. Large raw profiler CSVs remain local discovery artifacts, not durable copies.

## GPU producers and synchronization

Source sequence at the measured commit:

- `crates/hipfire-dispatch/src/pipeline/moe_program.rs::lower_decode`: Normalize → InputBasis → GateSide → Route → SharedDown → host-only MutationFence. CPU-deferred calls omit routed GPU GU/activation/down/combine; the architecture invokes the CPU splice after the sealed program.
- `pipeline/mod.rs::decode_gate_side_stage`: the mixed Q8-router/MQ-shared fixture uses separate router/selector projections and the already-prerotated shared gate/up projections. The all-MQ fused quartet does not apply to this mixed dtype set. Existing rotation reuse is retained.
- `pipeline/mod.rs::decode_shared_down_stage`: shared activation followed by the sigmoid-scaled shared-down residual projection. It updates the residual base the CPU eventually accumulates into.
- `cpu_exec.rs::moe_cpu_experts`: validates extents and capture/replay constraints, reuses pinned staging, optionally performs the trace-only drain, enqueues `x_rot`, indices, routing weights, and residual downloads, then takes one same-stream completion wait. Expert resolution, grouped CPU GU/SiLU/FWHT/DN, rank-ordered combine, and blocking residual H2D follow.
- `hip-bridge/src/ffi.rs::memcpy_dtoh_async` and `stream_synchronize[_default]`: copies and completion use the active stream or the same legacy stream. The device-wide synchronization in the splice is error cleanup, not the successful hot path.
- `rdna-compute/src/dispatch.rs::memcpy_htod_auto`: outside capture, the residual upload uses blocking `hipMemcpy` on the legacy stream. It was not changed.

The observed chain also includes preceding layer work still outstanding when the CPU reaches the splice: pre-MoE projection, residual add, and normalization. The wait cannot be attributed only to SharedDown's own kernel duration.

For the Qwen narrow multi-row path, `lower_prefill` places SharedDown **before** Route. Route is therefore the last producer for that recipe. A decode-only route-ready overlap design cannot simply be applied to multi-row verification. Existing row-width admission, rank order, rollback behavior, active-stream ordering, and graph/replay refusal were left untouched.

## Matched production completion wait

The successful capture used daemon-wrapped `rocprofv3` with kernel/HIP-runtime/memcpy CSV tracing through `serve_harness.py`, and graceful SIGINT cleanup to flush the daemon's trace. Profiling only the CLI or terminating a short `run` before daemon flush did not provide a usable kernel CSV; ptrace attachment was unavailable.

The profiled serve request had `HIPFIRE_CPU_EXEC_TRACE` **unset**: these are production single-completion-wait observations, not the extra diagnostic drain. Kernel and API timestamps share one daemon time base. Scope: the final decode token's sixteen CPU-deferred layers in an intentionally truncated 16-token request.

| Interval | Median, µs | Interpretation |
|---|---:|---|
| Completion-wait entry → last GPU producer completion | 88.65 | Pending ordered producer span |
| GPU busy within that producer span | 53.17 | Actual dispatched kernel execution |
| Inter-dispatch idle within that producer span | 35.40 | Serialized gaps; not pure computation |
| Last producer completion → fourth D2H completion | 20.9 | Copies and intervening queue gaps |
| Fourth D2H completion → wait return | 6.79 | Same-boundary host completion tail |
| Entire `hipStreamSynchronize` API call | 117.63 | Inclusive of the pending work above |

These are independently reduced cross-layer medians; they do not add exactly. Fourteen of sixteen waits begin during a producer kernel; two begin in a dispatch gap. Fifteen post-copy tails are approximately 6.6–7.0 µs; one is 170.1 µs, with an inclusive 280.0 µs wait. A single outlier does not justify a default change.

The trace does not show a large deferred-start stall or Hipfire software batching. It does show serialized GPU dispatch gaps, roughly 3.2 µs per boundary in this slice. Fusing suitable producers could reduce some gaps; the measurement alone does not prove that every gap is removable or identify a driver-internal cause.

The earlier preliminary interpretation that the inclusive API duration was mostly HIP overhead is **rejected**. It compared a complete host wait with only the last producer's kernel duration. Timestamp alignment, not that subtraction, establishes the intervals above.

## Existing wait-policy experiment

`HIPFIRE_HIP_WAIT=spin` already maps to HIP device scheduling flags before `set_device` (`rdna-compute/src/dispatch.rs:1684–1702`). It changes host wait policy, not kernels or synchronization count. No new polling loop, event, stream, or API was added.

Timing-disabled controls:

| Invocation, 20261010 | HIP wait policy | Control tok/s |
|---|---|---:|
| 104436 | unset | 52.4 |
| 104605 | unset | 53.7 |
| 104733 | unset | 51.2 |
| 110401 | spin | 53.1 |
| 110538 | unset | 52.0 |
| 110709 | spin | 52.8 |
| 110835 | spin | 52.1 |

Unset median 52.2 tok/s; spin median 52.8 tok/s. The difference is inside the observed unset range and does not establish an improvement. The original three baseline controls alone have median 52.4 tok/s, consistent with the requested approximately 52 tok/s optimized baseline.

Final diagnostic lines are **cumulative per-shape means**, not individual-splice samples; only the largest-call-count line per shape/invocation was retained. Across those lines, unset total splice is 0.55–0.63 ms and spin is 0.54–0.63 ms. Both policies retain `d2h_wait=0.12–0.13 ms`. GU remains approximately 0.20–0.26 ms and DN 0.11–0.14 ms, with no observed CPU GEMV regression or demonstrated wait reduction. Printed 0.01 ms resolution cannot resolve the small post-completion tail.

Even eliminating the typical post-copy return tail entirely would save only about 0.11 ms across sixteen splices in a roughly 19 ms decode step. This is a scale estimate, not a measured speedup. Spin was rejected and remains unset.

## Overlap decision and next bottleneck

The dependency is necessary under the existing operation order and CPU residual-combine contract, not an immutable hardware floor:

1. **Do not remove the D2H completion boundary.** CPU reads require completed rotated activations and expert IDs; final combine requires routing weights and the shared-updated residual. Removing it would expose unfinished data.
2. **Do not split one wait into two.** In decode, inputs become ready at Route before SharedDown, but copies currently queue after SharedDown. Earlier copies plus separate route/residual completion boundaries would add synchronization and complicate pinned-buffer lifetime/error cleanup. Shared activation and down take only a small part of the whole pending producer span; this is not a recovery of the entire traced wait. Multi-row ordering differs.
3. **Do not promote scratch hoisting as a gain.** Shape-only scratch preparation can legally precede completion, but warm resizing is already reusable and GU preparation/slot setup round to zero in these traces. No material preparation overlap is demonstrated.
4. **Next primary target: CPU GU/DN execution.** It is larger than the dependency wait. Profile and optimize the retained decoding/dot sequence rather than retuning Rayon or changing activation precision without a new accuracy contract.
5. **If targeting the GPU dependency path, target actual producers and dispatch gaps.** In this slice the pre-MoE projection is about 29 µs and SharedDown about 14.4 µs; faster kernels or an arithmetic-preserving fuse would address work rather than suppressing the wait. Mixed Q8/MQ gate-side dispatch and multi-row fallbacks must remain correct.
6. **Separate next handshake investigation:** blocking residual H2D is about 44 µs of host API wall versus about 2.9 µs of device copy in the profiled slice. This is not solved by deleting the D2H wait. An asynchronous upload would need an explicit staging-lifetime and stream-ordering design; no unsafe buffer-reuse shortcut was implemented.

## Verification and limits

All fourteen diagnostic/control generations finished with `stop` and coherent merge-sort implementations, including terminating base case, sorted merge, and remainder append. This exercises the actual CLI/daemon CPU-offload path and the spin alternative, but is not a numerical parity oracle or full serve/state acceptance suite.

The short profiler request intentionally ended at the token limit and reports `runaway=true`; it is attribution evidence **only**, not a passed correctness battery or a throughput claim. GPU timings cover one final-token layer sweep, not multiple fresh profiled processes. No model, kernel, state, arithmetic, or multi-row behavior changed; existing contract tests and parity oracles were intentionally unchanged and not rerun.

Only this checkpoint and its new receipt are permanent outputs. Existing checkpoints are unchanged; no performance gain is claimed.
