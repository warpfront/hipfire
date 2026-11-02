# Dense CPU offload: owned pinned async return — 2026-10-10

**Lifecycle:** `historical`. **Disposition:** measured dense AR improvement under this exact fixture; not product admission, a current default, or a transferable baseline. MoE throughput is outside this record's numerical claim. CPU-backend MTP remains disabled and was not tested.

## Decision and mechanism

Replace blocking CPU-return `hipMemcpy` in `hipfire-dispatch/src/cpu_exec.rs` with `hipMemcpyAsync` from reusable owned pinned storage on the active/default GPU stream. Dense GEMV computes directly into the pinned output span; no temporary-vector lifetime hazard and no extra output-staging copy. Activation and optional AWQ-scale downloads use the same stream and share their readiness boundary. CPU arithmetic, rotation/AWQ ordering, residual destination, dispatch coverage, and row extents are unchanged.

The HIP bridge owner permits queued GPU access through checked raw buffer extents, not mutable host slices. The next same-stream D2H completion retires the preceding upload before CPU access. Warm returns introduce no event or return-side host wait. Growth, stream/device changes, and teardown retire outstanding access; a failed drain retains storage rather than freeing a live DMA source.

This does **not** remove the CPU's need to await its GPU-produced input/residual. The preceding [GPU dependency investigation](2026-10-10-cpu-moe-gpu-dependency-wait-gfx1201.md) remains unchanged: required producer computation and ordered copies precede CPU arithmetic. An async return is a separate lever, not proof that the incoming dependency wait was unnecessary.

## Fixture and method

- AMD RX 9070 XT (`gfx1201`), HIP/ROCm 7.2, Ryzen 7 7800X3D; default Rayon thread count.
- Dense model `qwen3.5-2b.mq4`, MD5 `9ed6628f2df83ef4b1c062afd4a85bfb`.
- Prompt `benchmarks/prompts/gpu_offload_probe.txt`, MD5 `5835c71e471849b4a72e1dc8e39695e7` (59 prompt tokens).
- `HIPFIRE_GPU_LAYER_BUDGET=12`, `HIPFIRE_OFFLOAD_EXEC=cpu`: 12 of 24 layers spilled, 12/12 fully covered, no uncovered quants in every measured process.
- Three fresh processes per arm; each process uses five runs, two warmups, `--spec off --backend noslots --workload stateless --max-tokens 64 --json`, timing trace disabled. Same-session frozen binaries control drift.
- Baseline source `82a06e95d2b6a61aa150af766b22866acc74275b`; candidate is that working tree with the five-file owned async-return runtime/test change. Binary identities below pin what ran independently of subsequent commits.

| Binary | Frozen MD5 | Candidate MD5 |
|---|---|---|
| `hipfire` | `653bcaee005273dd4df8f353d95c8c52` | `d65c1093fbc9b2e1037393cef1813d91` |
| `daemon` | `23a7dccec849572bc68078ff25eab235` | `9192245e60995e340d7352d19277e467` |

## Measured decode throughput

Values are each fresh process's five-run median, not individual token samples.

| Arm | Process 1 | Process 2 | Process 3 | Median of process medians |
|---|---:|---:|---:|---:|
| Frozen | 36.8 tok/s | 37.0 tok/s | 37.2 tok/s | 37.0 tok/s |
| Owned async return | 42.6 tok/s | 42.8 tok/s | 41.2 tok/s | 42.6 tok/s |

Delta: **+5.6 tok/s, approximately +15.1%** for this fixture. The earlier separate-session frozen median was 37.5 tok/s; use the same-session 37.0 control here. Candidate process 3 was noisier; complete samples are preserved, not replaced by its best run.

[Durable raw evidence](data-2026-10-10-dense-cpu-async-return.json) contains all six full `hipfire bench --json` objects, including their samples arrays. Local discovery/protocol pointers: `.codeinsight+research/async-dense-baseline-20261010-113024/BASELINE.md` and `.codeinsight+research/async-candidate-20261010-115242/RESULTS.md`.

## Exercised correctness and limits

`hipfire-benchmark-runner` executed both actual hardware oracles:

- `cpu_offload_return_stream_and_reuse_parity`: bit-exact CPU-reference comparisons through the real offload helpers, same-stream GPU consumers and following CPU reads; MQ4/AWQ raw and prerotated inputs, HFQ6 and Q8, default plus two nonblocking streams, allocation growth/shrink and chained returns.
- `graded_cpu_moe_experts_multirow_parity`: real graded expert buckets, zero residual difference, accepted-prefix arithmetic and stream/buffer reuse across logical rows 3 → 2 → 3 → 1. This direct splice test does not enable or certify CPU MTP.

The unchanged MoE debug script completed with 16/16 CPU coverage and complete coherent merge-sort output. Separate dense serve checks at 512 tokens terminated naturally: complete longest-substring implementation and five completed chained answers; no runaway flags, with observed prefix-cache hits. The 64-token throughput fixture and initial 64-token text checks intentionally truncate output and are not task-completion acceptance.

Trace `h2d` now measures async enqueue API wall time, not DMA completion. A shorter splice trace alone cannot establish a throughput gain. This dense A/B also changes pageable activation transfer to pinned same-stream transfer; its gain must not be attributed exclusively to H2D synchronization. No MoE speedup, cross-model gain, CPU-MTP result, or current product performance is claimed here.
