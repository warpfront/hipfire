# Qwen4 shared token-batched prefill

## Status and purpose

**Status: planned.** This document records the agreed architectural direction and an implementation/verification plan for Qwen3.8 Flash-Next (`hipfire-arch-qwen4`). It is a cross-session reference, not a claim that batching is implemented, numerically accepted, or faster.

Follow [the design-record lifecycle](README.md). Runtime validation authority remains [../VALIDATION.md](../VALIDATION.md); benchmark protocol remains [../methodology/perf-benchmarking.md](../methodology/perf-benchmarking.md). Preserve this record; record later decisions and progress in separate linked records rather than rewriting historical intent.

This follows the shared sealed MoE work described in [sealed-granular-moe.md](sealed-granular-moe.md). It is not a request to redo that merge or replace its ownership model.

## Objective

Deliver real token-batched Qwen4 AR prefill through a shared execution description and centralized lowering. Keep architecture crates as small, typed descriptions and resource binders—not separate forward interpreters for decode and prefill.

The performance objective is layer-major matrix execution across prompt rows, reducing repeated weight reads and kernel launches. Wrapping the existing per-token loop in `Step`s is not sufficient.

“Centralized engine” here means the shared execution/dispatch/lowering machinery, not moving GPU layer execution into the host request scheduler.

## Current orientation: recheck before implementation

The following describes the code inspected when this plan was written. Symbol names are orientation anchors, not guarantees about a future checkout.

- `crates/hipfire-arch-qwen4/src/gpu_forward.rs`: `Qwen4GpuForward::forward_chunk_inner` embeds a chunk, then loops over tokens and layers. Most execution scratch is single-token. `execute_moe` binds `MoeParams` with `batch_size: 1`, seals decode, and executes `Step::Moe`.
- The same file owns direct HC/GDN/QSA/PLE execution. These operations do not yet constitute a shared layer program.
- `crates/hipfire-arch-qwen4/src/state.rs`: GDN owns recurrent and convolution state; QSA owns cache/pooling/selection state. QSA selection scratch currently holds one query's selection.
- `crates/hipfire-runtime/src/external_rows.rs` (reader) + `crates/hipfire-arch-qwen4/src/ple.rs` (row ids): PLE uses bounded staging, history-dependent lookup, and lease/epoch lifetimes. Multi-token PLE kernels exist, but surrounding execution remains scalar.
- `crates/hipfire-arch-qwen4/src/mtp_gpu.rs` and `mtp_spec.rs`: native MTP has separate state and token-at-a-time execution. Batching the target does not automatically batch MTP.
- `crates/hipfire-dispatch/src/pipeline/moe_program.rs`: shared decode/prefill stage lowering exists. Architecture consumers bind typed calls; they do not construct an expert-kernel interpreter.
- Qwen35 and Cohere already consume `MoePrefillParams` → `seal_prefill` → `Step::Moe` for token batches. Qwen35's `PrefillBatchScratch` is a useful existing layout reference.
- Qwen4's canonical top-10/QT44/QT53 decode route exists. Full prefill selection/lowering still has k=8 assumptions, lacks the necessary QT53 shared/grouped-down branches and Qwen4 G128 activation rotation, and has no Qwen4 prefill consumer. Some lower-level top-10 batched primitives already exist; their presence is not end-to-end support.

Prior profiling evidence is local scratch at `.codeinsight+research/qwen4/ar-pp-profile/`, including `summary.json`. It identifies token-serial execution as a structural limitation and records a separate router optimization. Do not use those measurements as a baseline for a different binary/model/prompt or as proof of this proposed path.

## Architectural contract

### Family responsibility

The family provides a typed description of each layer and binds its resources:

- Layer kind, dimensions, weights, formats, and exact numerical rules.
- HC stream/residual structure and PLE placement.
- Recurrent state, attention cache, PLE resources, and sealed expert ownership.
- Requested output rows and any consumer-required intermediate captures.

A conceptual layer is HC preparation → GDN or QSA → HC residual update → HC preparation → MoE → HC residual update, with PLE at its existing configured boundary. These are semantic operations, not proposed enum names or permission to reorder current arithmetic.

### Shared execution responsibility

The shared layer owns:

- Execution of the description for one row or a bounded token chunk.
- Shape-aware lowering, kernel selection, launch ordering, and resource validation.
- Scratch layout, lifetime, aliasing, and reuse.
- GEMV versus GEMM selection and indexed versus grouped expert execution.
- Stateful operation execution, causal visibility, and state advancement.

Use the existing `Step` and sealed-operation patterns where they fit. Do not introduce a second interpreter, public family-specific recipe language, or one public operation per kernel launch. Specialized mathematical kernels are acceptable behind shared semantic operations; not every primitive must be artificially generalized.

Exact public types are an implementation decision after inspecting current shared consumers. Prefer extending existing descriptors over a parallel API. Justify new shared abstractions with concrete consumers and reduced family orchestration.

### Execution shape

Current shape:

```text
embed/prefetch chunk
for token:
    for layer:
        execute scalar layer
    compute logits
```

Target shape:

```text
for bounded prompt chunk:
    bind/embed/prefetch token rows
    for layer:
        execute shared layer program on token rows
    compute requested output rows
```

One description must support decode and prefill. Different internal lowerings are expected; duplicated family-owned sequencing is not.

## Implementation phases and exit criteria

### Phase 1 — Establish the semantic baseline and shared contract

1. Inspect current HEAD, working-tree changes, shared Step APIs, and exported-symbol references. Preserve unrelated work. Recheck the orientation above rather than assuming an earlier session's line numbers are current.
2. Record exact model, fixtures, prompt bytes, binaries, settings, output tokens, and current failures. Preserve target and native-MTP behavior separately.
3. Map Qwen4 layer arithmetic, aliasing, state transitions, PLE placement, and consumer output/capture requirements into existing shared operations and the minimum missing semantic operations.
4. Define the typed layer description and its central execution owner. Identify a concrete existing-family consumer for common portions; do not rewrite unrelated families wholesale.

**Exit:** explicit ownership and shape contracts; reproducible scalar baseline; no unresolved ambiguity about causal state or output requirements. A pre-existing oracle failure remains a named blocker for that acceptance claim, not permission to loosen tolerance.

### Phase 2 — Unify execution ownership without changing arithmetic

1. Make the family bind the description/resources; place execution and lowering in shared code.
2. Preserve scalar launch order, HC aliasing, state mutations, PLE lifetime, output initialization, and sealed MoE validation.
3. Keep target and MTP using their shared applicable operations while retaining distinct state and output requirements.
4. Remove superseded family-owned orchestration as callers migrate; do not leave decode/prefill compatibility interpreters.

**Exit:** scalar numerical and state preservation, unchanged relevant product behavior, and no new per-token allocation/copy or launch regression attributable to abstraction. This phase alone is not a prefill speedup claim.

### Phase 3 — Add bounded layer-major token execution

1. Allocate reusable multi-row streams, projection/HC scratch, attention selection workspace, and MoE scratch. Budget memory by chunk size; respect PLE staging limits. Do not size every temporary to the entire context.
2. Preserve aliases deliberately: HC and PLE currently read/write `streams` in place. Establish when inputs must remain live rather than widening buffers blindly.
3. Batch independent projections with GEMM and batch row-local HC/pointwise work.
4. Execute stateful operations in causal order within each layer. Ordered internal recurrence is an acceptable first implementation; the surrounding projections must actually batch.
5. Use existing multi-token PLE primitives with the correct stream/query rows, history, and lease lifetime.
6. Compute only requested logits/capture rows. Ordinary AR prefill needs the final logits row; teacher-forced/speculative consumers may require more. Preserve the consumer contract explicitly.

**Exit:** chunked execution works for one row, multiple rows, and a partial final chunk; matrix work genuinely spans rows; state after a chunk supports correct subsequent decode. No claim of fully parallel recurrence is required.

### Phase 4 — Extend the shared MoE prefill route

This work can proceed alongside independent batch primitives once Phase 1 fixes the interfaces; integrate through Phase 3 rather than inventing a separate family loop.

1. Bind `MoePrefillParams` from Qwen4 and use existing expert binding, sealing, route readiness, fences, and common lowering.
2. Admit only the exact supported Qwen4 geometry/formats. Preserve missing-resource, illegal-QT53-placement, source/device/lease, and unsupported EP/replay rejection.
3. Complete top-10 routing/scatter/unscatter/combine, QT44 gate/up, G128 intermediate rotation, QT53 down, and BF16/QT53 shared-down through the common stages. Audit existing lower-level kernels before adding any.
4. Restore canonical token/selected-slot order before weighted combination. Preserve selected-weight normalization and output initialization semantics.
5. Use grouped expert matrix execution where supported and beneficial. A batched indexed kernel alone does not establish grouped weight reuse; selection should remain centralized and evidence-driven.

**Exit:** sealed batched calls match the scalar reference, including dirty-scratch reuse, and invalid calls fail before GPU work. Existing Qwen35/Cohere semantics remain intact.

### Phase 5 — Optimize stateful chunk internals behind the same contract

Prioritize from the new profile; do not assume the original bottleneck survives Phases 3–4.

- **GDN:** batch input/output projections first. Convolution history and recurrent matrix updates must remain ordered. A chunkwise recurrence/scan optimization needs independent numerical proof and must not leak sequencing into the family.
- **QSA:** batch projections/cache/query work while preserving each query's selected blocks and causal tail. Appending all chunk rows must not expose future tokens through pooled summaries or end-of-chunk selection. A conventional attention mask alone is not a sufficient argument.
- **PLE:** preserve causal depthwise history and token-dependent lookup history across arbitrary chunk boundaries. Retain lease/epoch cleanup and commit semantics.

**Exit:** optimized implementations satisfy the same semantic operation contract and chunk-equivalence checks. No architecture API fork is needed to select them.

### Phase 6 — Product and performance acceptance

1. Run claim-scoped kernel/dispatch and serving validation from `docs/VALIDATION.md`; record raw reports and inspect decoded text.
2. Compare matched fresh-process AR runs across useful prompt depths, including short inputs and multi-chunk inputs. Serialize GPU work. Record contention; do not call a contended comparison an uncontended performance result.
3. Profile candidate execution with rocprof. Confirm matrix-shaped execution, launch reduction/weight reuse where expected, remaining bottlenecks, and scratch memory cost.
4. Check subsequent decode and existing native-MTP behavior for regressions. Native-MTP batching itself is not implied by target prefill acceptance.
5. Update live documentation only for proven behavior. Keep measured fixture-bound evidence separate from product/admission claims.

**Exit:** correct user-facing AR prefill and continuation, regression checks complete, measured throughput/memory tradeoffs reported without a predetermined speedup promise.

## Invariants and verification matrix

The central correctness property is **chunk-boundary equivalence**: the same token sequence processed individually, in one chunk, or partitioned into chunks must produce equivalent required outputs and equivalent continuation behavior within established tolerances.

Cover:

- Chunk size one, multiple rows, partial final chunks, and several chunk partitions.
- Nonzero starting positions and pre-existing cache/recurrent state.
- GDN convolution-history wrap and recurrent-state advancement.
- QSA pooling-block boundaries, per-query selections, and absence of future-token leakage.
- PLE lookup/history and depthwise-history boundaries, including staged resource cleanup.
- MoE top-10 indices/weights, intermediate basis, canonical combine order, and dirty output/scratch reuse followed by required initialization.
- Final-row-only outputs versus all-row/intermediate capture consumers.
- Reset, repeated requests, and supported continuation/lifecycle paths.
- Target and native-MTP preservation with separate state.
- Existing shared-family tests and unsupported-format/ownership rejection before launch.

Keep regression tests for plausible semantic failures, not source-text wiring or incidental enum ordering. Run required Rust build/test checks; use numerical fixture/channel checks and actual serving execution as behavioral evidence. Do not relax tolerances or reinterpret an existing failure as acceptance.

For performance, use byte-identical committed prompts, exact model and binary hashes, fixed settings, warmup policy, and at least three fresh-process runs per comparison. Report prefill latency/throughput, launch/kernel evidence, memory use, and subsequent decode impact. Attribute improvement only to measured changes.

## Non-goals and boundaries

- No new model family, quantization quality campaign, or broader QT53 admission.
- No automatic Qwen4 Redline/PM4 replay or EP admission. Preserve existing refusals; physical multi-GPU proof requires appropriate hardware.
- No concurrent-request scheduling redesign or continuous-batching claim.
- No mandatory native-MTP batching in the initial AR deliverable; preserve it and reuse common primitives without silently claiming it was accelerated.
- No wholesale rewrite of other model families to demonstrate generality.
- No promise that GDN or QSA token dependencies disappear.
- No performance claim based only on Step unification, compilation, or a kernel name containing “batched.”

## Cross-session handoff

At each implementation session:

1. Read this plan, current shared-MoE design context, and the relevant validation route.
2. Locate the latest separate progress/decision record; verify its commit and artifacts against the checkout.
3. Record completed phase/acceptance scope, changed symbols, exact commands/results, evidence paths, blockers, and next actionable step. Keep unproven work explicitly planned or blocked.
4. Preserve baseline artifacts and unrelated user changes. Do not restart the old PR merge or repeat expensive baseline collection when the retained evidence still matches the exact inputs.
5. If a design decision changes, add a dated amendment linking this document and explaining the evidence/tradeoff. Do not silently redefine the objective to mean scalar Step execution.

Completion means both reusable centralized execution ownership and real, correct token-batched AR prefill—not merely either half.
