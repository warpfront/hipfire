<!-- SPDX-License-Identifier: Apache-2.0 -->

# Batch-invariant inference — design path (not yet implemented)

- **Date:** 2026-09-13
- **Status:** design only. No config flag ships: a capability flag without
  invariant kernels behind it would advertise a guarantee the engine does
  not make (the same contract violation class the 2026-09-13 audit flagged
  in the serve surface). This document is the implementation plan.
- **Context:** suite cells B8/C5 (WARN) and D1 (WARN) observe the
  limitation: the same prompt + seed produces different tokens across
  forward *shapes* — cold full-prefill vs cached-resume, solo vs
  concurrent, different chunk splits. Same-shape replay is byte-identical.

## Mechanism

Logits are floating-point reductions (GEMV over the model dim, attention
over the KV length, RMSNorm over the hidden dim). FP addition is not
associative, so the result depends on reduction order, and reduction order
depends on the launch geometry the dispatch layer picks per shape: grid
size, block size, tiling, split-K factor. A one-ulp logit difference flips
a sampled pick (any temperature) or a greedy near-tie; autoregressive
generation then amplifies it into divergent text. Reference:
[Thinking Machines, "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and [vLLM's shipped batch-invariance mode](https://docs.vllm.ai/en/latest/features/batch_invariance/)
— the industry template.

## Sources of shape dependence in this engine

1. **Prefill chunk tiling** — `prefill_chunk` (default 1024) splits the
   prompt; the GEMM shape of the last chunk varies with
   `prompt_len mod chunk`, and cold vs cached hits prefill different
   lengths entirely (the B8/C5 case).
2. **Attention reduction over KV length** — attention splits the KV walk;
   the split count varies with seq_len and the number of co-batched rows.
3. **Mixed-batch GEMV/GEMM** — a step's forward covers a ragged batch
   (verify rows + prefill rows + decode rows, `SlotBatch`); kernel
   wave/tiling choices vary with total rows.
4. **Sampling** — fixed by the RNG-parking fix (inactive slots no longer
   draw); the remaining sampling path is deterministic given identical
   logits.

## Design

Opt-in mode, config `serve.batch_invariant` (default **false**), surfaced
in `/health` capabilities only when the kernels below actually ship.

Scope: only the ops that feed logits need invariance — the final hidden
norm, the lm_head GEMM/GEMV, and attention (draft-quality parity for MTP
verify is likewise only needed where verify reads logits). Everything
upstream (KV writes, embeddings, MLPs mid-stack) already runs per-row or
is absorbed by the same three.

Per-op strategy (vLLM's approach, adapted):

| Op | Invariant variant | Cost |
|---|---|---|
| lm_head GEMM/GEMV | Fixed split-K: partition K into a constant number of stripes (independent of M and K padding), each stripe reduces in a fixed intra-stripe order, stripes combine in a fixed tree. Tile M to a fixed quantum (pad rows, mask out). | pad waste on the last M quantum; no split-K adaptivity |
| attention | Fixed KV split (flash-decoding style) with a constant split count per seq; combine partials with a fixed-order online-softmax merge. No atomics. | small: extra combine pass vs single-kernel fused path |
| RMSNorm / final norm | Reduce in a fixed order over the hidden dim with the block dim pinned (hidden dim is constant per model — already shape-invariant in most launches; pin explicitly). | none expected |

Non-negotiables: no atomics-based accumulation anywhere in the three ops;
launch geometry must be a pure function of (model, fixed quantum), never
of live batch composition; every invariant kernel gets a bit-equality
harness — same row computed inside 1, 2, 4, 8-row batches must be
bit-identical (the per-kernel invariance test vLLM's follow-ups use).

## Verification plan

- Unit: per-kernel bit-equality harness across batch sizes (host-runnable
  on any GPU box).
- Oracle: extend `test_serve_prefix_cache` with a
  `--batch-invariant` cell — run one prompt cold, warm, concurrent, and
  chunked; assert all four outputs byte-identical.
- Suite: B8/C5 upgrade from WARN to hard equality checks when the mode is
  on; D1's baseline comparison likewise.
- Perf: record the mode's tok/s delta in `docs/perf-checkpoints/` (the
  mode is opt-in precisely because this delta is expected to be negative).

## Effort

The three kernels + harness + oracle ≈ a focused week. The config surface,
oracle scaffolding, and suite upgrades can land first (as this document +
test scaffolding), but the flag itself must not ship until the kernels do.
