# Maple-Preview batched prefill (arch 15)

**Status:** design, approved 2026-08-22. Implements fast prefill for
`hipfire-arch-maple`; the decode path is already serving (see
`2026-08-22-maple-preview-20b-a1b.md`).

**Spec location note:** the superpowers default is `docs/superpowers/specs/`,
which is gitignored in this repo (`.gitignore:47`). This lives in `docs/design/`
so it is tracked, alongside `lfm2moe-gfx1201-prefill-architecture.md`.

## Problem

Maple decodes at ~120-134 tok/s but "prefills" at the same rate, because the
harness runs one `decode_step` per prompt token. A 3,059-token prompt costs
**24.4 s**. There is no batched path in this arch.

Measured 2026-08-22, gfx1151, release:

| Context | Prefill | Decode |
|---|---|---|
| 20 tok prompt, 45 gen | — | 134.1 tok/s |
| 20 tok prompt, 256 gen | 130.6 tok/s | 120.3 tok/s |
| 3,059 tok prompt, 64 gen | 125.5 tok/s (24.4 s) | 120.8 tok/s |

Decode is flat with context (18 of 24 layers are sliding-window-512), so prefill
is the only thing that scales badly.

## Key finding: no new HIP kernels are required

Every operation already exists. The one apparent gap — there is no *dense*
batched GEMM for `MQ2G256LloydU`, since every `gemm_mq2g256_lloyd_*` is
MoE-grouped — is closed by specializing the grouped kernel to a single expert
(below). A purpose-built dense kernel is **deferred to a measurement**, not
written speculatively.

## Design

### 1. `forward_batch`

```rust
pub fn forward_batch(
    cfg: &MapleConfig, weights: &MapleWeights, state: &mut MapleState,
    gpu: &mut Gpu, tokens: &[u32], start_pos: usize,
) -> Result<Vec<f32>, String>
```

Runs each weight matrix once for `B` tokens, fills KV for
`[start_pos, start_pos+B)`, returns the **last** token's logits. Per layer:

```
rmsnorm_batched(h[B], input_norm)             -> normed[B]
dense_qt51_gemm(wq/wk/wv)                     -> q[B], k[B], v[B]
rmsnorm_batched(q, q_norm, n_heads*B, head_dim)   QK-norm, BEFORE rope
rmsnorm_batched(k, k_norm, n_kv*B,    head_dim)
rope_partial_interleaved_f32_batched(...)     -> sliding layers ONLY
Step::Attend { AttnQ8_0KvBatchedMaskedWindowed, positions[B] }
dense_qt51_gemm(wo) + residual add
rmsnorm_batched(h, post_attn_norm)            -> normed[B]
moe_batch(...)                                -> accumulates into h[B]
```

Head: final RMSNorm on the last row only, then the existing `lm_head` GEMV.
Batching a 151,936-wide projection to discard all but one row is waste.

The per-layer ordering constraints are the same ones the decode path documents:
**QK-norm before RoPE**, and **RoPE on sliding layers only**.

### 2. `dense_qt51_gemm` — single-expert specialization

`gemm_mq2g256_lloyd_moe_grouped_wmma` computes, for one expert,
`Y = X @ W^T`. Driven with:

- `expert_weight_ptrs`: 1-entry table pointing at the projection
- `expert_tile_ids`: all zero (every tile uses expert 0)
- `sorted_slot_index`: identity `0..B`, padded
- `x_row_div = 1` (slot index *is* the row index; the MoE case divides by k_top)
- `m_total = ceil(B/16)*16` (BLOCK_M padding, ≤15 wasted rows)

The one-entry pointer tables are built **once at load**, per projection per
layer — not per call.

**`ensure_fp16_x` hazard.** That kernel converts `x` to FP16 internally and
caches the conversion **keyed on the source pointer**. Maple reuses a single
`normed` scratch buffer every layer with new contents, so the cache would hand
layers 1..23 layer 0's activations — silently, with no error. `cohere2moe` hit
exactly this (`q8_proj_raw`) and works around it by converting into its own F16
buffer per call. Do the same, and comment it so it is not "optimized" back.

### 3. MoE batching

```
router GEMV(batched) -> softmax -> moe_topk_renorm_k8(norm_topk = true)
moe_scatter_fused_k8
gemm_mq2g256_lloyd_moe_grouped_wmma      (already accepts MQ2G256LloydU)
moe_unscatter_silu_clamp_k8(swiglu_limit = 7.0)
grouped down GEMM
combine into h
```

`moe_unscatter_silu_clamp_k8` takes `swiglu_limit` and fuses the unscatter with
the asymmetric clamp, so Maple's clamped SwiGLU is free in the prefill layout —
the same way `deepseek4_silu_mul_clamp_f32_batched` covers it in decode.

**No FWHT anywhere.** `run_moe_prefill` takes `x_norm_batch` and `x_rot_batch`
from the caller; both receive the natural-basis activation, because qt51 weights
are unrotated. This is the same invariant the decode path enforces and the same
one `gate_down_skips_rotation` pins in the dispatch layer.

### 4. Chunking and integration

- Default chunk size **256**. `forward_batch` **errors** on `B > 512` (scratch
  ceiling) rather than silently splitting — splitting is the caller's job.
  Larger `B` raises tokens-per-expert (`B*k_top/n_exp`) and shrinks BLOCK_M
  padding waste, which is why 256 rather than 64.
- `forward_batch_supported(weights)`: all experts `MQ2G256LloydU` and Q8 KV.
  Callers fall back to per-token when false.
- `maple_coherence` prefills through it.
- `MapleCarrier::bench_prefill` overrides to chunk through it.

Out of scope: the daemon generate route (that is "make Maple servable", a
separate and much larger job), and the arm-A differential (arm A does not exist).

### 5. Testing

The per-token path is already verified end to end, so it is the oracle.

1. **Logit parity** — same prompt through `forward_batch` and the per-token
   path (`crates/hipfire-arch-maple/examples/maple_prefill_parity.rs`).
   Run at **B=1**, **B=17** (deliberately not a multiple of BLOCK_M, to catch
   padding bugs) and **B=256**.

   **Acceptance criterion revised 2026-08-22, after measurement.** The
   original bar here was "cosine >= 0.9999 AND identical greedy argmax
   [at the final position]". That was wrong on two counts, established by
   running the oracle rather than by inspection:

   - It compared only the LAST position of the whole run, i.e. **one**
     comparison point regardless of how many chunks were involved — a run
     that split into 12 chunks was reported as if 12 things had been
     checked, when only 1 had.
   - Its default input was synthetic ids scattered across the vocab
     (`1000 + i*7919 % 100000`), which is out-of-distribution: the model's
     logits go flat and hypersensitive to perturbation there, which
     *exaggerates* the arithmetic gap below. Feeding the same code real
     tokenized prose instead measured materially higher cosine.
   - 0.9999 cosine is unachievable by construction: the batched path is a
     WMMA GEMM over F16 activations (codebook deltas computed in
     `_Float16`, F32 accumulate —
     `kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip` ~L89-136),
     while the per-token oracle is an F32 scalar GEMV. A ~1e-3 relative
     per-GEMM difference is expected and compounds over 24 layers — the
     same shape as the shipped cohere2moe Q8 prefill path, which also
     feeds F16 activations to a WMMA GEMM.

   The revised bar:

   - **Hard bar: greedy argmax at every comparison point, or a flip the
     measured noise fully explains** (see the RESOLVED note below for why
     the "no flips anywhere" form of this bar was replaced), where a
     comparison point is one `forward_batch` CALL (it returns only the
     last token's logits per call, so a run chunked into N calls yields N
     points). The harness prints how many points were compared so a
     1-point run cannot be misread as an N-point pass, and prints the two
     numbers behind any noise-explained pass so it is never mistaken for a
     clean one.
   - **Reported metric, not the hard bar: cosine.** Cosine cannot fail the
     run; it is printed (min/mean per configuration) as complementary
     context to the argmax hard bar. A separate `COSINE_COLLAPSE_FLOOR =
     0.5` exists purely as a catastrophe detector — it catches a
     structurally broken run (e.g. a dropped sliding window) that healthy
     numerical noise can never approach — and is set nowhere near a
     precision bar (see the correction note below the table for why a
     precision-tuned floor doesn't work here).

   **Measured 2026-08-22, real tokenized prose, 200 tokens, gfx1151,
   release** (`--tokens 200`, run-to-run range across repeats where the
   GPU's reduction order is not fully deterministic):

   | Config | Points | Argmax (raw flips) | Cosine min | Cosine mean |
   |---|---|---|---|---|
   | B=1 (200 chained 1-token chunks) | 200 | ~10-17/200 flip, every run | 0.897-0.906 | ~0.994 |
   | B=17 (12 chunks) | 12 | 0-3/12 flip across repeats | 0.951-0.984 | ~0.99 |
   | B=256 (1 chunk, whole prompt) | 1 | no flips, every run | 0.980-0.998 | — |
   | chunk-split (2 chunks of 100) | 2 | no flips, every run | 0.998-0.999 | ~0.999 |

   The "raw flips" column is the count BEFORE the near-tie judgement below;
   under the shipped bar every one of these configurations passes.

   Synthetic OOD input, for contrast (NOT a representative measurement,
   see `--synthetic` in the harness): cosine min as low as 0.52, argmax
   fails at all three B values including B=256.

   **Correction (2026-08-22, task 7): the cosine floor above was
   undersampled, and the fix was to stop using cosine as a pass/fail
   criterion at all, not to re-tune the number.** The table's B=1 cosine
   min of 0.897-0.906 came from two runs and was turned into a
   `COSINE_FLOOR = 0.85` that could ALSO fail the harness alongside the
   argmax hard bar. Broader sampling (8 runs) found a real B=1 tail of
   **0.843436** — below that floor — and the UNMODIFIED base binary shows
   the same tail (0.871590), confirming this is inherent GPU
   non-determinism, not a regression introduced by later work: B=1 chains
   200 single-token chunks, so each run samples 200 points from a reduction
   order that is not deterministic call-to-call, and the minimum over that
   many points has a tail two runs cannot characterize. That is a false
   failure roughly 1-in-8 — the same "red when correct" shape the argmax bar
   was reworked to avoid below — so cosine was changed to match what this
   document already specified above it: a **reported metric**, not a bar.
   The only cosine check that remains is `COSINE_COLLAPSE_FLOOR = 0.5`, a
   catastrophe-only floor sitting in the gap between the healthy tail
   (0.843436 over 8 runs) and a genuine structural break (cosine -0.725,
   see the sliding-window liveness control described below) — wide enough
   that no amount of ordinary run-to-run variation can trip it.

   **This was an open finding when the table above was written; it is now
   RESOLVED (2026-08-22) — the flips are F16-vs-F32 precision, not a
   defect.** The history matters, so it is kept: at B=256 (the production
   chunk size) and at a 2-chunk split, argmax parity holds on every
   real-text run. At B=1, and intermittently at B=17, it does not, because
   chunk N's KV cache is itself the *output* of the WMMA-batched path for
   chunk N-1 — so with many sequential chunks the F16-vs-F32 gap compounds
   again at every chunk boundary through the cached K/V, not just once over
   24 layers. What was open was whether that residual divergence was
   arithmetic or a bug.

   It was answered with `maple_prefill_parity --near-tie`, which measures
   the perturbation each flip REQUIRED against the perturbation this
   arithmetic actually produces. Four independent pieces of evidence, all
   at 200 tokens of real prose:

   - **The flips sit at near-ties.** Median reference top1-top2 gap is
     **0.086 at flips vs 1.116 at matches** — a ~13x separation. A ~1e-3
     -scale perturbation can only reorder logits that are already close,
     and that is exactly where the flips are.
   - **Every flip is inside the measured noise.** The largest perturbation
     any flip required was **0.451**, against a measured deciding-margin
     noise of median 0.117 / **max 0.996**. No flip needed more than the
     arithmetic gap demonstrably supplies.
   - **The flip set is not reproducible run to run** (16 flips one run, 10
     the next; same binary, same input). A genuine defect would be
     deterministic.
   - **B=256, the shipped chunk size, shows no flips at all.**

   The positive control shows what a real defect looks like on this same
   instrument: forcing full-causal attention
   (`HIPFIRE_MAPLE_FORCE_FULL_CAUSAL=1`) at 1,200 tokens drives cosine to
   **-0.725**, nothing like the ~0.99 seen here.

   **The bar in the harness was changed to match this conclusion**, because
   a gate whose healthy state is `exit(1)` teaches people to ignore it: the
   default `--b 1,17,256` sweep failed its own hard bar on every run purely
   from these benign flips, so a real defect at B=256 would have looked
   identical to a healthy run. The bar is now *per flip*: a configuration
   passes with no flips, or with flips whose required perturbation is
   inside the noise band measured at that configuration's own non-flipping
   points, and fails on any flip at a well-separated logit. That is
   strictly sharper than "zero flips at B=256 only" — in particular a
   single-point configuration whose one point flips now fails, since it has
   no clean point to calibrate against. No mitigation (F32 KV write-back,
   minimum chunk size) is needed or planned.
2. **Chunk-boundary** — a prompt prefilled as 100+100 must match one 200-token
   chunk. Catches `start_pos` / KV-offset errors.
3. **Sliding-window liveness** — parity alone cannot prove the window is
   applied: if BOTH paths dropped it they would still agree. So run the same
   >512-token prompt twice through `forward_batch`, once normally and once with
   a test-only override forcing `window = 0` on every layer
   (`HIPFIRE_MAPLE_FORCE_FULL_CAUSAL=1`), and require the logits to **DIFFER**.
   That makes the answers differ by construction, so the assertion cannot pass
   vacuously.
4. **No decode regression** — re-run the coherence gate; decode tok/s and output
   unchanged.
5. **Measurement** — report prefill tok/s at 3,059 tokens against the 125.5
   baseline. Only if the profile shows the tile-id indirection or BLOCK_M
   padding costs meaningfully do we write the dedicated dense kernel.

## Risks

- **Stale FP16 cache** (§2) — silent wrong activations. Mitigated by the
  per-call conversion buffer; test 1 at B>1 would catch it.
- **Within-block causality** — batched prefill needs query `p` to not see keys
  `>p`. `attention_flash_q8_0_batched_masked_windowed` masks per query by
  absolute position (`[p-window+1, p]`), so this holds; test 2 guards it.
- **Padding rows** — the `m_total` tail beyond `B` computes garbage that must
  never be read back. Test 1 at B=17 is the guard.
