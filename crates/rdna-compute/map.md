# rdna-compute — map

> **Status:** `production` (vocabulary owned by [`docs/GLOSSARY.md`](../../docs/GLOSSARY.md)).
> **Layer:** Kernel select — sits on `hip-bridge`/`hsa-bridge` and below `saddle-core`/`hipfire-runtime`; see the layering table in [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md).

## Purpose

Kernel compilation, caching, and dispatch for RDNA GPUs. Owns `Gpu`/`GpuTensor`, the `DType`/family dispatch (`dispatch.rs`, `gemm.rs`, `gemv.rs`, `attention.rs`, `moe.rs`, `norm.rs`), `arch_caps`/`feature_flags` capability predicates, `compiler.rs`/`KernelCompiler` (chip → family → base `kernels/src/*.hip` selection and JIT cache), and the product-integrated Redline replay controller (`replay.rs`). The design rationale is the `//!` docs in [`src/lib.rs`](src/lib.rs).

## Gotchas

- Chip → family → base kernel selection is priority-ordered (`gfx1201.hip` overrides `gfx12.hip` overrides `base.hip`); a mis-named file silently falls back.
- Precompiled `kernels/compiled/<arch>/` blobs are preferred only when their hash matches; mismatched hash triggers `hipcc` JIT or a warned unvalidated blob.
- `Gpu` methods pick WMMA → specialized → baseline via `arch_caps` predicates — adding a new arch clause that a newer predicate already subsumes is redundant.
- `a3b_gqa8_oracle` requires `lab` and carries the original full-allocation decode dump comparison and VerifyAttn byte-identity gates. Native FP8 H16/KV2 uses the gfx1201 group-8 decode/verify exports under the existing admission flags; H24/group-6, explicit Q8 optimized routes, and the Q8 qualification table remain unchanged. Native FP8 PM-prefill and PM-Qprep default on only for their qualified exact-gfx1201 H16/KV2/D256 route; explicit flag `0` opts out and all existing shape, codec, recording/capture and other admission guards still apply. The already-blocked VMM mode is excluded from this target; existing production VMM refusal/fallback semantics are unchanged.
- `FP8_KV_QUALIFIED` has two exact-gfx1201 H16/KV2/D256 rows for F16-Q attention with native FP8 K/V and its PM Qprep route; it is not empty and is not qualification of FP8-quantized Q. Native FP8 KV selection and FA2 eligibility are independent and stackable. Existing codec/shape/eager-state guards and explicit flag overrides remain authoritative.
- A3B H16/KV2/D256 has its own Q-resident v2 Q8 pair (`attention_fp8_e4m3_fa2_gqa_qresident_v2_q8_a3b_gfx1201` direct, `..._range_a3b_gfx1201` range) sharing the dense KT48 body and 49,936 B dynamic LDS: one Raw Q allocation as E4M3 code plane + F32 scale plane with checked total-row offsets, exact-gfx1201 admission, and no Q8-image, historical 64-KiB/F16-record or `FP8_KV_QUALIFIED` F16-Q changes. PP16000 scope only: three original opt-ins (the PM down-A4 producer, the PM GDN chunk prep, `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI=1`) plus the C8 PM twin (now unconditional on exact gfx1201), not the stock default; qwen3.6-35b-a3b-d53.mq4rs 17478.9 / ornith-1.5-35b-a3b-d53.mq4rs 17383.8 strict medians, 6 fresh processes total, 18 measured samples (3 per process after 2 warmups), source dc307/tree fb056, CLI efa37fc4 / daemon 8a43d456, kit 860802. Tiny 522 (512 + tail 10) takes whole-batch native Direct before the F32 fallback; Range supports 1+ rows but GEN Range unobserved. The 522/128 GEN smoke was a coherent bounded prefix with mid-Python/length-runaway true, not a complete answer. Quality plus all three final CPU gates REQUIRED; no stock/default/release/tag/SITE claim.
- The gfx1201 native FP8 GQA8 tile scores heads in four two-head batches to reduce live Q registers; P.V retains two four-head halves. Group-6 scoring, per-head arithmetic order, ABI, launch geometry, LDS layout and admission guards are unchanged.
- Walked HIP plans bind live current registry recipes, not historical whole-source text; historical captures and pins remain provenance. Existing kind/compiler-flag/profile/symbol/shape/admission/pack contracts and strict embedded-image SHA checks remain enforced, with no kernel arithmetic, public API or feature-default changes.
- `examples/pm_loop.rs` (`--features lab`, dev-deps only) is the one-command emit → M7 → correctness → ABBA loop for PM production generators (`iu4_gemm`, `mq6_i8_gemm`, `qwen4_moe_sym` and the producers `iu4_c8_producer_pm`, `moe_rotate_i4_pm`, `gdn_chunk_prep_pm`; `--rip` is iu4_gemm/mq6_i8_gemm only, the 2:4-sparse `--mma` iu4_gemm only). Shared builders live in `examples/pm_loop/common.rs` (PRNG, `fn:`/`a3b:` shape presets, shipped-object selection, emulator launcher) and the producers share one generic `Workload` (`examples/pm_loop/prod.rs`: bitwise identity vs the shipped object plus an exact CPU reference where the repo has one). It times with `examples/support/abba.rs`, shared with `pm_decode_twins`: `pm_loop` calls a result `separable` by an exact sign test (p <= 1e-3) over the per-round paired ratios of each ABBA round, which cancels the between-round clock drift that defeats block-disjoint comparisons, and auto-extends by doubling ABBA rounds up to `--max-rounds`/`--time-budget-s`; the block-disjoint and twins' full-range rules stay in the record as `separable_block_disjoint` and `separable_full_range`, and `material` additionally needs `|block_ratio - 1| >= --min-effect` (default 0.5%). Run it only through `fabric-run.sh --class timing`.

## Crate map

<!-- crate-map:generated:begin -->

Modules, exported symbols and dependencies are generated, not committed: see
`map.generated.md` in this directory (`cargo xtask regen` writes it).

<!-- crate-map:generated:end -->
