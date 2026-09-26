# Sealed granular MoE execution

## Decision and scope

Continue PR #755 from `d9b05c454097317708367ed99fa9b1638b3772fe`. Retain its manifest-derived expert ownership, rank-local live resources, immutable source identity, route proofs, transactional publication, and existing model/generation owners. Recover the granular executable-program direction from PR #527 without porting its branch or adding another model owner.

The goal is actual shared execution, not a descriptive list around an unchanged monolithic function. A sealed public call remains the only authority to execute; its private lowering describes computation. A common runtime schedule owns collective ordering; architecture adapters supply operands and execute family-specific arithmetic, not a second schedule.

Non-goals: new model or topology admission, mixed TP×EP enablement, new kernels or quantization, a replacement generation loop, performance promotion, or product PM4 admission. Existing retained routes and fallbacks must keep their behavior. The post-review slot-order correction below intentionally replaces Qwen compact EP's rank-grouped routed reduction.

## Invariants

- All immutable source, shape, alias, local-slot, physical-device, scratch-capacity, and route checks remain enforced.
- A complete program is validated before its first side effect. A bad later rank or operation must not leave an earlier rank mutated.
- Preserve launch order and specialized fused kernels, including shared-expert treatment, CPU routing, mixed formats, Paro, expanded versus residual-fused down, and deferred combine where already supported.
- Preserve root-authoritative route bytes, destination-owned transport events, reverse source-reuse dependencies, and reduction leases. Qwen compact EP must fold routed contributions in global top-k slot order, matching the single-device combine; rank-partial routes for other families retain their existing fixed-rank fold.
- Shared contributions enter the correct residual or partial exactly once. No duplicate reduction or combine.
- No new per-token allocation for immutable plan construction. Do not cache borrowed mutable resource identity across reuse or reload.
- Failures retain the existing fail-stop/rollback behavior. No fallback introduced to hide an invalid program.

## Implementation sequence

### A. Granular computation

1. Define internal typed operations for existing decode and prefill arithmetic: route/shared computation, indexed or grouped projections, activation/basis transform, scatter/unscatter, and combine as applicable.
2. Derive the legal ordered program from the sealed call and current dispatch decisions. Keep genuinely fused kernel regions as explicit fused operations rather than multiplying launches.
3. Execute the derived operations through a single checked path. Remove the superseded whole-call arithmetic sequencers instead of leaving parallel production implementations.
4. Preserve all current sealed Qwen and Cohere consumers. The outer sealed-call API may remain, but must no longer hide an unchanged monolithic arithmetic executor.

### B. Collective program

1. Introduce a typed executable mesh schedule for preflight, zeroing, root route/compute, route distribution, rank-local contribution, named transport/combine, and residual completion.
2. Bind schedule identity to the admitted mesh/sealed contract. Resolve the collective for its actual program position rather than inferring a whole program's axis from its first reduction.
3. Migrate root-routed decode and all existing Qwen EP grouped-prefill/batch orchestration sites to the shared schedule. Preserve existing other-family rank-partial routes.
4. Gather Qwen's rank-local expert outputs in the global slot layout, run the ordinary single-device slot-order combine once on root, and byte-copy that finished partial to every rank. Architecture adapters expose resources and the existing combine; the shared executor owns stage order, rank traversal, and stop-on-error semantics.

Tracks A and B own disjoint dispatch and runtime/architecture files. They preserve existing call/parameter boundaries while integrating; no temporary compatibility aliases or duplicate final execution paths.

## Verification

Pinned baseline CLI, daemon, and `qwen35_sealed_moe_oracle` binaries are preserved locally under `.codeinsight+research/sealed-granular-20260913/baseline-bin/` before editing.

- CPU behavioral tests: legal program selection, exactly-once combine/reduction, invalid geometry/route/owner refusal before effects, error stops, decode and prefill schedule reuse. Do not test only enum names or source text.
- Build and test affected Rust packages, plus workspace tests as required by the repository. Format modified Rust files only; run narrow clippy targets.
- Baseline/candidate observation oracle: byte-identical model/prompt/settings, decode and chunked prefill, compare raw logits/router/KV/recurrent observations. Baseline warm-prefix discrepancies remain visible and must not be relabeled as candidate success.
- Serve battery/chain and failure/reuse checks on the exercised MoE route, with decoded text inspected.
- Dispatch retention: claim-scoped `redline_daemon_harness.py` capture, AQL contracts, and multi-position HIP/PM4 parity where the retained route is supported. This is not product PM4 admission.
- Physical EP: current-head EP2/EP4 route/residual and lifecycle oracles on distinct GPUs when accessible. Single-device/emulated evidence cannot close this row.

No performance win is claimed by this refactor. Qwen compact EP now restores the single-route floating-point association by gathering per-slot outputs before the ordinary combine; physical cross-mesh equivalence still requires distinct-GPU validation.

## Completion boundary

The production consumers execute the new computation and collective programs; obsolete hand-sequencing is removed; source/lifetime checks are not weakened; available numerical/state and serving checks run against the pinned baseline; any inaccessible fixture or distinct-GPU proof is explicitly recorded as a validation blocker rather than an acceptance claim.

## Implemented boundaries

- `hipfire-dispatch/src/pipeline/moe_program.rs` owns fixed-capacity `DecodeProgram` and `PrefillProgram` lowering and execution. Typed stages execute the existing arithmetic helpers; the public `Step::Moe(SealedMoeCall)` remains the checked entry, not the arithmetic implementation.
- `hipfire-runtime::ep::RootRoutedEpSchedule` resolves the actual named `moe` collective row for its layer. `execute_root_routed_ep` owns the all-rank preflight barrier, zeroing, root work, route transfer, contribution gathering, root slot-order combine, byte broadcast, and residual completion.
- Qwen's root-routed decode, `prefill_lane`, `forward_tick`, and `forward_prefill_batch_ep` supply adapters to that executor. `build_moe_prefill_params` is shared between preflight, execution, contribution-layout selection, and root combine.
- Existing noncompact rank-partial/HC3/HC4 execution is retained. `PrefillSkipAllReduce` explicitly preserves the existing prefill diagnostic policy; it is not a fallback.
- Schedule-admission regressions cover named-row and geometry refusal. A dispatch regression prevents CPU top-K from selecting shared-router fusion and skipping a required shared activation rotation. The slot-order follow-up rejects self-combining EP down paths before launch and reuses the ordinary decode/prefill combine kernels.

## Validation record — 2026-09-13

This is local change evidence, **not G5 acceptance, physical EP proof, a performance claim, or product PM4 admission**.

Fixture: `qwen3.6-35b-a3b.mq4p`, 19,757,996,288 bytes, SHA-256
`8eb3be6912aa9db2dcc89c3233b05ccdfe81a84a8d6ef43202f2098d7f2fc78f`.
Oracle prompt: `benchmarks/prompts/humaneval_3_below_zero.txt`, SHA-256
`430706f794ffabb60ec5818ca7c9fdbd281b97ef9d3ac6d6ba140a7a2498f5a5`.
Hardware: one gfx1151 GPU, HIP 7.2. Baseline source is the pinned commit above.

| Check | Observed result |
|---|---|
| Release product build and separate `deltanet,moe-oracle` example build | Passed |
| Affected library tests | Dispatch: 271 passed / 1 ignored; runtime: 690 / 7; Qwen35: 195 / 17; retained Cohere library: 12 passed |
| New schedule integration tests | 7 passed |
| Chunk-1 oracle, 8 decode positions | 10,258 observations across 16 fields byte-identical to baseline; fresh/warm comparison and four reset/unload/reload cycles passed |
| Chunk-16 oracle | Both binaries hit the same existing warm-prefix/fresh-state logit mismatch at position 117, element 0 (`0x40ebecec` versus `0x40ebe863`); `diff -qr` of their complete raw-array directories passed |
| Retained capture and shadow | Same stable 862 launches, 22 kernels, fingerprint `229e1cdc8cdbb0aa`; AQL contracts and 15-position PM4/HIP shadow parity passed, including GDN frame |
| Native serving battery and chain | Five turns each, byte-identical transcripts and request identities versus baseline, one terminal per turn, no empty responses/attractors/stream errors; each retains the baseline reasoning response capped at 128 tokens |
| Post-expert-mutation fault | Three fresh-process rounds passed: one rollback error, clean state attestation, retry matching fresh generation, unload |
| Formatting and narrow clippy | Modified-file rustfmt check passed; affected library and changed integration-test clippy targets completed with warnings |

The default-parallel workspace library run stopped at the unchanged
`rdna-compute::dispatch::tests::upload_raw_copy_failure_hip_frees_owner`
exact free-VRAM assertion (free memory increased by 2 MiB). A GPU-locked,
single-test-thread rerun passed all 245 rdna-compute tests; the remaining
redline-dispatch, redline-rocr, and saddle-core suites passed too. The original
workspace invocation is not described as green.

Oracle readback requires `HIPFIRE_GRAPH=0`: the default graph-enabled baseline
failed with HIP error 906 during an observation readback. Retained capture was
checked independently using the uninstrumented production daemon. Do not build
that daemon in the same Cargo invocation as the `moe-oracle` feature: its
observation hooks require an active collector. The existing expert-fault test
also needed its two stale helper calls updated with `expect_dflash=false` to
compile and exercise the Single AR route.

Measured binary MD5 identities:

| Binary | Baseline | Candidate |
|---|---|---|
| Native CLI | `89f039738824dc5bcb21e9ab2406adda` | `4124cfb5d3c398b1c8cd3529f7010a1d` |
| Production daemon | `cf627a1375949a326c51f9a24f6702f3` | `53953ae2cb6113a187f40e5d615b6bae` |
| Observation oracle | `f15a3719ee3d980623ce5ad406f93813` | `6a91b0e3975ed3455360f4f5f34868af` |

The feature-gated candidate fault daemon is
`31c05ef79d614767c345e705e3154c4d`. Raw reports, arrays, transcript comparisons,
source-file hashes, and binary copies are retained locally in
`.codeinsight+research/sealed-granular-20260913/`; these host-local artifacts
cannot promote a tracker milestone.

Distinct-GPU EP2/EP4 execution remains hardware-blocked: this run had one
physical GPU and no configured remote GPU host. Other model/quant/architecture
paths received source review and applicable CPU tests, not new GPU
qualification. No admission records or milestone receipts are changed.
