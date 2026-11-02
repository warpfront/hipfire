# A3 closure audit — unclosed routes

Base: `f3fa11ec40`. Selector extraction commits: `0f5d5b9d6f`, `9007323697`, branch `perf/jit-a3`, worktree `/home/kaden/ClaudeCode/warpfront/wt-jit-a3`.

**This is an incomplete reachability audit, not a closed ready-plan oracle. None of the five requested model/arch routes is certified closed by this artifact. Do not preload a partial inventory as if it were a complete model plan.** No `route_entries` implementation has been published. The source-derived transitive reachability and header-to-selector input mappings below remain unresolved; this is not a missing compiler or GPU prerequisite.

## Resolved scope decisions

Main clarified that G0 manifests are lower bounds: every observed required symbol must appear in a closed plan, but a closed plan may contain additional reachable requests. FN-gfx1151 observations stop before failed native prefill and cannot exclude downstream branches.

Explicit DFlash/DDTree/DSpark routes must refuse planning: the frozen input contains no separately admitted drafter header. Native MTP belongs to the primary model header and is in scope. The contract's `crate::flags::FeatureFlags` path is a typo; use `crate::feature_flags::FeatureFlags`, without an alias.

## Lower-bound inventories

All files are under `/home/kaden/qcal/release-0.4.2/jit-peacemaker/`. These inventories are unchanged and individually recipe/image pinned; they are not substituted for source reachability.

| Route | Manifest | Modules | Required symbols | Closure status |
|---|---|---:|---:|---|
| 27B gfx1201 | `27b-xts-gfx1201.routes.json` | 45 | 51 | unclosed |
| 27B gfx1100 | `27b-xts-gfx1100.routes.json` | 25 | 30 | unclosed |
| 27B gfx1151 | `27b-xts-gfx1151.routes.json` | 27 | 31 | unclosed |
| FN gfx1201 | `flash-next-gfx1201.routes.json` | 42 | 106 | unclosed |
| FN gfx1151 | `flash-next-gfx1151.routes.json` | 17 | 28 | unclosed; pre-failure lower bound |

The gfx1100 FN inventory is source-predicted and its G0 hardware fixture is inadmissible. It is not promoted by this audit.

Four live `tensor_ops` required exports are absent from the legacy registry declarations but present in measured ELF exports: `indexed_attention_attention_q8_batched`, `indexed_attention_attention_q8_batched_hg4`, `indexed_attention_cache_append_q8_batched`, `indexed_attention_decode_prologue_q8`. Do not drop them or infer that declared symbols alone are an export oracle.

## Forward-entry frontier

Paths in the following tables are relative to the worktree. This table identifies source roots; it is not a claim that every downstream edge has been walked.

| Surface | Entry/root | Selector/input frontier |
|---|---|---|
| dense prefill | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:2562` `forward_prefill_batch`; `:2765` `forward_prefill_batch_with_pbs_opts`; `:2808` inner | header projection dtypes and dimensions, PBS/row commitment, chunk ceiling/tails, KV axes |
| dense decode | `crates/hipfire-arch-qwen35/src/qwen35/forward.rs:1490` `forward_scratch`; `:1857` hidden path; `:2110` layer loop | entry-name-keyed cache, graph/recording state, dtype fusions, GDN/attention and LM-head dispatch |
| dense dispatch adapters | `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:4269`, `:4322`, `:4429`, `:4475`, `:4519` | plain/residual/fused gate-up/QKV/QKVZA keys; exact family lowering still must be followed |
| FN prefill | `crates/hipfire-arch-qwen4/src/gpu_forward.rs:2615` `forward_chunk`; `:2841` inner; `:2868` scoped | model scope, row count, HC state dtype and fusion level, expert placement, projection keys and QSA |
| FN decode/head | `crates/hipfire-arch-qwen4/src/gpu_forward.rs:2533` `forward_token`; `:2591` token/argmax path | ordinary versus retained execution, BF16/Q8 decode shadows, staged PLE and head projection |
| FN load checks/transforms | `crates/hipfire-arch-qwen4/src/gpu_forward.rs:955`, `:963`, `:2125`, `:2217`, `:2262` | expert/trunk symmetry checks, requested IU4 masks, BF16-to-Q8 decode transforms |
| FN native-MTP | `crates/hipfire-arch-qwen4/src/mtp_gpu.rs:1610` construction; `:1178` reset; `:1204` snapshot; `:1296` restore; `:1304` retained restore | draft forward, MTP head, snapshot/copy/rollback and KV append transitive calls have not been closed |

## Shared selectors extracted (behavior-preserving prerequisites)

The private launch wrappers continue to resolve the same process-frozen switches. Pure selection functions take those resolved switch values explicitly. No compiler, loader or module-cache implementation was changed.

| Helper | File:line | Inputs and choice |
|---|---|---|
| `mq6_x4_pm_gfx1151_selected` | `crates/rdna-compute/src/gemm.rs:221` | arch, native switch, M/N; gfx1151 native admission at M >= 2560 and N >= 2048 |
| `hc_down_tile_selected` | `crates/rdna-compute/src/gemm.rs:229` | arch, switch, M/K/rows, recording/capture; gfx1151 160x64 HC tile and gfx1201 split-K HC tile |
| `qwen4_moe_sym_native_selected` | `crates/rdna-compute/src/gemm.rs:243` | arch, native switch; gfx1201 remains native with gfx1151 kill switch disabled |
| `mq6_x4_tile_selected` | `crates/rdna-compute/src/gemm.rs:1012` | arch, resolved explicit/auto tile, Plain/BF16/Regions/HC-write kind, M/K/N; existing U3 measured table and BV12 divisibility rule |
| `lds256_tile_source_selected` | `crates/rdna-compute/src/gemm.rs:1030` | gfx1151 identity and entry; selected common versus Qwen4 LDS source/module, including 64x64 tile special case |
| `qsa_select_pm_shape_fits` | `crates/rdna-compute/src/tensor_ops.rs:3636` | arch admission, pooled dtype, rows, index heads/dim, blocks, query stride and output capacity; checked raw-offset bounds |
| `qsa_select_pm_selected` | `crates/rdna-compute/src/tensor_ops.rs:3650` | fits and frozen score/select switches; Hipcc/ScoreOnly/SelectOnly/Both |
| `qsa_gathered_pm_shape_fits` | `crates/rdna-compute/src/tensor_ops.rs:4820` | full capacity, KV heads and query heads; raw-offset and u24 bounds |

Pure selector parity tests cover native MQ6 threshold neighbors, HC threshold neighbors and capture modes, explicit/auto U3 tile admission, native MoE kill-switch behavior, LDS source identity, all four QSA switch arms, QSA rows 511/512/513, maximum pooled-block neighbors, checked-offset overflow, and gathered full-capacity bounds. Lazy switch reads retain their original short-circuit behavior on gfx1201 MoE and uncovered QSA requests.

## Specific unresolved source/image callsite families

These are unresolved *audit mappings*, not assertions that the source cannot be inspected. A ready plan must not be returned until their transitive callers and all shape branches are resolved.

| Callsite/source decision | Required unresolved input mapping |
|---|---|
| `crates/rdna-compute/src/gemm.rs:38314`, selection `:38376`–`:38396` | M/N, output dtype, overwrite, batch tile, gfx12 capability, U3 table selection and selected image exports; BF16/regions/HC-write cannot be blanket native |
| `crates/rdna-compute/src/gemm.rs:29597` `lds256_source` | `HIPFIRE_LDS_EPI_DIRECT` synthesized source and shared public-entry conflict handling; existing `OnceLock` source semantics must be preserved |
| `crates/rdna-compute/src/gemm.rs:638`, `:647` | private B1/B1S image functions, control bundle and frozen file overrides; these have not been extracted/shared with a planner |
| `crates/rdna-compute/src/gemm.rs:45904`, image ensures `:45950`, `:45952` | header/expert format, symmetry-check outcome, host mapping, row repeat, native/HIP source identity and export arrays |
| `crates/rdna-compute/src/attention.rs:6839` onward | gfx11 FA2 r3/twin/generic module, preconversion, source prefix/flags and all query/context shapes |
| `crates/rdna-compute/src/attention.rs:5948` family (base anchor) | gfx1201 resident FP8 Q8/A4 epilogue admission and fallback branch reachability for H24/KV4/D256 |
| `crates/rdna-compute/src/tensor_ops.rs:383`, `:533`, `:604`, `:799`, `:806`, `:812` | GDN norm/recurrence/rollback names and Q8-inline versus ordinary WMMA; exact callers and header state dimensions |
| `crates/rdna-compute/src/tensor_ops.rs:3712` `indexed_attention_select_rows8` | F32/BF16 pooled dtype, mirrored output, selector stride/capacity, capture versus ordinary paths and PM score/select image exports |
| `crates/rdna-compute/src/tensor_ops.rs:4836` `qsa_gathered_wmma_launch` | KV-format-specific producer (F32/Q8/FP8), native versus HIP attention, and all gathered capacity shapes |
| dense `forward_scratch_layers` and prefill dispatch adapters above | every selected GEMV/GEMM, normalization/GDN, KV write and attention helper must be expanded; sampler/head/capture helpers must also be included |
| FN `forward_chunk_scoped` and MTP methods above | sealed pipeline operations, all projections, PLE, expert router/scatter/combine, native-MTP head and rollback require transitive expansion |

No threshold/tail/max-context closure assertion has been made. The G0 default inventories remain lower bounds; no new complete inventory exists to perform the required subset comparison against.

## Exercised CPU evidence

All commands used `ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1`, `CARGO_TARGET_DIR=/home/kaden/qcal/release-0.4.2/target-jit-a3`, release/locked, `-j 16`, `-p rdna-compute --features deltanet`, and the absolute worktree manifest path.

- Original G0 route-data gate: 7 passed, 0 failed, 0.03 s (`artifact://119929`).
- First GEMM selector tests: 5 passed, 0 failed (`artifact://119957`).
- Expanded library gate initially found an ambiguous integer type in the new QSA test (`artifact://119969`); corrected the test row type to `usize`.
- Corrected complete library gate: 477 passed, 0 failed, 17 ignored, 0.13 s (`artifact://119971`).

These are CPU policy/library results, not proof of route closure, successful GPU execution, or A3 acceptance.
