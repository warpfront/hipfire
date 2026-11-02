# Design A — certified railgun DFlash window, gfx1201

Source-path convention: crate references below omit the leading `crates/`; `kernels/` references are checkout-relative and evidence references are relative to `R/dflash-depth/`.

## Decision and evidence boundary

Design only; no implementation, GPU execution, or correctness/performance validation was performed for this document. Source references are relative to `/home/kaden/ClaudeCode/warpfront/hipfire-beta`, inspected at HEAD `a0e571597f22a171840aa7a20f88ed9ca3074e6d`. Symbols, rather than line numbers alone, are the integration anchors if beta advances.

Build a **directly authored railgun window program**, not a new Redline capture. The admitted full-width path has one gfx12 IB containing draft embeddings, the five draft layers, draft head/selector, device proposal handoff, the existing 16-row verify, hidden staging commit, and verify head/argmax. Keep the DeltaNet SAVE and accepted-prefix repair around that IB initially. They are state transactions, not another target forward. Keep all existing inference kernels and their order; independently certify the small transport/host-computation substitutions listed below.

The target is approximately **2.5 ms/window saved**, not a measured result. `WINDOW-SPLIT.md:7-29,34-53,73-99` measures 40.84 ms/window, 34.14 ms verify including head, 296 eager draft launches, 822 verify-graph launches, and 24.2 ms AR on a retained single-IB tape. Those measurements were at beta `9cb35862c4`, not this inspected HEAD. Do not assume the kernel count or every removable gap is unchanged.

### A material constraint: the measured draft is DFlash2, not independent argmax

`raw/s0r1-prose-k16.err:87-97` identifies an all-sliding five-layer DFlash2 draft, W=2048, runtime B=16, IU4 verify off, and **lookup armed (copy+tail)**. On beta:

* `hipfire-runtime/src/dflash.rs:4999-5064`, `propose_candidates_device`, still downloads selector projection and top-K results and calls a **CPU sequential scorer**. Its name does not mean proposals are GPU-resident.
* `hipfire-runtime/src/dflash.rs:4795-4899`, `propose_inner`, selects each candidate using the previous selected token and a sequential f32 codebook dot product. Replacing it with the lm-head argmax is wrong.
* `hipfire-arch-qwen35/src/speculative.rs:9410-9454` performs a host lookup splice **after** proposal production and can shrink effective verify B. `dflash_spec.rs:1314-1396` chooses the first eligible prefix; `hipfire-runtime/src/spec_lookup.rs:547-560,588-626` uses measured prices and acceptance history.

Consequently **one fixed-16 IB cannot transparently implement every current default lookup window just by changing the dispatcher**. Do not disable lookup, force B=16 after a splice, or call two IBs one IB. This design has two explicit execution shapes:

1. **Whole16:** one IB, only when a pre-submit proof establishes that no host proposal transformation can change this window. Includes the DFlash2 device selector described below. An armed lookup can qualify if `NoTailAction` proves every legal tail decision declines.
2. **HostSplice:** the same logical window plan has two railgun segments, draft/selector then the existing host splice then verify at its actual B. Other B values keep the incumbent verify route until individually certified. This preserves default behavior; it is not evidence for the one-IB or fixed-16 claim.

`NoTailAction` is a pure, conservative predicate over an immutable controller snapshot: enumerate k=1..14 and offered=1..(15-k), construct exactly the `LookupBudget` used by `TailSource::splice`, call the existing `LookupController::plan`, then `clamp_tail_take`. Admit Whole16 only if none produces a positive take. Do not approximate prices or infer this proof from the previous window's `lookup=0`. Copy/probe choices made before draft remain with their existing owner. Diagnostics, sampled decoding, online tuning, PLD, n-gram overrides, repeat penalty and tree paths retain their existing semantics; they are not quietly admitted to Whole16. Report Whole16 coverage separately on prose and code. If it is low, the measured default workload cannot realize the advertised whole-window gain without another design for device lookup.

A faster backend can change a timing-trained lookup controller's later decisions. Byte identity is therefore a **same-input, same-window, same-controller-state** contract. The stream-identity gate freezes/replays controller decisions and B; an unfrozen production run additionally reports changed lookup decisions rather than mislabeling them numerical errors.

## 1. What exists, and what is missing

| Existing source | Actual capability / limitation |
|---|---|
| `rdna-compute/src/replay/railgun_shadow.rs:1-24,49-57,164-211` | M1 observes Redline launch records; developer-only M2 can execute railgun's lowering of admitted AR tapes. This is not a stand-alone DFlash program API. |
| `railgun/src/kernel.rs:350-398,444-487` | `WordRole` currently contains only `GdnFrame`; pointer interiors and grids are constants. There is no arbitrary context/row/head word family or device-written kernarg interface. |
| `railgun/src/kernel.rs:489-696,715-788` | `author` types launch facts and builds kernarg images. Missing/Unknown facts become non-PM4 nodes. Facts cover allocation-wide effects, not a proof of every index bound. |
| `rdna-compute/src/replay/railgun_shadow.rs:593-780` | `lower` can allocate railgun kernargs, but executable M2 requires one segment, byte equality with **Redline-owned** kernargs and its patching. Its emission transport can be reused; its recorder/planner must not be used for this feature. |
| `railgun/src/plan.rs:221-249,327-378` | gfx1201 pacing is post-dispatch NOP(64), and the plan derives hazard boundaries. Unknown nodes split programs; system barriers are required around foreign work. |
| `railgun/src/lib.rs:440-597`; `railgun/src/copy.rs:5-17,22-59,103-143` | Bounded copy/copy-batch authoring exists separately. Batch entries use slot indices, not pointer chasing. They are not automatically interleaved executable nodes in the M2 kernel tape. |
| `hipfire-arch-qwen35/src/dflash_spec.rs:3770-3813` | Old verify PM4 admission explicitly refuses non-Q8 KV. It does not establish that FP8 verify ISA is unsafe or uncertifiable. Do not relax this legacy gate as the implementation. |
| `rdna-compute/src/attention.rs:9936-9953` | `launch_verify_gqa` refuses **Redline recording**; setting that flag would silently select a different attention body. Direct railgun authoring must preserve ordinary HIP route decisions. |

Minimum extension: an owned prepared railgun executable independent of Redline recordings; bounded declared scalar/grid/interior patches; typed copy/fill emission; direct collection from existing launch builders without changing their numerical route; complete receipts for the actual draft+verify objects; and the exact selector/transport bridge. No general device-side dispatch graph, firmware change, new Redline capture, or general indirect-pointer analyzer is necessary.

## 2. Frozen implementation interfaces

These are **new interfaces to implement**, not existing APIs:

* `WindowShape`: `{position, logical_ctx_rows, projected_rows, full_cached_rows, compact_offset, ring_head, swa_w, full_w, max_position, max_tiles, verify_rows, draft_rows, generation}`. All logical arithmetic uses checked u64/usize; conversion to kernel i32/u32 is checked. `draft_rows=verify_rows=16` for Whole16.
* `WindowBindingKey`: model/draft identities, exact loaded object+toolchain digests, device, numerical-route flags, allocation identities/extents/generations, KV mapping generation, scratch capacities, context mode, GDN quant/EF/requant mode, extraction layers. Any change invalidates the prepared executable **before mutation**.
* `WindowInputs`: seed, mask ID, 16 absolute Q positions, the live K-position suffix, 16 physical verify positions, and any existing copy descriptors. Own the pinned upload storage until the transfer completion fence. Preserve `thlog.abs_positions()`, including compaction offsets; never synthesize all historical positions as `0..L`.
* `WindowHostDelta`: proposed updates to draft projection/upload watermarks and hidden ring head/written. Planning does not publish them. Commit once, after successful GPU completion, before accepted-prefix scatter.
* `WindowPlan`: ordered typed kernel/copy/fill descriptors, bounded patch sites, read/write allocation closure, shape-family key, host delta, completion surfaces, and either `Whole16` or `HostSplice`.
* `PreparedWindow::bind(&WindowShape, &WindowBindings, &WindowInputs) -> Result<BoundWindow, Refusal>` validates every patch without submitting. `submit(BoundWindow) -> Submission` consumes the bound generation. `Submission::complete() -> CompleteWindow` proves queue quiescence; failure distinguishes proven-quiescent from unknown/in-flight. Kernarg buffers, loaded kernels, IB and bound resources outlive completion. No in-flight patching.
* `WindowVerifyReceipt { same_as_hip_graph: bool, tape_populated: bool, generation: u64 }`. `same_as_hip_graph` is true only for a certified exact route, not merely a successful submit.

Retain one bounded cache of **32 prepared topology skeletons** per owner; evict only retired entries. Shape-dependent byte values are patched in-place in retired storage, not new allocations per window. Cache misses construct a plan before execution; no recapture forward and no speculative target advance to discover arguments. Prewarm the finite ordinary shapes off the decode critical path. Rare unsupported paths use the existing HIP implementation with an explicit refusal, not partially executed mixed programs.

### Authoring, not capture

Reuse `launch_maybe_blob` / `launch_kernel_blob` argument construction, but introduce an explicit plan sink independent of `gpu.replay.is_recording()` and `gpu.graphs.capture_mode`. For the target body, route choices must equal the current HIP-graph body; for draft, equal the current eager body. Freeze this distinction as a numerical-route policy, not a fake global recording flag.

The sink collects the existing kernel's exact code-object identity, grid/block/LDS, explicit blob, certified facts and declared changing fields. It does not issue launches. Materialize lazy scratch/JIT outside plan mode; route selection, scratch-generation invalidation and frame reservation cannot be accidentally executed twice during planning. Extract host bookkeeping into `WindowHostDelta` rather than running a dry forward and hoping its mutations are harmless. Route all D2D copies and memsets in the planned region through typed operations; an unhandled HIP API makes the complete plan refuse.

Use the current railgun copy primitive for D2D transport. For zero-initialization, add one direct-pointer `railgun_fill_u32(dst, value, words)` primitive with bounds and A1 write facts (the relevant lm-head zeros are word-aligned). Do not replace an ADD projection with SET to eliminate the fill: that can change signed-zero/numerical behavior. Copy/fill replacement is a separately certified byte transformation, not a claim of literally unchanged copy kernel symbols.

## 3. Shapes: finite topology, explicit patch expressions

Do **not** compile a program for every context length or pad an accept+1 projection to 16. Row count can change the GEMM route and numerics (`REPORT.md:48-52`).

Use these declared patch kinds only:

* `Scalar32 {field, checked expression, domain}` for loaded scalar args;
* `Interior64 {resource, checked byte_offset expression, required_extent}`;
* `GridAxis {axis, checked expression, domain}`; rebuild all live implicit grid words from the resulting geometry;
* existing `GdnFrame` reservation, with the same increments/order as HIP.

Expressions are a small pure AST: constant, named input, checked add/multiply, saturating subtraction where source uses it, min/max, division/modulo by a nonzero **load-time constant**, and ceil-div. No untyped arbitrary byte patches. Bind proof covers all scalar constraints, region endpoints, aliases and geometry. `KernelProgram::kernarg_image` remains the only full-image encoder.

Shape family = ordered kernel identities and projection segment row counts, not the changing addresses or position:

1. **Steady non-wrap:** delta = accept_previous+1 in 1..16. Preserve the exact delta-row route, including delta=1 GEMV versus multi-row GEMM and the selected draft dtype. All five noise/FFN bodies remain B=16; draft head remains 15 rows; verify remains 16.
2. **SWA wrap:** fc/hidden norm split at the same `ring_segments` boundary; retain each segment's actual row count. Context K/V splits at **both** source projection-ring and destination cache-ring boundaries. `hipfire-runtime/src/dflash.rs:3477-3496,3632-3735` is the authoritative formula. For the measured all-sliding W=2048 case there is one common ring modulus, so the new-row fill has at most two fragments. For legacy split/full drafts use the existing two-modulus loop, not an assumed two-fragment rule.
3. **Context concat/attention:** preserve `span_start`, `span`, `pos_base`, ring ordering and full-versus-sliding selection (`dflash.rs:3324-3343,3780-3878`). Copy byte counts and attention sequence length are bounded dynamic fields. Never bucket a true sequence length upward unless the same kernel already masks it and a separate byte oracle proves the superset transformation.
4. **Cold seed, reset, eviction, cache rebuild:** delta can exceed 16. Keep the current initialization/backfill route outside Whole16, invalidate old plans and resume after its watermarks are valid. This is not a steady-window speed claim.
5. **Capacity/sub-batch change:** target attention uses `max_tiles` and a partial-buffer capacity-dependent sub-batch (`attention.rs:9982-10095`). Preserve that exact decomposition. KV growth or a changed sub-batch is a binding/topology change, not merely a position patch.

Certification per family: exact blob/geometry reconstruction against an ordinary HIP plan; boundary cases delta=1/2/15/16, wrap-1/wrap/wrap+1 and context tile 127/128/129; all live hidden grid fields; out-of-domain refusal before work; no stale address after allocator reuse. Kernel receipts certify object identity/effects; the family manifest certifies these launch-domain obligations. Neither substitutes for the other.

**Rejected alternatives:** device-written kernargs require visibility to scalar/command-processor reads plus a producer/consumer certificate; changing kernel signatures to indirect shape structs would cease to be the same kernels. Neither is the minimum extension. A giant padded family changes projections/attention and has no identity certificate. Preauthoring 16 row variants alone is insufficient because of ring splits and selector/lookup dependencies.

## 4. Input transport and the proposal dependency

### Before the IB

Pack host-known seed/mask/positions into persistent pinned storage; issue async H2D copies on the owner HIP stream once, then transfer ownership to railgun with **one completion fence and entry acquire**. Existing APIs: `hip-bridge/src/ffi.rs:1925-1949` (`host_buffer`), `3070-3094` (`memcpy_htod_async`). `upload_prefill_batch_inputs` currently performs two synchronous copies (`qwen35/prefill.rs:2721-2741`); its bytes are the oracle.

This is the minimum exact substitute for in-program H2D. PM4 dispatch packets cannot invoke a HIP memcpy. Do not claim CPU-writable pageable memory is coherent GPU input. A mapped-host-load kernel is not the first implementation: although `MappedHostBuffer` exists (`ffi.rs:4238-4282`), its HIP-event retirement is not automatically proof that a separate PM4 queue has stopped reading it. Such an optimization needs its own host/system visibility and retirement certificate.

Enqueue `DeltaNetSnapshot::save_from_async_on` on that same stream before the handoff, without the synchronous `save_from` round trip. Its move before draft is safe only because draft reads target weights/accepted hidden and mutates no target KV or DN state (`speculative.rs:7525-7532`). This is the one deliberate reordering of a pure state copy; all inference arithmetic stays ordered. Do not also claim the SAVE's 0.38 ms kernel time disappears.

### Inside the IB

Preserve the embedding route selected on the inspected beta. `draft_dflash_forward_rank` already attempts the batched Q8 noise helper (`speculative.rs:7550-7568`); the measurement's 16 eager embedding launches are not automatically still available savings. Retain scalar token patches where the selected baseline uses scalar embedding kernels. Batched embedding is a distinct byte-certified substitution, not required for railgun.

For a selector-free draft, copy the existing 15 argmax words into `pbs.tokens[1..16]`; token[0] is the seed. For the **measured selector draft**, retain selector hidden projection and raw `topk_values_batched_f32` in their current order, and add `dflash_selector_greedy_exact`:

* Inputs: direct device pointers to predecessor/successor codebooks, projection[15,rank], top_ids[15,k], top_vals[15,k], seed; outputs scores[15,k] and block[16]. Codebooks are uploaded once, preserving the exact `SelectorCodebook::get_f32` values (`dflash.rs:346-362`). k is 1..16; rank is the loaded metadata value, not a hardcoded 256.
* One wave32, first k lanes each own one candidate. Iterate rows 0..14 sequentially. Each lane accumulates dimensions d=0..rank-1 in ascending order: `p*proj[d]`, then `*s`, then `dot+term`, each a separately rounded f32 operation. Explicit no-contraction operations and IEEE denormal behavior must match the Rust CPU scorer. Do not use a parallel reduction, reassociate, use FMA, or change FP16 conversion behavior.
* Share scores in LDS; lane 0 scans j=0..k-1 with the exact strict `>` comparison and first-candidate tie rule. Publish the selected ID to all lanes as the next predecessor. The seed is block[0]. Validate ID/rank/extent before dereference; a status word reports malformed input and prevents publication. Finite model inputs are the production domain; special-value behavior is exercised by the oracle rather than assumed.
* Greedy confidence is not needed to choose tokens. Preserve diagnostics by returning scores/top-K/projection at the final completion when requested and computing the existing CPU `exp` confidence afterward; do not replace CPU `exp` with GPU `exp` and claim byte identity. No temperature>0 support in this bridge.
* Store proposals in a dedicated immutable-for-window block buffer, then use `railgun_copy` into verify tokens. The draft and verify currently reuse `verify_scratch.argmax`/logits (`speculative.rs:7677-7678,7824-7835`); otherwise the verify would destroy the proposals needed by acceptance and traces.

The existing GPU kernels remain unchanged; the CPU scorer has moved and therefore needs its **own** numerical byte certificate. If this bridge fails that certificate, keep HostSplice; it is not acceptable to claim the full-IB design landed with selector argmax substituted.

## 5. FP8-KV facts and railgun-cert closure

Certify the exact loaded objects and recipe, not a symbol allowlist. `railgun-cert/src/lib.rs:195-311,351-440` serializes A1/A3/A4 facts and metadata differential; `railgun/src/kernel.rs:18-20,508-674` refuses Unknown or missing facts. A tier-D receipt is diagnostic lift plus identity and explicit obligations, not a formal proof of mathematical correctness (`kernel.rs:286-298`).

For the target GQA6 FP8 tile, source `kernels/src/attention_verify_gqa.gfx1201.hip:409-417`, expected explicit ABI is:

| Offset | Argument | Required effect |
|---|---|---|
| 0 | q | read |
| 8 | k_cache | read, including packed FP8 scale bytes |
| 16 | v_cache | read, including scales |
| 24 | partials | write |
| 32 | positions | read |
| 40,44,48,52 | n_rows,n_kv_heads,max_tiles,batch_offset | scalar, not pointers |
| 56 | scale_attn | constant f32 bits |

These offsets are a source-ABI expectation; assert against the actual code-object metadata and loader segment. Explicit fields end at byte 60; **do not hardcode the padded/hidden segment size**. Buffer descriptor construction/raw buffer loads at kernel lines 107-125 must preserve provenance to the K/V arguments. Native FP8 conversion is arithmetic, not pointer access.

The reduce (`:445-497`) reads partials at +0, writes out at +8, reads positions at +16, and has n_heads/max_tiles/batch_offset at +24/+28/+32. Source expected explicit end is 36. Preserve tile/reduction order and the actual host sub-batching. GQA8 sibling (`:429-438`) shares the tile ABI but is **not** the 27B shape. Generate its receipt when certifying the module; do not widen runtime admission to A3B merely because its object was analyzed.

The manifest must also cover the actual FP8 KV writer, RoPE/QK prep, baseline fallback tile+reduce if selected, all draft attention/conv/selector kernels, copy/fill helpers, norms/rotations/GEMMs, GDN pre-capture, recurrence, hidden commit and heads. `rdna-compute/src/kernel_registry.rs:1327-1331` is registration, not proof of receipt coverage. Record a complete node sequence/hash and fail a requested Whole16 plan if any node is `HipDirect` or `Graph`.

A1 is allocation-wide. Independently prove: positions in mapped KV capacity, max_tiles matching partial stride, partial rows fitting capacity, no overlapping outputs/inputs forbidden by restrict, one writer per KV row, and exact extraction/ring extents. VMM bindings need the allocation registry's mapped extents/generation; `hipMemGetAddressRange` alone is not a universal VMM extent oracle.

Do not weaken the analyzer to admit pointer tables. `peacemaker-ir/src/passes/kernargs.rs:33-42,2663-2689` deliberately refuses memory-derived pointer bases. Bulk snapshot tables and multi-layer replay tables therefore stay **outside** the initial IB. `railgun_copy_batch` is the reusable exception: its table contains indices into direct pointer kernargs (`railgun/src/copy.rs:9-17`), with prepare-time bounds and immutable table lifetime.

Use the existing gfx1201 cache table/pacing. Under missing G4 receipts use `RowPolicy::UnreceiptedSystem`, not invented weak barriers. Require silicon receipts for every weaker hazard row actually used before advertising the low-gap result. Preserve scalar-cache and WAR-prewriter boundaries; source `railgun/src/plan.rs:351-378,584-655`. A kernel's FP8 data type is not a reason to remove barriers.

## 6. State transaction and failure semantics

Let p be the seed position, B=16, a accepted drafts, r=a+1. The seed was emitted previously but **has not yet been forwarded at p**. End state is after r forwarded tokens; the newly emitted bonus is pending as next seed. It is not after a+2 forwards (`speculative.rs:5494-5524`).

| Transition | Owner and invariant before | Invariant after |
|---|---|---|
| Admit/prepare | Speculator owns live DN, target KV, draft scratch/thlog, hidden ring, snapshot, GDN tape. All old submissions retired. | No GPU/host semantic mutation. Entire plan, shapes, mapping and receipts validated. |
| SAVE + input staging | Live DN is S(p); snapshot may be old. | After HIP-stream fence snapshot holds S(p), scales, conv and EF. Snapshot immutable until repair completes. Input generation immutable. |
| Draft portion | Target state unchanged; accepted hidden valid through p; projection watermark c<=L. | Draft hidden/projections/KV caches contain exactly current eager results. Planned watermark becomes L but is not yet externally published. Selector block is retained separately. |
| Verify portion | Snapshot S(p) preserved; target KV prefix <p valid. | Live DN reflects B-row verify; KV slots [p,p+B) and tape[0,B) populated, final hidden/logits correct. Suffix beyond acceptance is speculative and must not become logical history. |
| Hidden staging commit in IB | Ring head h is a declared dynamic word; five staging planes written. | Same B rows copied to ring; planned head=(h+B)%capacity and written+=B. No host sync between producer and commit: hazard edge replaces it. `commit_staging_to_ring` :3562-3607 is split into GPU emit and host delta, not called afterward a second time. |
| Complete/readback | Queue owns all bound resources. | Completion releases them; publish host delta once, download retained proposal and verify picks. Existing CPU greedy accept and max_accept clamp run unchanged. |
| Accepted-prefix scatter | a fixed by existing decision. | Append only r hidden rows to draft context and `thlog.append_committed`. Do not append the bonus hidden. `dflash_commit_decision_borrowed` :5521-5549 remains the owner. |
| DN repair | Snapshot S(p), tape B rows; live state speculative. | Existing snapshot-source replay advances exactly r rows into live state, or the existing restore/replay fallback does. All four families valid at p+r; target KV suffix is logically ignored, not unnecessarily erased. |
| Next window | Host p advances by r, bonus is seed; lookup observes only completed window. | No stale staging, pointer table, frame or watermark reused. |

`DeltaNetSnapshot::save_from` currently synchronizes its bulk-copy path (`speculative.rs:1860-1941`). Keep its public synchronous contract; this feature uses the existing async variant under an explicit handoff.

`GdnTape::replay_gdn_from_snapshot` (`:2499-2612`) already has two fingerprinted table slots and immutable source state. Its two HIP launches stay outside the IB after accept; there is little launch-gap opportunity in two launches, and this avoids new indirect-table certification. Preserve the current frame allocator order/increments, including EF-off stochastic mode and check-mode replay. Do not infer frames from token position.

The current full-accept optimization is backend-labeled (`DflashVerifyReceipt.verify_on_hip`, :5373-5389; `keep_verified`, :5556-5589). Replace that semantic test at all callers with the explicit certified-state-parity receipt. Only a railgun route proven identical to the current verify can keep the B-row live state when r=B and EF is on. Ordinary HIP remains true; legacy retained PM4 remains false unless separately certified. No full-accept shortcut for EF-off or a changed GDN math route. This is an exported-symbol change: owner must run LSP references and migrate lane/eager callers together.

Refusal before submission: ordinary HIP route, no changes. Execution error: do not restore or free while PM4 may still run. Unknown quiescence quarantines owner/resources and returns error. Proven quiescence permits the existing safe-retry policy only after restoring recurrent state and the complete window host preimage, including frames, ring cursors, watermarks and lookup observation status. Ring data overwritten by a failed commit needs restoring or a demonstrated complete overwrite before retry. Simpler initial implementation: propagate execution errors after quiescence; automatic retries are not part of this feature. Check mode may keep its HIP twin result only after complete restoration.

## 7. Composer slices and acceptance gates

All slices are implementation assignments for the owner to delegate. Composers do not run shared project-wide checks. Interfaces above are frozen; only the integration owner changes shared API wiring.

| Slice | Bounded files / responsibility | Acceptance the owner runs |
|---|---|---|
| A1 typed shape binding | `railgun/src/kernel.rs`, `plan.rs`, standalone owned lowerer extracted from `railgun_shadow.rs:593-780`; retain legacy AR behavior | Host tests reject overflow, bad domains, alias/capacity mistakes, stale generations and in-flight patching. Reconstructed bytes match HIP blobs including hidden grid fields. One segment/no untyped holes. |
| A2 receipt closure | `railgun-cert`, corpus tooling/manifest and only analyzer rules demonstrated missing by actual objects | Every actual Whole16 node has exact-artifact facts; FP8 tile/reduce ABI table matches metadata; negative wrong digest/missing fact/hidden arg cases refuse. No blanket pointer/FP8 allowlist. |
| A3 selector/transport | `dflash.rs` proposal helpers, a new exact greedy selector kernel and wrappers; typed copy/fill; pinned owner inputs | GPU scorer scores and chosen IDs byte-identical to `propose_candidates_from_rows` for real codebooks, 15 sequential rows, ties, signed zeros/subnormals and adversarial predecessors. Copy/fill/input bytes exact; canaries unchanged. |
| A4 draft emitter | `draft_dflash_forward_rank`, `draft_dflash_block_rank`, `draft_forward_based`, `draft_dtod_at` | Existing eager and emitted HIP twin have identical per-layer hidden, ctx projections, K/V and proposal buffers for every delta and ring seam. No unhandled upload/memset/copy; no planning watermark mutation. |
| A5 verify emitter | `verify_dflash_block_inner_body`, pre-uploaded prefill entry, hidden commit and verify head | HIP graph versus emitted HIP twin: complete verify outputs, tape, live DN, KV, hidden ring and frame parity; same attention symbols, no recording-induced fallback. |
| A6 transaction integration | `spec_step_dflash`, speculator owner/admission, receipt and commit callers | Whole16 proof, HostSplice continuation and unsupported paths select correctly. a=0..15, budget clamp, EOS, restart/reset, ring wrap and pointer reuse maintain exact state. Failure before/after submit cannot double-commit or free in-flight resources. |
| A7 independent certification reviewer | Extend `rdna-compute/src/railgun_check.rs`, existing DFlash state examples, report manifests | Ladder below; reviewer veto on any byte mismatch, uncertified kernel, unproved queue handoff or changed proposal policy. |

A1/A2/A3 are independent. A4/A5 consume the frozen emitter interface and can be composed independently. A6 owns integration after those land; A7 owns the final gate, not the kernel author.

### Byte-identity ladder

1. **Host/static:** full artifact inventory, route/shape manifest, checked bounds and exact kernarg images. Include negative controls that alter one shape word, pointer generation, object digest and grid hidden field.
2. **GPU primitive:** selector, copy/fill, uploads and FP8 attention tile/reduce separately against the incumbent. GQA6/16 rows and all actual partial sub-batches; fixed and boundary contexts; reject rather than ignore unavailable facts.
3. **Phase:** eager draft versus emitted HIP; HIP graph verify versus emitted HIP. Restore identical preimages, not sequential runs on already advanced state.
4. **Transport:** railgun versus its exact HIP-node twin. Adapt the all-live-State comparison of `railgun_check.rs:1-39,183-268`; snapshot/restore fences are mandatory. Preserve input bytes and frame reservation on both arms. Compare immutable weight digests too. `sample(N)` alone is insufficient because its surfaces come from the facts being tested.
5. **Independent third arm:** current whole HIP-graph window versus Whole16, with controller decisions and window boundaries fixed. Compare proposals, logits, argmax, accepted/bonus, GDN S/scales/conv/EF, KV live+speculative written bytes, hidden staging/ring, ctx caches, cursors and frames. Scratch is initialized identically so complete allocation comparisons do not mistake uninitialized padding for computation.
6. **Adversarial state:** all a=0..15; EF on/off; two consecutive full accepts; budget-bound short commit; W and KV tile seams; reset/eviction/capacity growth; equal-size allocations reused at different generations; malformed metadata; failure/quarantine. Include byte flip and missing boundary negative controls. Byte flip must be caught; a removed barrier is not guaranteed to race on every trial, so require a dedicated hazard stress probe rather than pretending one clean trial proves safety.
7. **End-to-end:** same prose/code inputs as REPORT, fixed-decision identity arm, and separate live-controller semantics/performance arm. No AR-token-identity claim: REPORT already demonstrates row-count near-tie divergence. Design B is OFF throughout Design A certification.

## 8. Performance gate, architecture matrix, abandonment

Two estimates must not be added together:

* Literal rate substitution: 822*(3.3-2.2) us ≈ **0.90 ms** for verify and 296*(2.9-2.2) us ≈ **0.21 ms** for draft. This alone is only ~1.1 ms, not 2.5 ms.
* The unprofiled AR HIP-graph/PM4 difference is **1.7 ms on 761 kernels**; scaling that empirically to verify gives ~1.8 ms, and the report estimates ~0.8–1.0 ms for draft capture/transport. Thus **~2.5 ms is a reasonable target [INFERENCE]**, contingent on barrier cost, selector exactness, existing embedding improvements and Whole16 coverage. It is not established by multiplying average gaps. Do not double-count eliminated syncs in both rows.

Measure baseline HIP, verify-only railgun, draft-only railgun, Whole16, and the actual mixed default route separately, with identical numerical policy. Use unprofiled fresh-process A/B orders, warmup excluded, matching clocks/card ownership and end-to-end request wall. Record IB count, dispatch count, bytes uploaded, boundary count/rungs, Whole16 eligibility rate, per-window latency and total tokens. In-program work must still have GPU durations consistent with unchanged math. Profiler idle attribution is supplementary; `WINDOW-SPLIT.md:115-117` documents its perturbation and PM4 invisibility.

Admission is exact **gfx1201**, single GPU, dense 27B, supported resident/safely mapped FP8 KV, fixed-16 chain window and stable scratch. gfx1100, gfx1151, gfx1200, CDNA, TP, CB/VMM lane-table execution, tree verification and sampling retain their present route. Certifying a module on another architecture is not permission to enable the backend there. Do not change shipping AR transport or add any Redline capture.

Abandon the **single-IB performance claim** if complete receipts cannot be obtained without changing inference kernels, selector byte parity fails, Whole16 coverage is insufficient for a default-workload win, or unprofiled whole-window improvement is <=0.5 ms / within process spread after strict correctness gates. Retain only independently measured certified phase wins; report them as segmented phase acceleration, not completion of the one-IB target. Never weaken fences, change lookup, or swap numerical kernels to force the ~2.5 ms number. Reviewer approval is required before default-on admission.

## Amendment — 2026-10-25 UTC: loaded integer indices are not loaded pointer bases

**Ruling: choose (a), a conditional indexed-argument effect. Do not introduce a positions-as-scalars writer.** The reported `base-memory-derived` refusal of the 32 `kv_cache_write_fp8_e4m3_batched` nodes is accepted as the current result; no certifier or GPU check was rerun for this ruling. The existing writer object, FP8 conversions, grid and launch order remain unchanged.

This amendment supplies the missing proof rule in section5. It does **not** authorize dropping the OPAQUE bit, trusting a kernel-name allowlist, or accepting a pointer fetched through a device pointer table. The distinction is:

* acceptable **under discharged conditions**: `direct_kernarg_allocation_base + bounded_integer_offset`, where the offset includes a typed, immutable, host-validated `i32` load;
* still Unknown: a loaded value used as an address base, a pointer/descriptor assembled from loaded address words, or a direct base plus an offset whose extent, provenance or stability is unproved.

The existing A1 implementation conflates these when OPAQUE reaches the high address dword: `peacemaker-ir/src/passes/kernargs.rs:151-155` marks generic memory values OPAQUE, `Val::mix/join` propagates it (`:167-194`), and `resolve_base` rejects it (`:764-786`). A1 deliberately has allocation-wide effects, not extent proofs (`peacemaker-ir/src/kernarg.rs:11-29`). Preserve that separation: this new class identifies the allocation **conditionally**; a consumed bind witness discharges bounds/stability before the executable can use its effect summary.

### A9.1 Exact source and launch envelope

The inspected source is `kernels/src/kv_cache_write_q8_0_batched.hip:184-245`:

```
dst + positions[bid] * (n_kv_heads * (head_dim + 2))
```

in the singleton/legacy descriptor case. One workgroup per `(head,bid)`, block32, HD256. Each lane writes eight code bytes; lane0 also writes one F16 scale. Its arithmetic is the same source sequence at lines215-237.

**Do not overlook the current optional slot ABI.** The same object has `slot_descs` and `row_slot` arguments and branches at lines203-206. `kv_slot_desc.h:21-35` defines `k_base` as a loaded **byte offset**, not an allocation pointer; nevertheless arbitrary descriptor paths are not covered by this first rule. The initial conditional certificate must have the guards:

```
head_dim == 256
batch_size == 16
slot_descs == null
row_slot == null
grid == [n_kv_heads,16,1]
block == [32,1,1]
```

The source ABI expectation is dst +0, src +8, positions +16, n_kv_heads +24, head_dim +28, batch_size +32, slot_descs +40, row_slot +48, with padding before +40. Assert every offset and explicit/hidden segment against the **actual** loaded-object metadata; source layout alone is not the receipt. A variant object with a different ABI needs its own receipt and bindings, not these guessed bytes. Non-null slot pointers refuse this conditional case and keep the existing route; supporting slot descriptors is a separate bounded-table certificate, not a reason to relax this one.

Offline guarded analysis substitutes these exact kernarg constants into the **lifted original ISA**, follows its resulting CFG, and proves that neither optional table is dereferenced on the admitted path. It does not compile a new writer. If that path cannot be established, the guarded case remains Unknown. An argument known null and proven non-dereferenced in this case is encoded as a guarded constant, not as a fictitious allocation at address0.

### A9.2 Frozen fact and binding interfaces

Implement the following new fact vocabulary:

* `KernelStatus::Conditional`, distinct from both `Proven` and `Unknown`.
* `IndexedArgEffect`: access instruction PC/identity, direct base argument offset, access mode/class/cache bits/width, checked 64-bit address expression, index-load records, and launch/argument guards.
* `IndexLoad`: load PC, source pointer-argument offset, element type `I32`, byte-offset expression (here `4*workgroup_id_y`), and the extension/arithmetic semantics used by the consuming address.
* `IndexedBindWitness`: exact object/kernel/receipt/case identity, program/shape generation, sealed input generation and payload digest, bound scalar/geometry values, allocation identities/interiors/slice extents/mapping generations, and the successfully discharged predicates. This is runtime-owned and non-forgeable through the public launch API.

`IndexedArgEffect` describes the actual ISA expression, not a string supplied by the host. For this writer its semantic normal form is:

```
P_load = load_i32(P + 4*bid)
R      = H * 258
codes  = dst + zext64(P_load) * R + head*256 + lane*8 + {0,4}
scale  = dst + zext64(P_load) * R + H*256 + head*2       [lane == 0]
```

The certifier must preserve the actual signed-extension/multiply/carry operations; the above zext normal form is justified only after the host establishes `P_load >= 0`. Store widths are 4/4/2 bytes respectively. Metadata assertions, guards and machine-expression equality make this model applicable to the exact object. If the compiler emits equivalent different operations, prove that equivalence or retain Unknown; never overwrite the observed expression with this source formula.

Keep `RegionBound::AllocationWide` and normal allocation-wide RAW/WAR/WAW planning. A discharged indexed effect does **not** grant narrower hazard regions or permission to omit barriers between KV writers/readers. Its effect set is P read, src read, dst write, with the exact ISA read-cache classes. Optional guarded-null arguments have no dereference effect. `KernelStatus::Conditional` is PM4-ineligible until **all** its indexed and guard obligations have matching witnesses; `Unknown` remains ineligible regardless of any partial witnesses. The runtime may expose a separate `BoundCertifiedNode`, never mutate a cached global fact to `Proven` after one successful window.

Source consumers to migrate: `peacemaker-ir/src/kernarg.rs:20,96-105,257` for status/effect records; `passes/kernargs.rs` for analysis; `railgun-cert/src/lib.rs:132-191,195-311,574-615` for G8/serialization/summary; `railgun-corpus/src/receipt.rs:155-172` for the record; and `railgun/src/kernel.rs:113-116,167-184,309-345,508-514,664-674` for parsing/binding/admission. LSP references of `KernelStatus` were inspected for this ruling. Also update every serialized-status consumer discovered by symbol/reference lookup; an old `status != Unknown` check must not accidentally admit Conditional.

Bump the inspected receipt schema7/railgun-section1 to schema8/section2. Serialize guarded assumptions, indexed effects and required witness kinds in the hash-bound receipt, and fail closed on missing fields/unknown versions. Regenerate affected corpus receipts; do not silently reinterpret a v1 `proven` or `unknown` row as a conditional certificate. Preserve unguarded analysis results for diagnostics. The guarded case has its own case digest under the exact object receipt. A G8 differential checks the **complete guarded summary** against metadata permissions and reports its guard, not an unconditional proof.

### A9.3 Analyzer implementation — bounded, instruction-semantic, fail closed

Extend the existing abstract value with a **separate integer/address witness**, without changing legacy `roots/src/taint` meanings:

1. A qualifying ordinary 32-bit load from a directly rooted argument records `LoadedI32(source_arg, offset_expr, load_pc)`. Its source read must itself have a complete direct-base/bounded-offset proof. It remains OPAQUE to the old pointer lattice. A raw 64-bit memory load is not an integer-index leaf for this first class.
2. Track exact bit-vector expressions through moves, signed/zero extension, constant shifts, add/subtract and integer multiplication by a kernarg/constant stride. Track matching 64-bit low/high/carry chains as one address expression; do not infer a full address from its high half alone. Preserve actual truncation widths and signedness. Support select/CFG joins only when every reachable alternative is represented with the corresponding proved guard; unsupported joins widen to Unknown. Keep EXEC-partial writes conservative just as `write_v` does today (`passes/kernargs.rs:1319-1335`).
3. An address witness can be constructed only from **one live direct pointer argument plus an integer displacement expression**. The integer displacement may contain approved I32 index leaves and typed scalar/grid inputs. Pointer+pointer arithmetic, adding an opaque memory value without that witness, pointer values multiplied/shifted into an offset, float-to-address arithmetic, unknown operations, root/taint overflow, unmodelled calls and mixed partial pointer halves do not qualify.
4. At `resolve_base` / access recording, consult the complete witness as an alternative conditional resolution when ordinary resolution says `base-memory-derived`. Do not remove the old refusal for other addresses. Require a certificate for **every** unresolved memory access on the guarded CFG; one resolved store does not redeem another unknown access.
5. Emit the indexed conditions and read/write effects. Do not prove bounds from host type names, kernel names or `__restrict__`; the bind supplies actual value domains and resource extents. Actual loaded-pointer chains in `unattributable_bases_make_the_kernel_unknown` (`passes/kernargs.rs:2663-2689`) must keep exactly their refusal.

The certificate's symbolic address expressions are evaluated with checked wide host arithmetic and compared with their recorded machine-width semantics. Any wrap/sign discrepancy refuses. This is a small expression/guard evaluator for the specified operations, not a general SMT solver or a blanket “integer taint is harmless” rule. Unsupported code generation is an explicit missing rule and stays Unknown.

### A9.4 What the window bind proves, and when

Create a dedicated owner-managed **device-only positions allocation** for Whole16, at least64 bytes, used for this immutable input only. It must not share its allocation with tokens, counters, writable scratch or packed descriptors, because the hazard model is allocation-wide. The packed *host* upload slab may still contain other inputs. The 32 writer nodes can all reference this one sealed device P; other read-only verify consumers may share it.

The lifecycle is:

```
Retired -> HostChecked -> UploadInFlight -> SealedReady
        -> SubmittedReadOnly -> Retired
```

* `HostChecked`: before any submit, check the exact `[i32;16]` bytes that will be uploaded. They must equal the existing physical-position producer for this window (not an independently guessed logical/compacted position vector), be nonnegative, and be pairwise distinct. Negative positions fail before unsigned casts. For an ordinary contiguous window these are p..p+15; uniqueness is still checked, not assumed from last window.
* For **each** writer dst binding independently: validate H>0, HD256, B16, launch geometry and guards; validate all signed32 scalar products the kernel computes; compute R=H*258 in checked u64 arithmetic. Require every complete `[P[i]*R, P[i]*R+R)` row to lie inside the writer's intended KV tensor slice, not merely inside a large arena containing other layers. Include the bound dst interior. Require the allocation's actually mapped/writable intervals to cover those bytes and retain its mapping generation. Reserved VMM address space or a nominal max-context number is not enough.
* Validate P load domain `0<=bid<16` and 64 live P bytes. Validate src covers `16*H*256*sizeof(f32)` and source row/head/lane indexing. Validate pointer additions, alignments and code/scale stores, including the final scale at the last mapped row, without integer/address overflow. Validate restrict/no-alias requirements; in particular dst must not overlap P or src.
* **Writer uniqueness is two proofs:** checked unique P rows; plus the certified grid/address mapping partitions each selected row into disjoint head/lane code chunks and one scale store per head. Merely checking P uniqueness is not sufficient if the grid or block is patched incorrectly. K and V bindings and successive layer writers must also obey the expected non-overlap/ordered-hazard contract.
* `UploadInFlight`: copy exactly those checked bytes from an exclusively borrowed pinned stage; no mutation/reuse of that stage until the H2D completion fence. Couple the checked payload digest and P allocation generation to this upload token. A hash alone is not a GPU-content proof: the trusted upload operation and its completion/ownership establish that the checked bytes became device bytes.
* `SealedReady`: after upload completion, apply the railgun entry/system visibility acquire required by its existing transport/cache receipts. Check the program's full allocation/alias closure contains **no GPU write/atomic/copy destination** to P. All nodes must have complete guarded effects; a remaining Unknown effect anywhere in the single IB prevents this no-writer proof. Run this closure check for all aliases, not just argument names.
* The owning window holds an exclusive queue/resource lease from the start of upload through final completion. No other HIP stream, PM4 program, external owner, CPU mapping, asynchronous upload, allocator recycle or VMM remap may modify P during that lease. P is not mapped host memory. Read-only C qualifiers and merely placing a barrier are not enforcement. The ownership API must prevent enqueueing a writer through another alias while sealed; the existing single-owner window is responsible for excluding outside commands.
* `SubmittedReadOnly`: bind consumes the matching ready-generation witness and seals kernargs/IB too. Guards, geometry, pointers, P bytes and mappings cannot be patched in flight. Every indexed consumer uses that same generation. Retire/reuse only after a queue-quiescent completion fence. Unknown quiescence keeps P and its backing storage quarantined, not writable.

Thus bounds/uniqueness are proved from host bytes **before submission**, device identity is established by the completed exclusive H2D, and immutability is maintained until the last consumer by the resource lease plus full-program no-writer/alias proof. No per-window device readback or in-program validation kernel is required. A program cache hit does not reuse last window's bind witness.

### A9.5 Required tests and acceptance — owner runs, not the planner

**Analyzer/railgun-cert fixtures (same actual ISA frontend as production):**

1. Direct `dst + sext(load_i32(P+i))*stride` and zero-extended equivalent produce Conditional with dst-write/P-read, recorded load/access PCs, full address expression and exact cache bits. Exercise VMEM and SMEM index loads, carry across the low32 boundary, and the actual gfx1201 writer object.
2. A readonly arithmetic load that never influences an address does not become a pointer effect. An index source and destination argument exchanged in the fixture must change/refuse the facts; a malformed store width or hidden-grid dependency must be reflected, not ignored.
3. Loaded 64-bit pointer dereference, descriptor-built pointer base, two loaded32 halves reassembled as a pointer, mixed direct/loaded base through CFG join, partial-EXEC stale base, pointer multiplied into an alleged integer index, unsupported arithmetic, and index-source OOB/unknown address remain Unknown. Retain the existing two `base-memory-derived` negative fixtures unchanged.
4. Guarded-null writer analysis uses exactly the original object. Non-null or changed slot guards cannot consume its receipt. Negative/oversized H/B/HD and wrong block/grid cannot inherit the case. Branch analysis must show excluded table loads are unreachable under the recorded guards.
5. Receipt round-trip keeps expression widths, access PCs, guards, index source and condition status. Changing/removing any of these, object digest, case digest or schema fails runtime admission. G8 violations still fail. Default/legacy consumers cannot treat Conditional as Proven.

**Host binding/lifecycle tests:**

6. Accept rows0..15 and the last16 rows of a correctly mapped slice; reject -1, capacity, duplicate positions, truncated P, zero/overflowing H, wrong HD/B, wrong dimensions, correct arena but wrong tensor-slice extent, unmapped holes and last-row scale overflow. Exercise dst interiors and allocation/mapping generation changes.
7. Reject reused/stale input witnesses, a different upload payload, a submit before upload completion, and any post-bind kernarg/shape change. Attempted host stage mutation while upload is in flight and attempted P overwrite/remap/recycle while sealed must be rejected by the ownership API.
8. A same-IB direct store, aliased store, atomic, copy destination, unknown-effect node or separately queued writer targeting P makes Whole16 admission fail. A pure read consumer is allowed and remains in the hazard graph. Completion and unknown-quiescence fault paths retain/release P at the specified states.

**GPU gates after the host/cert gates:**

9. HIP writer versus same-object railgun writer: complete K/V code+scale bytes and canaries for real/synthetic F32 inputs, the actual16 positions, context-edge rows and nontrivial dst interiors. Verify P bytes remain unchanged. Compare all32 writer nodes in the window, not one representative.
10. Repeat the section7 third-arm whole-window byte/state ladder. No timing or full-window closure claim until all conditional effects are discharged and no hidden HipDirect segment remains. The new rule does not itself claim the reported32 nodes now certify; the owner supplies that artifact-specific receipt.

### A9.6 Applicability to other loaded-index kernels and bounded composer units

**Yes, the same fact class covers other kernels with a direct argument base and supported bounded integer indexing from an immutable host-staged resource**, including another use of the same sealed positions buffer, provided its own loaded-object address expression, source element domain, guards, capacity and uniqueness/overlap requirements are proved. Read-only consumers need bounds/stability but not artificial uniqueness if their actual semantics permit repeated reads. Position-dependent attention loop bounds additionally need their loop/range proof; this rule does not certify arbitrary loop behavior by association.

**No automatic coverage** for device-generated token IDs/indices, mutable counters, true pointer tables, nested `P[Q[i]]` indexing, non-null KV-slot tables, or arbitrary64-bit loaded offsets. For example the new device selector's codebook indexing cannot borrow a host-checked-positions witness for GPU-produced IDs. It needs an independently certified in-kernel range guard/producer-domain contract before the whole-window closure can pass. Until that exists it refuses or uses the declared segmented route. Do not extend this amendment by asserting all memory-derived arithmetic is equivalent.

Frozen slices, supplementing A1/A2 rather than changing inference code:

* **A2-index-analysis:** `peacemaker-ir` Conditional/IndexedArgEffect, guarded analysis and exact integer/address witnesses; owns analyzer fixtures1-4. Interface is the fact vocabulary above. No writer source change.
* **A2-index-receipts:** `railgun-cert` and `railgun-corpus` schema8/section2, guarded G8 and complete serialization; consumes the frozen fact vocabulary and owns fixture5. Old status paths fail closed.
* **A1-index-bind:** railgun typed witness consumption plus the window's dedicated sealed P allocation/lease; owns tests6-8 and allocation-wide effect emission. No weakening of cache/hazard rungs.
* **A7-index-review:** independently checks actual writer receipts and same-object gates9-10, including the no-writer closure across all32 nodes and optional-slot guard enforcement. Reviewer retains the veto.

If the exact object needs unsupported dataflow/CFG rules, report those specific rules and keep it refused; do not switch to scalar-position variants or a symbol exception without another design ruling. This amendment is the selected minimum conditional-index extension, not permission to claim an unmeasured certification or speed result.
