<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared-prefix serving: paged ownership, fair scheduling, and structured generation

- **Date:** 2026-09-05
- **Truth state:** **planned**. This document specifies implementation work; it does not change runtime defaults or claim benchmark results, feature admission, or GPU validation.
- **Source grounding:** checkout HEAD `d995164b8b837f8896fec336f2ef94ceabb03d94`, plus the working-tree source inspected on this date. HEAD alone is not an assertion that the working tree is clean.
- **Parent:** [Multi-slot engine: continuous batching + paged KV + MTP + vision](2026-09-03-multislot-paged-mtp-vision.md). Preserve its working paths and tests; extend its explicitly excluded cross-request sharing and scheduling functionality.
- **Authorities:** [Architecture](../ARCHITECTURE.md), [validation routes](../VALIDATION.md), [Redline contracts](../REDLINE.md), [measurement protocol](../methodology/perf-benchmarking.md). This spec does not replace those owners.

## 1. Decision

**Extend hipfire's existing slot engine into a bounded, cache-aware serving path. Do not embed vLLM/SGLang or build a third inference engine.**

Adopt these complementary ideas:

1. **vLLM/PagedAttention:** on-demand logical-to-physical KV mapping, immutable shared blocks, copy-on-write before mutation, explicit memory admission.
2. **SGLang/RadixAttention:** automatic longest-prefix lookup, retained prefix state, leaf-first eviction, and cache-aware scheduling.
3. **Modern serving practice in both:** iteration-level scheduling with a global work budget and chunked prefill; overlap independent CPU preparation with GPU execution only after ownership is explicit.
4. **SGLang structured generation:** share compiled constraint machinery, keep parser state private, and use token-safe jump-forward for provably forced output.
5. **SGLang workflow reuse:** ordinary multi-turn and branching requests benefit automatically; callers need neither a new DSL nor physical cache handles.

The primary objective is **successful requests meeting latency objectives per unit memory**, not maximum batch size, cache-hit count, accepted draft tokens, or kernel throughput in isolation.

The distinguishing hipfire requirement is hybrid state. A Qwen prefix is resumable only when attention KV, DeltaNet state, convolution history, scales, and error-feedback state all describe the **same processed token boundary**. Sharing KV alone is incorrect.

### 1.1 Attribution and corrections to the motivating comparison

The earlier conversational summaries are not the technical source of truth:

- vLLM supports **automatic** prefix caching. Its current design uses chained block hashes; explicit/manual prefix registration is not required. A radix tree is an alternative index, not a capability vLLM categorically lacks.
- Iteration-level batching predates vLLM; Orca described it in OSDI 2022. Chunked prefill and speculation also have independent prior art.
- SGLang's original API speculative execution generates beyond a stop boundary and reuses matching output for subsequent program primitives. It is not arbitrary speculative execution of side-effecting agent branches.
- Compressed-FSM jump-forward still performs model forward work to build state. It saves serial decoding iterations; forced text does not come with free KV.
- `torch.compile` is not a portable innovation to transplant into a Rust/HIP engine. SGLang also uses specialized kernels; compilation alone does not explain its design.
- Neither prefix caching nor a serving scheduler changes the attention equation. Paged addressing must preserve the model's causal and positional semantics.

Credit the [PagedAttention authors](https://arxiv.org/abs/2309.06180), [SGLang authors](https://arxiv.org/html/2312.07104v2), and [Orca authors](https://www.usenix.org/conference/osdi22/presentation/yu). The SGLang overlap announcement itself credits NanoFlow. Credit XGrammar separately if its algorithms or implementation are adopted.

hipfire is original work by Kaden Schutt and contributors, not a wrapper around these engines. Preserve [NOTICE](../../NOTICE), per-file contributor headers, and the boundaries in [PRIOR-ART.md](../../PRIOR-ART.md), particularly recurrent-state prompt caching. Those records do not claim that general prefix caching originated here. For any upstream code reuse, pin the exact source commit and file, inspect its actual license, retain required notices, and record modifications. This proposal selects independently implemented Rust/HIP mechanisms; it does not authorize unattributed copying or assume every dependency has the repository's license.

## 2. Grounded baseline: build on this, do not duplicate it

The following is a source inventory, **not admission evidence**. Source wins over older documentation and comments.

| Area | Existing owner and observed behavior | Required increment |
|---|---|---|
| Physical pages | [`page_pool.rs`](../../crates/rdna-compute/src/page_pool.rs): `PagePool`, `BlockTable`, `PAGE_TOKENS=128`, independent K/V byte strides, free list and reference counts | Immutable cached-page ownership, checked mutation/COW, generation-tagged handles, deferred reclaim |
| Slot addressing | [`slot_pool.rs`](../../crates/rdna-compute/src/slot_pool.rs), [`kv_slots.rs`](../../crates/rdna-compute/src/kv_slots.rs), [`kv_slot_desc.h`](../../kernels/src/kv_slot_desc.h): legacy slabs or paged descriptors; 32-byte descriptor ABI; checked full-page sharing into an empty destination | Wire sharing into serving; preserve descriptor ABI unless an independently reviewed kernel change requires otherwise |
| Paged forwarding | [`forward_slots.rs`](../../crates/hipfire-arch-qwen35/src/forward_slots.rs): provision write frontier, upload block tables before descriptors, descriptor-driven attention; paged path bypasses graph capture | Reserve all step resources before mutation; COW every written shared page; explicit completion ownership |
| Slot scheduler | [`serve_engine.rs`](../../crates/hipfire-arch-qwen35/src/serve_engine.rs), [`scheduler.rs`](../../crates/hipfire-arch-qwen35/src/scheduler.rs), [`slot_batch.rs`](../../crates/hipfire-arch-qwen35/src/slot_batch.rs): exclusive GPU thread, per-slot prefill chunks, decode and injected MTP verify rows | One global row budget, bounded waiting, fairness, transactional planning; current per-slot chunks can accumulate across slots |
| Ordinary batching | [`hipfire-engine/src/scheduler.rs`](../../crates/hipfire-engine/src/scheduler.rs), [`batch.rs`](../../crates/hipfire-generate/src/batch.rs): separate `ContinuousBatchScheduler`, narrow eligible cohorts, keyed lane generations | Reuse terminal/admission conventions; do not mistake this path for the slot engine or silently redirect existing families |
| Sessions | [`session_table.rs`](../../crates/hipfire-runtime/src/session_table.rs), [`prefix.rs`](../../crates/hipfire-runtime/src/prefix.rs): authoritative tokens, resident/swapped/cold sessions, continuation matching, LCP planning, last-token prefill requirement | Cross-session token-prefix index; snapshot-qualified resume rather than arbitrary metadata rewind |
| Ordinary cache planner | [`cache_plan.rs`](../../crates/hipfire-runtime/src/cache_plan.rs): Qwen exact-match miss/checkpoint resume; DeepSeek partial-prefix safety restrictions | Preserve family-specific safety. A longer token match is not permission to relax those restrictions |
| Hybrid snapshots | [`speculative.rs`](../../crates/hipfire-arch-qwen35/src/speculative.rs), [`weights.rs`](../../crates/hipfire-arch-qwen35/src/qwen35/weights.rs): `DeltaNetSnapshot`, recurrent matrices/scales, conv state, error-feedback residual | Cache-owned immutable boundary snapshots; private restored state per running request |
| Host/disk spill | [`swap/snapshot.rs`](../../crates/hipfire-runtime/src/swap/snapshot.rs), [`swap/mod.rs`](../../crates/hipfire-runtime/src/swap/mod.rs): gather/scatter slot KV plus extra model state, stamped snapshots; restoration allocates fresh pages | Account actual bytes and identity; transfer leases; failure before execution may yield a cold-prefill miss |
| Admission | [`admission.rs`](../../crates/hipfire-runtime/src/admission.rs): weights plus per-session context grants; slot admission evicts idle sessions, not active work | Distinguish physical capacity, unique resident pages, logical grants, future growth credits and non-KV state |
| Request/terminal protocol | [`serve/mod.rs`](../../crates/hipfire-runtime/src/serve/mod.rs), [`terminal.rs`](../../crates/hipfire-engine/src/terminal.rs), [`slots.rs`](../../crates/hipfire-daemon/src/slots.rs): submit/events; `(id, attempt_id)` correlation; `commit_ready` then commit/abort | Queue cancellation before a session exists; prevent publication/reuse while terminal commit is pending |
| Transport bounds | [`serve/mod.rs`](../../crates/hipfire-cli/src/serve/mod.rs), [`http.rs`](../../crates/hipfire-cli/src/serve/http.rs): HTTP admission; slot worker guard; slot submit channel is unbounded | Bound request counts **and bytes** before spawning workers; reserve control capacity so abort/reset cannot be starved |
| Grammar | [`saddle-core/src/grammar.rs`](../../crates/saddle-core/src/grammar.rs): shared JSON tool-call and DSML grammar machinery with token masks and state advancement | Integrate per-request constraints into slots; immutable compiled artifacts; explicit schema contract; safe forced-token runs |
| Client/bench | [`hipfire-client/src/lib.rs`](../../crates/hipfire-client/src/lib.rs): correlated native transactions and cancellation; [`bench_concurrency.rs`](../../crates/hipfire-cli/src/bench_concurrency.rs): `SlotDriver::start` currently returns an unavailable error | Restore a daemon-protocol slots benchmark before making concurrent performance claims |

**Terminology:** `KvBackend::{Contiguous,Vmm}` controls allocation/growth for another KV path. VMM is not the slot block-table/paged-attention feature. Do not add a misleading `--kv-backend paged` alias. KV encoding (q8/asym/fwht), physical allocation, and attention addressing are separate properties.

Additional source gaps: `LoadedModel.decoded_vocab` already shares an ID-ordered decoded-string table. The grammar matchers themselves remain request-local mutable state, not cached compiled schema automata. The inspected request path does not implement `response_format`/general JSON Schema; experimental slots reject tools, tool-result history, custom stops and logprobs. Qwen MTP accepts a grammar handle but does not consume it in-step; its emitter's post-acceptance grammar check is a fail-closed guard, not constrained speculative sampling. These are explicit implementation tasks, not capabilities this spec assumes already exist.

### 2.1 Scope and exclusions

Required release scope is a complete single-device slot route: paged sharing with hybrid-safe reuse, fair chunked scheduling, existing supported greedy MTP combinations, structured output, and actual HTTP/native validation. Begin with the existing Qwen3.5 dense 4B parent fixture, then validate MoE and other admitted cells separately. A dense pass does not certify MoE.

Preserve existing family routes outside this scope. Generic cache policy and interfaces must accommodate full-attention, sliding-window and recurrent models; no family advertises reuse without its own resume adapter and parity oracle.

Not prerequisites for this release: a new Python/JS runtime, agent orchestration DSL, speculative external tool execution, new draft training, PFlash, automatic quant-format changes, replacing existing TP/EP/PP, active-request preemption, disk-backed global radix caching, or cross-node prefill/decode transfer. Section 11 defines the scale-out decision boundary rather than pretending it is a single-GPU optimization.

For the new mode, this spec deliberately replaces the parent's **R-A4** admit-or-reject behavior with bounded waiting, **R-B6** no-COW limitation with C3, and ordinary capacity-exhaustion behavior in **E11** with S1/S4 reservations. A genuine shared-forward/device fault still fails all affected work. Parent positional, descriptor, MTP/VL and terminal guarantees otherwise remain requirements. Old routes retain their existing behavior until explicitly migrated.

## 3. Ownership and integration seams

Keep physical ownership, model semantics and request policy separate without creating parallel frameworks.

| Owner | Responsibility |
|---|---|
| `rdna-compute` | `PagePool`/`SlotPool`, checked refcounts and generations, COW copy descriptors, table staging, device completion leases; no tokenization, tenant policy or radix prompt index |
| `saddle-core` | Existing grammar and capability substrate; declarative resume requirements, not architecture-specific downcasts |
| `hipfire-runtime` | Existing session/prefix/admission/swap modules; token-keyed prefix index and cache identity; family-neutral step-budget and reservation data usable by arch crates without upward dependencies |
| `hipfire-arch-*` | Typed snapshot capture/restore, forward and verify, model-specific auxiliary state; Qwen slot step assembly continues here |
| `hipfire-loader` | Resolve one model owner and its supported feature combinations; combine model, device, KV and topology capabilities |
| `hipfire-engine` / `hipfire-generate` | Existing terminal policy and ordinary generation drivers; integration must not introduce an arch dependency into `hipfire-engine` |
| `hipfire-daemon` | Wire validation/correlation and forwarding; no second allocator, prefix index or scheduling policy |
| `hipfire-config`, `hipfire-cli`, `hipfire-client` | Typed configuration, HTTP admission/authenticated sharing domain, observable errors, client/benchmark adapters |

Do not move the slot scheduler upward into a crate that the arch crate cannot depend on. Shared pure policies needed by both scheduler stacks belong in existing lower-layer modules. Keep per-token forwards statically dispatched.

The logical API contracts below are **proposed operations**, not existing Rust symbols or a request to build a trait hierarchy:

| Operation | Input → output and invariant |
|---|---|
| Prefix lookup | identity + canonical input → longest **resumable** boundary and pinned handles, or miss; no GPU mutation |
| Resume planning | boundary + family requirements → snapshot/attention-window/last-token plan and byte cost; cannot return an incomplete state bundle |
| Step reservation | candidate rows + write intervals + scratch/snapshot/output needs → owned reservation or capacity result; failure leaves live request state unchanged |
| Submit step | reservation + stable descriptor generation → one in-flight completion ticket; all referenced storage outlives completion |
| Commit step | successful completion + accepted-token count → committed state boundary; no rejected draft data becomes reusable |
| Publish cache | immutable committed state + identity + policy → cache-owned leases; failure to cache cannot undo a successful request |
| Release | leases → cache-resident or reclaim-pending; never immediate reuse while a device or transport reader remains |

One integration owner changes the shared request/step contract first, then allocator, snapshot, scheduling and grammar implementation can proceed independently against it. Before modifying exported Rust symbols, enumerate callers with LSP references. Remove superseded operations and their callers together; do not leave aliases or two sources of page ownership.

## 4. Prefix cache: exact state, not approximate prompt similarity

### 4.1 Identity and isolation — C1

A cache domain binds:

- model content identity and load epoch; relevant sidecar/adapter identities;
- tokenizer vocabulary/configuration and prompt-template/normalization identity;
- architecture, numerical state ABI and position/attention policy;
- KV K/V encoding, strides, group/layer layout, recurrent quantization and error-feedback mode;
- device/topology and allocation epoch for device-resident artifacts;
- authenticated sharing namespace.

Keys use the **actual model input token sequence**, not rendered-character prefixes, user-turn FNV hashes, semantic similarity, filenames or slot IDs. Preserve hipfire's authoritative per-turn token stream and verbatim template-splice behavior. Re-encoding an old assistant transcript is not proof of a hit.

For visual input, include image-content digest, preprocessing/vision-tower identity, patch layout, embedding span, and actual positional/M-RoPE metadata. Identical image-placeholder tokens with different pixels MUST miss. Mutable URLs are not identities. Reuse before an image is permitted only if positional dependencies are proven independent; otherwise use the conservative whole-request multimodal identity.

Use canonical, versioned, length-delimited serialization. A cryptographic digest can accelerate lookup, but local radix edges are compared against exact tokens and metadata before reuse. Never treat a non-cryptographic hash collision as equality. Compute model/sidecar digests once at load, not per request or token. Benchmark md5 conventions are separate from cache security identifiers.

Ordinary temperature, top-p, seed and output constraints need not divide attention-prefix storage when the consumed tokens and forward semantics are identical. They DO remain private request state; none of their mutable state is inherited from a cache hit. Settings that actually change forward state belong in the domain.

Sharing is private to an authenticated trust domain by default. A client-supplied namespace/salt cannot grant membership in another domain. If no trusted identity exists, operate only in an explicitly declared single-owner domain or disable cross-session reuse; do not advertise anonymous multi-tenant isolation. Shared public templates require operator authorization. Do not log raw prompts, schema content or token keys in metrics. Eviction and latency can still expose aggregate resource contention; namespace partitioning is not a complete timing-side-channel defense.

### 4.2 One radix index over existing pages — C2

Use a CPU-resident compressed radix tree in the runtime prefix owner. Edges reference immutable token spans; nodes reference ordered existing physical-page leases and optional resumable-state checkpoints. Do not add a second independent block-hash eviction cache. A hash is an acceleration/index field within this ownership system.

Operations are longest-prefix lookup, split, insert, pin/unpin and leaf-first eviction. Splitting an edge changes metadata, not GPU data. If two completed requests contain the same prefix, choose one canonical cached entry; do not remap an in-flight duplicate solely to deduplicate it. Release the duplicate at a safe boundary.

Distinguish:

- **Matched tokens:** equal token prefix in the index.
- **Resident KV tokens:** matching attention rows that still exist.
- **Resumable tokens:** boundary for which every required state component exists.
- **Reused tokens:** forward rows actually skipped for this request.

Only the last quantity contributes to `cached_tokens`. A prefix match that lacks a recurrent checkpoint is not a hit in usage accounting.

Publish immutable **full 128-token pages** first, retaining the existing page geometry. Partial active tails stay private. The index can represent finer token boundaries, but that alone does not make a partial page shareable. A future page-size experiment must separately measure metadata, fragmentation and kernel behavior; do not transplant NVIDIA's block size or SGLang's historical one-token page size.

### 4.3 Page ownership and COW — C3

Extend the existing pool, not its GPU addressing equation:

- A host handle identifies pool epoch, physical index and allocation generation. Device tables may remain `u32` physical indices; the host validates generations before table upload. Stale handles must fail before dereference.
- Separate active/table references, cache ownership and in-flight/transfer leases. Physical bytes are charged once, regardless of reference count. Refcount underflow/overflow and duplicate release are errors, not wraparound.
- A page is logically private/mutable, sealed/shared, cache-only, reclaim-pending, or free. Cached pages are immutable even with one remaining owner.
- Before **any** KV write, inspect the entire write interval, not just append allocation. A write touching a shared/sealed page first reserves a private destination and copies the valid prefix in every K/V layer arena. Rebind that request's table only after copy ordering is established.
- Rewind can revisit a previously full shared page. Rejected speculative rows and last-token recomputation make append-only assumptions insufficient. Full-page-only publication does not remove the need for a write barrier.
- Do not mutate a shared last page in place. Either retain the full immutable page and allocate/copy a private prefix for the new boundary, or resume at the previous supported boundary and recompute.
- A failed multi-page/layer reservation releases only newly reserved resources and preserves the original table. Do not leave half-installed COW mappings.
- Copies preserve encoded bytes, including quant metadata and distinct K/V strides. No dequantize/requantize step in a cache copy.

Retain the existing checked full-page/empty-destination `share_prefix` contract; general mutation-safe attachment must not silently weaken it. Migrate its serving callers through the owned reservation/write-barrier operation.

### 4.4 Completion and eviction — C4

Host reference count zero is not GPU completion. Free/reuse requires every step, graph/tape, descriptor upload, D2H/H2D copy and snapshot reader to have released its lease. Track completion with the existing HIP event/stream facilities behind one owner. A synchronous implementation may retire at a proven step fence; later overlap cannot remove that proof.

Block tables and descriptors remain immutable while read by the device. Upload changed block tables before publishing descriptors. Double-buffering is allowed only with per-buffer completion ownership. No copied stack-pointer kernargs or speculative reuse of staging storage.

Evict oldest unpinned leaves; remove their lookup visibility before releasing page/snapshot leases. Keep useful ancestors while descendants use them. Reclaim cache-only pages before rejecting a new allocation. Bound CPU node/token metadata as well as device bytes; pruning a snapshot cannot leave a node falsely marked resumable.

Cached prefixes and running requests share the same page pool. A cache-retention limit is a ceiling, not a permanently reserved VRAM partition. Running work has priority over cache-only residency. Idle-session and prefix-cache owners must not each believe the same bytes are exclusively theirs.

### 4.5 Hybrid-state checkpoints and last-token semantics — C5

Define a resume boundary `p` as **exactly tokens `[0, p)` processed by the target**. The snapshot at `p` is state `S_p`; it cannot be combined with KV from another boundary.

The Qwen bundle includes all DeltaNet matrices, scales, convolution rings and indices, and error-feedback residuals. Include any other forward-affecting counters identified by the family adapter. Running requests restore into private mutable buffers; never share mutable recurrent state between forks. Use `DeltaNetSnapshot` and the established swap buffer inventory rather than another hand-written copy list.

Capture page-aligned checkpoints at selected completed prefill chunks and useful committed boundaries, with a byte-bounded LRU snapshot pool. Align chunks when capturing an intermediate boundary; a state at the end of a chunk cannot be relabeled as an earlier state. Do not copy full recurrent state after every token or for every tree node. If there is no affordable checkpoint, cache fewer boundaries and recompute the suffix.

Lookup chooses the largest boundary satisfying **all** layer requirements. Full attention needs its prefix; sliding-window attention needs the correct positional window; recurrent/compressed-state families need exact snapshot/ring state. A layer-local hit is insufficient. Unknown adapters return a miss rather than relaxing `CachePolicy` safety.

By default, resume at `p < prompt_len` and process the remaining suffix to obtain first-token logits. For a prompt that exactly matches cached tokens, select an earlier valid checkpoint/page boundary. Never run the last token again while leaving recurrent state at `S_prompt_len`; restore the corresponding earlier state first. Empty prompts follow existing tokenizer/BOS validation and cannot underflow `prompt_len - 1`.

Preserve the distinction between accepted token history and materialized target state: the newest sampled token may not yet have a KV row. Cache only the materialized committed prefix. Saved hidden/logit caching could avoid exact-hit recomputation, but is not required and must not be assumed.

DFlash/MTP drafter KV and hidden buffers are separate from the target cache. On resume, restore a separately identity-qualified drafter checkpoint, rebuild it by the existing reseed path, or choose the supported AR route **before execution**. Reusing the target prefix is not evidence that the drafter is ready.

### 4.6 Publication and session transitions — C6

GPU completion, accepted-token commitment, and the client terminal handshake are distinct events:

1. A prefill checkpoint may become shareable after successful forward completion, canonical input validation and sealing.
2. Speculative candidate rows never become shareable until target acceptance and state repair complete.
3. Generated-prefix cache entries and continuation eligibility become durable only after the existing `commit_ready` → client commit → `done` transaction succeeds.
4. While awaiting client commit, keep the session unavailable to another continuation. Abort/timeout removes unpublished generated state; a previously published prompt prefix remains valid if independently owned.

Cancellation is idempotent. It removes queued work by attempt identity even before `Accepted` supplies a session ID. Cancellation during execution latches immediately, suppresses later output and reclaims only after completion. Do not wait for a reply-channel write to discover every cancellation.

Model unload/reload or incompatible reconfiguration first stops admission, drains/aborts transactions, invalidates lookups and replay plans, waits for device/transfer completion, then releases allocations and advances epochs. Never select cached state by model filename after reload.

## 5. Memory admission and scheduling

### 5.1 Budget actual resources — S1

Separate four quantities: allocated pool capacity, uniquely resident pages, future growth reservations, and logical context limits. `max_seq` is not proof of physical allocation; summed session lengths are not unique page use.

For page size `B=128`, a page bundle's KV bytes are:

`page_bytes = sum_attention_layers B * (k_stride_bytes[layer] + v_stride_bytes[layer])`

Include quant scales/headers in those strides. The device budget also includes weights, private recurrent states, cache checkpoints, draft/head state, worst supported batch scratch, vision embeddings/tower scratch, graph/retained objects, descriptor staging and explicit safety headroom. Preflight this layout once, then use counters in the decode loop; never call a device-wide free-memory query per token. Preserve the unified-memory OOM guard and separately bound host/pinned/disk storage.

For the first robust release, use **completion-backed admission without active preemption**:

- At admission reserve credits for the request's maximum remaining target growth through `prompt + max_tokens`, any required COW tail, and the selected speculation workspace. Use checked arithmetic and existing context-cap rules.
- Shared physical prefix pages count once; the request's future private suffix and recurrent state count separately.
- Credits turn into allocated private pages as work advances. Allocation MUST NOT exceed the already granted credits. Cache-only pages are evictable backing, not an additional demand on top of the same pool.
- Invariant: unique non-evictable page bundles + unmaterialized growth/COW credits + reserved temporary workspace fit pool capacity. Account leases pending completion as non-evictable.
- If an individual request cannot fit on an otherwise empty supported route, reject it with the required/available capacity. If it could fit after running work finishes, queue it within finite limits.

This intentionally sacrifices some optimistic overcommit throughput to guarantee progress. Active preemption may later replace conservative grants only with a complete token/RNG/parser/recurrent restore contract and its own acceptance evidence; do not silently introduce it as an OOM fallback.

Idle sessions may relinquish physical resources using the existing snapshot/cold-recompute policy. Logical context grants must not permanently reserve scarce device credits for every idle conversation. GPU resident-state release, host snapshot accounting and logical session lifetime are distinct transitions.

### 5.2 One globally budgeted step — S2

At each completed step boundary:

1. Drain cancellation/terminal control; retire completed leases; process only a bounded amount of new admission work.
2. Pin eligible prefix hits and select admissible waiting requests. Validate reservations before changing live state.
3. Give runnable decode requests their base target progress allocation. Preserve per-request sampling/RNG isolation.
4. Allocate a nonzero prefill service quantum when prefill is waiting; distribute remaining rows by fair rotation. `prefill_chunk` remains a per-request maximum, not the total budget.
5. Allocate optional extra MTP verify/forced-token rows only inside remaining quotas. MTP decoding must not also receive an ordinary decode row.
6. Prepare, reserve, execute, synchronize/observe completion, repair/commit, then emit.

For a step, enforce:

`sum(prefill_rows + ordinary_decode_rows + verify_rows + forced_rows) <= max_batch_tokens`

MTP verify includes its seed/bonus-position work (`k+1` rows), not only accepted drafts. Bound draft-head work separately; the draft phase cannot bypass the fairness budget merely because it precedes `next_batch`. If a cycle cannot fit, reduce supported draft depth or run AR from the committed state before starting that cycle.

Size ragged row arrays and scratch to the global capacity. No unbounded vector growth, whole-context token cloning, full-tree scan or per-step tensor allocation on steady decode. Incremental cache insertion processes newly sealed pages only. Padded rows must not consume RNG, update state, write KV or emit tokens.

### 5.3 Fairness, cache preference and queue bounds — S3

Use age-aware fair rotation with cache locality as a **bounded tie-breaker**, not unrestricted longest-prefix-first:

- Preserve admission age across chunking and scheduling retries. Internal chunk/forced-token work cannot re-enter as a new young request.
- Aged eligible requests get service before younger cache hits. Use weighted deficit rotation between authenticated domains when multiple domains are enabled; equal weights initially.
- Within the same fairness band prefer lower uncached prefill cost, using resumable tokens rather than raw matched length. Do not cache-sort the entire queue every token.
- Set a maximum wait-to-service policy in scheduler ticks for already admitted runnable work. Wall-clock latency includes non-preemptible GPU steps and cannot be guaranteed by row counts alone.
- Require enough global rows for selected decode lanes plus the minimum prefill quantum; otherwise admit fewer lanes or reject the configuration. An endless decode arrival stream must not starve prefill.

A concrete initial policy avoids a tunable scoring model: maintain an oldest-first queue and a FIFO deficit ring for each domain. Once a request is skipped for one complete admission round, mark it aged; select aged requests before locality tie-breaks. If the oldest request is individually feasible but cannot yet acquire its full credits, stop admitting younger conflicting requests and let active work drain until it fits or its existing queue deadline expires. This bounded-backfill rule prevents a stream of small cache hits from indefinitely blocking a larger cold request. For admitted prefills, rotate a persistent cursor after each allocated quantum; with `N` runnable prefills and one available quantum per tick, each receives service within `N` completed ticks. Debit all target/verify/forced rows and bounded draft work; stalled consumers are not runnable. A non-preemptible vision-tower call remains a separately measured latency bound, not something this row-based guarantee conceals.

Keep the existing HTTP `serve.max_queue`, request byte limit and queue timeout authoritative. A queue count alone is insufficient: bound total canonical prompt/image bytes, grammar compilation work and pending output bytes. `serve.max_queue=0` currently means uncapped; the new robust multi-slot mode must reject that combination rather than reinterpret zero or silently inherit an unbounded queue.

Propagate one admission permit across HTTP, daemon and slot queues; these are stages, not additive independent allowances. Direct native clients must encounter equivalent limits. Acquire capacity before `thread::spawn`; a guard acquired inside an already spawned thread does not bound thread creation.

A separate bounded/control-priority path handles abort, terminal commit and shutdown. Full data queues must not block cancellation. Overload before streaming produces a typed capacity/queue error (HTTP 429 for bounded queue rejection, 503 for unavailable/poisoned backend); timeout and native error mapping stay consistent with the existing API. Never return successful empty output.

### 5.4 Failure and backpressure — S4

| Failure point | Required result |
|---|---|
| Validation/identity/unsupported requested constraint | Reject before GPU work; no cache/session mutation |
| Cache miss or unavailable host snapshot before execution | Cold-prefill only if the full request can be admitted; record miss, not hit |
| Step reservation cannot fit | Reclaim cache-only resources, reduce candidate step or wait; preserve already running requests and their state |
| Allocation invariant violated despite reservation | Fail closed as an engine/accounting fault, not a busy-loop retry |
| Per-request parser/sampling failure | Fail that request; release its resources at a safe boundary |
| Shared forward or uncertain device execution failure | Fail every affected request and quarantine uncertain resources; poison/recreate the engine if validity cannot be proven |
| Stream consumer stalls | Stop scheduling that request before its bounded event buffer fills; retain the committed state; abort on configured deadline |
| Client disconnect before terminal commit | Abort exactly once; no continuation from unpublished state |
| Late abort after terminal commit linearizes | Drain the already committed result per existing protocol; do not roll it back |

A failed GPU step is **not** an atomic transaction that can be undone by resetting lengths. Never retry the same partially executed token through another backend. Preparation failure before execution and execution failure have different recovery rules.

## 6. Speculation, multimodal input, and route combinations

### 6.1 Commit frontier — X1

Reuse the existing `SpecTarget`/MTP save-verify-repair contracts. A cycle tracks separately: input/accepted token history, materialized committed target rows, tentative verification rows, recurrent checkpoint boundary, and output already emitted. They must not be collapsed into one `seq_len`.

Before verify, reserve every potentially written page, COW destination and state-repair buffer. Preserve `positions[]` as the causal authority; allocation length is not an attention visibility boundary. On partial acceptance, repair recurrent state and set the committed frontier before cache publication. Stale rejected rows may remain physically present only while provably invisible and overwritten before future reads.

Check EOS, stop conditions, length, grammar completion and cancellation on every token in a burst. The bonus token is accepted history, not automatically materialized KV. Do not publish it as cached computation prematurely.

Preserve currently supported **greedy** slot MTP. Sampled, penalized, constrained or image requests must take their explicitly supported AR path until their joint verifier exists. Requested semantics may never be dropped to retain MTP.

If sampled speculation is later enabled, implement target-distribution-preserving acceptance using the actual conditioned proposal and target distributions: accept with `min(1, p(token)/q(token))`, and sample rejection correction from normalized positive `(p-q)`. Grammar masks, penalties and truncation order must be applied consistently per proposed prefix; hard zeros stay zero. Per-request RNG cannot depend on batch order. Greedy matching is not a sampled acceptance algorithm.

### 6.2 Compatibility contract — X2

| Combination | Target behavior / gate |
|---|---|
| Paged + shared prefix + text AR | Required; exact input identity, COW, completion leases |
| Qwen hybrid + shared prefix | Required only with complete state at the selected boundary; otherwise honest cold/suffix recompute |
| Existing supported KV tier + sharing | Encoded-byte copies and correct independent strides; per-tier oracle required |
| MTP + paged + shared prefix | Required for the existing greedy-supported cell; private tentative writes and target/drafter resume distinction |
| Sampling/penalties + prefix reuse | Required AR; sampler state is request-private |
| Grammar + slots + paged prefix reuse | Required AR after Section 7 lands; masks/parser/terminal semantics may not be bypassed |
| MTP + grammar or non-greedy sampler | AR selected before execution until exact joint acceptance is implemented and validated |
| Vision + paged | Preserve parent's batched vision behavior and M-RoPE; same-request MTP remains off |
| Vision + prefix reuse | Off until pixel/embedding/position identity and hybrid snapshots pass the dedicated image oracle; text-only prefixes are not assumed safe |
| Adaptive KV/CASK/window compaction + shared pages | Do not infer compatibility; in-place conversion/compaction requires private destinations and a new state identity |
| Paged + HipGraph/retained replay | Ordinary HIP until separately supported and certified; never reuse the plain-AR tape for ragged multi-slot work |
| Other model families / TP / EP / PP | Preserve existing route; no inherited slot/cache admission from a generic capability flag |

Loader capability resolution must describe combinations, not a collection of independent booleans whose Cartesian product is accidentally advertised. Combine carrier-declared properties with actual device/kernel, KV layout and topology support. Log/return the selected route and optimization bypass reason; reject missing requested semantics. An optimization bypass is allowed, a silent semantic downgrade is not.

## 7. Structured generation without a second grammar engine

### 7.1 Immutable artifacts, private cursors — G1

Extend the existing `saddle-core` grammar and runtime tokenizer integration. Separate reusable token/grammar metadata from per-request parser state. Compile outside the GPU loop into immutable shared artifacts keyed by tokenizer identity, grammar dialect/version, canonical schema, tool definitions and model framing config. Bound compilation time, input size, nesting, state count, mask-cache bytes and total artifact bytes; coalesce identical in-flight compilations.

The existing JSON tool-call and DSML guards are not evidence of arbitrary JSON-Schema enforcement. Add an explicit strict-output contract rather than relabeling a syntax guard. Preserve current tool framing behavior; strict validation is not silently applied to unrelated legacy requests.

Concrete migration sites are [`RequestContract` / `project_request_contract`](../../crates/hipfire-cli/src/serve/complete.rs), daemon request validation and projection in [`slots.rs`](../../crates/hipfire-daemon/src/slots.rs), the runtime `SubmitRequest`, the per-slot sampling state, and existing `SpecEmit`/terminal lowering. Keep the tool-call and DSML emitters model-specific; do not merge them into a parser containing architecture-ID branches. Remove slot tool/stop refusals only in the same change that supplies their complete behavior. Constrained slot requests use AR until the verifier applies constraints before acceptance, not only after a bad token has advanced state.

For the first strict schema implementation, publish the supported subset: JSON primitive types, objects (`properties`, `required`, boolean `additionalProperties`), arrays (`items`, `minItems`, `maxItems`), and `enum`/`const`. Enforce the supplied keywords; reject unsupported assertion keywords, external references, regex assertions, recursion and combinators before generation. Permit only documented annotation keywords. A recursive JSON syntax parser is not a finite-state automaton for arbitrary JSON Schema. Do not fetch remote schemas during request handling.

Implement `required` as object-key membership at the correct nesting level, not a substring search through emitted text. A required name occurring in a string value or child object does not satisfy its parent. Define duplicate keys as invalid in strict mode; treat escaped and unescaped spellings of the same decoded key identically. Validate numeric JSON syntax without admitting NaN/Infinity, and enforce `enum`/`const` by semantic JSON equality. Contradictory schemas must be rejected when compilation proves unsatisfiability; runtime dead ends remain typed constraint errors. These cases belong in A15 coverage.

Use the standard `response_format`/tool request surface through existing Rust request lowering; validate it end-to-end so no intermediate wire adapter drops the constraint. For strict successful output, final parsing and schema validation must agree with incremental generation. Truncation, cancellation or unsatisfiable constraints are not successful schema-conforming results. Streaming chunks need not each be complete JSON, but every emitted byte must belong to a valid prefix; terminal status states whether the object completed.

### 7.2 Masking and token boundaries — G2

A token contributes its decoded **bytes** to the incremental parser; handle UTF-8 fragments, escaping, tokens crossing delimiters and byte-fallback vocabulary entries. Precompute tokenizer byte metadata once. Cache masks by complete parser state only when sound; recursive parser stacks cannot be summarized by one FSM-state number.

Do not assume `tokenizer.decode([id]) -> String` preserves byte-fallback tokens: some decoders replace incomplete UTF-8 with a replacement character. Reuse the existing vocabulary cache only where it is lossless. Extend tokenizer-owned token-byte extraction/context handling for strict masks, then prove it against the tokenizer's incremental decoding behavior. Never build strict constraints on replacement-character artifacts.

Apply request penalties/biases according to the existing sampler contract, then ensure the hard grammar mask participates before probability-dependent filtering/renormalization. No later operation may resurrect a forbidden token. EOS is allowed only in an accepting grammar state under the output framing contract. An empty allowed set or invalid probability mass is a typed error, never an unconstrained-sampling fallback.

Each request owns parser cursor/stack, stop-matcher tail, detokenizer state, token-history penalties, reasoning/tool framing and RNG. Forks may share compiled artifacts but not mutable cursors. A speculative verifier advances a private cursor for each candidate and commits only the accepted prefix.

### 7.3 Conservative jump-forward — G3

Adopt the useful part of SGLang's compressed-FSM idea without its unsafe transplant points:

1. On a private parser cursor, find a bounded run for which there is exactly **one legal next token ID** at each step under all hard output constraints. Unique characters are insufficient: multiple tokenizations of the same forced text can change the next distribution.
2. Stop before a branch, EOS/stop boundary, output/context cap, or unsupported dynamic logit rule. If proof fails, ordinary constrained decode is the correct path.
3. Reserve pages and feed those exact token IDs through the existing chunked target forward, updating attention and hybrid state. A forced run uses the global step budget and does not jump the queue.
4. Advance histories, byte parser, finish counters and RNG exactly as the scalar constrained reference would. If the reference consumes a random draw for a probability-one choice, account for that draw; do not change later sampled output.
5. Commit/emission is token-by-token at the existing semantic boundary. No re-tokenization or rewriting of already emitted bytes. Requests needing model logprobs retain per-position scoring or do not take this optimization.

This is narrower than byte-string re-tokenizing jump-forward, deliberately. Do not promise a speedup for every schema. Success is fewer serial model iterations on eligible schemas with unchanged observable constrained behavior. Never claim probability-distribution equivalence from JSON validity alone.

## 8. CPU/GPU overlap and retained execution

### 8.1 Overlap only independent preparation — O1

First ship the synchronous ownership model. Profile actual CPU gaps before adding overlap; Rust dispatch may have a different bottleneck from Python serving.

The first overlap implementation prepares tokenized incoming requests, grammar artifacts, radix lookup candidates and immutable next-step metadata while the GPU executes the current step. A prepared plan carries request generation, model/pool epochs and resource leases. Revalidate at commit/launch if cancellation or current-step acceptance changed its assumptions.

Use bounded reusable staging buffers and HIP completion events; no host reads of future sampled tokens and no next-state mutation before the producer completes. Keep one GPU submit owner. A worker may perform pure host preparation but cannot independently mutate `PagePool`, session state or stream ownership.

Do not initially launch a request's next dependent decode step ahead of its EOS/grammar/acceptance decision. SGLang-style future-token lookahead is a separate optimization requiring cancellation, output buffering and speculative state accounting; hiding CPU work does not justify executing unvalidated work.

### 8.2 Replay safety — O2

Paged/ragged work continues through ordinary HIP until the exact route supports its dynamic tables, masks, positions, write ranges and lifetime model. The parent's paged capture bypass remains intact.

Any later graph or retained plan must key/bind batch geometry, row-to-slot map, KV/table identity, layer/quant layout, MTP masks and vision position/embedding state. Stable arena addresses alone are insufficient. Model reset, pool reconstruction, allocation rebinding and incompatible shape changes invalidate captured assumptions before use.

Follow [REDLINE.md](../REDLINE.md) for ABI blobs, artifact identity, dynamic bindings, effects and certification. Keep ordinary HIP, HipGraph, retained AQL/PM4, and experimental direct-KMD `redline` distinct. Never retry a replay execution failure through HIP for the same forward. Manual capture/parity evidence alone is not timed-arm route proof.

## 9. Configuration, observability, and workflow behavior

### 9.1 Typed controls

Retain `serve.multi_slot`, `serve.multi_slot_slots`, `serve.multi_slot_ctx`, `serve.multi_slot_prefill_chunk`, `serve.max_queue`, `serve.queue_timeout_ms` and request-size policy. Add the following **proposed typed fields** to the existing configuration registry, not direct environment reads:

| Proposed field | Meaning / conservative behavior |
|---|---|
| `serve.prefix_cache` | Boolean cross-session cache selection; off until the exact route passes release gates |
| `serve.prefix_cache_max_bytes` | Retained cache ceiling including hybrid snapshots; zero means no retained cache, never unlimited |
| `serve.max_batch_tokens` | Global trunk-row budget; validate against slot count, minimum prefill quantum and scratch limits |
| `serve.prefill_min_tokens` | Minimum service quantum when prefill is runnable; positive in mixed mode |
| `serve.max_queue_bytes` | Total canonical pending-input budget, positive and finite |
| `serve.stream_buffer_bytes` | Per-request bounded pending-event budget |
| `serve.stream_stall_timeout_ms` | Maximum stalled-consumer interval; abort after deadline |
| `serve.scheduler_overlap` | Off until independent preparation has race/lifetime and profile evidence |
| `serve.structured_jump_forward` | Off until scalar constrained-equivalence gates pass |

Resolve numerical defaults from the existing memory planner and checked route limits, then publish them in `CONFIG.md` with the implementation. They are not universal GPU tuning constants specified by this document. Keep the existing config precedence and validation path. No new user-facing knobs for block size, COW correctness, generation checks or fencing: these are mandatory invariants, not performance switches.

Schema validation is a requested semantic contract and remains active even with jump-forward off. Turning prefix caching off disables cross-request retention, not normal request-local attention KV. Disabling an optimization must leave a correct admitted route.

### 9.2 Measurements that prove the mechanism

Extend existing metrics/usage owners; do not create another telemetry service. Export bounded aggregate counters/histograms and per-request evidence:

- matched/resumable/reused/recomputed prefix tokens; miss reason; cold versus warm cache condition;
- unique private/shared/cache-only/in-flight page bytes, free capacity and reserved growth credits;
- checkpoint bytes/capture/restore time, COW count/bytes, eviction and spill counts;
- queued requests/bytes, rejection/timeout counts, admission-to-service wait and maximum service gap;
- prefill/decode/verify/forced row counts, active slots and actual selected route;
- grammar compile/cache/mask time, forced tokens, incomplete/invalid strict-output results;
- requested versus observed MTP/graph/replay route, bypass reason, draft cost and accepted tokens;
- client-observed TTFT, time to first visible answer where different, inter-token latency, end-to-end latency and successful-request goodput.

`cached_tokens` counts actual skipped target prompt computation, not matched tokens or draft work. Rejected/cancelled requests remain in denominators for overload reporting; their tokens cannot inflate successful goodput. Avoid request IDs as unbounded metric labels.

### 9.3 Branching and multi-turn clients

Ordinary requests with the same canonical prefix automatically benefit; cross-request reuse must not require a new session API. Concurrent branches acquire independent request identities, RNGs, parser cursors, mutable tails and recurrent state. They share only sealed prefix storage. A branch cannot hijack an already active continuation.

Keep `hipfire-client` responsible for Rust client conveniences, if later needed. No new orchestration language or server-side execution of user programs. A fork/join workflow joins application results, not incompatible KV histories. External tools execute only on accepted, committed tool calls; speculation never invokes them.

## 10. Delivery sequence and explicit cutover

Each stage is usable through the actual product route, not just a unit-test scaffold. Feature flags separate rollout risk; they do not permit semantic combinations to be silently ignored.

| Stage | Changes and owners | Exit criterion |
|---|---|---|
| P0 — Evidence and contracts | Existing CLI/client daemon-backed slots benchmark; source-pinned baseline; request/terminal and memory accounting contracts; capability matrix | Native/HTTP measurements prove which backend ran; unsupported arms fail rather than produce a misleading comparison |
| P1 — Ownership and admission | `PagePool`/`SlotPool` generations, COW/write barriers, completion leases; physical/future budget accounting; bounded admission/control channels | Page/state isolation, reservation failure and cancellation tests pass; ordinary paged AR remains correct with cache off |
| P2 — Hybrid-aware prefix reuse | Runtime radix index and session integration; Qwen checkpoint adapter; cache identity, publication and eviction | Cold/warm/branch/interleaved-session outputs and full state parity pass; actual reused tokens reduce prefill work |
| P3 — Fair continuous work | Global budget, age/deficit scheduling, prefill quantum, MTP quota and bounded streaming | Adversarial mixed load makes progress without starving prefill/old requests; OOM does not discard unrelated admitted work |
| P4 — Structured generation | Existing grammar refactor, explicit strict schema subset, per-slot cursor/masks/terminal handling; then forced-token batching | Strict requests never silently unconstrained; scalar versus accelerated state/output behavior passes; tool framing preserved |
| P5 — Composition and release | Existing MTP/VL/quant cells, lifecycle stress, source/API/docs migration; per-route default decision | Every advertised combination has its required oracle and product evidence; disabling optimization preserves behavior |
| P6 — Profile-gated overlap | Independent host preparation, stable staging, completion tickets | Correctness/failure matrix unchanged and measured host-gap benefit exceeds fixture noise; otherwise leave off |

P1 allocator work and P4 immutable grammar-artifact work may proceed independently after P0 freezes shared contracts. P2 depends on P1; P3 integration uses P1/P2 accounting. P5 cannot be declared complete from a text-only AR subset.

Keep the ordinary scheduler for families it serves until they have equivalent coverage. For a migrated family/configuration, select one execution owner once at load/request admission; never enqueue the same attempt into both engines. When a route is fully superseded, migrate its callers and remove obsolete gates/data fields in the same change. Do not create permanent compatibility shims merely to avoid updating callers.

Implementation release updates belong in `ARCHITECTURE.md`, `SERVE.md`, `CONFIG.md`, relevant CLI documentation and the normal changelog location. This planned document does not publish proposed settings as shipped defaults. Admissions remain solely in `admissions.yml` after earned evidence; historical checkpoint ledgers are immutable.

## 11. Scale-out: useful extensions, not hidden prerequisites

### 11.1 Cache-aware replica routing

For multiple **independent replicas** serving the same qualified model/state domain, a Rust routing policy can select by estimated completion cost rather than maximum prefix hit alone:

`estimated_cost = queue_delay + uncached_prefill_cost + decode_backlog_cost`

Use observed worker costs and load first; cache locality breaks ties among healthy compatible replicas. Worker cache summaries are advisory, generation-tagged and expiring. A stale hint may cause recomputation, never adoption of remote physical handles. Worker restart invalidates hints and session affinity; route an entire multi-turn continuation to one worker or cold-resume from canonical input.

Reuse the existing native HTTP/client/config owners; do not add a Python router. Preserve authenticated cache domains across hops. Never route to a model based only on its display name. Cancellation and deadlines propagate to the actual owner. Do not retry generation after visible output or uncertain commitment; an explicit pre-execution refusal can be rescheduled without duplicating a live attempt.

Acceptance requires a two-replica fixture with skewed popularity, stale hints, worker death and saturated warm-cache workers. Compare load-only routing versus locality-aware routing under identical arrivals, with p95/p99 latency, goodput, miss rate and rejected work. A high hit rate with worse tail latency is not success. This extension has no effect on a single-device deployment and is not part of its release bar.

### 11.2 Hierarchical cache and prefill/decode separation

Retain existing idle-session spill before building a distributed cache. A host-cache extension must transfer the whole qualified resume bundle, use bounded pinned buffers, validate length/version/digest before publication, and hold leases until all transfers finish. Restore remains a cache optimization; corruption/missing content before execution produces a miss. Device loss during restore is not an ordinary cache miss.

Prefill/decode disaggregation is a separate deployment project. Its contract must carry canonical tokens, exact model/adapter/tokenizer and state-layout identity, KV pages/window mapping, recurrent/conv/EF state, positions, and first-decode boundary. Negotiate a versioned transfer format with no raw pointers. Destination reserves resources before transfer; source retains leases until acknowledged; partial transfer never becomes visible. Cross-TP/EP reshards require an explicit transform and oracle, not byte copying.

Adopt only when measured TTFT/ITL objectives justify network/copy cost: transferable bytes / effective link bandwidth + serialization/restore delay is a lower bound on handoff. Consumer PCIe systems may lose; do not assume RDMA or promise throughput gains. Graphs and retained tapes are rebuilt locally, never transferred. These mechanisms require their own concrete transport design, failure protocol and route evidence before implementation admission.

## 12. Acceptance criteria and validation plan

### 12.1 Behavioral requirements to retain as regression coverage

Tests earn their place by observing state, output, memory ownership or progress—not implementation names, source strings or fixed error prose.

| ID | Scenario | Required observable result |
|---|---|---|
| A1 | Prefix lengths 0, 1, 127, 128, 129; exact match; prompt shorter than cache; divergent tail | Correct reusable boundary and first-token output; no recurrent double-advance or underflow |
| A2 | Two branches share full pages, then rewrite/append at a shared boundary | Original branch/cached bytes unchanged; private tails differ; no bystander KV writes |
| A3 | Random allocate/share/rewind/COW/release sequences; duplicate/stale handles | Page conservation, valid refcounts and generations; stale/invalid operations fail before mutation |
| A4 | OOM on the last required COW page or snapshot allocation | Failed reservation leaves old tables/state intact; unrelated active requests complete |
| A5 | Evict while pages are pinned by a step/copy; rapid reuse of same physical index | No use-after-free/stale descriptor; page becomes allocatable only after completion |
| A6 | Same tokens with changed model, KV mode, EF layout, template, namespace or image | Incompatible state misses; unauthorized sharing impossible |
| A7 | Qwen checkpoint restore and suffix extension | KV, matrices, scales, conv and EF state agree with the qualified reference at the same boundary |
| A8 | Checkpoint absent, evicted, corrupt host snapshot, new load epoch | Honest miss/cold recompute when safe; no false cached-token accounting |
| A9 | Last sampled token has not been forwarded; exact cached prompt | First-token logits and accepted history are correct; materialized-state count never includes an uncomputed row |
| A10 | MTP full reject/partial/full accept, page crossing, EOS/length inside burst | Only accepted target state persists; no candidate rows are cache-visible; AR reference contract holds |
| A11 | Abort before Accepted, mid-prefill/verify/COW, and while awaiting client commit | Exactly one terminal outcome; no orphan queue entry or reusable uncommitted continuation |
| A12 | Duplicate/late commit or abort after slot reuse; client EOF without terminal completion | Attempt/generation isolation; existing client reports premature EOF rather than success |
| A13 | Long cold prompt plus continuous short warm requests and mixed samplers | Both classes progress within declared service bounds; per-request RNG/penalties unaffected by slot reorder |
| A14 | Queue count/byte limit, slow consumer, disconnected stream, model unload | Finite resources, typed overload/stall errors, clean next request; control messages remain serviceable |
| A15 | Strict JSON subset, nested arrays/objects, escapes and split UTF-8 token bytes | Incremental acceptance matches independent final parser/schema validator; invalid schema rejected before generation |
| A16 | Grammar singleton-token run versus ambiguous byte-string tokenization | Accelerate only proven runs; identical accepted token/state/RNG behavior versus scalar constrained reference |
| A17 | Empty grammar mask, EOS before accepting state, stop/length inside forced run | Explicit failure/incomplete status; never unconstrained fallback or false schema-valid completion |
| A18 | Mixed vision/text/MTP and chunk boundary inside image embedding span | Parent interaction matrix preserved; image identity prevents false reuse; correct M-RoPE and per-slot isolation |
| A19 | HIP upload/launch/sync failure and replay-execution fault injection | Affected work fails closed; poisoned resources not reused; no same-forward fallback execution |
| A20 | Repeated model swap, cache on/off, idle spill/restore, repeated long soak | Bounded plateau after warmup, correct reset, no monotonic resource leak or stale state |

Use host property/state-machine tests for page/accounting and scheduler invariants; GPU oracles for encoded KV and recurrent state; actual HTTP/native interactions for terminal behavior. Host mocks cannot prove stream completion or device lifetime. Keep reference fixtures independent of the code that chooses reuse boundaries.

### 12.2 Existing routes and real gaps

- **Host behavior:** extend existing tests alongside page/slot pools, prefix/session/admission/swap, scheduler/terminal and grammar. Run targeted suites during implementation and the existing no-GPU route once at integration.
- **Numerical/state parity:** extend the applicable examples such as [`test_multislot_ops.rs`](../../crates/rdna-compute/examples/test_multislot_ops.rs), [`test_batched_attn_slots.rs`](../../crates/rdna-compute/examples/test_batched_attn_slots.rs), [`test_prefix_cache_equivalence.rs`](../../crates/hipfire-runtime/examples/test_prefix_cache_equivalence.rs) and [`test_swap_equivalence.rs`](../../crates/hipfire-runtime/examples/test_swap_equivalence.rs). Inspect their actual fixture/feature coverage before use. Extend them where COW/hybrid/quant/grammar state is not covered; absence of an oracle blocks the cell.
- **User-facing text:** [`scripts/serve_harness.py`](../../scripts/serve_harness.py) supports `--mode battery`, `--mode chain`, `--mode session`, `--model`, `--no-spawn`, `--sampling`, `--out`. Run against the exact configured product path and read decoded outputs. `battery` is an option value, not a positional subcommand in the inspected parser.
- **Concurrency:** complete the native `hipfire bench` daemon slots adapter in P0. The current `--concurrency` slots arm is not usable evidence. Extend the existing benchmark owner for committed arrival traces, mixed lengths, cache conditions and percentile/goodput records; do not substitute a sequential client loop for simultaneous requests.
- **Vision:** follow the dedicated image/stream/abort route in `VALIDATION.md`; the generic text harness does not cover pixels. Use a pinned HF reference for the exact image checkpoint where required.
- **Kernels/dispatch/graphs/replay:** use the applicable numerical oracle and `scripts/redline_daemon_harness.py` per the repository route policy; new paged/ragged coverage must actually exercise that route. Redline attribution additionally requires the certification ladder and same-report timed-arm route proof in `REDLINE.md`. If the available harness cannot prove a cell, record it blocked and add the missing coverage before promotion.
- **No retired gates:** `coherence-gate*` and undocumented historical scripts are not current acceptance, even if an old example or research result names them.

Use deterministic state for equality claims: production Q8 EF remains enabled; legacy stochastic Q8 is not a byte-parity oracle. Execution-preserving changes require exact state where the existing arithmetic path is unchanged. If batched kernels change reduction order, declare justified logit/state tolerances and examine near-tie token differences separately; never excuse state corruption as generic quantization noise.

### 12.3 Performance evidence and promotion

No numerical speedup is promised by this spec. Predeclare fixture-specific thresholds and latency objectives before measuring. Require at least three matched fresh-process samples; use more when the effect is comparable to spread. Each warm-cache sample deliberately primes the same cache inside that fresh process, then measures the same request trace. Separate JIT/DPM warmup from cache warmup; never accidentally warm only one arm's prefixes.

Required workloads:

1. No shared prefixes: cache-on overhead and ordinary single-stream decode regression control.
2. Repeated system/few-shot prefix with divergent suffixes: recomputed tokens, TTFT and unique KV bytes.
3. Interleaved multi-turn sessions and concurrent branches: hybrid snapshot value and isolation.
4. Mixed long-prefill/short-decode arrivals: p50/p95/p99 TTFT, ITL, completion latency and successful SLO goodput.
5. Structured JSON/tool output with both forced-token-rich and branch-heavy schemas: validity, compile cost, decode iteration count and latency.
6. Bounded-memory pressure, slow consumers and cache churn: reclamation, wait bounds and failures.
7. Existing MTP and multimodal interaction fixtures: benefit cannot hide regressions in supported combinations.

Record exact target/sidecar, prompt/token/image trace and both CLI/daemon identities; GPU gfx/ROCm/driver; KV/state modes; numerical policy; all route-affecting config; seeds; arrival trace; concurrency; cache warm/cold policy; raw samples and decoded text. Use the repository md5 convention and any additional mandatory pinned artifact SHA-256. Pin upstream revisions/hardware/quantization if making cross-engine comparisons; different model bytes or quants are not an engine-only A/B.

Single-stream regression measurements use `hipfire bench` with explicit backend/workload and no concurrency sweep. Concurrent goodput is a separate experiment, not a replacement for single-stream evidence. The new slots benchmark must refuse missing route evidence and count overload/errors rather than silently omit them.

Release only if every advertised cell passes correctness, all safety/progress invariants hold, the targeted benefit exceeds observed noise, and no protected workload suffers an unexplained regression. Investigate absolute deltas of 5% or more under project policy. Do not trade worse p99 or starvation for aggregate token/s without an explicit deployment-level decision. Source/CI green and historical measurements do not create an admission.

Before selecting defaults, the implementation campaign must record an explicit tuple of model/sidecars, architecture, KV/state format, topology, context range, batch range, schema subset and route; absolute p95/p99 TTFT/ITL objectives; and a maximum tolerated regression relative to its pinned baseline. Missing values block promotion, rather than being filled from favorable post-hoc runs. This is deployment-specific admission input, not a request to invent universal latency numbers for all GPUs.

## 13. Evidence sources

Primary external sources, consulted 2026-09-05. Moving documentation is design reference, not a pinned implementation dependency:

1. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), SOSP 2023 — physical block management and sharing.
2. vLLM, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) — chained identity, full-block reuse, allocation, eviction and cache isolation.
3. Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/html/2312.07104v2), NeurIPS 2024 — RadixAttention, cache scheduling, compressed FSM, frontend and API speculation.
4. Yu et al., [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022 — iteration-level scheduling.
5. vLLM, [Optimization and Tuning](https://github.com/vllm-project/vllm/blob/main/docs/configuration/optimization.md) — chunked prefill, decode-first budgets and recompute-preemption tradeoffs.
6. SGLang team, [v0.4: scheduler overlap, cache-aware load balancing and XGrammar](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) — independent preparation overlap, load-aware cache routing; credits NanoFlow and XGrammar.
7. Yin, Sheng and Zheng, [Fast JSON Decoding with Compressed FSM](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/) — jump-forward and tokenization-boundary hazards.
8. vLLM, [Hybrid KV Cache Manager](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md) — per-layer allocation/reuse constraints; that document is explicitly a historical implementation snapshot, not proof of current universal hybrid support.
9. vLLM, [Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md) — phase separation and connector boundaries, not a guaranteed throughput win.

**Completion of this specification is not completion of the implementation.** The release deliverable is the composed product behavior and evidence in Sections 10–12, with current-route documentation updated only when those gates are earned.
