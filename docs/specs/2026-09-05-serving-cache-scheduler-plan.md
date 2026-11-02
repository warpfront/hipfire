<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving cache scheduler — implementation plan

- **Date:** 2026-09-05
- **Status:** waves 0–5 advertised text cells composed on gfx1101 / ROCm 10. Defaults stay off (`serve.prefix_cache`, `serve.structured_jump_forward`, `serve.scheduler_overlap`). Not admitted. Vision+prefix stays off.
- **Spec:** [2026-09-05-serving-cache-scheduler-spec.md](2026-09-05-serving-cache-scheduler-spec.md) — this plan implements it; it repeats no guarantees the spec owns.
- **Base branch:** `feat/serving-cache-scheduler`, forked from `patch/oom-guard-multislot-tip` HEAD `d995164b8` (the spec's grounding checkout) with `feat/multislot-vision-mtp` tip `6f85c5da5` merged in. All slices branch from and PR back to this branch.
- **Recon:** four source-grounded inventories taken this date over the working tree; cited file:line below is from that recon, re-verify before editing (index drift).

## 0. Progress (2026-09-05)

| Wave | State | Tip commits |
|---|---|---|
| 0 merge | done | `6b07ef415` (layer-chunk + no D2H/H2D roundtrip) |
| 1 P0 contracts+bench | done | `f6904c85b`, `9d7110d46` |
| 2 P1 pool/admission + P4 grammar core | done | `b6b4a6a30`, `3f024ec4c` |
| 3 P2 prefix cache | done, default off | `0f648d091`, `d4faba38b`, `21f202f3d` |
| 4 P3 scheduler + wait + jump-forward planner | done; wait on; jump-forward default off | `4e2ed2423`–`f3222d63d` |
| 5 composition | **composed for advertised text cells** | Slot engine *and* single-slot loader probe `qwen3.5-4b.mtp` next to `qwen3.5-4b.mq4v2.hfq`. GPU `test_serve_prefix_cache --mtp-k 4` PASS on gfx1101 / ROCm 10: greedy MTP cold 0 / warm 256 / branch 256 matching tokens; sampled AR reuse 256/256 deterministic seed; grammar AR reuse 256 and `{"city":"Rome"}`; 4-hit soak then `reset` → reused=0. `serve_harness.py --mode chain` cached_tokens 0→89→192→265→346. `hipfire run --spec mtp` logs `MTP head loaded (sidecar .../qwen3.5-4b.mtp)`. Vision+prefix off (X2); tools/stop/logprobs refused; no A19 fault injection; full A20 (model swap/idle spill) not run; no `admissions.yml` row. |
| 6 overlap | **open, stays off** | `serve.scheduler_overlap` is a registered flag only — no host-prep overlap path exists, so there is no host-gap to measure. Spec §10: leave off. |
| 7 critique + gaps | **done** | Two-agent review of the landed waves plus a first-party audit; fixes: cache-ref reclaim leak + re-pin guards (page_pool), contiguous-residency resumability + node-bound rollback + O(1) eviction parents (prefix_index), prefill-tail livelock + aged-band recompute (serve_fairness), checkpoint byte ceiling enforced (checkpoint), capture-boundary relabeling guard + DN restore fail-closed + publish-failure cache-ref rollback + admission-time reclaim + honest reservation-invariant fault + reset releases radix cache leases (serve_engine), grammar compiler default-deny/type-arrays/schema-form additionalProperties/numeric equality/duplicate keys + required-open-object + mask fast paths (saddle-core, CLI validator aligned), atomic materialize_growth + checked arithmetic (admission), real pending-byte stream backpressure with stall deadline (http), spawn-before-acquire + json_object rejection + queue_bytes (daemon/CLI), bench drain deadline + no-bail commit + run-scoped ids (bench_concurrency). New gap cells: A10 long-generate MTP page-crossing, A13 mixed-load progress, A20 reset→re-warm (caught the reset cache-lease leak), A19 `--fault-publish` fail-closed oracle. Full GPU suite PASS gfx1101/ROCm 10; chain+battery harnesses complete end-to-end. |
| 8 residual fixes | done, `2390e4e0f` | Follow-up audit: 6 fixes (25–30 in summary) — completion pin leak, greedy/sampled gate alignment, bare-object schema, pool-epoch handles, reused_tokens stat, incremental maxItems + tests. |
| 9 verification | **done** (`a2e013b6e`) | Full campaign: workspace --lib green; GPU oracle extended + PASS (A10 forced full-reject seam, A13 concurrent, A20 free-page plateau, A19 HIP-bridge fault injection upload/launch/sync); wave-8 fix 25's slot-clear deletion found + fixed (reset cell caught it); chain `0→89→192→265→346` re-baselined on single-chunk prompts; typed-refusal probes pass; bench slots arm runs (0 rejected). New precise gaps: strict json_schema over ChatML fails closed (think-framing vs mask; §7.2 cursor follow-up), admission/forward page-accounting mismatch under pool pressure, legacy batch-arm JIT kernel error (out of X2 scope). |

Host tests: sidecar_probe 3 + loader stem probe 1. GPU: `test_serve_prefix_cache --mtp-k 4` PASS; single-slot `hipfire run --spec mtp` loads `qwen3.5-4b.mtp`. Pre-existing `oversized_pool_is_refused_not_allocated` fails when OOM guard is inactive.

## 1. What already exists (do not rebuild)

| Asset | Where |
|---|---|
| Paged allocator, 128-token pages, split K/V strides, checked full-page `share_prefix` into empty dst | `crates/rdna-compute/src/page_pool.rs` |
| Optional paged mode + block-table staging + 32 B descriptor ABI | `slot_pool.rs`, `kv_slots.rs`, `kernels/src/kv_slot_desc.h` |
| Paged forward with write-frontier provisioning, tables-before-descs ordering, graph bypass | `hipfire-arch-qwen35/src/forward_slots.rs` |
| Multi-slot engine: chunked prefill, decode, greedy MTP verify/repair, VL, adaptive MTP retire | `hipfire-arch-qwen35/src/serve_engine.rs`, `scheduler.rs` |
| Sessions/LCP/cache-plan/swap-snapshot/admission scaffold | `hipfire-runtime/src/{session_table,prefix,cache_plan,admission,swap}/...` |
| JSON tool-call + DSML grammar matchers (request-local) | `saddle-core/src/grammar.rs` |
| commit_ready→commit/abort terminal protocol, per-request cancellation | `hipfire-client/src/lib.rs`, `hipfire-cli/src/serve/*` |
| serve.* config registry incl. `serve.max_queue=0` means uncapped | `hipfire-config/src/lib.rs` |

## 2. Gap matrix (recon verdicts)

| Spec | State | Anchor |
|---|---|---|
| C3 generations | absent; bare `Vec<u32>` refcounts, ABA undetectable | page_pool.rs:82 |
| C3 sealed/COW | absent by design ("no copy-on-write"); contract-only immutability | page_pool.rs:266 |
| C3 checked rc errors | underflow/double-release silently swallowed | page_pool.rs:344-364 |
| C4 completion leases | absent; rc==0 ⇒ immediate free+reuse, no HIP event/fence | page_pool.rs:346, slot_pool.rs:188 |
| C1 identity | `SnapshotStamp` has model_hash only; no template/tokenizer identity | swap/snapshot.rs:36 |
| C2 radix index | absent; sessions match on `convo` user-turn hashes, not tokens | session_table.rs:170 |
| S1 reservations | absent; pages provisioned lazily inside forward; one slot's OOM rejects ALL active | forward_slots.rs:3739, serve_engine.rs:2061 |
| S2 global budget | absent; static `max_batch`, panic-assert inside forward, no throttle | serve_engine.rs:359, forward_slots.rs:3628 |
| S3 fairness/queue | absent; admit-or-reject, no waiting queue, no ageing | serve_engine.rs:2783 |
| S3 queue bytes | count-only; no aggregate byte cap; 0=uncapped must be rejected in multi-slot | serve/mod.rs:266 |
| G1 artifacts | schema+cursor fused in mutable per-request `Matcher`; no JSON Schema/`response_format` | grammar.rs:185 |
| G2 mask order | multi-slot path masks AFTER sampling (fail-closed guard), never before | spec_emit.rs:332 |
| G2 byte fidelity | `decode()` is `from_utf8_lossy` → U+FFFD breaks byte-fallback masks | tokenizer.rs:788 |
| G3 jump-forward | absent | — |
| P0 bench | `SlotDriver::start`/`DaemonDriver::start` bail; no public multi-inflight `Engine` API | bench_concurrency.rs:200,347 |

## 3. Cross-slice contract freeze (must land first — P0 gate §3 of spec)

New module `crates/hipfire-runtime/src/serve_contract.rs` (+ re-export): family-neutral data types for the seven proposed operations — `CacheDomain` (C1 identity fields incl. template + tokenizer identity), `PrefixLookup`, `ResumePlan`, `StepReservation`, `StepTicket`, `CommitBoundary`, `PublishLease`, `MissReason`. These are plain owned structs/enums; arch crates consume them, nothing depends upward. `rdna-compute` gets the matching physical types (`PageHandle{phys,epoch,generation}`, `PageState`, `LeaseClass`, `CowPlan`) in `page_pool.rs`. Everything downstream codes against these; no trait hierarchy, no arch downcasts.

Config keys (all typed, in the existing Serve registry block): `serve.prefix_cache` (false), `serve.prefix_cache_max_bytes` (0), `serve.max_batch_tokens`, `serve.prefill_min_tokens`, `serve.max_queue_bytes`, `serve.stream_buffer_bytes`, `serve.stream_stall_timeout_ms`, `serve.scheduler_overlap` (false), `serve.structured_jump_forward` (false). Defaults resolved from the memory planner, not hard-coded.

## 4. Waves

### Wave 0 — Merge (blocking)
- **M0 MergeVLBases:** resolve VL-forward conflicts; keep no-roundtrip fix AND layer-chunking; build affected crates. Everything else waits on this.

### Wave 1 — P0 contracts (after M0; 2 parallel slices, disjoint files)
- **P0-contracts** (hipfire-runtime + hipfire-config): land §3 contract module + all config keys + `serve.max_queue=0` rejected when multi-slot on. Exit: workspace builds, keys visible in `hipfire config get`.
- **P0-bench** (hipfire-client + hipfire-cli): public multi-inflight API (`submit_streaming`-style pending registration exposing the existing commit_ready handshake); implement `DaemonDriver`/`SlotDriver` start+run on the wire protocol; refuse unsupported arms loudly. Exit: `hipfire bench --backend both` runs slots arm on a daemon; rejected/timeout counted, not omitted.

### Wave 2 — P1 ownership + P4 grammar artifacts (after contracts; parallel, disjoint crates)
- **P1-pool** (rdna-compute): page generations, `PageState` machine (private/sealed/cache-only/reclaim-pending/free), checked refcount errors, lease classes, `CowPlan` (reserve-private + copy valid K/V prefix preserving strides/quant bytes, transactional failure), deferred reclaim queue, host completion-lease API (HIP event-based; sync fence acceptable initially), slot generations in SlotPool. Preserve descriptor ABI and `share_prefix` contract. Exit: A2/A3/A5 host property tests green; paged AR unchanged with cache off.
- **P1-admission** (hipfire-runtime + cli serve admission): physical/future-credit accounting (page_bytes from strides, growth credits, checked arithmetic), bounded waiting queue with bytes cap, unified permit HTTP→daemon→slots, control path never starved. Exit: A14 overload/stall typed errors; A13 progress bound test harness scaffolding.
- **P4-grammar-core** (saddle-core): split immutable compiled artifact from per-request cursor for both matchers; strict JSON Schema subset compiler (subset per spec G1, reject unsupported keywords pre-generation); tokenizer byte-exact token metadata path (no lossy decode). Exit: A15 host tests; behavior of existing tool-call requests unchanged.
- **P4-contract-plumbing** (cli complete.rs + daemon slots.rs + runtime SubmitRequest): `response_format` end-to-end; per-slot mask BEFORE sampling in serve_engine (kill the post-acceptance guard on the slots path); keep tools/stop refusals until their full behavior lands. Exit: strict requests never silently unconstrained (A17).

### Wave 3 — P2 prefix cache (after P1-pool)
- **P2-index** (hipfire-runtime): CPU radix tree over token spans, pin/unpin, leaf-first eviction into pool's deferred reclaim; C1 domain identity incl. image digests; cache-only pages evict under pressure.
- **P2-qwen-adapter** (hipfire-arch-qwen35): `DeltaNetSnapshot`-based checkpoint capture at page-aligned boundaries, byte-bounded LRU snapshot pool, restore-into-private, last-token semantics (resume at p < prompt_len or earlier boundary, never double-advance recurrent state), drafter reseed-or-AR decision at admission.
- **P2-wire** (serve_engine + session_table): lookup→pin→resume-plan in admit path, publish sealed pages at commit boundaries, C6 publication transitions. Exit: A1, A6-A9 parity oracles via existing prefix/swap equivalence examples extended for hybrid state; `cached_tokens` counts reused-only.

### Wave 4 — P3 scheduler (after P1+P2 accounting)
- **P3-budget** (qwen35 scheduler/serve_engine): global `max_batch_tokens` enforced at scheduling (replace panic-assert), prefill quantum, decode base allocation, MTP verify quota inside budget, per-slot reservation-before-mutation so one slot's OOM fails only that slot (S4).
- **P3-fair** (runtime policy + serve_engine): oldest-first queue + aged marking + deficit ring per domain, bounded tie-break on cache locality, max wait-to-service in ticks, bounded stream buffer/stall timeout. Exit: A13 adversarial mixed load, A10 inside-budget MTP, A11/A12 abort/commit matrix.

### Wave 5 — P5 composition, then P6 overlap (gated)
- MTP/VL/quant combination oracles (X2 matrix), lifecycle soak (A20), docs/CONFIG migration, per-route default decision, admissions evidence per §12. P6 overlap only with measured host-gap benefit; `scheduler_overlap` stays off otherwise.

## 5. Acceptance & evidence rules for every slice

- Validation route by change class per docs/VALIDATION.md: kernels/dispatch/graph → `scripts/redline_daemon_harness.py`; user-facing generation/state → `scripts/serve_harness.py` (battery/chain/session, read decoded text). Retired coherence-gate scripts are not evidence.
- Perf numbers: ≥3 fresh-process runs, prompt md5, model+binary md5s, byte-identical prompt files under `benchmarks/prompts/`.
- A1–A20 IDs from spec §12.1 are the regression targets; host property tests for pools/scheduler, GPU oracles for state, real HTTP/native for terminal behavior.
- Before editing exported symbols: LSP references; cross-file renames via LSP only. Superseded operations and callers removed together; no shims.
- Each slice: skip build-clogging during parallel work; integration owner runs full check at wave boundaries.

## 6. Non-goals (per spec §2.1, do not let slices drift)

No Python/JS runtime, no orchestration DSL, no active preemption, no disk radix cache, no P/D disaggregation, no new draft training, no PFlash. Scale-out (§11) is a later decision, not part of this plan. `--kv-backend paged` alias forbidden.

## 7. Risks

- **Merge semantic drift (M0):** the two VL fixes overlap in the same function; resolution must be hunk-level, verified by build + a VL smoke before P2 touches the engine. Mitigation: P0-bench exercises only text paths; VL smoke gate at Wave 3 start.
- **Completion-lease API surface (P1-pool):** HIP event ownership inside rdna-compute is new; if event wiring blocks, ship host-side sync-fence retirement first (spec allows), overlap later.
- **Session tokens vs radix identity:** sessions key on `convo` hashes; the radix index keys raw token spans — the two coexist; session continuity stays, radix adds cross-session reuse. Do not merge them.
- **Grammar eager scope:** strict JSON Schema subset only; the existing tool matchers keep model-specific framing. Byte-exact vocab table is prerequisite to any mask claim — G2 ordering change without it ships nothing.
