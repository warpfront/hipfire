<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving cache scheduler — work summary and remaining gaps

- **Date:** 2026-09-05 (review pass appended same day)
- **Branch:** `feat/serving-cache-scheduler` (pushed to `origin` at `ghazni101/hipfire`)
- **Tip:** `a2e013b6e` `fix(slots): restore batched-path slot release; add wave-9 verification cells`
- **Spec:** [2026-09-05-serving-cache-scheduler-spec.md](2026-09-05-serving-cache-scheduler-spec.md)
- **Plan:** [2026-09-05-serving-cache-scheduler-plan.md](2026-09-05-serving-cache-scheduler-plan.md)
- **GPU:** gfx1101 (AMD Radeon RX 7700 XT), HIP 7.15, ROCm 10 container (`local/rocm-base:10.0.0`)
- **Model:** `qwen3.5-4b.mq4v2.hfq` + `qwen3.5-4b.mtp` sidecar

---

## Review pass (2026-09-05, wave 7)

Two parallel code-review sweeps (structured-generation wave; daemon/admission/bench wave) plus a first-party audit of the ownership/scheduler core. Every finding below was verified against source, fixed, and covered by a test or an oracle cell.

### Correctness bugs found and fixed

| # | Area | Bug | Fix |
|---|---|---|---|
| 1 | `page_pool.rs` | `release_cache_ref` with in-flight refs left the page in limbo — neither free nor in `reclaim_pending` — leaking it permanently | Mirror `dec_table_ref`: enter `ReclaimPending`, freed by `drain_completed`; `add_{cache,inflight}_ref`/`refcount_inc` refuse `ReclaimPending` pages; `drain_completed` also requires zero table/cache refs |
| 2 | `prefix_index.rs` | Resumability compared a checkpoint boundary against a cumulative count of valid pages (gaps below a boundary could be compensated by valid pages above) | Walk tracks the gap-free contiguous resident prefix; a checkpoint is resumable only while every page below it is resident (A8) |
| 3 | `prefix_index.rs` | A chain insert that hit the CPU-node bound partway left its already-inserted nodes in the tree without adding them to `total_nodes` — the bound silently stopped binding | `create_chain` rolls its partial chain back on bound failure; accounting stays exact |
| 4 | `prefix_index.rs` | Eviction found a leaf's parent by an O(nodes) scan per leaf | Nodes carry a parent pointer; eviction unlinks in O(1) |
| 5 | `serve_fairness.rs` | A prefill whose remaining tail was smaller than `prefill_min_tokens` could never be granted — the slot livelocked in prefill forever (hidden at default min=1) | A grant that completes the prompt (`rows == uncached`) is always allowed; sliver control still applies to non-completing grants |
| 6 | `serve_fairness.rs` | `aged` was never cleared — one skipped round promoted a request ahead of the whole queue forever | Age recomputed per round: served requests leave the aged band and can be re-aged |
| 7 | `checkpoint.rs` | Insert "break"ed out of the eviction loop when everything was pinned and inserted anyway — the byte ceiling was soft | Refuse the capture (drop the blob, return `CheckpointId::NONE`); the boundary stays honestly unresumable |
| 8 | `serve_engine.rs` | Checkpoints captured the live DeltaNet state (`S_next_pos`) but were labeled at the page-aligned boundary below — any chunk ending mid-page poisoned every resume from that boundary (spec §4.5 relabeling ban) | Capture only when the published boundary equals the state boundary; mid-page publications are KV-only |
| 9 | `serve_engine.rs` | DN checkpoint restore was fire-and-forget (`let _ = restore_to(..)`) — a failed restore left shared KV paired with zeroed recurrent state | Restore first, fail-closed to the cold path on error |
| 10 | `serve_engine.rs` | Publish failure orphaned the cache refs taken before `publish_sealed_pages` | Refs rolled back on failure (tree-unchanged guarantee from fix 3 makes this exact) |
| 11 | `serve_engine.rs` | `reset` replaced the radix without releasing its cache leases — every published page stranded CacheOnly; the pool drained to 0 free pages within a few reset cycles (caught by the new A20 re-warm cell) | `PrefixIndex::release_all` + `drain_completed` wired into Reset |
| 12 | `serve_engine.rs` | Admit had no page-pressure reclaim (spec §4.4: reclaim cache-only pages before rejecting work) | Cold admit evicts oldest unpinned radix leaves when free pages cannot back suffix + generation budget |
| 13 | `serve_engine.rs` | The step-reservation shrink path trimmed phantom rows (accounting said fit, batch unchanged) | After draft dropping, a remaining overflow is an accounting fault: fail closed per S4 |
| 14 | `serve_engine.rs` | Continuation-admit compiled the JSON schema with `.ok()` — a compile failure admitted the request silently unconstrained after `Accepted` went out | Compile hoisted before any session mutation; failure rejects before `Accepted` |
| 15 | `serve_engine.rs` | Jump-forward greedy gate (`> 1e-6`) disagreed with the sampler's argmax gate (`== 0.0`) — tiny temperatures took the RNG-consuming path, breaking G3.4 draw accounting | Gate aligned to `temperature == 0.0`; `DrafterDecision::Reseed` now declared truthfully |
| 16 | `grammar.rs` | Unknown keywords fell through `_ => {}` — `if`/`then`/`else`, `additionalItems`, and arbitrary unknowns compiled with constraints ignored | Default-deny; annotations remain the only silent accepts |
| 17 | `grammar.rs` | `{"type":["integer","null"]}` fell into the inference path and compiled to unconstrained `Any`; schema-form `additionalProperties` flattened to `true` | Both rejected before generation; CLI validator aligned (same refusal set, doc parity) |
| 18 | `grammar.rs` | `json_equal` was serde `PartialEq` — variant-strict (`1 != 1.0`), stranding matchers mid-generation on numeric enums | Semantic numeric equality (integer-exact when possible, f64 otherwise, recursive on composites); `number_could_grow_to_match` accepts semantically-equal spellings |
| 19 | `grammar.rs` | Duplicate object keys (including escaped spellings) merged silently via serde | Raw byte scan decodes keys and rejects duplicates at accept time; same-key-in-sibling-objects stays legal |
| 20 | `grammar.rs` | Required name without a `properties` entry rejected as unsatisfiable even when the object was open | Compiles as an `Any` property when `additionalProperties: true`; genuinely unsatisfiable only when closed |
| 21 | `grammar.rs` | Recursion heuristic rejected finite schemas on repeated property names | Dropped — `$ref` (already rejected) is the only recursion vector |
| 22 | `admission.rs` | `materialize_growth` = release-then-charge: over-materialize discarded other requests' credits and charged anyway; releases saturated on underflow, inflating free capacity | Atomic materialize (verify-then-mutate); releases are checked errors; checked arithmetic in `AdmissionController` |
| 23 | `http.rs` | Stream backpressure byte bound computed `forwarded/forwarded == 1` byte per pending slot — the 16 MiB bound could never trip, and a Full channel blocked the forwarder forever with no deadline | Real pending-byte counter shared producer↔consumer (saturating, never wraps); stall = retry loop within `stream_stall_timeout_ms`, then typed abort; regression tests for trip/resume/deadline |
| 24 | `daemon` | Slot worker threads spawned before the concurrency check (spec §5.3 anti-pattern); `queue_bytes` hardcoded 0 so the wait-queue byte cap could not bind; `json_object` accepted then silently unconstrained (front end dropped it) | Pre-spawn bound at the dispatch site; `canonical_prompt_bytes` wired into `SubmitRequest.queue_bytes`; `json_object` rejected typed at both front end and daemon |

Perf hardening: `is_token_allowed` gained bounded fast paths (structural first-byte set + inert string content) so the vocabulary scan stops cloning+reparsing for inert tokens; the scanner also feeds duplicate detection.

### Verification (all in the rocm10 container, gfx1101)

- `cargo test --workspace --lib`: all suites green (the only failure, `oversized_pool_is_refused_not_allocated`, is the documented pre-existing environmental case — passes with `HIPFIRE_OOM_GUARD=1`; the 7 `tests::update_*` self-installer tests also fail on the base commit in this container).
- GPU oracle `test_serve_prefix_cache --mtp-k 4`: PASS — greedy/sampled/grammar reuse cells, plus new A10 (96-token generation crossing page boundaries replays identically), A13 (long cold prefill + short warm request both complete), A20 (soak, reset → cold, **re-warm after reset reuses again** — the cell that caught fix 11), and A19 `--fault-publish` (injected publish failure → honest miss with identical output).
- End-to-end HTTP (multi-slot + prefix cache serve): `serve_harness.py` chain `cached_tokens 0→89→192→265→346` (matches the pre-change reference) and battery complete; streaming strict-schema requests get typed A17 rejections instead of false successes; `json_object` and unsupported schema keywords are typed rejections.

### Remaining gaps (unchanged or newly precise)

- **Vision + prefix reuse (X2/A18)** — still off; needs the pixel/embedding/position oracle.
- **Slots tools / stop / logprobs** — upstream `ad6004ac0` restored tool turns on
  the daemon slot path: `tools` are now **supported** (a tool call finishes with
  `tool_calls`; a tool message without `tool_call_id` is a typed 400). `stop`
  and `logprobs`/`top_logprobs` refusals stay (spec: remove only with complete
  behavior).
- **`admissions.yml` / ARCHITECTURE.md / default-on** — still gated on the full §12 evidence tuple.
- **P6 overlap** — flag only, stays off until a measured host gap exceeds fixture noise.

### Wave 10 — gap closure pass (2026-09-12)

| # | Gap | Resolution |
|---|---|---|
| W10-1 | Forward page-accounting mismatch under pool pressure | Admit-time page demand now wired on BOTH admit paths: cold admit AND continuation admit estimate `ceil((uncached + max_tokens)/128)+1` pages, evict radix leaves then LRU idle sessions, and typed-reject (`page demand exceeds pool`) instead of dying mid-forward. Verified: 8-page pool rejects a 9-page demand at admit; small requests unaffected. |
| W10-2 | Strict `json_schema` over ChatML rejected requests with thinking | Framing-aware grammar cursor: `SubmitRequest.started_in_think` + `GrammarConstraint.{in_think,think_open_id,think_close_id}` defer the schema mask until `</think>`; think-tag tokens never reach the matcher. Verified e2e: schema+thinking returns valid JSON in `content` with reasoning in `reasoning_content`. |
| W10-3 | Early per-token schema pruning | `is_token_allowed` now refuses tokens whose first non-whitespace byte cannot begin a value conforming to the expected `SchemaNode` (schema-position stack on `scan_raw`/`RawScan`; `value_next` byte set). Also fixed two latent fast-path bugs: `:` refused after object keys, `]` refused at empty-array positions. 8 new tests; 201 saddle-core tests green. |
| W10-4 | Mid-page radix divergence refused (`MidPageDivergence`) | Partial-page nodes: `Node.first_page_skip` lets an edge begin mid-page while its `PageHandle` still covers a full 128-token page; `split_edge` forks at any token via a zero-page marker; `walk` derives `Handle.token_offset` from `child_base - first_page_skip`. `MidPageDivergence` removed. GPU cell: zero publish refusals; A13 mixed `reused=512` (divergent tail cached). |
| W10-5 | A19 fault injection below the HIP bridge | `HIPFIRE_FAULT_HIP=<upload|launch|sync>` seam in `hip-bridge` (arm-after-load, atomic countdown). `--fault-hip=<class>` cell: typed rejection, no fake Done, sync poison → fresh-engine recovery byte-identical. All 3 classes PASS. |
| W10-6 | A20 model swap + idle spill/restore | `--a20` cell: slot pressure spills an idle session (evictions=1), named reentry restores it (restores=1, reused ≥ prompt); two shutdown+respawn cycles each cold→warm with stable `free_pages`. PASS. |
| W10-7 | A13 adversarial multi-domain wait bounds | Concurrent phase now mixes samplers (2 greedy + sampled + penalized, distinct convos) with a per-request 120 s wait-bound assertion. PASS. |


### Wave 11 — deep-review fix campaign (2026-09-12)

A full deep review (three parallel sweeps + first-party verification of the
pool/index/fairness/checkpoint cores) found two P0s, four P1s, and a set of
P2 design gaps. Every finding below was fixed and verified; all 39 workspace
lib suites green (saddle-core 210, hipfire-runtime 698, hipfire-arch-qwen35
261); GPU compose oracle + A19/A20 fault cells PASS; live serve regression
matrix PASS.

| # | Area | Bug | Fix |
|---|---|---|---|
| P0-1 | `serve_engine.rs` | Prefix-cache publication ran between the forward and its error check: a failed step sealed/published unwritten pages and captured a mid-layer DN checkpoint, permanently poisoning the radix | Publication (and checkpoint capture) gated on `fwd.is_ok()`; spec §4.6/§5.4 |
| P0-2 | `grammar.rs` | Malformed keyword VALUES silently compiled to no constraint (`{"enum":"hello"}` → `Any`, `required` as string → empty, non-numeric `minItems` → 0): strict requests ran unconstrained and reported success | Every present-but-wrong-typed keyword value is a typed `InvalidSchema`; empty `enum` rejected as unsatisfiable (CLI validator aligned) |
| P1-2 | `serve_engine.rs` | The COW write barrier never executed its copies: the forward wrote through the still-bound sealed mapping and `commit_cow` rebind to uninitialized destinations (reachable via edited-history continuations) | `execute_cow_copies` runs the per-layer K/V d2d copies at plan time; commit (rebind) moved BEFORE the forward; commit failure fails only that slot |
| P1-3 | `serve_engine.rs` | Prefix-hit restore was fail-open on the share step: `share_published_pages` failure fell to cold prefill on top of an advanced DN state (silent corruption) | DN state reset on ANY non-converting exit of the hit path |
| P1-4 | `serve_engine.rs` | Dropping an MTP draft in the reservation-shrink path wedged the slot forever (`remaining_prompt` was already taken; every readiness predicate false) | Draft seeds stashed per-slot; a dropped draft restores the seed and retires `mtp_active`. Pre-draft budget gate caps drafted slots at `max_batch_tokens/(k+1)` in FairQueue age order |
| P1-5 | `serve_engine.rs` | Continuation admits reset `last_published_boundary` to 0: every turn re-took cache refs on already-published pages — orphaned refs stranded pages away from the free list (monotonic pool drain) | Published boundary persisted on the session (`Session::published_boundary`), seeded from reuse at admit, clamped at `begin_turn`, reset on swap/cold |
| P1-6 | `serve_engine.rs` | Plain-AR slots never set `decoding`: decode rows were classified as prefill (FairQueue Phase-1 never applied) and generated pages published mid-flight (C6 violation) while the Done path never published | `decoding=true` set at the prefill→decode transition; Done-time `publish_generated_prefix` on the plain-decode terminal; mid-step loop skips decoding slots; vision-guard (`rope_delta != 0`) on all publication sites |
| P1-7 | `grammar.rs` | Dead-end wedges: one-byte-deep pruning let doomed continuations commit (nested numeric enum divergence, nested fraction under `integer`, nested minItems/maxItems, `additionalProperties:false` unknown keys, duplicate keys) then refused completion forever — burn to max_tokens then reject | Scan-level flags (`unknown_key`, `required_missing_on_close`, `min_items_unmet_on_close`, `array_over_max`, `number_dead_end`) are typed errors at the token that proves the dead end; structural pruning refuses `,` after all keys of a closed object, refuses `}` under unmet `required`, forces items under `minItems`, forces close over `maxItems`; key-string filter prunes characters that diverge from unused known keys; nested numeric dead ends (fraction under `Integer`, impossible `Const`/`Enum` targets) classified BEFORE the phase flip |
| P1-8 | `grammar.rs` | `is_token_allowed` at a number/literal tail was O(vocab × output_len): clone + full rescan + reparse per candidate (measured 135× slower than adjacent positions) | Number-tail fast path (pure continuation bytes at Any/Number/Integer positions — dot-free under `Integer`) plus an exact structural-byte refusal at literal tails; the O(n) simulation remains the authority for everything else |
| P1-9 | `http.rs`/`slots.rs` | Terminal-ack wait had no deadline (a zero-window client pinned the admission permit forever); engine-originated queue rejections surfaced as 500, never the spec's 429 | Terminal ack bounded by `stream_stall_timeout`; queue-full/timeout/cancelled rejections classified `overload` daemon-side and mapped to 429; refusal messages mapped to 400 |
| P2-1 | `serve_engine.rs`/`serve_wait.rs` | Waiter deadlines denominated in ticks (1 tick ≈ 1 ms assumption false in both directions); new Submits admitted ahead of queued waiters; re-enqueue after a failed retry slid the deadline forever; no cancellation for queued work | Wall-clock `Instant` deadlines (`queue_timeout_ms`); newcomer parks when the room is non-empty; original admission age preserved across retries (`pending_repark_age` handoff); `EngineCommand::CancelWaiting` + `SubmitRequest::request_tag` wired from the daemon abort path |
| P2-2 | `serve_engine.rs` | Dead pub-ref scheme: refs taken by the caller before insert were orphaned whenever the index declined to adopt pages | Cache-ref ownership moved INSIDE the index (one ref per node slot, released on rollback/eviction/release_all); callers only `seal`; orphaned-ref class eliminated structurally |
| P2-3 | `prefix_index.rs` | No ceiling on retained KV page bytes (only checkpoints were bounded) | `PrefixIndex::max_retained_bytes` enforced at insert with leaf eviction; `retained_bytes()` observable; wired to `serve.prefix_cache_max_bytes` |
| P2-4 | misc | `seal()` missing `check_phys`; schema-cache key `unwrap_or_default()` collision; 24 stats `expect`s poisoning callers' threads; sequential-VL silently dropping penalties/min_p; mid-string `<think>` un-masking; jump-forward planning during think; dead role chunk (compiler-verified); idle-eviction TOCTOU; stale demo examples; bench attempt-id reuse across repetitions; `AdmissionGuard` doc drift; CLI/compiler `items`-bool drift; huge-int const f64 confusion | All fixed (small, each with tests where host-testable) |

New regression tests: 9 saddle-core tests (nested max/minItems pruning, AP:false
key pruning + close forcing, duplicate-key pruning, malformed keyword values,
huge-int const, number-tail soundness). Verification:

- `cargo test --workspace --lib`: 39 suites green.
- GPU compose oracle (gfx1101 container): greedy cold/warm/branch 0→256→256,
  sampled deterministic, grammar JSON, A10 (long-generate + forced
  full-reject), A13 (mixed + concurrent samplers), A20 soak free_pages flat
  (20,20,20,20) — the continuation-leak fix's direct evidence — reset →
  cold → re-warm 256.
- A19 fault cells: publish-fault honest miss; HIP upload/launch/sync typed
  rejection + same-engine exact recovery.
- Live `hipfire-serve` (Ornith-1.5-9B MQ4, multi-slot 2, prefix cache on,
  q8): chain cached_tokens 0→40→83→126 (restarted engine: 0→89→192→265→346
  canonical), radix cold 0/warm 256/branch 256, strict json_schema returns
  schema-valid JSON with `finish=stop` under the framing-aware cursor
  (text-based `</think>` detection added: a tokenizer without the special
  close id no longer defers the mask forever), typed refusals 400, 6-vs-2
  concurrent all-complete.
- hipfire binary md5 `dd8e629baa6da05fc7bed9de92141d30`; daemon md5
  `dac823c79ed4354aa1a26a60f3910685`.

Residual (documented, out of scope): FairQueue remains an eligibility mask
rather than the allocation authority (verify rows pre-subtracted; forced rows
ride prefill); `ServeCapacityAccount` stays an unwired (correct, tested)
primitive; key-filter pruning covers AP:false objects only (duplicate unknown
keys under open objects still fail at completion); engine-side stall skip on
a bounded event channel remains open (CLI-side byte bound + deadline are
enforced).

### Wave 12 — parallel-generation verification campaign (2026-09-12)

A proper two-slot parallel load verification (two concurrent ~4k-token-prompt
+ 1600-token-generation requests, ~200 of 782 KV pages under load, live
Ornith serve) exposed TWO real concurrency bugs that every prior test missed
— the earlier "concurrency" cells used trivial generations that completed
inside a single scheduling regime:

| # | Layer | Bug | Fix |
|---|---|---|---|
| 1 | `admission.rs` | Sessions were granted their FULL context cap (`cap_tokens` = 50k tokens ≈ 1.7 GiB of KV credit). After trunk weights, the second concurrent session's grant did not fit the budget, so every request parked behind the resident one until its session CLOSED — the two slots were effectively serialized for long generations despite free pages and free rows on both slots. | Request-sized grants (spec §5.1: "reserve credits for the request's maximum remaining target growth through `prompt + max_tokens`"): the engine grants `prompt + max_tokens` clamped to the cap; new `AdmissionController::resize` (verify-then-mutate) grows/shrinks the grant per continuation turn BEFORE any mutation, fail-closed on budget refusal. |
| 2 | `serve_fairness.rs` | `starved_oldest` counted any request with `uncached_prefill_tokens > 0` as "unserved" — a multi-tick prefill is permanently "unserved" by that test, so starvation latched for the WHOLE prefill and the engine's bounded-backfill mask restricted eligibility to only the oldest request. A decoding slot + a prefilling slot serialized at the scheduler even after fix 1 (trace: the prefilling session contributed zero rows for 43 s, then ran full chunks the instant the decoder finished). | "Unserved" now means received NOTHING this step; a request with a grant this step is being progressively served. The legitimate bounded-backfill case (oldest feasible request with zero service) still fires (regression-tested both ways). |

Also: the container's default 30 s wall-clock queue timeout (wave 11 made it
honest) surfaced as a 429 for a request queued behind a 75 s generation —
deployment knob, not a bug; raise `HIPFIRE_SERVE_QUEUE_TIMEOUT_MS` when
long queue waits are expected.

Verification (live `ornith-1.5-9b-mq4-multislot-hipfire`, 2 slots, prefix
cache on, both fixes in):

- Slot trace: both sessions ADMIT 0.2–0.6 s apart; prefill publications
  INTERLEAVED across slots within the same scheduler step
  (`2048` on slot 0 and `1024` on slot 1 at the same timestamp); both
  slots decode with MTP and retire adaptively independently.
- Parallel pair (12.5k KV tokens each): 62.2 s vs 88.7 s sequential
  (1.43×); aggregate decode throughput 51.6 tok/s vs ~37 solo
  (bandwidth-bound batching).
- Coherence/isolation: both parallel outputs are coherent on-topic essays
  with zero cross-slot topic bleed.
- Soak: 3 fresh cold parallel rounds, drift 7.8%, no 429s, retries=0,
  queue drains to 0 — no slot wedge or page-leak signature.
- `cargo test --workspace --lib`: 39 suites green (incl. new
  `resize_grows_within_budget_and_refuses_beyond`,
  `resize_shrink_returns_credit_and_unknown_session_refuses`,
  `request_sized_grants_admit_a_second_session_a_full_cap_grant_would_block`,
  `partial_prefill_grant_does_not_latch_starved_oldest`).

---

## What was done

### Waves 0–4 (prior commits, `6b07ef415`–`f3222d63d`)

| Wave | What landed |
|---|---|
| 0 merge | Folded `feat/multislot-vision-mtp` ViT layer-chunking (no D2H/H2D roundtrip) into the branch base |
| 1 P0 | `serve_contract` types (CacheDomain, PrefixLookup, DrafterDecision); multi-inflight bench client API; daemon slots benchmark scaffold |
| 2 P1 | `PagePool` generations, sealing, COW, leases, deferred reclaim; `SlotPool` integration; bounded admission with physical/future budget accounting |
| 3 P2 | Token-keyed radix `PrefixIndex` with domain isolation; Qwen `DeltaNetSnapshot` checkpoint pool + resume planner; prefix-cache lookup → pin → resume-plan in the admit path; sealed-page publication at commit boundaries |
| 4 P3 | Global `max_batch_tokens` budget; prefill quantum; per-slot COW isolation (S4); `FairQueue` age-aware scheduling with deficit ring; bounded `WaitQueue` (replaces R-A4 reject); `forced_token_run` planner for JSON Schema (default off) |

**Defaults:** `serve.prefix_cache=false`, `serve.structured_jump_forward=false`, `serve.scheduler_overlap=false`. All off.

### Wave 5 — P5 composition (this session, `fd3364798`–`aeeda2259`)

#### C1 cache-domain identity (`fd3364798`)

Replaced zeroed tokenizer/template stubs and file-size proxy with real SHA-256 digests:

- `Tokenizer::vocab_digest()` — SHA-256 of decoded vocabulary in id order
- `Tokenizer::config_digest()` — SHA-256 of merge ranks, special tokens, BOS/EOS/EOT, BPE/SentencePiece flags
- `HfqFile::content_digest()` — SHA-256 of arch_id, metadata JSON, ordered tensor manifest, file length (path/inode/mtime excluded)
- `serve_contract::sha256_len_prefixed()` / `sha256_file()` — shared digest helpers
- Sidecar digests (MTP, VL) hashed independently via `find_mtp_sidecar` + `HfqFile::content_digest` or `sha256_file`

#### Token-keyed page handles (`fd3364798`)

- `PrefixIndex::lookup_with_pages()` returns the actual sealed `Handle` list along the matched prefix, not a length-keyed side map
- Two equal-length different prefixes return distinct physical pages (`lookup_with_pages_isolates_equal_length_prefixes` host test)
- Removed `published_handles: HashMap<u64, Vec<Handle>>` side map from `Rig`
- Admit path shares `hit_handles` directly into the destination slot via `share_published_pages`

#### Exact-match reuse (`fd3364798`)

- `reused_tokens_from_plan(boundary, prompt_len)` now returns `boundary.min(prompt_len)` — a boundary covering the entire prompt is a full hit, not zero

#### MTP sidecar discovery (`3803a0573`, `aeeda2259`)

- `mtp_sidecar_candidates()` / `find_mtp_sidecar()` in `crates/hipfire-arch-qwen35/src/mtp_head.rs` — probes both `with_extension("mtp")` (last extension) and the stem after stripping `.hfq` + quant suffixes (`.mq4v2`, `.mq4`, …), so `qwen3.5-4b.mq4v2.hfq` finds `qwen3.5-4b.mtp`
- Slot engine (`serve_engine.rs`): both bundled-miss and bundled-error fallbacks use `find_mtp_sidecar`; C1 sidecar identity uses `find_mtp_sidecar`; miss message lists all probed paths
- Single-slot loader (`hipfire-loader/src/lib.rs`): both fallbacks use `find_mtp_sidecar`; error message lists probed paths

#### GPU oracle (`3803a0573`)

`crates/hipfire-runtime/examples/test_serve_prefix_cache.rs` — compose oracle with `--mtp-k N`:

| Cell | Result |
|---|---|
| Greedy MTP cold | reused=0, generated >0 |
| Greedy MTP warm identical | reused=256, tokens match cold |
| Greedy MTP branch (divergent suffix) | reused=256, tokens differ from Italy |
| Sampled AR (temp 0.8, seed 7) | reuse 256/256, deterministic (same seed → same tokens) |
| JSON-Schema AR | reuse 256, output parses as `{"city": "Rome"}` |
| A20 soak (4× warm) | reuse 256 each, tokens match cold |
| A20 reset | `SlotEngine::reset` → reused=0 (allocation epoch bump drops radix) |
| Stats | admitted=11, reused_tokens=2304, prefix_hits=0 |

#### serve_harness chain (`fd3364798`)

`serve_harness.py --mode chain --no-spawn --port 11524 --sampling greedy --max-tokens 48 --max-think-tokens 0`:

```
cached_tokens: 0 → 89 → 192 → 265 → 346
```

#### Single-slot MTP load (`aeeda2259`)

`hipfire run --spec mtp` on gfx1101 logs:

```
MTP head loaded (sidecar /home/ghazni/.hipfire/models/qwen3.5-4b.mtp): n_embd=2560 vocab=248320
qwen35 MTP speculator enabled (compressed-serial, K=3)
```

The subsequent generate fails with an open-think-span validator (too few tokens for the think contract) — unrelated to sidecar discovery.

### Host tests

| Suite | Count | Status |
|---|---|---|
| `sidecar_probe_tests` (mtp_head) | 3 | ok |
| `mtp_sidecar_probe_tests` (loader) | 1 | ok |
| `cache_identity_digest_tests` (tokenizer) | 2 | ok |
| `lookup_with_pages_isolates_equal_length_prefixes` (prefix_index) | 1 | ok |
| `should_lookup_prefix` policy (serve_engine) | 3 | ok |
| `reused_tokens_from_plan` policy (serve_engine) | 4 | ok |

### Documentation

- `docs/CONFIG.md` — 7 new `serve.*` keys documented (prefix_cache, prefix_cache_max_bytes, structured_jump_forward, scheduler_overlap, max_batch_tokens, prefill_min_tokens, max_queue_bytes)
- `docs/SERVE.md` — experimental prefix-cache section (opt-in only, not admitted)
- `docs/specs/2026-09-05-serving-cache-scheduler-plan.md` — §0 progress table updated with wave 5 evidence

### P6 overlap

`serve.scheduler_overlap` is a registered config flag only (`hipfire-config`, default false). No host-prep overlap path exists in the engine — there is no CPU gap to measure. Per spec §10: leave off.

---

## What still exists as gaps

### Not admitted

- **No `admissions.yml` row.** Spec truth state remains **planned**. Completion of code ≠ admission, default-on, or product claim.
- **Defaults stay off:** `serve.prefix_cache`, `serve.structured_jump_forward`, `serve.scheduler_overlap` all `false`.

### Vision + prefix reuse (X2 / A18)

- `should_lookup_prefix(prefix_cache, has_visual) = prefix_cache && !has_visual` — vision requests skip the radix entirely.
- Needs a pixel/embedding/position identity oracle before vision+prefix can be enabled.
- No VL model oracle has been run on the slot engine with prefix cache.

### Slots tools / stop / logprobs

- `validate_generate_caps` in `crates/hipfire-daemon/src/slots.rs` still refuses:
  - `stop` (non-null) → rejected
  - `logprobs` / `top_logprobs` → rejected
  - (`tools` was removed from this list by upstream `ad6004ac0`: tool turns are
    supported on the slot path now — a tool-call finish emits `tool_calls`.)
- Spec: remove refusals only when complete behavior lands in the same change.

### A19 — HIP fault injection / fail-closed reuse

- **Closed (wave 10):** `HIPFIRE_FAULT_HIP=<upload|launch|sync>` injects below the HIP bridge; `--fault-hip=<class>` proves typed rejection, no fake Done, sync poison → fresh-engine byte-identical recovery. All 3 classes PASS on gfx1101.
- Pool/engine-seam fault path landed earlier: `HIPFIRE_FAULT_PREFIX_PUBLISH=1` fails the first publication; `--fault-publish` proves honest miss + unchanged output + no cache-ref leak.

### A20 — Full lifecycle soak

- Landed: 4-hit warm soak, reset → cold (reused=0), **re-warm after reset reuses again** (allocation epoch drop + cache-lease release), post-reset output matches the original greedy run.
- **Closed (wave 10):** `--a20` cell covers idle spill/restore (evictions=1 → restores=1, reused ≥ prompt) and two shutdown+respawn model-swap cycles (cold→warm, stable `free_pages`). PASS on gfx1101.

### A10 — MTP page-crossing / partial-accept cache-visibility

- Landed: 96-token MTP generation (crossing generated-page boundaries with verify/repair writes) replays identically from cache on the warm run — candidate rows are provably not cache-visible; non-aligned prompts exercise SuffixRecompute against the capture-boundary guard.
- Still open: an explicit forced full-reject sweep (prose prompt driving τ→1.2) asserted against cache visibility.

### A13 — Mixed-load progress bound

- Landed: long cold prefill submitted before a short warm request — both complete; cold miss stays a miss (no false reuse).
- **Closed (wave 10):** adversarial phase now mixes samplers (2 greedy + sampled + penalized, distinct convos) with a per-request 120 s wait-bound assertion. PASS on gfx1101.

### P6 — Scheduler overlap

- Flag only, no implementation. Stays off until a measured host-gap exceeds fixture noise.

### `admissions.yml` / `ARCHITECTURE.md` / changelog / default-on

- Blocked until every advertised cell (including vision) has its oracle and product evidence.

---

## Commit history (this branch, newest first)

```
a2e013b6e fix(slots): restore batched-path slot release; add wave-9 verification cells
2390e4e0f fix(slots): wave-8 residual audit — pin leak, greedy gates, bare objects, maxItems
890669a6d fix(scheduler): page-align prompt-completing prefill rounds so checkpoints are capturable
4fbd66c3e docs: serving cache scheduler work summary and remaining gaps
c88347efe fix(slots): review pass — ownership, fairness, grammar, and backpressure hardening
aeeda2259 fix(loader): probe stem .mtp sidecars like the slot engine
3803a0573 feat(slots): probe stem .mtp sidecars and compose prefix reuse
fd3364798 feat(slots): C1 content digests and GPU-verified prefix reuse
bffd2d9a9 docs: record wave 0–4 landing on serving-cache-scheduler
f3222d63d feat(slots): bounded wait on busy slots and optional jump-forward
e94572339 feat(slots): enable strict JSON Schema on the multi-slot path
6679deddf feat(runtime): bounded waiting-room policy for slot admission
9edd7e4c3 feat(grammar): conservative forced-token-run planner for JSON Schema
802dfca02 feat(slots): consult FairQueue for per-tick eligibility
df1efcf9c feat(slots): global row budget, prefill quantum, and per-slot COW isolation
4e2ed2423 feat(runtime): age-aware fair queue policy for slot scheduling
21f202f3d feat(slots): wire prefix cache lookup, COW barrier, and publication
d4faba38b feat(qwen35): hybrid-state checkpoint pool and resume planner
0f648d091 feat(runtime): token-keyed prefix radix index with domain isolation
3f024ec4c feat(slots): response_format contract, lossless token bytes, pre-sample mask seam
9d7110d46 feat(bench): multi-inflight client API and daemon slots benchmark
f6904c85b feat(serve): freeze cache/scheduler contracts and bound admission
b38e56b65 docs: serving cache scheduler spec and implementation plan
b6b4a6a30 feat(pages): generations, sealing, COW, leases, deferred reclaim in PagePool/SlotPool
6b07ef415 merge: fold feat/multislot-vision-mtp ViT layer-chunking into serving-cache-scheduler base
```

---

## Key files touched (this session)

| File | Change |
|---|---|
| `crates/hipfire-runtime/src/serve_contract.rs` | `sha256_len_prefixed`, `sha256_file`, `Digest` type |
| `crates/hipfire-runtime/src/tokenizer.rs` | `vocab_digest()`, `config_digest()`, digest tests |
| `crates/hipfire-runtime/src/hfq.rs` | `content_digest()` |
| `crates/hipfire-runtime/src/prefix_index.rs` | `lookup_with_pages()`, `Handle` collection in walk, equal-length isolation test |
| `crates/hipfire-runtime/Cargo.toml` | `sha2 = "0.10"` dep, `test_serve_prefix_cache` example registration |
| `crates/hipfire-runtime/examples/test_serve_prefix_cache.rs` | GPU compose oracle (greedy/sampled/grammar/soak/reset) |
| `crates/hipfire-arch-qwen35/src/mtp_head.rs` | `mtp_sidecar_candidates()`, `find_mtp_sidecar()`, sidecar probe tests |
| `crates/hipfire-arch-qwen35/src/serve_engine.rs` | C1 real digests, `lookup_with_pages` admit path, `find_mtp_sidecar` for slot engine + C1 sidecar identity, `reused_tokens_from_plan` exact-match fix, `published_handles` removal |
| `crates/hipfire-loader/src/lib.rs` | `find_mtp_sidecar` for single-slot loader fallbacks, probed-path error message, loader stem probe test |
| `docs/CONFIG.md` | 7 new `serve.*` keys |
| `docs/SERVE.md` | Experimental prefix-cache section |
| `docs/specs/2026-09-05-serving-cache-scheduler-plan.md` | §0 progress table with wave 5 evidence |

### Wave 8 — residual fixes (2026-09-06)

Six residual bugs found by a follow-up audit and fixed:

| # | Area | Bug | Fix |
|---|---|---|---|
| 25 | `serve_engine.rs` | Pin leak on normal completion — `lookup_with_pages` pins the radix path; `unpin` was only called on miss and `ClientGone`. Every successful prefix-cache request leaked a pin, preventing eviction of the matched path. | `idx.unpin(domain)` added to the normal completion path (Eos/MaxTokens) after `publish_generated_prefix` |
| 26 | `serve_engine.rs` | `request_sampled` gate (`temperature > 1e-6`) disagreed with the sampler's greedy gate (`temperature == 0.0`). Temperatures in `(0.0, 1e-6]` were sampled but classified as greedy by `request_sampled`, enabling MTP with a greedy verify that can't reproduce the sampled pick. | Gate aligned to `temperature > 0.0` (matches sampler's `== 0.0` exactly) |
| 27 | `grammar.rs` | `compile_object` rejected valid `{"type": "object"}` (no `properties`) as "object schema missing properties". A bare object type is valid JSON Schema meaning "any object". | Absent `properties` treated as empty vec; `additionalProperties` and `required` handling already work with empty properties |
| 28 | `serve_engine.rs` | Both publish sites constructed `PageHandle` with `epoch: 0` instead of the pool's actual epoch. | `epoch: rig.pool.page_pool().map(|p| p.epoch()).unwrap_or(0)` at both sites |
| 29 | `serve_engine.rs` | `InFlight.reused_tokens` set to `0` for continuation admits, but continuation hits do reuse tokens (`plan.reused`). Stats undercounted. | `reused_tokens: plan.reused` |
| 30 | `grammar.rs` | `maxItems` not enforced incrementally — the matcher allowed emitting item starts beyond `maxItems`, creating dead ends that only surfaced at terminal validation. The `Frame::Array.count` field existed but was dormant. | `count_root_array_items` helper scans the raw buffer in the EOF branch of `parse()`; when the root schema is `Array { max_items: Some(N) }` and the item count exceeds N, the matcher errors immediately |

Regression tests added (`saddle-core --lib`):

| Test | What it guards |
|---|---|
| `bare_object_type_without_properties_compiles` | `{"type":"object"}` compiles and accepts any object |
| `bare_object_type_with_required_only_compiles` | `{"type":"object","required":["name"]}` compiles and accepts `{"name":"x"}` |
| `max_items_blocks_extra_item_incrementally` | 3rd item after `maxItems:2` errors before array close |
| `max_items_allows_up_to_limit` | Items up to `maxItems:3` accepted; 4th errors |
| `max_items_empty_array_ok` | `maxItems:0` accepts `[]`, rejects `[1` |
| `count_root_array_items_helper` | Unit test for the byte-scanning helper (nesting, strings, commas) |

Verification: `cargo test --workspace --lib` — all suites green (191 saddle-core, 261 hipfire-arch-qwen35, 0 failures across workspace).

### Wave 9 — end-to-end verification pass (2026-09-06)

Full verification campaign on gfx1101 / ROCm 10 container (GPU shared with a
co-tenant workload holding ~7.75 GB; serve verified at 2 slots × 2048 ctx —
the default 4×8192 shape does not fit alongside the co-tenant and was not a
gate). Host: `cargo test --workspace --lib` green under `HIPFIRE_OOM_GUARD=1`
(the `oversized_pool` case passes with the guard; the 7 bin-level
`update_*` self-installer failures are documented environmental).

#### Found and fixed

| # | Area | Bug | Fix |
|---|---|---|---|
| 31 | `serve_engine.rs` | Wave-8 fix 25's edit **deleted** `slots[s] = None` at the run_loop batched-completion path instead of inserting beside it — every MTP-path termination wedged its slot, and `reset` always rejected "requests in flight" (caught by the A20 reset oracle cell) | Restored the clear beside the unpin; `commit_sampled_token` (sequential path) had kept its own |

#### New verification cells (all GPU-PASS)

| Cell | Mechanism | Evidence |
|---|---|---|
| A10 forced full-reject | `HIPFIRE_FAULT_MTP_FULL_REJECT=1` seam in `mtp_batched_verify_accept_from_batch` forces τ=1 every cycle | Generated sequence equals accepting-MTP greedy; warm replay identical — rejected candidate rows provably never cache-visible |
| A13 concurrent | 4 simultaneous requests (3 warm + 1 cold) against 2 slots | WaitQueue losers complete with byte-exact outputs (1.2–4.4 s) |
| A20 plateau | `EngineStats.pool_free_pages` sampled per step | Soak free pages flat (10,10,10,10); reset returns to 26 — no leak across the lifecycle |
| A19 fault-hip | `HIPFIRE_FAULT_HIP=upload\|launch\|sync[:N]` injects the first bridge-call failure after arming (arm-after-load semantics) | launch/upload → typed rejection + same-engine exact recovery; sync (device_synchronize) → engine poison + fresh-engine recovery; all byte-identical (spec §5.4 S4) |

Oracle pool note: the compose oracle moved to `cap_tokens: 2048` (32 pages).
At 16 pages, wave-8's pin-release fix made cached paths evictable again and
the A13 publications legitimately evicted the Italy prefix — correct §4.4
behavior that the soak's `reused>=PAGE` assumption collided with. The suite
now tests steady-state boundedness, not incidental eviction.

#### Harness + product-route evidence

- `serve_harness.py --mode chain` (multi-slot + prefix-cache serve, HTTP):
  `cached_tokens 0→89→192→265→346` reproduced on the final binary — these
  are single-chunk prompts, the case `890669a6d` (page-aligned prefill
  rounds) revived.
- `battery` complete (5 distinct prompts, cached=0 correct); `session`
  complete (8 turns to ctx 1012, `cached_tokens=977` on the final turn; the
  retrieval gate needs `--max-tokens ≥ 512` for visible answers).
- Typed-refusal probes over HTTP: unsupported schema keyword, `json_object`,
  `tools`, `stop`, `logprobs` → all typed rejections, per contract.
- `hipfire bench --concurrency 2 --backend both`: **slots arm runs** on the
  daemon protocol (2 slots, 0 rejected; 33.2 agg tok/s vs noslots 37.3 at
  k=2 — P0 exit criterion met; not a perf claim).

#### New gaps (this pass made them precise)

- **Strict `json_schema` over ChatML fails closed** (NEW): a *valid*
  supported-subset schema via `/v1/chat/completions` is typed-rejected with
  "grammar constraint allows no token" — the reasoning (think) framing
  requires tokens the schema forbids before JSON can start. The engine-level
  raw-prompt schema path (oracle grammar cell) works. Follow-up: §7.2's
  framing-aware grammar cursor (schema masks apply to the post-think answer
  span). Never falsely succeeds — fail-closed.
- **Admission vs forward page-accounting mismatch under pressure**: with a
  resident session + published cache near pool capacity, admission granted a
  request the forward could not back ("PagePool: need 8 more pages but only
  0 free") → forward-level typed rejection instead of admit-level
  wait/reclaim/reduce (spec §5.4 S4 row 3). Fail-closed; small requests
  unaffected. Follow-up: wire admit-time eviction to the forward's actual
  page demand or park on shortfall.
- **Legacy batch (beta) arm**: 2/2 requests rejected at k=2;
  `kv_cache_write_q8_0_independent` fails JIT compile on gfx1101
  (sequential/batch prefill kernel variant — the slots path uses the paged
  kernels and is unaffected). Pre-existing dispatch-fallback class; needs
  triage but is outside this spec's X2 matrix.
- A failed multi-page allocation in the daemon's slot route may briefly hold
  freed pages in reclaim-pending (free count drifts 0→2→5 between retries)
  before `drain_completed` retires them; all counts recovered.
