# Multi-slot engine: continuous batching + paged KV + MTP + vision, together

- **Date:** 2026-09-03
- **Branch:** `feat/multislot-vision-mtp` (audit base: a448eb627 + working tree)
- **Status:** branch-implemented / gap analysis attached (§8)
- **Validation model:** `Qwen/Qwen3.5-4B-MQ4` — trunk `qwen3.5-4b.mq4`,
  MTP sidecar `qwen3.5-4b.mtp`, vision sidecar `qwen3.5-4b.vl` (one model
  carrying all three features is the point of the fixture).

## 1. Goal

One serving engine — the multi-slot `SlotEngine` — where four capabilities
operate **simultaneously and correctly**:

1. **vLLM-style continuous batching**: requests join and leave a rotating
   batch without draining in-flight work; chunked prefill mixed with decode.
2. **Paged KV**: per-slot logical context over shared physical pages
   (block-table addressing), allocated on demand up to a page budget.
3. **MTP speculative decode**: vLLM-style batched draft/verify inside the
   same rotating batch, including under paged KV.
4. **Vision (VL)**: image requests prefilled through the same batched path
   (external embeddings + M-RoPE), including under paged KV, and batched
   alongside MTP and plain text slots.

The branch's earlier work built these mostly separately, with feature
combinations gated off as they interacted badly (SP-gate history:
59c731dd0 refused paged×MTP; e8b9bc6af dropped penalties). This spec
defines the target behaviour as checkable requirements, so the remaining
combinations can be closed instead of gated.

### Non-goals

- Preemption of *active* (generating) sessions, request queueing beyond the
  daemon's bounded worker pool, priority scheduling.
- Copy-on-write page sharing / prefix caching under paging beyond the
  full-page fork contract (`share_prefix`), defragmentation.
- Multi-image requests; sampled (non-greedy) MTP acceptance on the slot
  engine; MoE MTP heads on the batched head path.
- CUDA-graph-style capture **under paged** (plain launches; capture stays
  available for the legacy flat pool).

## 2. System overview

```
HTTP /v1/chat/completions, /generate
        │  (hipfire-daemon: main.rs worker per request, slots.rs handle_generate)
        ▼
SubmitRequest {prompt_tokens, visual_data?, sample params, continuation}
        │  EngineCommand::Submit (mpsc)
        ▼
┌────────────────────────── SlotEngine thread ──────────────────────────┐
│ (exclusive GPU owner)                                                 │
│ run_loop step:                                                        │
│   1. drain commands (new requests join mid-decode)                    │
│   2. [vl_sequential only] per-token VL path                           │
│   3. MTP draft: k serial head steps per decoding MTP slot             │
│   4. VL tower: vision_forward once per new image request              │
│   5. Scheduler::next_batch → SlotBatch (+ MTP verify-row injection)   │
│   6. forward_batch_slots_graphed_opts (paged ⇒ plain path)            │
│   7. sample_per_slot (greedy argmax / penalize-then-argmax / top-p)   │
│   8. MTP verify/accept per draft slot (lm_head view + DN tape repair) │
│   9. MTP head prefill fill per prefilling MTP slot                    │
│  10. commit tokens per slot (EOS/max_tokens/ctx-cap/client-gone)      │
└───────────────────────────────────────────────────────────────────────┘
   │                        │                      │
   ▼                        ▼                      ▼
SlotPool (+PagePool)   DeltaNetState[n]      MtpSpecState[n] (private
(block tables/descs)   (per-slot recurrent)  head KV, DN snapshot)
```

Key files: `crates/hipfire-arch-qwen35/src/{serve_engine,scheduler,
slot_batch,forward_slots,mtp_spec,mtp_head}.rs`, `crates/rdna-compute/src/
{slot_pool,page_pool,kv_slots,sampling}.rs`, `kernels/src/kv_slot_desc.h`,
`crates/hipfire-daemon/src/{slots,main}.rs`, `crates/hipfire-arch-qwen35-vl/`.

## 3. Requirements

Each requirement is numbered for the gap analysis in §8. MUST items are
correctness; SHOULD items are behaviour quality.

### A. Continuous batching

- **R-A1 (MUST)** New requests join mid-decode: commands are drained at the
  top of every engine step; the next batch mixes the newcomer's prefill
  chunk with other slots' decode rows. A fully idle engine blocks on recv
  instead of spinning.
- **R-A2 (MUST)** Chunked prefill: a prefilling slot contributes at most
  `prefill_chunk` rows per step; decoding slots contribute one row each.
  A long prompt must never head-of-line block a decoding slot for more
  than one chunk.
- **R-A3 (MUST)** Slot finish releases the slot immediately (`slots[s]=None`,
  work cleared); other slots continue. EOS/max_tokens sessions stay resident
  and idle (KV + recurrent state intact) for multi-turn reuse; client-gone
  closes the session and frees the slot at once.
- **R-A4 (MUST)** Admission: cold prompt `len < cap` else rejected with an
  actionable message; multi-turn continuation is a strict token extension
  matched by user-turn hash (or tool-result re-entry), metadata-only rewind
  to the reuse point; extension past cap is rejected and the session closed.
  When no slot is free, an LRU **idle** session is evicted to host swap and
  restored on its next turn; if none is evictable the request is rejected
  ("all slots busy"). Active sessions are never preempted.
- **R-A5 (MUST)** Per-slot sampling: temperature/top_p/top_k/seed/min_p and
  repeat/presence/frequency penalties are honoured per slot. All-greedy
  unpenalized steps take the batched argmax path; penalized slots take a
  penalize-then-argmax prepass; sampled slots take per-slot top-p sampling
  with per-slot RNG state.
- **R-A6 (MUST)** Failure isolation: a per-slot error rejects only that
  slot; a forward error rejects every active slot (never a silent empty
  200) and leaves no poisoned KV/DN behind; a HIP sync/upload failure
  poisons the engine and fails every active request loudly.
- **R-A7 (MUST)** Finish conditions per committed token: EOS terminates and
  is never emitted; `max_tokens` terminates with the terminal token retained
  in the session store (so the next turn's strict extension sees it);
  client-gone releases everything; `sess.tokens.len() + 1 >= cap` stops
  decode before the KV overflows.

### B. Paged KV pool

- **R-B1 (MUST)** Opt-in (`HIPFIRE_SLOTS_PAGED=1`, page-count override
  `HIPFIRE_SLOTS_PAGED_PAGES`, default = legacy-equivalent page count).
  Page size 128 tokens; one K arena and one V arena per attention layer;
  LIFO free list; per-physical-page refcounts; preflight refuses
  over-budget arenas.
- **R-B2 (MUST)** Per-slot block tables (`page_indices` + `live_tokens`)
  uploaded to per-slot device staging sized `ceil(cap/128)` entries; a table
  that outgrows staging fails the step with a loud error, never a
  out-of-bounds write. Upload order per step: block tables **before** the
  descriptor table (activation mutates it).
- **R-B3 (MUST)** Write-frontier provisioning: before the KV write every
  active slot is provisioned to `set_seq_len(max_pos+1)` for the step, so
  the write frontier has a page; pool OOM at provision fails the step
  closed **before any GPU write**.
- **R-B4 (MUST)** Device descriptor ABI is 32 bytes (`block_table: u64,
  legacy_k_base: u64, legacy_v_base: u64, seq_len: i32, page_tokens: i32`);
  the host staging buffer for the table is `n_slots × 32` bytes. Paged
  descriptors carry `page_tokens=128`, zero legacy bases, `seq_len =
  live_tokens`; legacy descriptors carry null `block_table` and per-slot
  slab bases.
- **R-B5 (MUST)** Shrink keeps pages (only `live_tokens` drops); grow
  allocates the page delta or errors; the per-slot cap is enforced
  host-side in `set_seq_len`.
- **R-B6 (SHOULD)** Prefix sharing is the CoW-free full-page fork contract:
  share only full pages of a source into an **empty** destination;
  violating it is silent corruption by contract, so no engine path may
  share except at that contract.
- **R-B7 (MUST)** Under paged, hipGraph decode/verify capture is bypassed
  (plain step), and the single-slot WMMA slab-reduction fast path is
  excluded; the descriptor-driven batched kernels are the only KV
  addressing path.

### C. MTP (multi-slot, batched verify)

- **R-C1 (MUST)** Head loading: bundled trailer first, then `<stem>.mtp`
  sidecar; a miss while MTP is requested logs loudly and serves AR; the
  head is shared across slots, per-slot private state (head KV, DN
  snapshot, prev_hidden) allocated lazily.
- **R-C2 (MUST)** Cycle: draft = k serial head steps from the slot's seed;
  the scheduler emits no rows for a decoding MTP slot; the engine injects
  `k+1` verify rows (`[seed, c1..ck]` at `next_pos..next_pos+k`) into the
  batched forward; lm_head over the slot's verify rows (view of `x_batch`)
  decides greedy acceptance; committed tokens (accepted drafts + bonus)
  advance `next_pos` and `set_seq_len`.
- **R-C3 (MUST)** Partial accept repairs only the private DeltaNet state by
  replaying the per-layer verify tape over the accepted rows (Q8 state
  only); the trunk KV is untouched — accepted rows were written at their
  positions by the verify itself, and rejected rows are stale-but-covered
  (every later write is position-ordered, and kernels bound causal sweeps
  by `positions[]`, never `seq_len`).
- **R-C4 (MUST)** MTP prefill goes through the batched chunked path; after
  each chunk the head's private KV is filled from that chunk's `x_batch`
  rows (single batched head call), the chunk's last normed row becomes
  `prev_hidden`, and when the prompt drains the step's sampled token is the
  first draft seed.
- **R-C5 (MUST)** Gates: penalized requests decode AR (verify acceptance
  cannot reproduce penalize-then-argmax); VL requests never activate MTP;
  `mtp_verify_fits_cap` retires MTP (keeping only the newest committed
  token) before a verify frontier could fail the provision closed; adaptive
  retire sends a slot back to AR after two consecutive 32-cycle windows
  with mean advance < 2.2.
- **R-C6 (MUST)** Paged compatibility: the MTP path contains no flat-slab
  (`slot_kv_view`) writer — prefill is the scheduler's block-table path,
  verify is the batched forward, repair touches private buffers. MTP under
  paged is **allowed** (graph capture being off under paged is the only
  consequence).
- **R-C7 (MUST)** Burst terminators: EOS/max_tokens/client-gone/ctx-cap are
  enforced on **every** committed token of an accepted burst, not just the
  last; the burst stops emitting at the first terminator.

### D. Vision (VL) on the batched path

- **R-D1 (MUST)** `.vl` sidecar discovery: `HIPFIRE_VL_FILE` override, else
  `<stem>.vl` sibling of the trunk; sidecar wins over inline tensors;
  vision config comes from the sidecar.
- **R-D2 (MUST)** Request build (daemon): single image; prompt spliced as
  `<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>`; exactly one
  contiguous pad run (multi-image rejected); M-RoPE positions built for
  every prompt token; VL forces a `Cold` continuation (no prefix reuse
  across image turns).
- **R-D3 (MUST)** Batched execution: the vision tower runs once per request
  on the engine thread; the embedding matrix is uploaded to per-slot device
  staging; the scheduler holds the slot's rows until embeddings exist;
  `embedding_scatter_ext_batched` overwrites pad rows from the per-slot
  matrix (per-row base pointers re-uploaded every step); any VL row in a
  step puts the whole step on M-RoPE (`pos3`), with text rows `[p,p,p]`
  bit-identical to 1D RoPE.
- **R-D4 (MUST)** VL decode rows: `pos3 = [pos + rope_delta; 3]`, no
  embedding splice; VL sampling and commit behave exactly like text slots.
- **R-D5 (MUST)** Sequential fallback (`HIPFIRE_VL_SEQUENTIAL=1`) keeps
  working on the legacy flat pool and is refused under paged (at rig build
  and at admit); `slot_kv_view` asserts `!is_paged()` as defence in depth.
- **R-D6 (MUST)** A prefill chunk boundary may fall inside the image-pad
  run: visual embeddings splice in prompt order across chunks, positions
  continue across chunks, and the post-image text attends to image KV rows.

### E. Feature-interaction matrix (every cell defined)

| Combination | Required behaviour |
|---|---|
| text + text (2 slots) | independent generations, no cross-bleed |
| text + VL mixed step | VL rows carry pos3/ext_emb; text rows `[p,p,p]`/`-1`; both correct |
| text(MTP) + VL in one step | MTP verify rows `[p,p,p]`/`-1`; VL slot prefills in the same step; per-feature slots unaffected |
| MTP + paged | allowed; plain forward path; frontier guard active |
| VL + paged | allowed (batched VL is the paged-compatible path) |
| MTP + VL same *request* | refused (MTP off for the image request; AR decode) |
| penalties + MTP | request decodes AR |
| penalties + VL batched | penalize-then-argmax works on VL slots too |
| sequential-VL + paged | refused at build and admit |
| anything + over-cap | rejected with actionable message, engine continues |
| pool OOM mid-step | step fails closed; all active requests rejected; engine survives |

## 4. Invariants (load-bearing, each with an enforcing site)

1. **`positions[]` is the only causal/positional authority** in `_slots`
   kernels — never `desc.seq_len` (MTP verify writes past the committed
   frontier by design). `kernels/src/kv_slot_desc.h`.
2. **One engine thread owns the GPU.** All launches, uploads, tower runs
   happen on it; the channel front-end never touches HIP.
3. **Descriptor/block-table freshness**: block tables upload before descs;
   descs upload only on the dirty flag; graph replays run the host-side
   bookkeeping (`advance_slot_seq_lens`, ext-emb pointer refresh)
   out-of-band.
4. **Stale KV is covered, never read**: writes are position-indexed; the
   newest session token may trail KV by one row; rejected MTP rows sit
   above the frontier and are overwritten before any read can see them.
5. **Fail-closed over fail-silent**: pool OOM, staging overrun, forward
   errors and sync failures reject loudly; nothing degrades to a
   successful empty response.
6. **Session ids are monotonic and never reused**; a closed session's
   slot is released exactly once.
7. **MTP head KV staleness is harmless** (v1 head attention reads only its
   own position, attention ≈ V), which is what makes head-KV rollback
   unnecessary; the trunk verify is the correctness gate.

## 5. Edge cases (each must have a defined, testable outcome)

- **E1** Prompt of exactly `cap-1`, `cap`, `cap+1` tokens (cap+1 cold
  rejected; cap-1 decodes then stops at the cap).
- **E2** KV write frontier exactly on a page boundary (positions 127/128
  crossing): block-table row 1 must exist before the write (the
  write-frontier provision).
- **E3** Prefill chunk boundary inside the image-pad run, and at
  prompt-minus-one (one-token final chunk).
- **E4** MTP verify frontier at `cap-k-1 .. cap` (guard retires to AR at
  the boundary instead of failing the step).
- **E5** MTP accept spread: 0 accepted (full reject + DN repair), partial,
  full k+1 (bonus), EOS inside the accepted burst, `max_tokens` inside the
  burst.
- **E6** `max_tokens=1` on MTP and VL requests.
- **E7** Client disconnect mid-prefill and mid-MTP-cycle (slot released,
  session closed, engine continues).
- **E8** Concurrent mixed load: 1 VL + 1 MTP slot; 2 VL; 2 MTP; VL on both
  slots then text follow-ups.
- **E9** Continuation after an MTP turn (resident: head KV extends);
  continuation after swap-restore (head KV reset); continuation at the cap
  (rejected + closed).
- **E10** Over-budget admission (prompt+max_tokens > ctx) refused daemon-side.
- **E11** Paged pool exhaustion (page budget < demand): fail closed, all
  active rejected, engine still serves the next request.
- **E12** Eviction: an idle session evicted while another slot decodes; the
  evicted conversation resumes correctly (swap restore or cold re-prefill).
- **E13** Penalized request when MTP is loaded → AR decode, penalize-then-
  argmax output (no repetition attractor).
- **E14** ≥9 slots under paged: descriptor staging sized `32×n_slots`
  (R-B4) — the allocation must not rely on allocator round-up.
- **E15** Multi-turn conversation across the page boundary (context grows
  past 128, 256, … tokens: block tables grow; shrink-on-rewind keeps pages).

## 6. Validation plan

**Host-side (no GPU):** `cargo test -p rdna-compute` (page_pool, slot_pool
incl. provisioning/sharing/dirty tests), `-p hipfire-arch-qwen35`
(scheduler batching rules, `mtp_verify_fits_cap`, staging sizing).

**GPU-side (container, ROCm, gfx1101; engine exclusive):** matrix over
{paged, legacy} × {AR, MTP} × {text, VL} + the interaction cells of §3.E,
each against the 4B fixture:

1. Serve smoke: coherent answers, no `[slot-trace]` errors, log shows the
   expected feature lines (paged pool banner, MTP head loaded, VL load).
2. Correctness: fact-recall prompts with facts placed before/after the
   image and across chunk boundaries; image questions answered from image
   content; multi-turn recall.
3. Byte-identity where the spec demands neutrality (unpenalized AR text
   legacy vs batched is *allowed* to diverge only via the documented
   numerics classes; MTP output at greedy must be ** accepted-token-
   identical** to AR in exact arithmetic, and statistically equivalent in
   Q8 — eyeball every decoded output).
4. Concurrency: parallel streams on both slots; assertions: no cross-slot
   bleed (each answer matches its own prompt), no engine panic, clean
   rejections for over-cap/over-budget.
5. MTP evidence: `HIPFIRE_MTP_TRACE=1` shows draft/accept cycles under
   paged; adaptive-retire line appears on prose; decode tok/s ≥ AR on code.
6. Edge cases E1–E15 above, each with its expected outcome.

## 7. Performance expectations (secondary, non-gating)

- Batching ≈1.36× at 8 slots (SP3 measurement); paged pays a small
  descriptor-kernel overhead vs the single-slot WMMA slab path — acceptable
  for the scheduling wins.
- MTP must not lose >5% vs AR on code prompts at 1 active slot after the
  batched-verify rewrite (85dde5df0 claimed +18%); prose may retire to AR
  via the adaptive window.
- Penalized decode may pay a prepass cost (measured ~25% at 9B); acceptable.

## 8. Gap analysis against this spec

Audit performed 2026-09-03 against a448eb627 + working tree; validation
matrix in `.codeinsight+research/mtp-paged-vl-validate-2026-09-03/RESULTS.md`
(Qwen3.5-4B-MQ4 fixture). Statuses: **closed** (fixed this session),
**pass** (implemented + GPU-verified), **open** (documented, unfixed).

| Req | Status | Evidence / note |
|-----|--------|-----------------|
| R-A1–A3 | pass | join-mid-decode, chunked mix, slot release exercised by every concurrency test |
| R-A4 | pass | continuation HIT traces; LRU eviction/restore untouched; over-cap rejections |
| R-A5 | pass | greedy argmax fast path + penalize-then-argmax (penalized_no_loop) |
| R-A6 | pass | fail-closed pool OOM (Run C) rejects active requests, engine survives |
| R-A7 | pass | EOS/max_tokens/ctx-cap/client-gone all exercised (edges phase) |
| R-B1–B3, B5–B7 | pass | paged pool banner, provision, fail-closed OOM (Run C), graph off under paged |
| R-B4 | **closed** | `descs_dev` was `n_slots×24` B (pre-`legacy_v_base` layout) vs the 32-byte ABI — masked by the 256 B pool round-up ≤8 slots, fatal at 9+. Now `n_slots×DESC_BYTES`; proven at 9 slots (Run G) |
| R-C1–C4, C6 | pass | paged×MTP gate removed (working tree); 5060 verify cycles under paged, mean advance 2.73; byte-coherent output |
| R-C5 | pass | penalized request → zero MTP trace lines; VL request → mtp off; frontier guard + adaptive retire in place |
| R-C7 | pass | burst terminators enforced per committed token (max_tokens=1, EOS mid-burst) |
| R-D1–D4, D6 | pass | batched VL under paged; chunk straddle both orders; mixed VL+MTP step correct |
| R-D5 | pass | sequential fallback serves on legacy; paged+sequential refused at build (Run F) |
| §3.E matrix | pass | mixed text(MTP)+VL step, VL×2, MTP×2, penalties×MTP, paged cells all exercised |
| E1–E15 | pass | see RESULTS.md run table; E14 (≥9 slots) is the descs fix's regression test |
| — | **closed** | text follow-up after an image turn lost the M-RoPE `rope_delta` (every follow-up row phase-offset by ~|delta| positions vs the image turn's KV) — "image follow-up echoes previous answer" symptom. Now: `Session.rope_delta` → `PendingWork.pos3_delta` → scheduler phases + verify-row injection; fixed on paged and legacy |
| — | **closed** | prompt+max_tokens > ctx was admitted and silently truncated at the cap; now rejected up front daemon-side (vLLM-style) with an actionable message |
| — | open | CLI-side belt still counts only `max_tokens`+1024 vs `multi_slot_ctx`; consider unifying with the daemon belt |
| — | open | PM4/Redline parity for the batched-VL scatter + mrope kernels (owed from a448eb627; no new kernels this session) |
| — | **closed** | KV-cache tier was hardwired Q8_0 across the whole slot engine (`per_pos_bytes = n_kv_heads·(head_dim/32)·34`, q8-named kernels only). Now tier-generic across the full static ladder — q8, asym{2,3,4}, fwht{2,3,4} — under BOTH the legacy slab pool and the paged pool, with batching/MTP/VL unchanged: (a) `SlotKvTierPlan::resolve` (saddle-core) publishes per-tier K/V bytes-per-pos and rotation-table geometry; (b) `SlotPool`/`PagePool` track separate K/V strides and descriptors carry distinct `legacy_k_base`/`legacy_v_base`; (c) the five rotated tile-batched attention kernels and six batched K-writers were ported to descriptor addressing (`kv_offset_for_k/v` + trailing `slot_descs`/`row_slot`, byte-identical when null — ABI pinned by `kv_slot_desc_port_tests`); (d) the Q8 batched writer gained a `use_v_base` arm so rotated-V writes resolve their own slab base; (e) `forward_slots` dispatches per tier (`kv_write_slots` / `tier_attend_slots`); the single-slot WMMA reduction and WMMA prefill tiles stay q8-gated; (f) swap snapshots carry both strides (format v2) and a per-tier `kv_dtype_tag`; (g) the engine resolves the tier via a dedicated site policy (`qwen35-slots`): full ladder accepted, q8 kept as the slots default because the rotated slots paths lack a GPU validation pass — an explicit `HIPFIRE_KV_MODE=asym3`-style request opts in. Sequential-VL (`HIPFIRE_VL_SEQUENTIAL=1`) is refused under rotated tiers (its flat slab view is q8-only). GPU-validated 2026-09-03 in the ROCm container on gfx1101 (evidence: `.codeinsight+research/kv-tier-validate/RESULTS.md`): (a) kernel ports bit-exact on all six rotated tiers — zero-base parity, non-zero-base understudy isolation, composite-write byte equality, q8 `use_v_base` both directions (`crates/rdna-compute/examples/test_kv_slot_desc_ports.rs`, committed, re-runnable); (b) end-to-end slots serve smoke per tier under legacy AND paged pools (asym3/asym4/fwht2/fwht3/fwht4: coherent greedy answers, correct tier banners and stride math), MTP trace shows healthy draft/accept cycles under asym3+paged (546 lines, advance 3-4), 2-slot concurrency probe PASS (no cross-slot bleed), q8 regression unchanged |
| — | open (by design) | MTP verify greedy-only; MTP head attention v1 (≈V); multi-image unwired; sequential-VL drops penalties; no CoW/defrag under paged |

## 9. Full-branch audit addendum (2026-09-03, second session)

Deep review of the whole branch (all 36 commits) + hardening pass, base
dffd24f33 + working tree. Host tests: rdna-compute 243, qwen35 199, runtime
600, daemon 22 — all pass. GPU validation on gfx1101 (container
`local/hipfire-ornith:dev-mtp`, Qwen3.5-4B-MQ4 trunk + .mtp + .vl, 2 slots ×
8192 ctx, greedy, thinking off): AR/MTP × legacy/paged — smoke 4/4 each,
**MTP-vs-AR greedy identity 4/4 byte-identical on both pools**, full
regression battery 11/11 on both pools (evidence:
`.codeinsight+research/multislot-hardening-2026-09-03/`). Binaries under
test: hipfire `7ba549f1e467947f325ccc734e8b2bc2`, daemon
`f685a8295f7950fcb32d3c278b35d263`.

Bugs found and fixed this session:

1. **MTP verify lm_head consumed PRE-norm residual** (`mtp_spec.rs`):
   `pbs.x_batch` is the pre-final-norm residual (the AR path
   `final_logits_per_slot` norms each row with `output_norm` before its
   GEMV) but the batched verify fed the raw view to
   `mtp_trunk_verify_lm_head` — acceptance argmax skewed by the per-channel
   γ scale, breaking the MTP ≡ AR-at-greedy identity (bonus tokens were
   never the AR pick). Fixed: `rmsnorm_batched` the verify rows into
   `state.verify_hidden` staging (matching that buffer's post-norm
   semantics); `prev_hidden` now copies the staged row directly. Proven by
   the 4/4 identity result above.
2. **DN state skew on mid-burst termination** (`mtp_spec.rs` +
   `serve_engine.rs`): the accept rule stops only on `eos_id`, but
   `commit_sampled_token` also terminates on auxiliary eot, `max_tokens`,
   and ctx-cap; the DN repair replayed the FULL `advance` rows, leaving the
   recurrent state past the store for the rest of the conversation. Fixed:
   the burst is clamped to `min(remaining max_tokens, remaining ctx room)`
   and to the first auxiliary terminator before repair, so `advance` ==
   the rows the store will hold. Trace-verified (`(budget 1)` clamps a
   full accept to one committed token) and by multi-turn recall
   ('PORT-9') after a max_tokens=2 MTP burst.
3. **rope_delta hole on verify-only steps** (`serve_engine.rs`):
   `inject_mtp_verify_tokens` derived the pos3 need from scheduler-batch
   emptiness; an MTP-decoding slot emits zero scheduler rows, so a
   verify-only step (the steady state for a single active MTP slot
   continuing an image conversation) dropped pos3 entirely — phase-shifted
   verify queries against the image turn's KV. Fixed: work-level
   `force_pos3` (same predicate as the scheduler's `any_pos3`) passed into
   the injection. GPU-verified: image turn → text follow-up with MTP
   cycles active answers "Red" (both pools).
4. **Compressed-sidecar MTP head panicked the engine thread** at first
   draft (`mtp_scratch.logits_compressed` never allocated on the slots
   path — only the sequential speculator called
   `ensure_compressed_logits`). Fixed at both slot-engine alloc sites.
5. **Sampled requests activated MTP** (verify is a bare greedy argmax —
   cannot reproduce a sampled pick): now gated `temperature <= 1e-6`,
   mirroring the penalized gate. Default temperature 0.0 keeps MTP on.
6. **Repeat-window ratchet-down**: the per-step clamp wrote the effective
   window back into `sample_params[s].repeat_window`, which the next step
   read as "requested" — freezing the window at the first step's session
   length for the whole generation. Fixed: requested window carried on
   `InFlight.repeat_window_req`, clamped fresh each step.
7. **`max_batch` ignored `mtp_k + 1`**: `prefill_chunk < mtp_k + 1`
   (operator knobs) overflowed `pbs.max_batch` on the first verify step —
   an engine-thread panic (serve-hang class). Fixed: sized
   `max(prefill_chunk, mtp_k + 1) × n_slots`.
8. **Adaptive-retire counters leaked across requests** on a slot
   (`clear_work_slot` didn't reset them) — a request ending mid-window
   left the next one near retirement with a mixed window mean. Reset now.
9. **Engine-channel `Disconnected` reported as a normal "stop"**
   (`slots.rs`): an engine-thread death surfaced as a clean truncated
   answer with the session leaked resident. Now an internal error envelope
   + session close.
10. **`.vl` sidecar weight-load failure was silently discarded**
    (`carriers.rs` `.map_err(..).ok()`): a corrupt/truncated .vl yielded a
    model whose image gate stayed open with no weights (sequential path
    panics on unwrap; slots path rejects misleadingly). Now a loud
    load-time error.
11. **HFQ data-region truncation**: an intact header+index over a short
    file panicked the mmap slice path or silently zero-filled the pread
    tail. Both read paths now detect the overrun/short-read and fail
    loudly.
12. **CLI/daemon admission-belt divergence** (§8 open item): the CLI
    counted `max_tokens + 1024` (one image is ~1700+ pad tokens). In
    multi-slot mode the CLI now rejects only what the exact daemon belt
    can never admit (`max_tokens` alone beyond ctx) and defers the rest —
    the daemon's 400 (already correctly mapped) is the single authority.
    **§8 CLI-belt open item: closed.**
13. **Continuation admits had no budget belt**: a multi-turn request in
    the (cap − max_tokens, cap) window decoded until silent truncation.
    The engine now rejects the request up front (actionable message) while
    keeping the session (retry with smaller max_tokens still continues).
14. **Robustness cluster**: `image_url` now under the same 40 MiB cap with
    an actionable remote-URL error; `top_k > 64` rejected (was silently
    capped) with a 400-mapped message; `"vl": true` in the slot-mode
    loaded ack (was hardcoded false); `HIPFIRE_VL_FILE` set-but-missing
    warns instead of silently falling back; kv-mode fallback warning
    echoes the rejected value and chosen default;
    `vit_attention_qtiled_f32` falls back to the naive kernel for
    out-of-contract head shapes instead of asserting (engine-thread
    panic class); `attention_flash_bf16_windowed` asserts the tile
    kernel's head_dim%128==0 contract; `launch_asym_flash_batched` asserts
    head_dim%32==0 && <=512 (register-array ceiling);
    `PrefillBatchScratch::free_gpu` frees the three VL buffers and the
    allocation projection counts them; `SlotPool::share_prefix` completes
    the fork (live_tokens + desc.seq_len) instead of trusting the caller;
    `set_seq_len` refuses lengths beyond the i32 descriptor ABI;
    `MapleState::new` routes through MAPLE_POLICY instead of hardcoding
    Q8; stale doc comments corrected (penalty forwarding, batched-kernel
    kernarg trailing params, maple head-quant help).

Still open (unchanged from §8): PM4/Redline parity for the batched-VL
scatter + mrope kernels; the by-design MTP limitations. Config-schema
note: `models`/`aliases` now belong in `models.toml`; older inline
`[aliases]` config.toml files fail loudly at parse.
