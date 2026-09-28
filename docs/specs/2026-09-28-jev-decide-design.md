# Jev-style decide endpoint — Design Spec

> Status: **DRAFT — awaiting review** · Date: 2026-09-28 · Addendum §12
> (session mode, v1.1; design approved 2026-09-28)
> Author: nwoolmer
>
> A local, Jev-compatible decision endpoint: unstructured state plus typed
> questions in, calibrated-probability answers out, with **no decode loop**.
> Every answer is one next-token distribution read after a prefill, restricted
> to single-token option codes. Wire-compatible with TypeSafe's Jev
> (`POST /v1/systemone`) so existing Jev clients and benchmarks run unchanged.

## 1. Goal

Serve Jev's three question types — `choice`, `score`, `noul` (yes/no) — from
whatever model hipfire has loaded, fast enough to be used as a routing /
filtering / guardrail primitive, and measurably comparable to Jev on public
benchmarks.

Phasing:

- **v1 (this spec):** logit readout on the loaded model; raw (unfitted)
  probabilities.
- **Later (separate spec):** a trained or fitted calibration layer (Jev's RLCD
  is the reference), and a dedicated classifier model.

## 2. Non-goals (v1)

- **No calibration training or fitting.** Probabilities are the model's own.
- **No multi-token option labels** (tree scoring over real keys). See §9.
- **No batching of question suffixes across slots**; questions run sequentially
  after one shared prefill.
- **No dedicated classifier model.** The daemon's `flock` on
  `~/.hipfire/daemon.pid` enforces one daemon per machine, so a second resident
  model is a separate design.
- **No participation in the continuous-batching slot engine.**

## 3. Background: what Jev is and why this design

Jev (TypeSafe AI, released 2026-09-15) never generates text. It returns
probabilities over caller-defined answers in 70–500 ms. Public evidence (the
external probe study "Jev's Architecture Unmasked" by archerhume, the TypeSafe
launch post, and three open reimplementations):

- **One readout, no decode loop.** A 255-option request is as fast as a
  2-option one, and `output_tokens` doesn't track latency.
- **Shared state, isolated questions.** Token usage is exactly additive across
  questions. A fact placed in a sibling question is invisible to the others
  (scored 0.00); the same fact placed in the state scores about 0.9.
- **Options influence each other.** Adding an irrelevant option shifts the
  odds between the others, so the full option list is in context.
- **Calibration comes from training** (their RLCD); the architecture is
  conventional.
- **Confidence formulas.** Verified against Jev's committed answers:
  `choice` confidence = (p_max − 1/K)/(1 − 1/K), with |err| ≤ 0.02, i.e.
  rounding. `score` confidence ≈ p_max (mean err 0.006).
- **Jev's published input limits** are about 64,000 tokens for the shared state
  plus all questions, and about 32,000 tokens for the state plus the longest
  single question (source: Flavio Copes, "A deep dive into Jev",
  https://flaviocopes.com/jev/).

Open reimplementations (open-alternative-jev, openjev) read single-token
**letter** labels at one position, capped at about 26 options. Jev accepts up
to 255.

### Why snapshot/restore instead of tree masks

hipfire's tree verify is only exact for the Llama family. For Qwen3.5/3.6
hybrids, `spec_step_ddtree_batched`
(`hipfire-arch-qwen35/src/speculative.rs`) advances the GatedDeltaNet state
through the tree in linear order. Sibling branches therefore contaminate each
other's recurrent state. That is tolerable for speculative decoding, which
re-verifies the accepted prefix, but not for question isolation or option
probabilities. `DeltaNetSnapshot` save/restore is exact on every supported
architecture.

### Why single-token codes scale to 255

In the Qwen3.5 vocabulary (248,044 entries):

- all 26 single letters are single tokens;
- 544 of 676 two-letter uppercase codes are single tokens after a space (562
  with no space);
- digits `0`–`9` are always single tokens, because the pre-tokenizer splits
  digits one at a time.

## 4. Architecture and layering

This follows `docs/ARCHITECTURE.md`: the daemon does dispatch only;
architecture-specific GPU work goes in `hipfire-generate`; arch-neutral logic
goes in `hipfire-engine`.

| Crate | New piece | Responsibility |
|---|---|---|
| `hipfire-engine` | `decide` module | Request types and validation, option-code assignment (tokenizer-checked), prompt rendering, answer assembly (softmax over label logits, confidence, score mean, code→key mapping), usage accounting. Pure; no GPU. |
| `hipfire-generate` | `decide.rs` | GPU runner: chunked state prefill → snapshot → per-question restore / suffix prefill / logits gather → final restore and cleanup. Qwen3.5 and Llama paths. |
| `hipfire-daemon` | `"decide"` message | Dispatch; one reply `{"type":"decided","id",answers,usage,timing}`. |
| `hipfire-cli` serve | `POST /v1/systemone` | Body cap, validation, admission, `ensure_model`, forward, shape the response. |

Run `scripts/check-crate-maps.py` after adding files.

### 4.1 GPU runner (`hipfire-generate/src/decide.rs`)

1. Render and tokenize every question's full prompt (§5), checking Jev's
   per-question token limit as each is rendered (§7), so an over-long
   state fails on the first question. Prefill the shared prefix via
   `qwen35::forward_prefill_batch` (Llama: `llama::forward_prefill_batch`),
   starting from position 0 of a reset model. The prefix's own logits are
   never read, so this prefill skips the full-vocab download
   (`decide_prefill_logits(…, want_logits = false)`).

   A single-question request has nothing to share: it is one plain full
   prefill of that question's prompt (`prefix_len = 0`), with no snapshot
   allocation and no restore.
2. Snapshot: record `P = seq_pos`. On Qwen3.5, also
   `DeltaNetSnapshot::new_for` + `save_from(&dn_state)`.
3. For each question, in request order:
   1. restore (`seq_pos = P`, `restore_to(&mut dn_state)`);
   2. prefill the question suffix;
   3. download `scratch.logits`;
   4. gather the logits of that question's label token ids.
4. Cleanup:
   - leave the model reset: `seq_pos = 0`, recurrent state reset via
     `common::reset_qwen35_recurrent`;
   - clear `conversation_tokens` and the prefill checkpoints, so the AR
     prompt cache can't match decide tokens.

   A plain decide does not preserve a cached chat conversation; it behaves
   like serve's existing per-request reset. Session mode (§12) keeps it.
   - call `DeltaNetSnapshot::free_gpu` explicitly. It has no `Drop`, so
     skipping this leaks device memory.

Preconditions (409 otherwise):

- `pp == 1` and no expert parallelism (`m.ep` is `None`);
- no `kv_adaptive`;
- an un-compacted KV cache after the pre-decide reset
  (`KvCache::compact_offset == 0`, read through the carrier hook
  `decide_kv_compact_offset`). Positional rewind is only valid on an
  un-compacted cache.

KV eviction (CASK / TriAttention, `m.eviction`) may be configured; `hipfire
serve` loads every model with it when config has `cask` on. It is safe
because eviction cannot run inside a decide: `Eviction::maybe_evict` is only
called by the generate loops (`ar.rs`, `qwen.rs`, `vision.rs`), never by
`qwen35::forward_prefill_batch` / `llama::forward_prefill_batch`, which are
the only model calls the decide hooks make. What bounds a decide under
eviction is the KV buffer: with `compact_offset == 0` the prefill writes KV
slot = position, and the cache holds only `m.physical_cap` slots (by default
`cask_budget + cask_beta + 256`, clamped to `max_seq`), with no bounds check
in the prefill. So with eviction configured the longest question prompt + 1
must be <= min(`Eviction::budget()`, `m.physical_cap`), else 422 naming the
limit and without `required_max_seq` (a larger `max_seq` does not raise it).
Under the current `CaskConfig` the load requires `max_seq >= budget + beta +
4` and `physical_cap >= budget`, so `physical_cap` never binds and the limit
is the budget; the `min` is defensive. Likewise the 409 compacted-cache check
below is defensive: the pre-decide rollback resets `compact_offset`, so it
cannot fire today. Decide prompts (hundreds to a few thousand tokens) are far
below the usual budgets.

A client disconnect does not interrupt a decide in v1. The loop is
sub-second, and mid-loop abort would need the daemon's per-request terminal
registry; serve simply discards the reply.

A debug request field `_debug_no_snapshot: true` makes the runner reset and
prefill each question's full prompt from position 0 instead of restoring.
Gate 1 (§10) compares the two modes. It is a daemon-level knob only:
`gates.py` talks to the daemon directly, and serve does **not** forward it
(serve forwards only `state` and `questions`).

Multi-slot mode (the experimental slot backend owns the weights) is refused
with 409 `decide unsupported in multi-slot mode`.

## 5. Prompt layout

Every question shares a byte-identical prefix. State comes first; everything
question-specific follows the snapshot point.

```
<|im_start|>system
You answer classification questions about the state. Reply with one option code.<|im_end|>
<|im_start|>user
State:
{state}
                                                   ◀── snapshot (end of shared prefix)
Question: {instructions}
Options:
A. {criteria for option 1}
B. {criteria for option 2}
…
Answer with the option code only.<|im_end|>
<|im_start|>assistant
<think>

</think>

                                                   ◀── logits read here
```

- **State rendering.** A string is used as-is. An object or array is
  pretty-printed as JSON, so backtick `field` references in instructions
  resolve naturally.
- **Tokenization.** Each question's full prompt is rendered and tokenized
  on its own, via `hipfire_engine::prompt::batch_render_prompt_tokens` with
  `enable_thinking = false` (the model's Jinja chat template; `ChatFrame`
  fallback).
  - The snapshot point is the longest common token prefix across all the
    questions' sequences, capped at `min_len − 1` so every suffix is
    non-empty.
  - Every question is therefore evaluated on exactly the tokens of its own
    full prompt, with no tokenization-boundary assumptions.
- **Labels:**
  - `choice`: codes `A`–`Z`, then two-letter codes, in criteria order.
  - `score`: levels labelled `0`…`k-1`, with each level's description.
  - `noul`: `Yes`/`No`, with the criteria descriptions when provided.
- **Code check.** Per request (not cached), every candidate code is checked
  to encode to exactly one token, after stripping a leading BOS that
  tokenizers with `add_bos_token` (Llama / Mistral style) prepend in
  `Tokenizer::encode`; the same BOS-stripped encoding supplies the label's
  token id. Codes that fail are skipped. If fewer than K valid codes remain,
  the request gets a 422.

## 6. Request lifecycle (serve)

1. `POST /v1/systemone` with `{model, state, questions}`. Serve checks body
   size and JSON well-formedness only.
   - The daemon is the single validation authority (§7), as with
     `img_generate`. `hipfire-cli` does not depend on the GPU crates that
     own the validation code.
2. **Model selection.**
   - If `model` names a local hipfire model (registry tag or path), serve
     calls `ensure_model`.
   - Any other value, including Jev's `jev-latest`, means the currently loaded
     model.
   - If no model is loaded, return 503 with an explanatory message.
   - The response `model` field is the hipfire model actually used, named
     as chat and `/health` report it (the registry tag when a tag was
     requested); the filesystem path only when no served name is known.
3. **Admission.** Exclusive gate, the same as a sequential chat request. When
   saturated: 503 + `Retry-After`.
4. Forward `{"type":"decide","id",state,questions}` to the daemon.
   - The daemon replies with exactly one `decided` message: either
     `answers`/`usage`/`timing`, or `error: {status, message,
     required_max_seq?}`.
   - `decided` is not a lifecycle event type, so `hipfire-client` routes it
     to the control waiter, and `Engine::request` suffices.
   - If the longest prompt exceeds `max_seq`, the daemon returns 422 with
     `required_max_seq`. Serve then reloads once via `ensure_model(…,
     Some(required_max_seq))` and retries, up to the same context ceiling
     chat stops at (`MAX_SEQ_CEILING` = 393,216 + 1,024). Above it serve
     returns the daemon's 422 as-is, `required_max_seq` included, without
     reloading. (Jev's 32,000-token per-question limit, checked first, keeps
     `required_max_seq` far below the ceiling in practice.)
5. Shape Jev's response `{model, answers, usage}`.
   - Per-phase timings (state prefill, per-question suffix, readout) go in an
     `x-hipfire-timing` header and the log. No extra body fields are added.

## 7. Wire format (Jev-compatible)

Request validation, returning 422 with the question and field named:

- `state`: string, JSON object or JSON array.
- `questions`: non-empty map, name → question, at most 128 questions. Each
  question needs `instructions` (string) and a `type`:
  - `choice`: `criteria` is an object with 2–255 keys (key → description);
  - `score`: `criteria` is an array of 2–10 level descriptions, low → high;
  - `noul`: optional `criteria` `{true, false}`.

Token limits (hipfire bounds chosen to match Jev's published limits, §3),
checked by the daemon after rendering, 422 naming the limit and without
`required_max_seq`:

- state + the longest question (that question's full rendered prompt) ≤
  32,000 tokens; checked per question as it is rendered, so an over-long
  state fails on the first question; these bounds are counted in the loaded
  model's tokens including chat-template overhead, so a state Jev accepts
  near the limit may be refused here;
- shared state + all questions (the `usage.input_tokens` accounting, shared
  prefix counted once) ≤ 64,000 tokens.

Criteria keys (and question names) keep request order: `hipfire-engine`
enables serde_json `preserve_order` itself.

Answers:

- `choice`: `{type, choice, probabilities: {key: p}, confidence}`, where
  confidence = (p_max − 1/K)/(1 − 1/K).
- `score`: `{type, score, probabilities: {"0".."k-1": p}, confidence}`, where
  score = Σ i·p_i and confidence = p_max.
- `noul`: `{type, noul: p}` with no confidence field, matching Jev.

All three:

- Probabilities are a softmax over the question's label logits only.
- Probabilities are **not rounded**. Jev rounds to 2 dp; full precision keeps
  calibration measurable.
- `usage.input_tokens` = state-prefix tokens + Σ suffix tokens, i.e. additive,
  as observed for Jev. `output_tokens` = 0.

## 8. Errors

| Condition | Response |
|---|---|
| Malformed request / limits exceeded (incl. > 128 questions) | 422, naming question and field |
| State + longest question > 32,000 tokens, or state + all questions > 64,000 tokens | 422 naming the limit, no `required_max_seq` |
| Model architecture has no snapshot path (neither Qwen3.5 nor Llama family) | 400 `decide not supported for <arch>` |
| No model loaded and `model` not a local model | 503 |
| KV eviction configured and the longest prompt + 1 exceeds min(eviction budget, `physical_cap`) | 422 naming the limit, no `required_max_seq` |
| KV cache compacted (`compact_offset != 0`) after the pre-decide reset | 409 |
| Prompt exceeds the loaded `max_seq` | 422 + `required_max_seq`; serve reloads once up to `MAX_SEQ_CEILING`, above it returns the 422 as-is |
| Fewer than K valid single-token codes | 422 |
| Admission saturated | 503 + `Retry-After` |
| Daemon crash / GPU error | 500; the existing serve restart path |
| Multi-GPU (pipeline or expert parallel), `kv_adaptive` active, continuous-batch lanes active | 409 |
| Multi-slot mode (experimental slot backend) | 409 `decide unsupported in multi-slot mode` |

## 9. Deferred

- **Real-key labels via tree scoring.** Exact on the Llama family; would
  need per-path evaluation on hybrids. Build it only if jev-bench's
  `descriptions` and `order` experiments show letters losing to keys.
- **Batched question suffixes** via `forward_batch_slots`. It supports
  multi-token prefill per slot but has no slot-fork API. Needs a D2D copy of
  the arena range and DN buffers.
- **Calibration layer.** A per-model, per-type temperature file, then trained
  calibration.
- **Dedicated classifier model.** Blocked by the one-daemon-per-machine
  `flock` invariant.

## 10. Testing

**Unit tests, no GPU** (`hipfire-engine`, `no-gpu-ci.sh`):

- validation: every 422 path and limit;
- code assignment against a Qwen3.5 tokenizer fixture: single-token and
  collision-free;
- golden prompt tokens, including the prefix + suffix = full-prompt identity;
- answer assembly: the confidence formulas checked against sampled committed
  Jev answers, the score mean, and the code→key mapping.

**Serve tests:**

- a `decide` branch in `serve/fake_daemon.py`;
- route tests for the 422 paths, `jev-latest` → loaded model, 503 with no
  model loaded, and the timing header.

**GPU correctness gates** (`scripts/jev_eval/gates.py`; none may FAIL to
merge; run on Qwen3.5 (DeltaNet restore) and a Llama-family model):

Prefilling a prompt in one call and prefilling it as prefix + suffix give
slightly different logits. This drift is deterministic and is an inherent
precision property of kernels whose route or rounding depends on the batch
composition. It is not a decide defect
(`.superpowers/sdd/split-prefill-investigation.md`). The kernels involved:

- the int8-activation MMQ GEMM routes: HFQ4G128 on 16-aligned batches on
  gfx1151, and the generic MQ4 route at batch ≥ 128;
- the Q8 GatedDeltaNet state, which is requantised once per launch rather
  than per token;
- the f16 WMMA flash-prefill attention, which is not bit-row-invariant at
  head dim 128.

Suffixes of 1–3 tokens also take the per-token path. Gate 1 therefore runs
in two modes.

1. **Snapshot exactness.**
   - **(a) Exact mode.** The daemon is launched with those routes pinned to
     their row-invariant alternatives: `HIPFIRE_HFQ4G128_MMQ=0`,
     `HIPFIRE_MMQ=0`, `HIPFIRE_DN_REQUANT_PER_TOKEN=1`,
     `HIPFIRE_FLASH_PREFILL=0`. The following must hold bit for bit
     (Δ = 0):
     - the snapshot decide equals the `_debug_no_snapshot` decide for every
       question;
     - each question of a multi-question request equals a single-question
       decide of that question alone (a single-question decide is one plain
       full prefill, `prefix_tokens == 0`), so restore leaves no residue.

     `EXACT_ENV` in `gates.py` pins today's batch-composition-dependent
     routes; when PR #768's default-on gfx1151 routes land it must be
     extended with their disable switches.
   - **(b) Default mode.** This is checked per question:
     - drift_q is the |Δ log p| between that question's snapshot and
       full-prefill label distributions;
     - floor_q is the same question's path-noise floor: one-call prefill vs
       token-by-token prefill of the same prompt, measured each run by the
       `split_prefill_probe` example;
     - the gate requires drift_q ≤ 2 × floor_q.

     The probe's one-call answer must reproduce the daemon's full-prefill
     answer (Δ ≤ 1e-6). Otherwise the floor was measured on a different
     prompt or kernel route, and the gate fails.

     Why this rule: (a) is the exactness proof, and (b) only guards the
     default kernels against gross regressions. The floor comes from a
     different noise source (the per-token path) than the drift does (the
     split point), so it is a scale, not a bound, hence the factor 2.
   - **(c) Order invariance.** Every ordering of the questions gives
     bit-identical answers (Δ = 0), at a fixed split point.
   - The decide must not return the 400 "cannot disable thinking" error.
2. **Isolation.** The "secret code" probe, judged relative to a baseline
   with no code anywhere:
   - the gate passes iff |P(code | code in a sibling question) − P(code | no
     code)| ≤ 0.05 and P(code | code in the state) − P(code | no code)
     ≥ 0.3;
   - if only the sensitivity half fails, the model cannot do the probe, and
     the result is INCONCLUSIVE. That is not a failure, but it is reported.
3. **Model left clean.** A greedy `generate` after a decide yields exactly
   the same tokens as a `generate` after a plain `reset`. This covers both a
   prompt that shares a prefix with the decide state (must not hit a stale
   LCP cache) and one that doesn't.
4. **No leak.** The daemon's own device memory (GTT + VRAM, from its amdgpu
   DRM fdinfo) grows by less than 64 MiB over 1,000 decides.
   - The device-wide sysfs sum over all cards is reported alongside, and is
     used as the fallback when fdinfo can't be read.
   - It isn't the criterion because other processes on a shared machine move
     it.
5. **Error replies.** Two requests are sent, and a normal decide must still
   succeed after each:
   - a state longer than the loaded `max_seq` returns 422 with an integer
     `required_max_seq` greater than `max_seq` (with `--cask`: a state over
     the eviction limit returns 422 naming the limit, without
     `required_max_seq`);
   - a malformed request returns 422;
   - a state over Jev's 32,000-token per-question limit, and four long
     questions over the 64,000-token request limit, each return 422 naming
     the limit, without `required_max_seq`.

   `gates.py --cask` configures every daemon with the cask* params serve
   sends (defaults from `~/.hipfire/config.json`), so all gates, including
   1a exact-mode equality, run with eviction configured; it adds gate 0
   (per daemon: the over-limit 422 is returned, proving eviction is on, and a
   normal decide succeeds).

## 11. Evaluation

Harness in `scripts/jev_eval/`. The benchmark repos are external, cloned to
`~/repos/jev-evals`. Their committed Jev answers were verified 2026-09-28 to
align row-for-row with examples rebuilt locally.

- **jev-bench** (Running-Dolphins): 12 tasks × 500 examples, plus 6
  experiments (out-of-scope, language, option count, order, repeat,
  descriptions). A wrapper sets `jevbench.API_URL` to the local endpoint;
  `jevbench.py` itself is unmodified.
- **jev-ood-calibration** (scienthoon): 900 synthetic tickets (choice / score
  / yes-no), plus OpenBookQA, CommonsenseQA and a HellaSwag 2k subset. A small
  adapter writes its JSONL result schema; its `metrics.py` scores it.
  - Public sets are rebuilt as the first ≤ 2,000 validation rows, filtered,
    then shuffled with `random.Random(1)`.
- **Report.** Per task: accuracy and top-probability ECE (10 bins) vs Jev's
  committed answers, with a per-type breakdown.
  - Jev baselines:
    - jev-bench accuracy ranges from 0.70 (yelp-stars) to 0.99 (sms-spam);
    - weak calibration on ledgar (ECE 0.14) and yelp-stars (ECE 0.15);
    - synthetic tickets: `score` at 0.447 accuracy, ECE 0.306.
- **Latency.** decide vs the same model generating the answer through
  `/v1/chat/completions`, on the same examples and hardware. p50/p95, split
  into state prefill vs per question. Jev's own latencies include the network
  and are not comparable.
- **Models:** Qwen3.5-4B and Qwen3.6-27B.

**v1 is done when** all four GPU gates pass and the evaluation report exists.
There is no accuracy bar; accuracy is a property of the model, and calibration
is the later phase's job.

## 12. Session mode (v1.1)

> Addendum, 2026-09-28. The shape was approved by the human; this section
> fixes the details. §1–§11 still hold for plain (`state`) requests.

A plain decide resets the model before and after (§4.1), which destroys the
chat conversation the daemon has cached. An agent that asks a routing
question mid-conversation pays twice: the decide prefills the conversation
as `state`, then the next chat turn re-prefills the whole conversation. In
session mode the conversation itself is the state, and the cache survives.

### 12.1 API

`POST /v1/systemone` accepts a hipfire extension:

```json
{"model": "…", "messages": […], "tools": […], "tool_choice": "…", "questions": {…}}
```

- `messages`, `tools`, `tool_choice`: exactly what the agent sends to
  `/v1/chat/completions`. Send the same values the chat requests use. They
  change the rendered tokens (a Qwen template renders `tools` into the
  system turn), and any difference turns a cache hit into a miss.
- `questions`: unchanged (§7).
- Exactly one of `state` / `messages`; both or neither is a 422. `state`
  requests keep v1 behaviour, including the reset.
- The response is unchanged: `{model, answers, usage}`.
  - `usage.input_tokens` = E + Σ(len_i − E), where E is the conversation
    length in tokens (§12.3).
  - The `x-hipfire-timing` header adds:
    - `mode: "session"`;
    - `start`: `extend`, `resume` or `cold`;
    - `cached_tokens`: conversation tokens reused, not prefilled;
    - `conversation_prefill_tokens`;
    - `committed`.
  - `prefix_tokens` = E.
- Daemon wire:
  `{"type":"decide","id","messages","tools"?,"questions","_debug_no_snapshot"?}`.
  The reply is unchanged. The daemon arm is unchanged too:
  `handle_decide_message` dispatches on the mode.

### 12.2 What the chat path caches

These findings shaped the design.

- **Contents after a Qwen AR generate.** `m.conversation_tokens` holds:
  - the rendered prompt (`ar.rs:3937`);
  - every generated token, thinking tokens included, through the raw
    commit (`ar.rs:315-340`);
  - `<|im_end|>` plus the `\n` trailer, when the turn ended on EOS
    (`ar.rs:4824-4868`). A length-capped turn has neither.

  With no eviction, `seq_pos == conversation_tokens.len()` and the
  DeltaNet state is at that position.
- **Next turn: render.** The next turn renders the full `messages` through
  `build_cached_history_jinja`. It splices each assistant turn's verbatim
  generated tokens from `asst_turn_cache` and re-supplies the generation
  primer when the template renders history turns bare
  (`ar.rs:3055-3135`).
- **Next turn: LCP.** It then compares the render with
  `conversation_tokens` (`ar.rs:3171-3176`):
  - a strict forward extension sets `seq_pos = lcp` and prefills the rest
    (`ar.rs:3438-3449`);
  - otherwise it resumes from the latest prefill checkpoint ≤ lcp, on
    qwen35 only (`ar.rs:3290-3390`). Checkpoints are taken at the first
    prefill chunk and then every `ckpt_interval()` (2,048) tokens, at most
    8 (`ar.rs:704-723`, `3880`, `4301`);
  - failing that, a cold reset that keeps `asst_turn_cache`
    (`ar.rs:3391-3434`).

  The `done` event's `cached_tokens` is the reused length
  (`ar.rs:380-404`).
- **Eligibility** (`ar.rs:3025-3029`): `messages` present, no eviction,
  `HIPFIRE_QWEN_PROMPT_CACHE` not `0`, and a non-empty conversation. Under
  eviction every Jinja chat turn cold-resets (`ar.rs:3465-3470`).
- **Serve** sends `reset` before a chat request only when the model is
  neither `cache_capable` nor `continuous_batch_capable`
  (`complete.rs:1668`).
  - `cache_capable` means arch 5|6|9|10|12|14 (`daemon main.rs:2093`).
  - Qwen3.5/3.6 (`Qwen35Carrier`, arch 5/6) therefore keeps the
    conversation between turns.
  - Llama-carrier archs 0/1 are reset before every chat turn. Session mode
    works for them, correctly but cold every time.
- **Templates.**
  - They always render a generation prompt (`add_generation_prompt =>
    true`, `prompt_frame.rs:1438`), so `messages` cannot be rendered on its
    own.
  - They may render a history turn differently depending on what follows:
    Qwen drops past reasoning before the last user query.
  - Qwen3.5 renders history assistant turns bare, while Qwen3.8 re-emits
    the empty think block (`prompt_frame.rs:1540-1556`).
- **v1's rollback** also clears `asst_turn_cache` (`common.rs:1265`).

### 12.3 Rendering and the conversation end

**Rendering.** For each question, R_i is the chat path's cached render of
`messages + [user: question text]`, produced the same way the chat path
renders a turn:

- the same Jinja template and `tools`;
- the same per-message normalisation (`maybe_normalize_prompt`, as the
  daemon's `generate` arm applies it);
- assistant turns spliced from `asst_turn_cache`, with the chat path's
  primer rule;
- `enable_thinking = false`.

The question text is v1's question block with no state: `Question: …`,
`Options:`, the labelled options, then `Answer with the option code only.`.
No decide system prompt is added; the conversation's own system turn
stands.

R_i is a full render of the longer conversation, so any history rewrite
the template applies when a user turn follows is already in it. The next
chat turn renders it the same way. Nothing assumes that a render of
`messages` alone is a prefix of R_i.

**Conversation end E.**

1. L = the longest common prefix of every R_i and a probe render (user turn
   `.`), capped at min len − 1.
2. E = the position of the last `<|im_start|>` in R_0[..L], which is the
   opener of the appended turn.
3. With no `<|im_start|>` token, E = L. The committed conversation then
   includes the user-turn header, which costs at most a later cache miss.

Every R_i shares R_0[..E]. The suffix R_i[E..] is the question turn plus the
closed-think assistant opener.

E stops before the opener so that the common agent case is an exact match:

- the cached conversation ends with `<|im_end|>\n`;
- the decide prefills nothing;
- the next chat turn prefills exactly what it would have without the
  decide.

**Open assistant turns.** The template closes every turn it renders. Suppose
the cached conversation ended mid-turn: a length cap left no `<|im_end|>`,
and that turn was not stored in `asst_turn_cache`. The render then
re-tokenizes the turn and adds the terminator. If the tokens agree, that is
a delta; otherwise it is a resume or cold start. Either way the result is
correct.

**Cost.** Every question renders and tokenizes the whole conversation, as
v1 does the state. That is O(questions × conversation) CPU.

### 12.4 Lifecycle (daemon)

1. **Plan.** This step is read-only on KV, DeltaNet, `seq_pos` and
   `conversation_tokens`. Like a chat render, it may refresh
   `asst_turn_cache`'s LRU order. It parses the request, checks the
   preconditions and limits, renders, computes E, and picks the start.
   Reuse is eligible only when all of these hold:
   - no eviction;
   - `HIPFIRE_QWEN_PROMPT_CACHE` is not `0`;
   - `compact_offset == 0`;
   - `conversation_tokens` is non-empty and
     `seq_pos == conversation_tokens.len()`.

   The start is then one of:
   - **extend**, when `conversation_tokens` is a prefix of R_0[..E]. This
     includes an exact match. Unlike the chat path's exact-match edge, an
     exact match is safe here: every question suffix has at least one
     token, so no token is re-applied to the DeltaNet state;
   - **resume** from the latest prefill checkpoint ≤ lcp. This needs
     checkpoint resume on and no speculator. The checkpoint is restored
     through `decide_restore`, and later checkpoints and tokens are dropped,
     as in the chat path;
   - **cold**, otherwise. This is the attested rollback with
     `asst_turn_cache` moved out and back. The chat path's own cold start
     keeps that cache too, and its entries are keyed by content, not
     position.
2. Prefill the delta R_0[from..E] without logits. In the normal agent case
   it is empty.
3. Snapshot at E (`decide_save`).
4. Per question: restore (except for the first question), prefill R_i[E..]
   at E, and read the labels as in v1.
5. Restore to E and commit:
   - `conversation_tokens` = R_0[..E], and
     `seq_pos = E = conversation_tokens.len()`;
   - DeltaNet is at E;
   - prefill checkpoints ≤ E are kept, and the decide takes none;
   - `asst_turn_cache` is untouched;
   - KV past E holds the last question's suffix. It is never read
     (attention reads [0, pos]) and the next prefill overwrites it, just as
     after a chat checkpoint resume.
6. Free every snapshot on every path.

**Speculator loaded** (DFlash/MTP, `m.speculator`). The drafter keeps its
own context and checkpoint ring, which the decide hooks do not advance, so
the decide never commits:

- **extend:** take an extra snapshot at the cached end first, and restore
  it last. The cache is left exactly as found, and the next chat turn
  re-prefills the delta.
- **resume:** not used.
- **cold:** end with the attested rollback. The cache did not match
  anyway.

**Failure.** Any error after planning fails closed with the attested
rollback, then returns 500. This is the chat path's invariant that a
retained cache never holds uncommitted state (`ar.rs:3317-3324`).

**Rollbacks.** Session mode never resets a reusable conversation. The only
rollbacks are:

- the cold start, when the cache does not match (the chat path would have
  reset too);
- speculator + non-extend;
- fail-closed after an error.

`handle_decide_message` reports each one, so the daemon bumps
`state_epoch`.

**Debug path.** `_debug_no_snapshot` renders as above, then runs v1's
full-prefill path over each R_i: reset, one full prefill per question,
reset. It is the S2 reference and ends reset, like v1.

### 12.5 Preconditions, limits, eviction, thinking

**Preconditions.** The v1 409s and 400s apply unchanged: pp/EP,
`kv_adaptive`, multi-slot, active lanes, unsupported arch. Session mode also
returns 400 when no Jinja chat template is available (none in the model, or
`HIPFIRE_JINJA_CHAT=0`). Session mode must render exactly like the chat
path, and that is Jinja.

**Limits.**

- Jev's limits apply to the question part only: each suffix R_i[E..] ≤
  32,000 tokens and the suffixes together ≤ 64,000. Over either, 422 naming
  the limit.
- The conversation is bounded by context: longest R_i + 1 ≤ `max_seq`,
  else 422 + `required_max_seq`. Serve then reloads, which drops the cache,
  so the retry starts cold.

**CASK / eviction.** Session mode never reuses under eviction. The chat
cache is off there (`ar.rs:3025-3029`) and every chat turn cold-starts
(`ar.rs:3465`), so there is nothing to keep. The rule:

- always start cold; the rollback zeroes `compact_offset`;
- the decide never calls `maybe_evict`;
- the limit is v1's: longest R_i + 1 ≤ min(budget, `physical_cap`), else
  422 without `required_max_seq`.

Without eviction there is no compaction. `compact_offset == 0` is also
required for reuse, as a defensive check.

**Thinking.** Question turns render with `enable_thinking = false`, because
the readout needs a closed think block. Reuse assumes the conversation
renders the same under that setting.

- This holds for Qwen3.5 templates. Their history rendering does not read
  `enable_thinking`, and stored thinking turns replay as whole envelopes.
- It fails for a template whose history rendering reads `enable_thinking`,
  or for a thinking-on turn stored without reasoning. Those get a different
  primer and fall back to resume or cold: correct, but slower.

### 12.6 Serve

When `messages` is present, serve projects `messages`, `tools` and
`tool_choice` with the chat projection (`project_request_contract`):

- `normalize_openai_messages`, with `reasoning_content` for Qwen3.5/3.6;
- the default system prompt;
- the `tool_choice` projection.

It then forwards `messages` and `tools`. When the caller also sent `state`,
serve forwards it too, so the daemon returns the 422. A projection error,
such as an invalid `tool_choice`, is a 422. There is no reset. Model
selection, admission, the `required_max_seq` retry, keep-warm and response
shaping are as in §6.

### 12.7 Errors (additions to §8)

| Condition | Response |
|---|---|
| Both or neither of `state` / `messages` (with a model loaded) | 422 |
| `messages` not a non-empty array of chat messages; `tools` not an array; bad `tool_choice` (serve) | 422 |
| Chat template render fails (e.g. the template raises on the message order) | 422 |
| Session mode without a Jinja chat template | 400 |
| A question suffix > 32,000 tokens, or all suffixes > 64,000 | 422 naming the limit, no `required_max_seq` |
| Conversation + longest question > `max_seq` | 422 + `required_max_seq` |
| Under eviction: conversation + longest question + 1 > min(budget, `physical_cap`) | 422, no `required_max_seq` |
| GPU error mid-session | 500; the model is rolled back and the cache lost |

### 12.8 Gates

These are added to `scripts/jev_eval/gates.py`, alongside the v1 gates.

**What "equal" means.** A session answer is compared with a one-call full
prefill of the identical token sequence R_i from a reset model
(`_debug_no_snapshot` in session mode). A plain decide whose `state` is the
rendered conversation is not comparable: it wraps the state in the v1
system prompt and `State:` block, so its tokens differ.

- **S1 cache reuse.** Chat turn 1 from reset. The baseline is turn 2 with
  no decide; the test repeats turn 1 from reset, runs a session decide,
  then turn 2.
  - **(a) No delta, default kernels.** The decide's `messages` end with the
    assistant reply. Pass requires:
    - turn 2 greedy text bit-identical to the baseline;
    - decide `start = extend` and `conversation_prefill_tokens = 0`;
    - turn 2 `cached_tokens` == the decide's `prefix_tokens` (== the
      baseline's).
  - **(b) Delta, exact env.** The decide's `messages` include the next user
    turn. The split point moves, so this runs in exact mode. Pass requires:
    - identical text;
    - `conversation_prefill_tokens > 0`;
    - turn 2 `cached_tokens` == E, which is greater than the baseline's.
  - S1 is INCONCLUSIVE, not FAIL, when the baseline itself gets no cache
    hit (Llama carrier, `--cask`). It must PASS on Qwen3.5.
- **S2 exactness (exact env).** Per question, Δ = 0 between the session
  answer and the reference, both for a warm start (`extend`) and for a
  start with another conversation cached (not `extend`).
- **S3 no leak.** 1,000 session decides alternating two conversations in
  pairs, so `extend` and `cold` both run and are counted. The daemon's
  fdinfo growth must be < 64 MiB (gate 4's measure).
- **S4 stale cache (exact env).** Conversation C, with its own conversation
  cached, gives answers W. Then:
  - with an unrelated conversation cached (`start = cold`), the answers
    must equal W;
  - with a conversation sharing a ~2.5k-token prefix cached
    (`start = resume`), the answers must equal W (Δ = 0);
  - the next chat turn on C must reuse the decide's conversation
    (`cached_tokens` == E).

  S4 is INCONCLUSIVE if no checkpoint precedes the shared prefix, since the
  resume path is then not exercised.
- **S5 error replies.** Both, neither, an empty `messages` and a
  non-list `messages` each return 422. After them, a session decide on the
  committed conversation must start `extend` with zero delta, proving the
  refusals left the cache untouched.
- **`--cask`.** Every start is `cold`. S1 reports INCONCLUSIVE, S2 and S4
  compare answers only, and S3 and S5 expect `cold`.

**v1.1 is done when** S1–S5 pass on Qwen3.5-4B and the v1 gates still pass.
On a Llama-carrier model S1 may be INCONCLUSIVE and the rest must pass.

### 12.9 Deferred

- Taking DeltaNet checkpoints during the decide's conversation prefill, so
  that a later divergent chat turn can resume inside it.
- Tokenizing only the question tail at the `<|im_start|>` boundary, instead
  of the whole conversation per question.
- Reasoning controls for session questions (thinking-on agents), and a
  commit that also advances a loaded speculator.
