# Jev-style decide endpoint — Design Spec

> Status: **IMPLEMENTED — v1 + v1.1 session mode** · **DESIGN — v1.2 calibration (§13)** · Date: 2026-09-28 · Addenda §12, §13 · Results: `bench/jev/`
> (session mode, v1.1; design approved 2026-09-28) (calibration, v1.2; shape approved 2026-09-29)
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
- **Calibration layer.** The per-model, per-type temperature is specified in
  §13 (v1.2). Trained calibration stays deferred (§13.9).
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
   with no code anywhere. This runs in exact mode (`EXACT_ENV`, §10.1a),
   because in default mode a sibling question shifts the shared-prefix
   split point, and split-point drift alone can move answers by up to the
   noise floor (§10.1b), which would swamp the 0.05 threshold:
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
  - With `continuous_batch_size > 1`, chat turns run in batch lanes that
    keep their own caches, not `m.conversation_tokens`. A session decide
    then never finds the conversation cached: reuse is effectively never
    available and every start is cold. That is correct, only slower.
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
- **Fail-fast order.** Every render re-renders and tokenizes the whole
  conversation, and up to 128 questions run under exclusive admission, so a
  hostile oversized conversation must be refused before the questions
  render. The probe (§12.3) renders first:
  - over the eviction limit (below): 422 before any question renders;
  - over `max_seq`: 422 after R_0 only. `required_max_seq` is estimated as
    R_0's length with question 0's text swapped for the longest question's
    (each text tokenized alone) plus 16 tokens of slack. It only sizes
    serve's reload; the retry re-checks exactly.

  Then each R_i is checked as it is rendered: the eviction limit, and Jev's
  limits on the suffixes past a provisional E computed from the probe and
  R_0. The final E can only be shorter, so every early refusal is one the
  final check would also make. The exact limits and `max_seq` are checked
  once every R_i is rendered.

**CASK / eviction.** Session mode never reuses under eviction. The chat
cache is off there (`ar.rs:3025-3029`) and every chat turn cold-starts
(`ar.rs:3465`), so there is nothing to keep. The rule:

- always start cold; the rollback zeroes `compact_offset`;
- the decide never calls `maybe_evict`;
- the limit is v1's: longest R_i + 1 ≤ min(budget, `physical_cap`), else
  422 without `required_max_seq`.

Without eviction there is no compaction. `compact_offset == 0` is also
required for reuse, as a defensive check.

**Multimodal content.** Session decide is text-only. Serve returns 422 for a
`messages` content part that is not `text` (e.g. `image_url`), instead of
letting the chat projection flatten it to its text parts and drop the image.
A direct daemon request with array content fails to parse (422).

**Security.** `cached_tokens` and `start` in `x-hipfire-timing` show whether
a prefix of the request is in the daemon's cache, so a caller can learn
whether another client's conversation shares that prefix. Chat's own
`cached_tokens` already reveals the same.

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

Before projecting, serve refuses (422) a `messages` that is not a non-empty
array, one in which no message has a role the projection keeps (it would
reach the daemon as only the injected default system message, which the
daemon accepts), and non-text content parts (§12.5). The message is the
daemon's: `messages must be a non-empty array of chat messages`.

It then forwards `messages` and `tools`. When the caller also sent `state`,
serve forwards it too, so the daemon returns the 422. A projection error,
such as an invalid `tool_choice`, is a 422. There is no reset. Model
selection, admission, the `required_max_seq` retry, keep-warm and response
shaping are as in §6.

### 12.7 Errors (additions to §8)

| Condition | Response |
|---|---|
| Both or neither of `state` / `messages` (with a model loaded) | 422 |
| `messages` not a non-empty array of chat messages, or none with a chat role; `tools` not an array; bad `tool_choice` (serve) | 422 |
| A non-text `messages` content part, e.g. `image_url` (serve) | 422 |
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
- **Prefill-built vs decode-built state (human ruling).** S2 and S4 are
  split like gate 1:
  - **Prefill-built** conversations must be exact: Δ = 0 against the
    reference. This covers a cold start, a checkpoint resume, and an extend
    over a conversation a decide committed.
  - **Decode-built** conversations are held to a noise bound instead. This
    is a session decide that extends right after a chat turn, whose reply
    tokens the model generated token by token.
  - **Why the split.** The KV and DeltaNet state the decode steps leave is
    not bit-identical to a one-call prefill of the same tokens, even with
    the exact env pinned. This is the same precision property as §10's
    gate 1b: per-token kernels differ from batched prefill.
  - **The bound.** Per question, the drift against the reference must be at
    most 2 × that question's one-call vs token-by-token floor. The floor is
    measured on every run by `split_prefill_probe` in the env the gate runs
    in (exact), on a v1 decide over the short conversation's text with the
    session questions. The probe loads with `max_seq` 512, so S4's ~2k-token
    conversation reuses this floor.
  - **Reporting.** The detail line prints each question's drift and 2 ×
    floor, and names the floor's source.
- **S2 exactness (exact env).** The reference runs last, because it ends
  reset and clears `asst_turn_cache`.
  - **(a) Prefill-built.** Δ = 0 per question in two cases: a start with
    another conversation cached (not `extend`), and an `extend` over the
    conversation that decide committed.
  - **(b) Decode-built.** A warm `extend` right after the chat turn stays
    within 2 × the floor per question.
- **S3 no leak.** 1,000 session decides cycling three conversations, two of
  which share a long prefix. A chat turn per cycle re-creates the prefill
  checkpoints, so `extend`, `resume` and `cold` all run and are counted.
  The daemon's fdinfo growth must be < 64 MiB (gate 4's measure). On an
  arch without prefill checkpoints (the Llama carrier) resume cannot run:
  S3 passes on `extend` and `cold` alone.
- **S4 stale cache (exact env).** Conversation C, with its own chat turn
  cached, gives answers W (decode-built). Then:
  - with an unrelated conversation cached (`start = cold`), the answers
    must equal the one-call reference (Δ = 0);
  - with a conversation sharing a ~2.5k-token prefix cached
    (`start = resume`), the answers must also equal the reference (Δ = 0),
    so resume == cold exactly;
  - W must be within 2 × the floor per question of the reference;
  - the next chat turn on C must reuse the decide's conversation
    (`cached_tokens` == E).

  S4 is INCONCLUSIVE if no checkpoint precedes the shared prefix, since the
  resume path is then not exercised. On an arch without prefill checkpoints
  (the Llama carrier) it compares the answers only and is PASS or FAIL.
- **S5 error replies.** Both, neither, an empty `messages` and a
  non-list `messages` each return 422. An over-long conversation (over
  `max_seq`; with `--cask` over the eviction limit) with 128 questions
  returns 422 (+ `required_max_seq` > `max_seq` without `--cask`) within 4×
  the time of the same refusal with one question + 0.25 s, proving the
  fail-fast order (§12.5). After them, a session decide on the committed
  conversation must start `extend` with zero delta, proving the refusals
  left the cache untouched.
- **`--cask`.** Every start is `cold`. S1 reports INCONCLUSIVE, and S2 and
  S4 compare answers only. Every leg is prefill-built there, so every Δ must
  be 0. S3 and S5 expect `cold`.

**v1.1 is done when** S1–S5 pass on Qwen3.5-4B and the v1 gates still pass.
On a Llama-carrier model S1 may be INCONCLUSIVE and the rest must pass (S3
on `extend` + `cold`, S4 on answers only).

**S6a (speculator) evidence.** The next-turn text identity is weak evidence:
speculative decoding is lossless, so the greedy text would match even if the
decide had disturbed the drafter. The meaningful check is `cached_tokens`
equality with the no-decide baseline. Only a DFlash drafter is gated; MTP is
not.

### 12.9 Deferred

- Taking DeltaNet checkpoints during the decide's conversation prefill, so
  that a later divergent chat turn can resume inside it.
- Tokenizing only the question tail at the `<|im_start|>` boundary, instead
  of the whole conversation per question.
- Reasoning controls for session questions (thinking-on agents), and a
  commit that also advances a loaded speculator.
- The DFlash chat path's cache ends at `<|im_end|>` without the `\n`
  trailer the render has, so a session decide after a DFlash chat turn
  prefills one token (S6a tolerates it). Store the trailer there as the AR
  path does.
- `split_prefill_probe` loads with a hard-coded `max_seq` 512. A longer
  prompt used to be an illegal GPU memory access; it is now refused with a
  message before the prefill. Sizing `max_seq` from the input would let the
  floor be measured on S4's ~2k-token conversation instead of reusing
  CONV_C's.
- A thinking-on S1a variant: the chat turns run with thinking on, the
  decide renders thinking off, and the next turn must still reuse the
  cache (§12.5 "Thinking").
- An end-to-end serve GPU test: `/v1/systemone` session requests through
  `hipfire serve` with a real daemon (the route tests use a fake daemon; the
  gates talk to the daemon directly).

## 13. Calibration (v1.2)

> Addendum, 2026-09-29. The shape was approved by the human; this section
> fixes the details. §1–§12 still hold. With no calibration configured,
> every answer is bit-identical to v1.1.

### 13.1 What and why

v1 returns the model's raw probabilities, and they are miscalibrated in
both directions. On the reported rows (`bench/jev/`), Qwen3.5-4B's
banking77 top-probability ECE is 0.188 against Jev's 0.099, while its
synthetic `score` ECE is 0.091 against Jev's 0.306.

v1.2 adds **temperature scaling**: one scalar T per (model, question
type), applied to the label logits before the softmax. It is the first half
of §9's deferred calibration layer and the smallest fitted calibration
there is:

- one parameter per type;
- it can never change which answer wins (§13.3);
- it is fitted in seconds from answers the endpoint already returns, so it
  needs no new endpoint and no logit dump.

Trained calibration (Jev's RLCD) stays deferred.

### 13.2 API and configuration

**The Jev wire is unchanged.** Request, response, answer shapes and
confidence formulas are exactly §7's. A client can neither select nor see
calibration in the body. A `calibration` field in an HTTP request body is
ignored: serve never forwards it.

**Config keys.** Three request-scoped keys in the `hipfire-config` schema,
set in the per-model overlay (`~/.hipfire/models.toml`):

| Key | Type | Default | Range | Scope |
|---|---|---|---|---|
| `decide.calibration.choice` | number | `1.0` | 0.05–20 | `Request` |
| `decide.calibration.score` | number | `1.0` | 0.05–20 | `Request` |
| `decide.calibration.noul` | number | `1.0` | 0.05–20 | `Request` |

They are not registry-backed (`registry_allowed = false`) and have no
environment alias, so they are not part of the daemon's process config.
Legacy JSON spellings are `decide_calibration_{choice,score,noul}`.

```toml
[models."qwen3.5-4b.mq4".overrides.decide.calibration]
choice = 1.3
score = 1.9
noul = 0.7
```

- The inline form `overrides = { decide = { calibration = { choice = 1.3 } } }`
  is equivalent.
- CLI: `hipfire config qwen3.5-4b.mq4 set decide.calibration.choice 1.3`.
  For a model with no catalog record yet (a path-only model such as
  `qwen3.8-27b.mq4-xts`) this creates the record with its path, as for any
  per-model key.
- A key that is not set means T = 1 (raw) for that type.
- The schema also accepts the keys in the global `config.toml`, as it does
  every request key. That would apply one T to every model, which is wrong
  for calibration; the docs say to set them per model.

**Serve.** After `ensure_model`, serve reads the three values from the
config resolved for the model actually used. That is the same
`resolved_for_model` layering chat uses for `generation.temperature`
(registry card < global < per-model < env).

- If any value differs from 1, serve adds
  `"calibration": {"choice": Tc, "score": Ts, "noul": Tn}` (all three) to the
  daemon `decide` message.
- Otherwise it adds nothing, so an uncalibrated model's daemon message is
  byte-identical to v1.1.
- This applies to plain and session requests alike. A `required_max_seq`
  reload re-resolves the config.
- Serve resolves the config on every request, so `hipfire config … set` takes
  effect on the next decide without a reload.

**Daemon wire.** `{"type":"decide", …, "calibration"?: {"choice"?, "score"?, "noul"?}}`.

- Absent or `null` means the identity. Each key is optional and defaults to 1.
- Values must be numbers in [0.05, 20].
- A non-object, an unknown key or an out-of-range value gets a 422 naming
  `calibration.<key>` (§13.8).
- `hipfire-daemon/src/main.rs` does not change:
  `handle_decide_message` already receives the whole message.
  (`daemon_lines` sits exactly at its 4882 ceiling.)

**Timing header.** When the applied calibration is not the identity, the
daemon's `timing`, and so `x-hipfire-timing`, carries
`"calibration": {choice, score, noul}`. It is diagnostic and never appears
in the Jev body.

**Why the request path and not `LoadedModel`.** Both were considered; the
request path is the consistent and cleaner one.

- **Consistency.** Per-model settings that shape a request's *output*, such
  as `generation.temperature`, `top_p` and `prompt.system`, are resolved by
  serve per request and sent in the daemon message. Only settings that change
  what the load *allocates* (`kv_cache`, `speculation`, CASK) travel as load
  params into `LoadedModel`. Calibration is an output setting.
- **No reload.** A changed T applies to the next request. On `LoadedModel`
  it would need a reload of the weights.
- **Ratchet.** `LoadedModel` would need a new load parameter parsed in the
  daemon's load arm, which sits at the `daemon_lines` ceiling. The request
  path adds no daemon lines.
- **Gates stay raw.** Direct daemon clients (`gates.py`, the eval harness)
  choose calibration per message and are raw by default, so the v1/v1.1
  gates keep testing the raw readout.
- **Backend-agnostic.** Nothing is tied to a loaded LLM. The engine applies
  T to whatever label scores a decide backend produces (§13.3).

### 13.3 Math

Let z₁…z_K be the question's label scores: the gathered next-token logits,
in `labels_for` order. The answer uses

  p = softmax(z / T), computed in f64 as exp((zᵢ − max z)/T) / Σⱼ exp((zⱼ − max z)/T).

Everything downstream is §7's formula applied to the calibrated p:

- `choice`: `probabilities` = p; `choice` = argmax; confidence =
  (p_max − 1/K)/(1 − 1/K);
- `score`: `probabilities` = p; `score` = Σ i·pᵢ; confidence = p_max;
- `noul`: `noul` = p_Yes = σ((z_Yes − z_No)/T).

Invariants:

1. **T = 1 is bit-identical to v1.1.** Division by 1 is exact in IEEE 754,
   so the f64 values are the same bits.
2. **T > 0 preserves order.** The argmax and the ranking are unchanged, so
   `choice`, the `noul` ≥ 0.5 decision and accuracy do not move. The
   probabilities, confidences and the `score` mean do.
3. **Direction.** T > 1 flattens (less confident); T < 1 sharpens.
4. **Shift invariance.** softmax((z + c)/T) = softmax(z/T). Recorded
   probabilities give back the label scores up to a constant:
   log pᵢ = zᵢ − logsumexp(z), so softmax(log p / T) = softmax(z / T)
   exactly. Fitting and evaluation therefore work from the answers the
   endpoint already returns (§13.4, §13.6).

**Backend-agnostic.** The engine API is
`assemble_answer(question, label_scores, T)` plus `assemble_answers(questions,
per_question_scores, &Calibration)`. By invariant 4 the scores may be
logits or log-probabilities.

- A future decide backend, such as the Laya ModernBERT decision model under
  research, hands its per-label scores to the same functions.
- Its temperatures are fitted by the same protocol (§13.4) on its own
  answers, and stored under its own model id.

### 13.4 Fitting protocol

- **Unit.** One T per (model artifact, build, question type), pooled over
  every held-out source of that type.
- **Data.** Held-out rows only (§13.5). They are answered raw (no
  calibration configured), one question per request, by the same build and
  serve configuration as the reported rows:
  - build: `eval/jev-decide-mq4-lloyd` at 237bb7bba, i.e. `feat/jev-decide`
    plus PR #768;
  - config: KV q8, CASK on, budget 16384.
- **Row.** `{source, type, probs, gold}`: `probs` as served, in
  answer-key order, `gold` an index into them, ℓ = log p. There is no text,
  and no option keys, which would dominate the file size.
- **Objective.** Mean negative log-likelihood (log loss):
  NLL(T) = −(1/N) Σₙ log softmax(ℓₙ / T)[yₙ].
- **Pooling.** At most 300 rows per source (the first 300, in the order the
  held-out builder kept them), so no source dominates a type.
- **Optimiser.** NLL is convex in β = 1/T.
  - The derivative is dNLL/dβ = mean(E_q[ℓ] − ℓ_y), and it is monotone,
    since the second derivative is mean Var_q[ℓ] ≥ 0.
  - Safeguarded Newton on it, with β bracketed in [1/20, 1/0.05].
  - A fit that lands on a bracket end is an error and is not shipped.
- **Uncertainty.** 100 bootstrap resamples of the pooled rows
  (`random.Random(0)`) give a 95% interval for T.
- **Ship rule.** Decided on held-out rows only. A type ships its T iff:
  - the held-out NLL drops by ≥ 1% relative, and
  - the 95% interval excludes 1.

  Otherwise the type stays at T = 1, and its key is omitted. T is rounded
  to 3 decimals.
- **Output.** `bench/jev/<model>/calibration.json` holds:
  - per type: T, its interval, n per source, raw and calibrated held-out
    NLL, and `ship`;
  - the `decide_calibration` object to configure.

  The fitter also prints the `models.toml` snippet.
- **Reported rows never feed the fit.**
  - `calibrate.py fit` only accepts a directory named `heldout/`, and the
    ship rule reads nothing else.
  - The reported rows are only ever *evaluated* (§13.6), after T is fixed.
- **Build pinning.** T belongs to a build's numerics. The PR #768 prefill
  routes move 27B answers by up to about 1 log-prob (§11 caveats). Refit
  when the serving build's decide numerics change, for example when #768
  merges, is dropped, or its routes change defaults.

### 13.5 Held-out data and row selection

**Reported set R** is every state that appears in a reported row:

- (a) the 12 jev-bench tasks' fixed-seed rows: each task constructor at
  n = 500 with `jevbench.SEED` = 0, served from the same dataset cache;
- (b) every row of both reported runs' saved raw answers
  (`jevbench/raw/*.jsonl`), including the six experiments. `x-oos` adds
  CLINC out-of-scope test rows;
- (c) jev-ood-calibration's `data/val.jsonl` (300 tickets × 3 questions);
- (d) OpenBookQA, CommonsenseQA and HellaSwag validation rows as the harness
  rebuilds them: the first 500, 1,221 and 2,000 rows.

**State identity.** `state_hash` is the SHA-256 of the state text after
`strip()`:

- a string state is used as-is;
- an object or array is JSON with sorted keys, `(",", ":")` separators and
  non-ASCII kept (`ensure_ascii=False`), UTF-8 encoded.

R is committed as `bench/jev/heldout/reported_state_sha256.txt` (sorted,
unique). The build fails if (b) is missing for either reported model, so R
can never silently shrink.

**Candidates.**

- jev-bench candidates come from each task's *own* constructor, with its
  loader redirected (`hf_rows` → the held-out split, n = 400, seed 1;
  banking77's CSV URL `test.csv` → `train.csv`). Field and label mapping
  are therefore the reported code's.
- Every candidate is asked the *reported* question, with the same
  instructions and the same criteria in the same order.
- A candidate whose label is not among the reported criteria is dropped,
  for example a MASSIVE intent absent from its test split.

| Source | Type | Reported rows from | Held-out candidates from |
|---|---|---|---|
| banking77 | choice | `test.csv` | `train.csv` (same repo), constructor order, first 400 |
| massive-en, massive-it | choice | test | validation, `hf_rows` seed 1, 400 |
| clinc150 | choice | test (`plus`) | validation (`plus`), seed 1, 400, in-scope only |
| ledgar | choice | test | validation, seed 1, 400 |
| ag-news | choice | test | train, seed 1, 400 |
| sms-spam | noul | train (its only split) | train, seed 1, 400; disjoint by R-exclusion only |
| duplicates (QQP) | noul | validation | train, seed 1, 400 |
| doc-yesno (BoolQ) | noul | validation | train, seed 1, 400 |
| offensive | noul | test | validation, seed 1, 400 |
| yelp-stars | score | test | train, seed 1, 400 |
| sentiment-it | score | test | validation (324 rows), seed 1 |
| synth-choice, synth-score, synth-noul | choice, score, noul | `val.jsonl` (`generate.py` seed 0) | `generate.build_synthetic(400 tickets, label_noise 0.05, seed 7)`; disjoint by R-exclusion only |
| openbookqa | choice | validation (first 500) | train: first 400 rows, filtered, `Random(1)` shuffle, as the harness builds validation |
| commonsense_qa | choice | validation (all 1,221) | train: as openbookqa |

HellaSwag's train split is not used. The approved public held-out sources
are the OpenBookQA and CommonsenseQA train splits, and `choice` already has
nine sources. HellaSwag's reported validation rows are still in R.

**Selection, per source, in candidate order.**

1. Drop every candidate whose `state_hash` ∈ R.
2. Drop repeats within the source (the same `state_hash` again).
3. Keep the first 300.

This gives 17 sources: 2,700 choice, 900 score and 1,500 noul rows per
model. A CPU dry run of the builder on 2026-09-29 kept 300 from every source
and found R to have 10,253 states. Its losses were:

- **sms-spam: 66 of 400.** Both draws sample 100-row pages of the one split,
  so pages overlap.
- **massive-it: 2.** Short utterances whose Italian text also occurs in the
  test split.
- **each synth type: 15 of 400 tickets.** They are identical to a
  `val.jsonl` ticket, from the placeholder-free templates. Held-out synth
  therefore slightly under-represents those templates. This is accepted.
- **Label filter:** 1 MASSIVE row each (EN, IT).
- **Every other source: none.**

**Proof of no overlap.**

1. **By construction.** H ∩ R = ∅, because rule 1 drops every candidate in
   R. `heldout.py build` asserts it per source, and `heldout.py verify`
   re-checks it from committed files alone:
   - the R hash list;
   - the manifest, which pins each source's examples file by SHA-256;
   - the work-dir examples.
2. **Different splits support it, but do not replace it.** 11 of 12
   jev-bench sources and both public QA sets come from a split that holds no
   reported row. Only sms-spam and synth share a split or generator with
   reported rows. A different split still does not guarantee different
   text: massive-it's validation split repeats two test utterances. That is
   why rule 1 runs on every source.
3. **By the fitter.** Labels as well as rows: `calibrate.py fit` refuses any
   directory but `heldout/`, so no reported label reaches the fit.

**What is committed.** Held-out text stays out of the repo, the way
jev-bench keeps its own `raw/` and `data/` out. It lives in the work dir
`~/.cache/hipfire-jev-calib/`. Committed:

- `bench/jev/heldout/manifest.json`: per source, the type, split, seed,
  counts drawn / label-filtered / excluded / kept, and the examples SHA-256;
- the R hash list;
- per model, `bench/jev/<model>/heldout/<source>.jsonl` answer rows, which
  hold no text.

### 13.6 Evaluation protocol

**Do the saved predictions suffice? Yes. No re-scoring, no GPU rerun.**

- jev-bench's committed-style `predictions/*.jsonl` are **not** enough.
  They keep `p_top` (4 dp) and `correct` only. `p_top` alone cannot be
  rescaled, because the rest of the distribution sets the normaliser of
  softmax(log p / T).
- `jevbench.save` also writes `raw/*.jsonl`, which holds each row's full
  `prob` distribution at full precision. The calibration-set adapter's rows
  keep full `probs` too.
- Both reported runs have these files, in the eval worktree
  (`~/repos/hipfire-jev-eval/bench/jev/<model>/`, untracked). There are 12
  jev-bench tasks + 6 experiments + 4 sets for 27B, and 3 tasks + synth for
  4B.
- Every saved probability is > 0 (minimum 7.9e-12), so log p is finite.

**Freeze.** `calibrate.py freeze` copies the reported rows into the repo as
text-free rows, `bench/jev/<model>/probs/`, with probabilities kept to 12
significant digits. It checks each jev-bench row against that task's saved
prediction: the top probability within 5e-5 of `p_top`, and the same
correctness. This proves they are the reported rows. It records the source
files' SHA-256 in `probs/SOURCES.json`. After the freeze, the evaluation no
longer depends on the throwaway eval worktree.

**Report.** `report.py` reads the frozen rows and `calibration.json` and
reports, per jev-bench task and per calibration set × type:

- n;
- Jev accuracy, hipfire accuracy, Jev ECE;
- hipfire ECE raw and calibrated;
- hipfire NLL raw and calibrated;
- for `score` rows, the mean |E[score] − gold| raw and calibrated.

Details:

- "Calibrated" applies each type's shipped T (T = 1 for a type not
  shipped) with the same `apply_t` the fitter uses.
- The report asserts that no row's argmax moved.
- ECE is jevbench's 10-bin top-probability ECE.
- The raw column is recomputed from full-precision probabilities, so it can
  differ in the third decimal from the v1 report, which used `p_top` rounded
  to 4 dp, when a value sits on a bin edge.

**No bar.** The numbers are reported as found. A type whose calibrated ECE
is worse on the reported rows is reported that way, not refitted: refitting
to reported rows would leak them into the fit.

### 13.7 Gates and tests

**Unit, no GPU** (`scripts/no-gpu-ci.sh`):

- `hipfire-engine`:
  - `parse_calibration`: absent / `null` / partial / unknown key /
    non-object / out of range;
  - T = 1 bit-identical to the v1 formula;
  - T scales the scores before the softmax;
  - shift invariance, i.e. recovery from log p;
  - flatten / sharpen with the argmax kept;
  - the choice confidence formula on the calibrated p;
  - noul = σ(log-odds / T);
  - the score mean on the calibrated p;
  - `assemble_answers` applies each question type's T;
  - plain and session requests carry `calibration`.
- `hipfire-generate`: the timing echo appears only for a non-identity
  calibration.
- `hipfire-config`: the keys' scope, default, range and flags; nested and
  inline `models.toml` overrides load and round-trip; out of range is
  refused.
- serve (fake daemon):
  - a per-model override is forwarded with all three values;
  - with no override, no `calibration` key is sent;
  - a client's body `calibration` is never forwarded.
- Python (`scripts/jev_eval/test_*.py`):
  - `apply_t` identity and shift invariance;
  - the fit recovers a known T on synthetic data, and the NLL is minimal at
    the fit;
  - a fit at a bound raises;
  - ECE matches jevbench's binning;
  - `summarize` keeps the argmax;
  - freeze row shapes;
  - the fit refuses a non-`heldout/` directory;
  - `state_hash` normalisation;
  - selection drops R members and repeats and caps at 300;
  - `verify` catches an overlap and a changed examples file.

**GPU:**

- **L1 live equivalence.** Calibration-branch build, Qwen3.5-4B, daemon
  driven directly.
  - Precondition: the same raw request repeated is bit-identical. If it is
    not, the result is INCONCLUSIVE, reported with the Δ.
  - For 5 held-out examples per source, and for one session-mode pair
    (cold, then extend):
    - the calibrated answer equals `apply_t(raw answer)` within 1e-9 per
      probability;
    - `choice` is unchanged;
    - confidence and score follow §7 on the calibrated p;
    - `timing.calibration` echoes the temperatures.
  - An explicit `{"choice":1,"score":1,"noul":1}` is bit-identical to no
    calibration.
  - A malformed `calibration` gets a 422, and a normal decide succeeds after
    it.
- **v1 and v1.1 gates.** `gates.py` (raw) still passes on Qwen3.5-4B with
  the calibration-branch build.
- **`heldout.py verify`** passes.

**v1.2 is done when:**

- every unit test, L1 and the gates pass;
- both models have `calibration.json`, regenerated reports (raw vs
  calibrated) and a `models.toml` snippet.

Writing the snippet into the user's `~/.hipfire/models.toml` is a separate,
human-approved step (§13.9).

### 13.8 Errors (additions to §8 and §12.7)

| Condition | Response |
|---|---|
| `calibration` present but not an object (daemon wire) | 422 `calibration must be an object {choice, score, noul}` |
| `calibration` has a key other than `choice` / `score` / `noul` | 422 naming the key |
| A `calibration` value that is not a number in [0.05, 20] | 422 `calibration.<key> must be a number in [0.05, 20]` |
| `decide.calibration.*` invalid in `models.toml` / `config.toml` | config load error: `hipfire config … set` refuses it; serve's `ensure_model` fails (500) |

Serve never sends an invalid value, because the schema validates on load.
The 422s guard direct daemon clients.

### 13.9 Caveats and deferred

- **Older binaries reject the new keys.** `load_catalog_toml` refuses any
  unknown override key. Once `~/.hipfire/models.toml` carries
  `decide.calibration.*`, every hipfire built before this change fails to
  load the catalog and stops at startup or model resolution. That includes:
  - the installed `~/.hipfire/bin`;
  - other worktrees' builds;
  - the eval build.

  Write the keys only once every hipfire that reads that catalog is at or
  past this change, or point older binaries at another `HIPFIRE_HOME`.
- **Deferred:**
  - per-task or per-K temperatures, vector or matrix scaling, and trained
    calibration (RLCD);
  - shipping stable temperatures through the registry
    (`registry_allowed`), once a fit holds across builds;
  - a separate session-mode fit. Session answers use the plain-mode T,
    since the question block is the same text; only the context differs.
