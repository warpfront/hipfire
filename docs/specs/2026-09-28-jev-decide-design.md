# Jev-style decide endpoint — Design Spec

> Status: **DRAFT — awaiting review** · Date: 2026-09-28
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

   A decide does not preserve a cached chat conversation; it behaves like
   serve's existing per-request reset.
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

Token limits (Jev's, §3), checked by the daemon after rendering, 422 naming
the limit and without `required_max_seq`:

- state + the longest question (that question's full rendered prompt) ≤
  32,000 tokens; checked per question as it is rendered, so an over-long
  state fails on the first question;
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
