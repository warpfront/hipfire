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

1. Render and tokenize the shared prefix (§5). Prefill it in
   `prefill_max_batch(gpu)` chunks via `qwen35::forward_prefill_batch`
   (Llama: `llama::forward_prefill_batch`), starting from position 0 of a
   reset model.
2. Snapshot: record `P = seq_pos`. On Qwen3.5, also
   `DeltaNetSnapshot::new_for` + `save_from(&dn_state)`.
3. For each question, in request order:
   1. restore (`seq_pos = P`, `restore_to(&mut dn_state)`);
   2. prefill the question suffix;
   3. download `scratch.logits`;
   4. gather the logits of that question's label token ids.

   Between questions, check the out-of-band `abort` flag.
4. Cleanup:
   - leave the model reset: `seq_pos = 0`, recurrent state reset via
     `common::reset_qwen35_recurrent`;
   - clear `conversation_tokens` and the prefill checkpoints, so the AR
     prompt cache can't match decide tokens.

   A decide does not preserve a cached chat conversation; it behaves like
   serve's existing per-request reset.
   - call `DeltaNetSnapshot::free_gpu` explicitly. It has no `Drop`, so
     skipping this leaks device memory.

Preconditions: refuse to run when KV eviction (CASK) is active. Positional
rewind is only valid when `compact_offset == 0`.

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
- **Tokenization.** The prefix and suffix are tokenized separately and
  concatenated at a newline boundary; the Qwen pre-tokenizer always splits on
  `\n`. The template comes from the model's own chat template via
  `JinjaChatFrame` with thinking closed (`ClosedThink`), falling back to
  `ChatFrame`.
- **Labels:**
  - `choice`: codes `A`–`Z`, then two-letter codes, in criteria order.
  - `score`: levels labelled `0`…`k-1`, with each level's description.
  - `noul`: `Yes`/`No`, with the criteria descriptions when provided.
- **Code check.** At model load, and cached per model, every candidate code is
  checked with `tokenizer.encode(code).len() == 1` in its readout form. Codes
  that fail are skipped. If fewer than K valid codes remain, the request gets
  a 422.

## 6. Request lifecycle (serve)

1. `POST /v1/systemone` with `{model, state, questions}`. Serve checks body
   size and validates the request (§7) before touching the daemon.
2. **Model selection.**
   - If `model` names a local hipfire model (registry tag or path), serve
     calls `ensure_model`.
   - Any other value, including Jev's `jev-latest`, means the currently loaded
     model.
   - If no model is loaded, return 503 with an explanatory message.
   - The response `model` field is the hipfire model actually used.
3. **Admission.** Exclusive gate, the same as a sequential chat request. When
   saturated: 503 + `Retry-After`.
4. Forward `{"type":"decide","id",state,questions}` to the daemon.
   - The daemon verifies that state plus the longest suffix fits within
     `max_seq`. Serve requests a larger context through `ensure_model`'s
     `min_max_seq`.
5. Shape Jev's response `{model, answers, usage}`.
   - Per-phase timings (state prefill, per-question suffix, readout) go in an
     `x-hipfire-timing` header and the log. No extra body fields are added.

## 7. Wire format (Jev-compatible)

Request validation, returning 422 with the question and field named:

- `state`: string, JSON object or JSON array.
- `questions`: non-empty map, name → question. Each question needs
  `instructions` (string) and a `type`:
  - `choice`: `criteria` is an object with 2–255 keys (key → description);
  - `score`: `criteria` is an array of 2–10 level descriptions, low → high;
  - `noul`: optional `criteria` `{true, false}`.

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
| Malformed request / limits exceeded | 422, naming question and field |
| Model architecture has no snapshot path (neither Qwen3.5 nor Llama family) | 400 `decide not supported for <arch>` |
| No model loaded and `model` not a local model | 503 |
| KV eviction active | 409 |
| Prompt exceeds the maximum `max_seq` | 422 |
| Fewer than K valid single-token codes | 422 |
| Admission saturated | 503 + `Retry-After` |
| Daemon crash / GPU error | 500; the existing serve restart path |
| Client disconnect | `abort`; the daemon stops between questions and runs the §4.1 cleanup |

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

**GPU correctness gates** (all must pass to merge):

1. **Snapshot exactness.** For each question, decide's label log-probabilities
   equal a from-scratch prefill of that question's full prompt (no snapshot),
   within tolerance. Run on Qwen3.5 (DeltaNet restore) and a Llama-family
   model.
2. **Isolation.** The "secret code" probe: a fact in a sibling question does
   not raise another question's probability; the same fact in the state does.
3. **Model left clean.** A greedy `generate` after a decide yields exactly
   the same tokens as a `generate` after a plain `reset`. This covers both a
   prompt that shares a prefix with the decide state (must not hit a stale
   LCP cache) and one that doesn't.
4. **No leak.** Device memory is flat over 1,000 decides.

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
