<!-- SPDX-License-Identifier: Apache-2.0 -->

# SCS suite — serving-cache-scheduler black-box test suite

`scripts/scs_suite.py` is a 69-cell black-box test suite for the
`feat/serving-cache-scheduler` branch, driven entirely over the
OpenAI-compatible HTTP surface of a running `hipfire serve` container. It was
built to exercise every externally observable feature of the branch against
the live `qwen35-4b-mq4v1-multislot` container (4 slots, ctx 50000, prefix
cache on, `json-schema-strict-v1`, jump-forward on, `.mtp` + `.vl` sidecars
present, queue timeout 300 s).

It complements, and does not replace, the other verification layers:

| Layer | Tool | What it sees |
|---|---|---|
| Host unit tests | `cargo test --workspace --lib` | pool/index/fairness/grammar internals |
| GPU compose oracle | `test_serve_prefix_cache --mtp-k N` | engine-level reuse + fault cells (A19/A20) |
| Product harness | `scripts/serve_harness.py` | generic serve semantics, chain/battery/session |
| **This suite** | `scripts/scs_suite.py` | **the branch's user-facing contracts end to end**, incl. areas no other layer covers over HTTP (capability advertisement, 503/cancel/stall behavior, schema×cache composition, strict-JSON framing, error-status mapping) |

## Usage

```bash
# against the default container (127.0.0.1:8420)
python3 scripts/scs_suite.py                       # full suite, ~5-8 min
python3 scripts/scs_suite.py --list                # list all cells
python3 scripts/scs_suite.py --area C              # one area (A B C D E F G V H)
python3 scripts/scs_suite.py --cell E2 --cell D3   # specific cells
python3 scripts/scs_suite.py --json-out report.json --docker-name <container>
```

Exit code 0 iff no FAIL. A companion JSON report (per-cell status, evidence,
prompt md5s, health snapshot, container binary md5s) is written after every
cell, so a killed run still leaves a partial report. All request timeouts
scale with `--timeout-scale`; the default non-stream timeout is 150 s (the
daemon's queue timeout on this container is 300 s, so an admission park fails
the cell before the daemon would time it out).

**Exclusivity:** the suite assumes it is the only client of the serve
endpoint. Concurrent external traffic produces spurious failures in the
timing-sensitive cells (D1/D3/D6 parallelism and overload, B7/B8/C2/C5
determinism) and skews the VRAM headroom the crash-sensitive areas probe.
The runner waits for a resident model before starting, but not for a quiet
endpoint.

## Coverage map

| Area | Branch feature | Cells |
|---|---|---|
| A | capability advertisement (`/health` capabilities, per-model projection on `/v1/models`, `/stats`, Prometheus `/metrics`, 404/OPTIONS) | A1-A6 |
| B | generation + sampling contract (shapes, SSE frames, usage chunk, reasoning separation, coherence battery, greedy/seeded determinism, sampling params, finish=length) | B1-B10 |
| C | prefix cache correctness: cold/warm reuse, warm replay identity, A10 long-gen replay, branch + mid-page fork (W10-4), multi-turn chain, false-reuse guard, concurrent shared-prefix isolation (COW), soak stability, gen-param independence | C1-C9 |
| E | strict JSON Schema: valid output + determinism, streaming schema, framing-aware cursor (W10-2), enum/const, min/maxItems, required/closed, nested, unsupported keywords (P0-2), malformed keyword values, union type, bare object, structural errors, integer dead-end (P1-7), `json_object`/`text` refusal, schema×cache composition | E1-E16 |
| F | typed refusals on the multi-slot route: tools, stop, logprobs, tool-role messages, reasoning_effort/think-cap/budget, min_p range, `n`, `logit_bias` | F1-F9 |
| G | HTTP limits + error mapping: invalid JSON, unknown model, max_tokens bounds, 413 Content-Length, seed validation, empty messages, ctx overshoot | G1-G7 |
| V | vision×radix isolation (A18: repeated image request must show `cached_tokens 0`) | V1 |
| D | scheduler/admission: two-slot parallel generation + progressive service (wave-12), four-slot saturation, queue-full 503 + Retry-After, post-burst drain (pop_ready regression), cancel storm, long/short prefill interleave, queue timeout (env-dependent) | D1-D7 |
| H | lifecycle canaries: post-gauntlet health + long-prompt liveness, cold-cycle stability, idle eviction (env-dependent), model swap (out of scope for a single-model container) | H1-H4 |

Design invariants enforced across cells:

- **Reuse preserves semantic correctness and request isolation.** Greedy replay is byte-compared where the execution shape is unchanged. B8/C5/D1 record cross-shape divergence as WARN because batch-invariant reductions are design-only; `seed` does not promise cold/warm or solo/batched byte identity.
- **Eyeball-equivalent coherence.** Every long generation goes through an
  n-gram attractor detector (uniq / maxfreq / trigram thresholds mirroring
  `serve_harness.py`); genre prompts additionally assert on-topic substrings.
- **Prompt md5s recorded** for every fixture (in the JSON report), per the
  byte-identical-prompt rule.

## Results on qwen35-4b-mq4v1-multislot (2026-09-13)

Seven full runs. Runs 1–5 were executed while a second agent ran concurrent
workload on the same GPU and endpoint; runs 6–7 are the clean re-verification
(quiet endpoint, hipfire the only GPU client, fresh engine each run). The
container runs binary `dd5faaeaf87c68532c8f57876b4f4b16` (image built
14:23 +04) for ALL runs — the branch's post-17:00 fix commits
(`e80f3e110`…`13f9a27fc`) are NOT in this image, so these runs do not verify
those fixes; rebuilding the container is the follow-up.

### Poisoning forensics

* VRAM attribution via amdgpu sysfs: hipfire stopped → **174 MiB used**;
  hipfire idle-resident → **9181 MiB / 12272 MiB (75%)**. The 99% readings
  taken earlier (12.76/12.87 GB) carried ~3.5 GiB of a foreign workload.
  `/usr/bin/rocm-smi`, present early in the campaign, later disappeared from
  the host.
* The clean runs (6–7) had zero daemon deaths, zero OOM, and a passing vision
  cell — the run3/run5 crashes, the run5 vision OOM, and the run1 admission
  parking do **not** reproduce without the co-tenant and are attributed to
  it. Keep them in mind as a deployment-shaped risk (the engine's hipMalloc
  failure mode is poison-and-exit), but they are not branch bugs on the
  evidence available.
* The "phantom `cached_tokens`" (cold prompts reporting 640–768 reused) was
  observed only in run2 under co-tenant conditions and not in runs 3, 4, 6,
  7 — unconfirmed; treat with suspicion.
* B8/C5/D1 expose the documented kernel batch-invariance limitation: clean-engine
  divergence across cold/warm or solo/batched shapes is a warning, not a cache
  correctness failure. Same-shape greedy replay and cross-request isolation remain
  hard contracts.

### Clean verdict (runs 6–7, quiet engine, old binary)

### Confirmed real on the clean engine (every clean run)

| Cell | Symptom | Severity / class |
|---|---|---|
| E2 | Streaming strict-schema generation emits **all content tokens, then no terminal chunk, no `finish_reason`, no `data: [DONE]`** — the stream stalls and the server closes it ~30 s later. In run7 the streamed content itself was grammar-corrupt (`"leg\": "` mangled key + tab run). Non-schema streams (B2/B3) are fine. | **P1** — SSE contract break + content corruption on the schema stream path |
| E3 | Valid schema + default thinking → typed 400 `grammar constraint reached an unsatisfiable state` (fail-closed), or — in runs 6/7, immediately after E2's stalled stream — a **150 s admission park** (post-stall wedge signature). | **P1** — framing cursor (W10-2) ineffective + stall-follow-on wedge |
| B8 | Same seed + identical request produced different outputs across cold/warm execution shapes | **WARN** — batch-invariant reductions are design-only |
| C5 | Multi-turn final-turn replay differed from the original in-context answer while history and cache accounting remained valid | **WARN** — same cross-shape limitation |
| D1 | Batched concurrent outputs differ from solo baselines; coherence + topical isolation held | **WARN** — same batch-shape sensitivity |
| E12 | `minItems > maxItems` → **500** (server_error), message proves the CLI validator detected it ("contradictory schema") | P2 — error-status mapping |
| E15 | `response_format {"type":"text"}` → **500**. `text` is the OpenAI *default* type | P2 — compat + status mapping |
| F8 | `n: 2` → **200** (silently accepted, single choice) — `validate_generate_caps` documents `n` as refused | P2 — refusal gate not reached on the serve path |
| F9 | `logit_bias` → **200**, silently ignored | P2 — same class |
| G3 | `max_tokens: -5` → **200** (silently treated as absent → uncapped); `0` and `400000` correctly 400 | P2 — validation gap |
| G5 | `seed: -1` → **500**; SERVE.md documents "400-style error" (fractional seeds correctly 400) | P2 — contract drift |
| G6 | `messages: []` → **500** server_error | P2 — client-input error as 5xx |
| G7 | WARN: ctx overshoot → **500**; server stays healthy; should be 4xx | P2 — status mapping |
| H2 | Post-flood warm miss: identical prompt re-run gets `cached_tokens 0` after the D3 overload (runs 4, 6, 7) | P2 — reuse reliability after admission pressure; needs `pool_free_pages` visibility to distinguish eviction from publication failure |

### Poisoned / not reproducible clean

| Symptom | Runs | Verdict |
|---|---|---|
| Serve/daemon death (clean exit 0 mid-schema-area; `hipMalloc: out of memory` at vision forward poisoning the daemon) | run3, run5 only | **Co-tenant VRAM pressure** — clean runs never die; VRAM attribution above. Re-test only if a real co-tenant returns. |
| Vision request OOM | run5 only | same |
| Admission parking (700-token requests park in the daemon wait queue until the 300 s timeout; widens to all long prompts) | run1 only, on an engine with hours of foreign traffic | **Attributed to the co-tenant** holding slots/VRAM. The signature to watch for: `queue_depth` frozen ≥1 with no admission and no rejection log line. H1's long-prompt canary guards this. |
| Phantom `cached_tokens` (cold prompts reporting 640–768 reused from another passage's chain) | run2 (C2, C6, H2); run1 C1 | **Unconfirmed** — never reproduced clean (runs 3, 4, 6, 7). Keep the C1/C6/H2 guards; if it ever reproduces on a quiet engine, capture the JSON evidence immediately. |

### The run1 degradation episode (first run, aborted)

The very first full run (against an engine with ~80 requests of prior
foreign traffic) showed a *progressive admission wedge*: first seeded/sampled
~700-token requests parked in the daemon wait queue for exactly
`queue_timeout_ms` (300 s) while greedy requests kept being served; within
minutes **every** ~700-token prompt parked, including previously-successful
ones, with `queue_depth 1` frozen and no admission/rejection log line.
Restarting the container cleared it completely; the identical requests then
passed. A fresh-engine full run (run2) did not reproduce the parking, but did
reproduce the phantom `cached_tokens` (above). Interpretation: a slow
state-corruption class in the admission/radix path that several hours of
traffic (including vision + schema requests from before the suite) can
trigger. The H1 long-prompt liveness canary exists to catch this class.

### What passed on the clean engine (stable green)

Capability advertisement and telemetry (A1-A6), non-stream/stream shapes and
usage chunks (B1-B3, B5), coherence battery (B6), greedy determinism (B7),
sampling-param acceptance (B9), finish=length (B10), the entire
cold/warm/branch/fork/COW/soak cache block (C1-C4, C6-C9), strict-schema
valid output + streaming-assembly + min/maxItems + nested + all typed schema
rejections (E1, E5-E11, E13, E14, E16 — schema prompts DO compose with the
prefix cache), the multi-slot refusal set (F1-F7), HTTP limits (G1, G2, G4),
vision skips the radix with no OOM (V1), the whole scheduler block (D2-D6:
flood 429s with Retry-After, drain after burst, cancel storm, progressive
service), and the post-gauntlet canaries (H1, H2 cold-side).

## Relationship to the parallel fix campaign

The later fix campaign repaired grammar, stream accounting, and error contracts.
Seed parking makes sampling deterministic for identical logits, but does not make
shape-dependent kernels batch-invariant. B8/C5/D1 therefore remain observational
warnings until the opt-in mode planned in `2026-09-13-batch-invariance-design.md`
lands; they are not acceptance failures.

## Post-merge re-verification (2026-09-14, upstream v0.3.1 merged)

Container rebuilt from merge commit `de1455476` (upstream/master @ v0.3.1
merged into the branch). Clean single run on a fresh engine:
**63 pass / 2 fail / 1 warn / 3 skip** (report:
`.codeinsight+research/scs-post-merge-2026-09-14-r3.json`).

Fixed by the post-17:00 commits + upstream merge (all verified green):
E2/E3/E4 (grammar stream + framing cursor + dead-end), E12, E15, F8, F9,
G3, G5, G6, G7, H2.

Still failing (unchanged, the known determinism cluster): **B8** (seeded
sampling cold/warm divergence) and **C5** (multi-turn final-turn replay).

Contract updates from the merge (cells updated, now green):

* **F1/F4** — upstream `ad6004ac0` restored tool turns on the daemon slot
  path: tools are accepted, `tool_calls` finish is emitted, tool-result
  continuations work; a tool message without `tool_call_id` is a typed 400.
  The old refusal assertions were replaced with support assertions.
* **B4** — upstream's fail-closed terminal turns a think span truncated by
  `max_tokens` into a 500 `unsafe multi_slot terminal: open_think`. The cell
  now uses `max_tokens=3000` (span closes, separation verified) and asserts
  the truncated-think 500 contract explicitly.

Note: a second back-to-back suite run on the same engine hit a host OOM
(daemon 16.4 GB anon RSS after ~480 cumulative requests) — consistent with
the long-lived-engine degradation class noted above; keep one suite run per
engine lifetime.

## Notes for triage

- The JSON reports contain full per-cell evidence, all fixture prompt md5s,
  the `/health` snapshot (config actually in force), and container binary
  md5s — attach one to any issue.
- The determinism cluster (B8/C2/C5, D1-WARN) and the phantom-`cached_tokens`
  cluster point at the same subsurface: greedy/sampled bit-exactness under
  changing batch composition, and the radix walk's verification of matched
  pages. Engine-side, `EngineStats.pool_free_pages` / `reused_tokens` are not
  exposed over HTTP — adding a stats surface would let black-box runs
  distinguish eviction from publication failure (H2) and honest reuse from
  accounting lies (phantom).
- The run3/run5 daemon deaths are attributed to the second agent's GPU
  workload (see poisoning forensics): with the endpoint quiet they do not
  reproduce across two full clean runs. If the engine's hipMalloc
  failure mode (poison the daemon, exit 0) is ever hit again under a real
  co-tenant, treat it as an engine robustness gap, not just deployment
  sizing.
- The run1 degradation episode (admission parking) was only observed on an
  engine that had served hours of foreign traffic first; if it recurs,
  capture `/stats` (`queue_depth` frozen at 1) plus `docker logs` — the
  signature is requests parked in the daemon wait queue with no admission
  and no rejection until the queue timeout.
