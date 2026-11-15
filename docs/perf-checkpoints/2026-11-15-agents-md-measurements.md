# Measurements formerly in AGENTS.md and CLAUDE.md

**Lifecycle:** `historical`  
**Authority:** Dated, fixture-bound **Measured** evidence only. Never current defaults, automatic baselines, or admission decisions.  
**Claims:** Measurement under each item's exact fixture and method, as originally recorded. Not transferable across model, quant, GPU, prompt, route, or method.

**Source:** `AGENTS.md` and `CLAUDE.md` at `a3a096297d` (origin/beta-candidate),
moved verbatim on 2026-11-15 when `AGENTS.md` became the rules document
(fixture values now in [`fixtures/pins.toml`](../../fixtures/pins.toml), fixture
prose in [`docs/fixtures.md`](../fixtures.md)). Each item keeps its own date;
2026-11-15 is the move date, not a measurement date. Pins named below:
`pin:model/qwen3.8-flash-next.mq6q8-pleq8` (undated Flash-Next items),
`pin:model/qwen3.8-flash-next.mq4`, `pin:model/qwen3.8-flash-next.mq4-xts`,
`pin:prompt/lru_cache_pep8_strict`, `pin:prompt/prose_river_short`,
`pin:prompt/merge_sort_thinking_off`, `pin:prompt/humaneval_3_below_zero`,
`pin:kld/gfx1151.a4.wt2.pre-canonical`, `pin:kld/gfx1151.a4.code24.pre-canonical`.

## 1. Flash-Next (Qwen4) MTP and bench measurements (from AGENTS.md § 5 "Pinned Flash-Next bench fixture")

Caveats and measurements below without a date were taken on the
`qwen3.8-flash-next.mq6q8-pleq8` pin unless they say otherwise.

- **Measured, not an 80% claim:** on 2026-09-23, gfx1151, HIP 7.2, greedy,
  KV q8, `max_seq=2048`, graph off, three fresh daemon processes per mode and
  byte-identical committed prompts, median code decode was AR 19.4 versus
  MTP 20.1 tok/s (end-to-end 17.1 versus 17.2); prose decode was AR 17.9
  versus MTP 18.0 (end-to-end 15.6 versus 17.1, with large run-to-run
  spread). Daemon md5 `d09ff1d3220b818c4f4d06a3c60cc85b`; prompt md5s:
  code `df5dedc8040ce70ba55080c4548e6024`, prose
  `07a7880965142971dbb3cc7493f8fb94`. All three MTP runs emitted AR's
  exact 227 code and 256 prose token IDs. Local raw reports:
  `.codeinsight+research/qwen4/mtp-parity/runs/moe-buffer-fix/perf-v2/`.
- After fixing routed/shared MoE scratch aliasing, restoring the learned final
  HC mixer, and reselecting QSA at source-correct boundaries, conditional
  draft agreement was 0.708/0.532/0.476 for prose and
  0.922/0.932/0.889 for code at steps 1/2/3. The remaining throughput gap
  to a claimed 80% gain is verifier/replay cost, not evidence that the
  trained drafter is defective. Do not extrapolate these fixture-bound
  measurements to other architectures or prompts.
- **Later measurement (2026-09-28, same prompts, fixture-bound):** after the
  few-row verify, rollback and draft-head work in
  [`docs/perf-checkpoints/2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md`](2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md),
  MTP decode was code ~55 and prose ~37 tok/s against AR ~33 on the same
  build, with greedy MTP ids equal to AR's. The GPU was shared with external
  processes; read the checkpoint's method before citing a number.
- **On `qwen3.8-flash-next.mq4` (2026-09-28, fixture-bound):** with the
  MQ6G256V2 head's re-score, greedy MTP geomean over nine committed prompts
  was 59.5 tok/s against AR 33.7 (55.4 without the re-score); the same file
  with a Q8_0 head reached 59.2 at AR 33.1. Method and caveats:
  [`docs/perf-checkpoints/2026-09-28-qwen4-mtp-mq6-rescore-gfx1151.md`](2026-09-28-qwen4-mtp-mq6-rescore-gfx1151.md).
- `hipfire bench` measures this model. The old fail-closed load had two
  causes, both gone. The qwen4 contract admitted exactly `max_seq` 2048
  (lifted in `432fa5c68`; the limit is now 262144, and an omitted value gets
  32768). Bench's `load_params` also sent
  `max(memory.max_seq, generation.max_tokens + 1024)` — the configured 32768,
  or 4096 + 1024 = 5120 with `memory.max_seq` forced to 2048
  (`822b48fb5:crates/hipfire-cli/src/main.rs:3000-3001`, removed in
  `294c4ad9d`). `--matrix` now sizes the load to the run (largest `--pp`, or
  `--ctx` + `--tg`, plus 32), and `--max-seq N` overrides it
  ([CLI.md](../CLI.md#bench-context-max_seq)). Measured on 2026-10-04 on
  a Strix Halo (gfx1151) host (one card, `qwen3.8-flash-next-gptq3.mq4`
  `8b15b6fe…00972`): `--matrix --pp 8192 --tg 32 --runs 3 --warmups 1` loads
  `max_seq` 20064 (the default `--ctx` tops out at 20000). pp8192 was
  1927.3 / 1921.8 / 1918.3 tok/s, fast-PPT-limited for 48–55 % of each run.
  The standard path keeps the automatic 32768.
- Decode numbers are not comparable across instruments: the raw decode probe
  measured 22.74 tok/s (ctx128, graph off, kv q8), the serve path ~19.9 tok/s.
  Same model, different measurement; never average or compare them across.

## 2. A3B MoE DFlash fixture, best observed (from AGENTS.md § 5 "Pinned A3B MoE DFlash fixtures")

Fixture: `hipfire bench ~/.hipfire/models/qwen3.6-35b-a3b.mq4-awq-mi300x --spec dflash --runs 5 --warmups 3 --max-tokens 256 --kv-mode q8 --backend noslots --workload stateless --json`
(target `pin:model/qwen3.6-35b-a3b.mq4-awq-mi300x`, draft `pin:draft/qwen36-35b-a3b-dflash-mq4.hfq`).

- `benchmarks/prompts/merge_sort_thinking_off.txt`
  - md5: `253c7ac50857fe6d0e10fb0d2c5e35c0`
  - best observed post-MoE tape replay fix: `151.00 tok/s`, tau `2.711`,
    accept rate `0.5422`, `45` cycles, `168` emitted tokens.
- `benchmarks/prompts/humaneval_3_below_zero.txt`
  - md5: `37c5aad9f9efe93b5c47f27256bdf149`
  - best observed before the MoE tape replay optimization: `127.61 tok/s`,
    tau `3.714`.

## 3. Cold-cache gfx1151 discriminator (from AGENTS.md § 5 "Quality pins")

Fixture: `pin:model/qwen3.8-27b.mq4-xts`, `eval_hipfire --scoring-mode prefill
--max-chunks 24`, ROCm 10.1 / HIP 7.16 / clang 24, pre-canonical pin
generation (2026-10-08).

Cold-cache gfx1151 discriminator (not pins): WT2 measured 0.070054
(`2355f76c…`) and 0.069878 (`a2f550a9…`); code24 measured 0.050417
(`4e78f8e4…`) and 0.050338 (`9caca85b…`). Only sequence 0 differed.
After one cache-populating WT2 run, both scored warm WT2 runs reproduced
`2355f76c…`; code24 warmup and both scored warm runs reproduced `9caca85b…`.
This ties the observed instability to the cold-JIT path; its root cause is
not established, and empty-cache Halo determinism is not claimed.

## 4. Prompt-structure τ sensitivity, 27B-3.5 DFlash (from CLAUDE.md, 2026-04-24 / 2026-04-26)

**One newline character can swing τ by 17% on 27B DFlash.** Two prompts
that tokenize to the same number of tokens (e.g. both 232) but with
different whitespace patterns produce dramatically different draft-target
acceptance:

```
PEP-8 strict (\n\n\n between top-level defs):    27B-3.5 LRU max=120  → 161 tok/s τ=8.07 (deterministic ±2%)
Single-blank (\n\n between top-level defs):      27B-3.5 LRU max=120  → 184 tok/s τ=9.42 (range 173-204)
```

**Why:** identical token COUNT, different token SEQUENCE → different
prefix-conditioned distribution shape at each position → different
draft/target argmax alignment → different τ. Same model, same flags,
same kernels, same binary md5.

**How to apply:** ANY tok/s or τ comparison across sessions, agents, or
commits MUST use byte-identical prompts. Embed prompts as committed
files (not heredocs in scripts that get reformatted by editors), and
record the prompt md5 alongside the result. A 14% perf delta from a
whitespace cleanup is invisible in code review but catastrophic for
benchmarking. Discovery cost ~6 hours of phantom-regression chasing
on 2026-04-24 (rocBLAS, DKMS, firmware, kernel cache, mold/sccache,
DPM — all null) before isolating to a single newline.

**Corollary**: agent-to-agent perf claims that lack prompt md5 are
unverifiable. Don't accept "X agent got Y tok/s" without reproducing
on the exact prompt bytes they ran.

**Mitigation (Phase 1 implemented):** The engine collapses all 3+ consecutive
newlines to exactly 2 before tokenization. This eliminates the whitespace-
variance source entirely, making PEP-8 and single-blank prompts tokenize
identically.

**DEFAULT ON since 2026-04-26.** The original Phase 1 ship gated this behind
`HIPFIRE_NORMALIZE_PROMPT=1` opt-in, but empirical bench showed it's worth
+24% τ on PEP-8 code prompts (159 → 196 tok/s on 27B-3.5 LRU DFlash) without
correctness cost. Opt out with `HIPFIRE_NORMALIZE_PROMPT=0` (or
`prompt_normalize=false` in config) only when raw `\n{3,}` whitespace is
semantically load-bearing. See:
- `crates/hipfire-runtime/src/tokenizer.rs:maybe_normalize_prompt()` — engine impl
- `crates/hipfire-runtime/examples/encode_prompt.rs` — verification utility
- commit 9a2c667 — root cause + bench data behind the default flip
