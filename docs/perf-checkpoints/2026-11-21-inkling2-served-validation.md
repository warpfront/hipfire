# Inkling-2-Mini served validation — 2026-11-21

**Lifecycle:** `historical`

**Disposition:** fixture-bound **measured** evidence for arch-24 served cells
under the exact fixture/method below. Single-turn coherence cells PASS and are
citable as scoped cells; the multi-turn chain cell FAILS on a code-grounds
1024-row KV cap (finding inside) — this record is evidence of both, not an
admission decision, not a perf claim, not transferable across model, quant,
GPU, prompt, route, or method.

## Fixture

- Host: Windows, gfx1100 (25.8 GB VRAM, HIP 7.16), second device gfx1036
  untouched. Toolchain: therock-dist-windows-gfx110X-all-10.1.0
  (`amdhip64_7.dll` runtime 71626385 / driver 71626385). ROCm 10.1.
- Model: `D:/hipfire-models/inkling2-mini-mq4.hfq`, 13927188480 bytes, md5
  `5f59320ff25abba935cc9ef9f31548b2`, sha256
  `9536b3fde8f38511032df41aaf1124cc81cb839a9deeb333c650a0b2961eebe3`.
- Binaries (built in the measuring worktree from `a3265308e`,
  `cargo build --release -p hipfire-cli -p hipfire-daemon`): hipfire md5
  `814dad0c253aa40332e867aca8b26ab9`, daemon md5
  `8c79362e5ad63b02cf14d59e505b6790` (differ from the 05-report pair by
  build-path embedding only; same source commit, cross-binary decoded output
  byte-identical).
- Route: served `hipfire serve` via `scripts/serve_harness.py`
  (`--mode battery` / `--mode chain`), embedded chat template, greedy
  (`temperature=0.0`), `--max-tokens 2048 --max-think-tokens 256`,
  speculation off, kv auto (legacy backend effective), max_seq automatic.
  `HIPFIRE_HOME=D:/hipfire-models/inkling2-home` for every load.
- Battery prompts (built-in 5-genre, md5): code
  `43ca0d15712d3dfb777b51ae76d8fd5f`, reason
  `640e0fd4f55996cb175a422f0a12cef5`, factual
  `8f66b4c97988825bd8e7840aaf44357e`, prose
  `8fe0ad36f61bcf4992cc9df81cdf3817`, instruct
  `8bed8e2d056dc1d47dccae9d32dbecf4`. Chain reuses the same five in order.

## Method

- Battery: 2 sizing runs then the cited run (2048/256,
  `HIPFIRE_SERVE_ALLOW_INCOHERENT=1` so flags report instead of killing the
  JSON dump): exit 0, full per-turn text in `battery.json`. First 512-cap run
  fail-closed by design (length cutoffs); a concurrent duplicate run died with
  ConnectionResetError (two serves, one GPU/port — operator error, discarded).
- Chain: run 1 (2048/256, forgiven) then run 2 (identical + `--serve-log`)
  after a host TUI hiccup; both show the identical progressive signature, so
  environmental cause is excluded. Daemon error lines define the finding.
- Determinism: same `run` prompt twice at `--temp 0`, two fresh processes per
  binary pair (05-built and 07-built), stdout diffed.
- Goldens: 4 llama.cpp F32 thinking dumps (`D:/hipfire-models/inkling2-golden/`,
  `-n 48 --temp 0`) vs our 48-token `--temp 0` served outputs; word-prefix
  agreement counted after tags (python word split).
- KLD: `eval_hipfire` example built from the same tip, run against the
  artifact; `.kldref.bin` search on `D:/hipfire-models/` and in-repo.
- 2k: single `hipfire bench --prompt-file` (8310-char prompt, md5
  `24cf76efd8db6e8c4878966f96e6538d`); exact prompt_tokens unreported
  (prefill refused before counting).

## Result — battery (PASS on coherence)

Cited run: `turns=5 runaway=0 empty=0 attractor=2 retrieval_miss=0
avg_decode=14.0tok/s`. Verdict lines: code stop/615 (prefill 2800ms, decode
15.3), reason stop/134 (3406ms, 15.9), factual stop/338 (1783ms, 14.4), prose
finish=None/0 + ATTRACTOR, instruct stop/206 + ATTRACTOR (2825ms, 10.5).
Raw arrays: gen=`[615, 134, 338, 0, 206]`, prefill_ms=`[2800, 3406, 1783,
None, 2825]`, decode_tok_s=`[15.3, 15.9, 14.4, None, 10.5]`.
All five turns complete and on-spec (correct merge_sort + docstring + examples;
210 miles with steps; exactly three sentences; four-sentence story; exactly
five numbered tips), each closed with `<|end_message|>`. The two attractor
flags are analyzed detector artifacts: prose thinking holds the story twice
(draft + polished revision); instruct holds the five items in thinking and
again in the answer (the task demands it). Full verbatim texts are pasted in
the validation report (`docs/plans/local/inkling2-validation-report.md` §1),
not repeated here.

## Result — chain (FAIL, code-grounds finding)

Both runs: `turns=5 runaway=0 empty=2 attractor=0`. T1 stop/615 (prefill
3133ms, decode 14.6), T2 stop/151 (prefill 53096ms, decode 11.2 — stateless
full re-prefill of ~655-token history at ~71ms/token), T3 partial 101 words cut
mid-decode, T4/T5 zero chars. Raw arrays (run 2): gen=`[615, 151, 0, 0, 0]`,
prefill_ms=`[3133, 53096, None, None, None]`,
decode_tok_s=`[14.6, 11.2, None, None, None]`.
Daemon root cause (serve log): `inkling2 gpu fwd: visit 0 KV overflow at pos
1024 (rows 1024)` — decode-side on T3 (`attempt=3`), prefill-side on T4/T5
(`attempts=4,5`), `retryable=false rolled_back=true`. The 05 route is
stateless (cold reset + full history re-prefill per turn), so cumulative
length grows monotonically and every multi-turn conversation dies at ~1024
rows; the forward has no eviction (03-documented fail-closed design). Clean
Err, no NaN/garbage/hang — but served multi-turn past ~1024 cumulative tokens
is blocked, with empty HTTP-200 content on T4/T5. Single-turn cells (< 1024
tokens) are unaffected. No patch per brief 07; follow-on brief owed (eviction
/ sliding re-prefill / honest limit).

## Result — determinism, goldens, KLD-stop, 2k

- Determinism PASS: byte-identical stdout across fresh processes (both binary
  pairs); cross-pair decoded output identical. Prompt md5
  `1aacd3c05cf9695cc799acc59581938d` (= golden g1 prompt).
- Goldens (uniform-MQ4 vs F32, diverge by design; per-prompt agreement-end, no
  pass/fail): g1 ends word 8/35, g2 ends word 18/32, g3 ends word 8/26
  (budget-truncated tail), g4 ends word 15/36. Every prompt on-theme with the
  golden's factual answer (Paris / 4 / haiku form / Jupiter+gas-giant).
  Chat-template tags (`<|message_model|><|content_thinking|>`) vs golden
  `[Start thinking]` are harness differences, not model divergence.
- KLD STOP: `eval_hipfire` exits 1 — `detected arch_id=24 (HFQ header)` then
  `unsupported arch_id 24: eval_hipfire wires dense 0|1 (llama), 5|6 (qwen35),
  13 (gemma4), 14 (glimmer), 11 (lfm2)` (matches
  `eval_hipfire.rs:386-391`). No `.kldref.bin` for Inkling-2 anywhere checked.
  No reference invented, no pin claimed; evaluator wiring + reference
  generation is a separate follow-on brief.
- 2k: `bench` exit 1, empty stdout — `inkling2 prefill failed: inkling2 gpu
  fwd: visit 0 KV overflow at pos 1024 (rows 1024)`. Same cap as the chain
  finding; SWA-wrap/extent-edge questions unreachable on this route. No tok/s
  at 2k (nothing ran).

## Numbers, not adjectives

Single-turn served decode (greedy, this fixture): 10.5–15.9 tok/s across
battery/chain T1–T2 turns (unoptimized; Phase 4 owns performance). 05 bench
smoke (8 client tokens): daemon wall median 20.1 tok/s. No cross-cell
comparisons claimed; served numbers never averaged with CPU parity numbers.
