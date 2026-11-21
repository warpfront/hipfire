# Inkling-2-Mini MQ4L / imatrix calibration plan (Task 3)

Status: plan only (2026-11-21). READ-ONLY investigation — no quantize run was
executed, no `crates/*` file was modified. All commands below are staged to run
from the `feat/inkling2-mini-support` worktree
(`C:\Users\Jan\AppData\Local\Temp\opencode\hipfire-inkling2`), which owns the
Inkling-2 ingest code; line cites say `worktree:` vs `beta:` (this checkout,
`origin/beta`) explicitly.

## 0. Baseline and hypothesis

- Uniform artifact: `D:\hipfire-models\inkling2-mini-mq4.hfq` (13.9 GB, arch 24,
  6790 tensors: 428 F16, 6144 per-expert MQ4G256, 170 MQ4G256V2) — built by the
  worktree quantizer, which dequantizes NVFP4/MXFP8 experts to F32, splits the
  3-D stacked tensors per expert, regroups w13 rows, and encodes each expert
  with `quantize_mq4g256` inline (`worktree:crates/hipfire-quantize/src/pipeline.rs:3414-3517`).
- Measured gap (impl log 2026-11-21): per-weight qt44 F32 decode correlates
  0.995 with the F32 GGUF oracle but SQNR is only ~20 dB (mse 4.3e-6 vs signal
  4.3e-6…4.3e-4), while token-0 full-logit parity is r≈0.19 with argmax
  disagreement. Plausible mechanism: small uniform-4-bit errors on the
  1152-wide experts flip discrete top-8 routing and compound over 48 visits.
  Hypothesis only — the F32-GGUF forward control (same `forward.rs` on oracle
  weights) must first separate forward bugs from quant noise.
- Fix path: imatrix-graded **MQ4L** (qt 52 `MQ4G256V2Lloyd`): byte-identical
  136 B V2 group container plus a per-tensor F32[16] `lloyd_levels` codebook
  sidecar (`<stem>.lloyd_levels.weight`, shape `[16]`;
  `beta:crates/hipfire-quantize/src/quant_mq4v2_lloyd.rs:1-20`,
  `beta:crates/hipfire-quantize/src/quant_mq4v2_lloyd.rs:800-815`).

## 1. What MQ4L is (mechanics that constrain the recipe)

- Format aliases (all select qt 52): `mq4v2-lloyd`, `mq4v2l`, `mq4l`,
  `mq4g256v2l`, `mq4g256v2-lloyd`
  (`beta:crates/hipfire-quantize/src/pipeline.rs:1545-1549`). The product tiers
  are NOT formats: `--tier xt|base|pro` only lifts fixed-tier classes
  (`beta:crates/hipfire-quantize/src/model_filter.rs:309-347`).
- Fit: 25 Lloyd-Max iters, ~2M-value stride-subsampled pool, uniform init →
  free fit → E4M3 snap → warm-start constrained iters → ±2-neighbor
  coordinate-descent polish
  (`beta:crates/hipfire-quantize/src/quant_mq4v2_lloyd.rs:21-24,522-580`).
  Lloyd needs **no imatrix** (fit is on rotated weight values weighted by
  per-half `sc²`); imatrix enters only via AWQ pre-scale and graded routing.
- Gate: end of run prints a per-family weighted-RMS table (families keyed by
  `mlp_down/mlp_gate/mlp_up/attn_o/attn_qkv/ssm/other` substring match,
  `beta:crates/hipfire-quantize/src/quant_mq4v2_lloyd.rs:195-217`) and **refuses
  to write the artifact** unless the E4M3-constrained gain over uniform is ≥
  12% on every family (`beta:crates/hipfire-quantize/src/pipeline.rs:4514-4524`).
  Expect `mlp_down`/`mlp_gate` to carry the decision on this model.
- K-shape: `quantize_mq4g256v2_lloyd(w, m, k)` zero-pads a trailing partial
  group (`beta:crates/hipfire-quantize/src/quant_mq4v2_lloyd.rs:602-607`), so
  the 1152-wide expert `down_proj` (K=1152, 1152 % 256 = 128) is encodable —
  unlike the `supports_g256` (`inner_k % 256 == 0`) MoE-dispatch gate that
  would silently route it elsewhere
  (`beta:crates/hipfire-quantize/src/pipeline.rs:5955-5956`).
- Proven precedent (copy the shape): `D:\hipfire-models\DaisyRose-27B\quant-mq4l.sh`
  runs `--format mq4v2-lloyd --tier $tier --imatrix $IM --awq --awq-fix-la-head-order`
  for tiers `base/pro/xt`, ~523–554 s/tier on 27B
  (`D:\hipfire-models\DaisyRose-27B\kld\quant-times.tsv`). Expect ~10 min/tier
  for Inkling-2-Mini (26.4B) on the same box.
- GPU fit is opt-in (`HIPFIRE_QUANT_GPU=1`, `quant-mq4l-gpu.sh`); default CPU
  path is the reference. Do not use it for the first calibration.

## 2. Calibration input: imatrix sources (pulled 2026-11-21)

Primary: bartowski (merged-layout, use this one):
- `D:\hipfire-models\bartowski-imatrix\thinkingmachines_Inkling-2-Mini-imatrix.gguf`
  (90,752,064 B, MD5 `d48f8d10c93105b1eaf2ab0bdf48f3c8`) + calibration
  `thinkingmachines_Inkling-2-Mini-calibration-v6.txt` (1,135,011 B).
- Built with b12600 (merged #32694) — same layout as the forward. Corpus:
  prose + tool/reasoning through the chat template with `--parse-special`.
  Per-tensor `layouts/` recipes published in the repo.
- Fetch: `hf download bartowski/thinkingmachines_Inkling-2-Mini-GGUF --include "*imatrix*" --include "*calibration*" --local-dir D:\hipfire-models\bartowski-imatrix`
- Use as `$IM` in §5 (replaces the ggml-org path below).

Cross-check: Atomic ladder (sensitivity reference, NOT a direct imatrix):
- `D:\hipfire-models\atomic-metrics\` (dataset `AtomicChat/Inkling-2-Mini-GGUF-metrics`):
  `results/summary.tsv`, `ladder/ladder.json` (profile `moe-inkling2`),
  `ladder/AD-*.types`, `imatrix/imatrix.stats.txt`, `imatrix/corpus-manifest.json`.
- Key rules: attn + controllers + embed/head stay Q8_0; `ffn_gate_up_exps`
  carries the quant (q6_k/q5_k/q4_k/iq*), `ffn_down_exps` is q8_0 → q5_1/q5_0/iq4_nl
  (1152-wide rows can't take K-quants); IQ2_M keeps iq4_nl on the 8 busiest
  down layers (3,4,5,15,16,17,18,19). Measured: AD-Q4_K KLD 0.0080/top-1 96.3%
  vs preset Q4_K_M 0.0165/94.7%; AD-Q6_K 0.0037 ≈ Q8_0 quality. Ceiling here is
  ~97.7% (Q8_0 vs BF16), not 99%+.
- Corpus: 4.76M tokens, 87×4096 neutral 30-lang, `--parse-special`, shards 0-1
  on BF16, 2-5 on Q8_0 proxy. Built from PR #32695 (`6196d98`) — GGUF names
  (`blk.N.loop_*`, `ffn_down_exps`) do NOT map 1:1 to the merged layout, so do
  not feed this imatrix to hipfire; use its ladder to pick which experts/tiers
  matter for MQ4L/AWQ.
- unsloth: no downloadable imatrix (tag only, Dynamic-3.0 internal). Ignore.

Legacy ggml-org path (superseded by bartowski above, kept for provenance):

- Source: `ggml-org/Inkling-2-Mini-GGUF/imatrix.gguf` (90,752,032 B, sha256
  `553662ef…`, per plan §0). **Not present locally** (no `*matrix*` under
  `D:\hipfire-models` outside DaisyRose/llama-build objects) — fetch once:
  ```powershell
  hf download ggml-org/Inkling-2-Mini-GGUF imatrix.gguf --local-dir D:\hipfire-models\inkling2-imatrix
  ```
  Verify size == 90752032 bytes before use.
- Inspect which keys the loop-arch converter actually emitted (names decide
  §4). From a Python shell, list tensor names/shapes ending in
  `.in_sum2` / `.counts` and check for `ffn_gate_exps` / `ffn_down_exps`
  (`blk.{N}.ffn_{gate,down}_exps.weight.in_sum2` shape `[K, n_mat]` is the MoE
  form; dense-only `[K]` entries are skipped by the legacy dense table with a
  warning, `beta:crates/hipfire-quantize/src/calibration.rs:741-747`, but the
  MoE paths read the raw GGUF directly so multi-mat is fine there):
  ```powershell
  python C:\Users\Jan\AppData\Local\Temp\opencode\inspect_gguf.py D:\hipfire-models\inkling2-imatrix\imatrix.gguf
  ```
  (`inspect_gguf.py` already exists in that temp dir.) If the converter used
  non-`ffn_*_exps` names for the loop MoE, record the exact names here before
  touching code — every consumer below keys on them.

## 3. Which tensors to calibrate (and which to leave alone)

Calibrate (Lloyd codebook + AWQ pre-scale where imatrix covers them):

- ALL routed experts, both projections, all 24 layers × 128 experts:
  `{base}.experts.{E}.gate_up_proj.weight` ([2304, H], regrouped) and
  `{base}.experts.{E}.down_proj.weight` ([H, 1152]), where
  `{base} = model.llm.layers.{N}.mlp`. These are the 6144 MQ4G256 tensors in
  the uniform artifact and the hypothesized gap source. Gate_up K (= hidden,
  %256-clean) and down_proj K=1152 (padded last group) are both encodable.
- Dense attention + output: `attn_q/k/v/o_proj`, `q/k/v_input_norm` linears,
  `attn_res.attn_queries.*`, `output_res_query`, `output_gate` — via the
  standard `--awq` dense path keyed by `safetensors_to_ggml_name`
  (`beta:crates/hipfire-quantize/src/calibration.rs:554-608`); exactly which
  Inkling names map is verified by the run log (`imatrix(awq): no entry for …
  → plain …` lines mean a miss, not an error).
- `lm_head` / `embed_tokens`: do NOT hand-pick — use `--tier xt` (lifts embed
  only) → `base` (+lm_head) → `pro` (+ssm_out, n/a here) exactly as the
  DaisyRose precedent, and compare (§6).

Keep lists (already wired — do not touch):

- `is_inkling2_keep_f16` forces F16: `.controllers.`, `res_query`,
  `attn_rel_bias`, `.q_l2_clip.`, `.rel_l2_clip.`, `.attn_sink_proj.`,
  `conv*`, `*scale*` under `model.llm.`
  (`worktree:crates/hipfire-quantize/src/model_filter.rs:46-58`).
- Generic gates: `should_quantize` keeps norms/biases F16 and routes
  `model.visual.*` to the vision path
  (`beta:crates/hipfire-quantize/src/model_filter.rs:24-41`); Q8 fixed tier
  (router `mlp.gate.weight`, embed, lm_head, attn) applies unless
  `--no-q8-router` (`beta:crates/hipfire-quantize/src/model_filter.rs:349-375`,
  `beta:crates/hipfire-quantize/src/pipeline.rs:2620`).
- `mtp.safetensors` is excluded from the trunk ingest (impl log); vision
  refused in Phase 1. No `--include-vision`, no `--include-prefix` beyond the
  smoke run.

## 4. Integration caveat (Task 4 work, not this plan's commands)

The current worktree Inkling branch quantizes experts with bare
`quantize_mq4g256` inline (`worktree:…/pipeline.rs:3493-3502`) and never
enters the MoE-imatrix machinery, which keys on 3-D stacked parents
`…layers.{N}.mlp.experts.{gate_up_proj|down_proj}` →
`blk.{N}.ffn_{gate,down}_exps.weight`
(`beta:crates/hipfire-quantize/src/quant_mq.rs:690-707`,
`beta:crates/hipfire-quantize/src/pipeline.rs:6100`). Consequences:

- `--format mq4v2-lloyd` alone will Lloyd-encode the *dense* tensors (the
  `use_mq4v2_lloyd` dense arms exist,
  `beta:crates/hipfire-quantize/src/pipeline.rs:7019-7024,7204`) but the
  per-expert inline call ignores it — experts would stay uniform MQ4 unless
  the Inkling branch is taught the Lloyd encoder + `lloyd_levels` sidecar
  emission (`beta:crates/hipfire-quantize/src/pipeline.rs:8441-8442` pattern).
- Per-expert AWQ (Route A, `beta:crates/hipfire-quantize/src/pipeline.rs:6151-6199`)
  and graded routing (`HIPFIRE_MOE_GRADED`, down_proj-only by De-risk note,
  `beta:crates/hipfire-quantize/src/pipeline.rs:6207-6257`) likewise need the
  Inkling parent names mapped into `safetensors_to_imatrix_key` or a sibling.
- Do NOT pass `--awq-fix-la-head-order`: Inkling has no `linear_attn` geometry
  and the flag hard-errors when it is absent
  (`beta:crates/hipfire-quantize/src/calibration.rs:713-720`).
- Start graded work only after uniform-MQ4L verifies (§6): graded emits mixed
  per-expert dtypes needing its merged decode kernel, and `HIPFIRE_MOE_GRADED`
  requires `--imatrix` or it exits(2)
  (`beta:crates/hipfire-quantize/src/pipeline.rs:1588-1593`).

## 5. Exact CLI flow (copy-paste; run in the worktree)

```powershell
$Q = 'C:\Users\Jan\AppData\Local\Temp\opencode\hipfire-inkling2\target\release\hipfire-quantize.exe'
$IN = 'D:\hipfire-models\Inkling-2-Mini'
$IM = 'D:\hipfire-models\bartowski-imatrix\thinkingmachines_Inkling-2-Mini-imatrix.gguf'
cargo build --release -p hipfire-quantize   # in the worktree; isolates $HOME below
$env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
```

0. Smoke (prefix run, ~minutes) — proves the Task-4 wiring closes accounting
   before burning 10 min/tier:
   ```powershell
   & $Q --input $IN --output D:\hipfire-models\inkling2-mini-mq4l-smoke.hfq --format mq4v2-lloyd --tier xt --arch-id 24 --include-prefix 'model.llm.layers.0.' 2>&1 | tail -30
   ```
1. Uniform-MQ4L baseline, no imatrix (Lloyd is weight-only; valid alone):
   ```powershell
   & $Q --input $IN --output D:\hipfire-models\inkling2-mini-mq4l-xt.hfq --format mq4v2-lloyd --tier xt --arch-id 24 2>&1 | tee D:\hipfire-models\inkling2-quant-mq4l-xt.log
   ```
   Required log lines: the `=== MQ4V2-Lloyd weighted-RMS by family ===` table
   with `e4m3% ≥ 12` on every family (else exit 2, by design), and the
   `Recipe census` showing `MQ4G256V2L` + `F32` (`lloyd_levels`) entries.
2. AWQ treatment (needs the §2 download):
   ```powershell
   & $Q --input $IN --output D:\hipfire-models\inkling2-mini-mq4l-xt-awq.hfq --format mq4v2-lloyd --tier xt --arch-id 24 --imatrix $IM --awq 2>&1 | tee D:\hipfire-models\inkling2-quant-mq4l-xt-awq.log
   ```
   Optional second arm isolating expert noise: `HIPFIRE_AWQ_EXPERTS=down` (down
   only) vs unset (gate_up+down), `beta:…/pipeline.rs:6159-6172`; and
   `HIPFIRE_AWQ_EXPERTS=none` as the dense-AWQ-only control.
3. Tier ladder (only promote winners from steps 1–2): repeat with
   `--tier base` (and `pro` if `ssm_out`-class tensors exist on this arch).
4. Record for every artifact: `sha256sum`, the full stderr log (contains the
   family table + census), and `hipfire_recipe_census` from metadata.

## 6. Verification (in order — stop at the first red)

1. **Lloyd gate**: the run's own family table (`e4m3%` column) ≥ 12% on
   `mlp_down`/`mlp_gate`/`mlp_up`; artifact refused otherwise. Compare
   `rms_e4m3` across tiers.
2. **Token-0 logit parity** (available today, no loader needed): re-run the
   worktree CPU forward (`forward_token` + `lm_head_logits`,
   `worktree:crates/hipfire-arch-inkling2/src/forward.rs:584-648`) on each
   candidate artifact to `D:\hipfire-models\cpu-logits-tok0.txt` and score
   with the existing script:
   ```powershell
   python C:\Users\Jan\AppData\Local\Temp\opencode\compare_logits.py
   ```
   (oracle `D:\hipfire-models\oracle-logits-tok0.txt` vs cpu; reports max/mean
   abs diff, argmax agreement, Pearson r — `compare_logits.py:1-39`). Success
   bar: r clearly above the uniform 0.19 baseline *and* argmax agreement on
   the MANIFEST prompts; an r jump with unchanged argmax is still routing
   flips, not a win.
3. **Per-weight SQNR**: decode one NVFP4 + one MXFP8 expert per layer-kind to
   F32 from the candidate vs the uniform artifact vs the F32 GGUF; target is
   SQNR well above the uniform 20 dB (the 0.995-correlation figure must move
   in dB, not just in r).
4. **Golden MANIFEST parity**: once the loader serves the model,
   `D:\hipfire-models\inkling2-golden\MANIFEST.txt` prompts must reproduce the
   pinned thinking-phase text; then `serve_harness.py battery + chain` per the
   support plan, and only then a KLD/`.kldseq` pin (`eval_hipfire --scoring-mode
   prefill --max-chunks 24`, AGENTS.md §5) as the bitwise bar for fusion work.
5. **A/B hygiene**: every comparison reuses byte-identical prompts, records
   artifact + binary md5s, and eyeballs decoded text (tight-stddev suspicion
   rule, AGENTS.md testing playbook §0).

## 7. Rollback and cost

- Rollback is the kept uniform artifact `inkling2-mini-mq4.hfq` (13.9 GB) plus
  the step-1 no-imatrix MQ4L file: every later treatment must beat step 1,
  not just uniform MQ4. Never overwrite an artifact; the writer refuses
  existing `--output` anyway.
- Cost: ~10 min/tier CPU (DaisyRose 27B precedent: 523 s pro / 554 s xt,
  `kld/quant-times.tsv`); 3 tiers + 2 AWQ arms ≈ under an hour wall. Peak RAM
  is the F32 decode working set — reuse the release binary and default
  threads; lower `--threads` only if bandwidth-bound.
- Known dead ends, do not retry: `mq2v2` (product-rejected KLD),
  `HIPFIRE_MOE_GRADED` before uniform-MQ4L verifies (mixed-dtype decode
  kernel dependency), `--awq-fix-la-head-order` on this arch (hard error),
  GGUF weight input (removed in 0.5.0 — the F32 GGUF is oracle-only).

## 8. Open inputs this plan needs from the run

- Actual imatrix key names for the loop MoE (assumed
  `blk.{N}.ffn_{gate,down}_exps.weight`; §2 inspection decides).
- Whether dense AWQ finds entries for Inkling attention names (run log
  `imatrix(awq): no entry` lines).
- The family table values for `mlp_down`/`mlp_gate` — the go/no-go number.
