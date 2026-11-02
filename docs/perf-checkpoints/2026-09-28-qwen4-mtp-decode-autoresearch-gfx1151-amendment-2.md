# Amendment 2 — Qwen4 native MTP decode autoresearch (gfx1151) — 2026-09-28

**Lifecycle:** `historical`

**Amends:** [2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md](2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151.md)
and [amendment 1](2026-09-28-qwen4-mtp-decode-autoresearch-gfx1151-amendment-1.md)
(both unchanged). Same fixture, prompts, harness and gate; this records the
continuation on the same branch (`d554080d1` → `f22d7e220`).

**Disposition:** measured local deltas on `gfx1151`. Not a G5 admission, not a
retained-replay certification, not a cross-architecture result, not a product
speed-floor update. External `llama-server` processes shared the GPU during
the continuation.

## Result

| | start (run 343, `d554080d1`) | end (run 359, `f22d7e220`) |
|---|---:|---:|
| MTP decode, code | 60.9 | **66.4** |
| MTP decode, prose | 41.5 | **44.1** |
| geomean | 50.2 | **54.1** (+7.8%) |

Target outputs are bitwise unchanged and MTP ids equal AR ids on every run.
Harness acceptance is unchanged (code 3.48 tokens per window, prose 2.12).
Two changes alter draft numerics only (the target verifies every token):
the vocabulary-front draft head and the four-wave HC read in the head; on the
10-prompt sweep every prompt keeps its acceptance except `bare_factual`, which
loses one draft to a token outside the front (1.74 → 1.42 tokens per window).

## Validation (end state, daemon md5 `7ac8aa1a480b339283859f9bc261e81f`)

- Breadth: the 9 committed sweep prompts at 200 tokens: MTP ids equal AR ids
  on all; MTP/AR 1.39x-2.19x.
- Non-English: zh/ja/ru prompts (40-53% of tokens outside the draft front)
  keep the pre-change acceptance (0.99 / 0.97 / 1.21 tokens per window): the
  full-vocabulary fallback holds.
- Serve route: `scripts/serve_harness.py --mode battery --sampling greedy
  --mtp on --thinking off --max-tokens 400 --kv q8 --max-seq 2048`: 5/5 turns
  `finish=stop`, no runaway/empty/attractor; answers read and correct (merge
  function, 210 miles, axial tilt, four-sentence story, five practices).
- Unit tests: `hipfire-arch-qwen4` 56, `hipfire-dispatch` 286, `rdna-compute`
  286 pass.

## What changed

- **GDN verify rollback.** The verify's capture kernel wrote the recurrent
  state after every row (4 × 3 MB per GDN layer at 4 rows); those writes sat
  dirty in the MALL and were evicted during the next MoE's router/shared GEMV
  (83 µs instead of 49 µs after every GDN layer, found by per-layer medians).
  Now the verify writes only the last row's state, its convolution output and
  gate/beta land in the capture buffer, and a partial acceptance re-runs the
  kept rows of all 36 GDN layers in one launch from the untouched pre-verify
  ring slot (bitwise the verify's own prefix; ~1.2 ms per rollback). The
  persistent GDN step orders only LDS at its barriers.
- **Draft head front.** The MQ2 ranking copy of the LM head is laid out token
  ids `[0, 100000)`, then EOS and the control tokens, then the rest; a draft
  step ranks only the first 100 276 rows (60% less read) unless an input token
  outside them was fed in the last 64 steps. The top-8 exact Q8 re-score maps
  rows back to token ids. (MQ2 + top-8 re-score drafts the Q8 head's own
  argmax: identical acceptance to drafting with the Q8 head on 10 prompts.)
- **MTP head.** The trunk's fused QSA prologue (8 launches → 1 per step); the
  HC read down projection on four waves per row with the activation fused;
  QSA selection on the rank-parallel kernel and the reuse pass as one block;
  `fc_hidden` both branch rows in one launch.
- **Verify launches.** Grouped gate/up O4 × R2 instantiation for few rows;
  HC read up reads each LDS scale/quant once for all rows; GDN capture projects
  straight into its buffer; QSA selection mirrors its final row into the
  persistent indices for any row count.

## Rejected (measured)

- Q8_0 copies of the head's HC read weights: +0.4% but acceptance fell on
  long-context prose (`glimmer_prefill_1024` 1.97 → 1.84).
- Re-measured adaptive-depth window costs: prose shifts to K = 1-2 windows at
  unchanged tok/s.
- MQ6 few-row GEMV with x staged in LDS (4x slower), two weight rows per wave
  (slower); its cost is linear in rows (~16 µs per extra row per 33 MB), issue
  bound rather than x-traffic bound. Q8_0 HC down with two weight rows per
  block: 2.7x slower (half the blocks).
- Head launch trims (no wide-state copies, HC write reusing the read's norm):
  bitwise, ~100 µs per cycle, below run-to-run noise.
- `HIPFIRE_MTP_PAIRING=aligned|aligned-head`: mixed per prompt.
