# Chat-template eval fixtures (item #37 — ChatScaffold→jinja for qwen)

Fixtures for `examples/template_cache_eval.rs`, which measures chat-template
prefix-cache behaviour (forward-extension LCP) through hipfire's own minijinja
env (trim_blocks + lstrip_blocks + strict + pycompat). NOT yet wired as the
production qwen render path.

- **qwen35-official-reference.jinja** — the official Qwen3.5/3.6 chat_template,
  extracted verbatim from `qwen3.5-0.8b.tier1.mq4.hfq` metadata. Interleaved
  thinking (drops prior-turn reasoning once a new user turn arrives). Reference
  for comparison.
- **qwen35-froggeric-v20.jinja** — froggeric/Qwen-Fixed-Chat-Templates
  (https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), version
  "qwen3.6-froggeric-v20", **Apache-2.0** (compatible with this repo).
  Adds `preserve_thinking` (keep history reasoning → 100% KV forward-extension),
  agentic-loop fixes (retry/stall/empty-think), and engine compatibility
  (avoids loop.previtem / Python-only filters).

## Measured (qwen3.5-0.8b tokenizer, 2-turn colors conversation)
| template  | preserve_thinking | LCP / turn1_kv | forward-extension |
|-----------|-------------------|----------------|-------------------|
| official  | (interleaved)     | 18/59 = 30.5%  | no                |
| froggeric | false (default)   | 18/59 = 30.5%  | no (≡ official)   |
| froggeric | true              | 59/59 = 100%   | YES               |

preserve_thinking=true gives plain-LCP 100% caching for qwen — no verbatim
token-splice and no DeltaNet rewind. Coherence note: qwen's current ChatScaffold
already feeds reasoning-in-history (verbatim splice) and is validated-coherent,
so froggeric+preserve is the same reasoning-in-history rendered via the trained
template — low coherence risk. NEXT: wire qwen generate() to render via this
(flag-gated) + reasoning_content from asst_turn_cache + run both coherence gates.

- **qwen38-27b-mq6-official.jinja** — the Qwen3.8 27B chat_template, extracted
  byte-exact (8952 bytes, sha256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`) from
  `tokenizer_config.chat_template` in the HFQ metadata of registry artifact
  `qwen3.8:27b-mq6` (`https://huggingface.co/hipfire-models/qwen3.8-27b/resolve/main/qwen3.8-27b.mq6`,
  21750254592 B), via HTTP Range of the metadata JSON (file offset 14310342).
  Identical to the template in local `qwen3.8-27b.mq4r`, `.mq4-pro`, and
  `.mq4-xt.asym.hfq`. Natively supports `reasoning_effort` ∈ {low, medium, xhigh}
  (default xhigh when undefined; high/max raise). Used by the
  `prompt_frame.rs` effort-probe regression.
