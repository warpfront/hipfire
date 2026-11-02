# Qwen4 tiny profiles (S0 source contract)

Developer-only CPU references: `tiny-moe` (pinned upstream equations) and `tiny-dense-dev`
(non-upstream dense equation). Neither is a production admission and neither proves a real
Qwen4-27B config; both exist so S1/S2/S3 can lower complete tiny checkpoints against per-op dumps.

## Commands (clean CPU venv, no GPU, no network if the vLLM source cache is warm)

```bash
cd <worktree>
export HIPFIRE_QWEN4_ORACLE_CACHE=$HOME/.hipfire/datasets/cachequality-qwen4-upstream   # or --cache-dir
PY=/home/kaden/qcal/release-0.4.2/qwen4-27b/s0-cpu-venv/bin/python
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-moe       --out /tmp/q4-tiny-moe   --seed 3800
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/q4-tiny-dense --seed 3800
# GPU-compatible variant (same seed/dims/tokens): RoPE base 1e7 and context 1024 on the main AND MTP config; marker id stays tiny-moe
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-moe --out /tmp/q4-tiny-moe-gpu --seed 3800 --rope-theta 10000000 --max-position-embeddings 1024
# optional components (both default present, byte-identical default output): drop PLE and/or native MTP
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/q4-dense-nople  --seed 3800 --no-ple
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/q4-dense-nomtp  --seed 3800 --no-mtp
# width-5120 knob (1-2 layers only; reduced projections/FFN)
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --hidden-size 5120 --layers 1 --out /tmp/q4-dense-5120
$PY -I -m unittest discover -s crates/hipfire-arch-qwen4/reference_oracle -p 'test_*.py'   # run from the repo root
```

`-I` ignores `PYTHONPATH`; `HIPFIRE_*` variables still apply. The system Debian torch crashes
(allocator error); use only the clean venv. Output per profile: `config.json`, `tokenizer.json`,
`tokenizer_config.json`, `model.safetensors` (BF16 + three I64 PLE buffers), `forward.npz`
(every op), `states.npz` (helper-capture states, plus DynamicCache states after prefill and chunked decode when the stack has an attention layer), `mtp_step0.npz`,
`manifest.json` (envelope, seed, source hashes, token ids, full tensor inventory with
shape/dtype/sha256, per-dump shape/logical dtype/sha256, negative-control results).

## Frozen envelope (root of `config.json`)

```json
{"hipfire_profile": {"id": "tiny-moe" | "tiny-dense-dev", "developer_only": true,
  "equation_source": "pinned-upstream" | "non-upstream-dense",
  "ffn": {"kind": "dense", "intermediate_size": N}
       | {"kind": "moe", "num_experts": E, "top_k": K, "intermediate_size": N,
          "shared_intermediate_size": M | null, "norm_topk_prob": bool}}}
```

PLE stored-geometry marker (S1 schema; values are read from the instantiated pinned `Qwen4ExpTextNGramEmbedding`, not constants):
`hipfire_profile.ple = {"valid_rows": 18}` and top-level
`qwen4_ple = {"version": 1, "multipliers": [3 x i64], "head_vocab_sizes": [7, 11], "head_offsets": [0, 7], "padded_rows": 24}`.
No other marker keys are emitted; row width, shard count and head counts come from `text_config`
(`ple_embed_dim`, `split_ngram_parts`, `heads_per_ngram`). Exactly one PLE table is supported by this marker.

Outer `qwen4_exp` / `Qwen4ExpForConditionalGeneration`, inner `qwen4_exp_text`; layer types use the
checkpoint spelling (`full_attention`; HF normalises it to `qwen_sparse_attention`). The config
also carries `full_attention_interval=2`, `mamba_ssm_dtype`, `mtp{num_hidden_layers,layer_types,hybrid,rope_theta}`,
`mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings=false`. Tiny vision tower tensors are
inventory component `vision` (never executed); filter by `component` for the text-only inventory.
`shared_intermediate_size = null` is **not constructible** from pinned HF (the sparse block always
builds a shared expert) and is rejected, not faked. Dense config omits every MoE key.

## Declared gate construction and shrink constraints

hidden16 vocab32 depth2 heads2/kv1 head8; GDN key1/value2 dims 8/8 conv2; layers `[linear, full]`;
experts3 top2 intermediate8 shared8; HC2 lowrank4; PLE on layer 1 (embed16, ngram3, heads/ngram1,
vocab-base7, divisible-by8, split1, conv2, eos2); QSA heads1 kv1 dim4 budget4 ratio2; rope
theta 10000 partial .5; max positions 64; output gate sigmoid; eager attention.

`--rope-theta` / `--max-position-embeddings` (defaults 10000 / 64, byte-identical output) override both for the main and the
cloned MTP config (`text_config.rope_parameters.rope_theta`, `max_position_embeddings`, `mtp.rope_theta`, tokenizer `model_max_length`); non-default
runs add `manifest.rope`, built from the instantiated main config, MTP config, MTP layer `self_attn.config` and a recomputed rotary `inv_freq`
(the MTP step feeds the main `rotary_emb` cos/sin to the cloned layer); any disagreement is refused. `rope_theta` must be finite and > 0;
`max_position_embeddings` an integer >= the 9 fixed tokens. The profile id/marker is unchanged; name the output directory for the variant.

Optional components (`--no-ple`, `--no-mtp`; `ProfileOptions.with_ple` / `with_mtp`, default true). Pinned HF supports a PLE-free model
(`ple_layer_ids` empty, one cache state slot); the MTP head is built here, outside HF, so `--no-mtp` constructs nothing. Absence follows the S1 schema and is
cross-checked against the instantiated model, the written inventory and the written `config.json` (a disagreement aborts generation):

- `--no-ple`: `text_config.ple_layer_ids = []`, the seven PLE scalars (`ple_embed_dim`, `ple_conv_kernel_size`, `ngram_size`, `heads_per_ngram`,
  `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`) absent, no `qwen4_ple`, no `hipfire_profile.ple`; no PLE layer, tensor or buffer;
  no `conv_state1`/`conv_state2` states; `ple_eos_history` and `ple_cached_history` are recorded as `status: skipped` with the reason, not omitted. `seed` is kept (not in the S1 PLE list).
- `--no-mtp`: no `text_config.mtp`, `mtp_num_hidden_layers` or `mtp_use_dedicated_embeddings`; no `mtp.*` tensor; no `mtp_step0.npz` (no empty or fabricated dump: `forward.npz` and
  `states.npz` are the real backbone dumps, bit-identical to the default's for the same options); `manifest.mtp = {present: false, constructed: false, ...}`, no `equation_sources.mtp`, RoPE evidence lists no MTP checks.
- Variants add `manifest.options.with_ple|with_mtp` (non-default only) and `manifest.optional_components` (`ple.layers`, inventory tensor counts, `cache_state_slots`, omissions); default manifests are unchanged.

Constraints the shrink must respect (all enforced by pinned HF validation or observed):

- PLE: `ple_embed_dim % ((ngram_size-1)*heads_per_ngram) == 0` (16 % 2); PLE only on linear layers; eos id required.
  Primes after `base-1`: heads 7 and 11 -> **valid 18 rows, padded 24** (divisible by 8), row width 8, 3 I64 buffers.
- QSA: `indexer_kv_heads == 1`, `budget % ratio == 0`, `rotary_dim (= head_dim*partial) <= indexer_head_dim` (4 <= 4).
  Selection only bites with >= 3 complete blocks (>= 6 visible tokens): the fixed text has 9 tokens.
- MoE: HF needs `num_experts > 0` and a shared expert; `top_k < experts` so the router boundary exists.
- Fixed text `hi<|endoftext|>to hi.` -> 9 ids, EOS (id 2) mid-sequence to exercise EOS-aware n-gram history.
- The requested seed is used exactly by default. QSA/router top-k boundary ties are not screened: selection is whatever pinned
  `torch.topk` does on CPU, deterministic within the pinned environment. Opt-in QSA screening: see "QSA tie-screened seeds" below (router ties stay unscreened).
- Linear-attention-only stacks (1 layer) cannot run cached/chunked decode in pinned HF
  (`DynamicCache.get_seq_length` needs an attention layer): no `prefill.*`/`chunked.*` states. `states.npz` is still emitted with
  `captured.layer<L>.{conv_state0,recurrent_state0,conv_state1,conv_state2}` derived from the helper captures (GDN conv input and
  final recurrent state, PLE short-conv input, last token ids); with a cache these are asserted bit-equal to the `DynamicCache` states.
- Width 5120 is bounded to 1-2 layers; every projection except the hidden axis is reduced. The two
  `[H,H]` MTP fusion projections (2 x 5120^2 = 52M elements, ~105 MB BF16) are the structural floor; guard 80M elements.

## QSA tie-screened seeds (opt-in, `--screen-qsa-ties`)

Default generation is unchanged (seed used exactly, `manifest.seed.requested == effective`, no `qsa_tie_screen` key; the seed-3800 fixtures are not touched).
With `--screen-qsa-ties`, `--seed` is the **start** seed: seeds `start, start+1, ...` are tried sequentially (none skipped) and the first seed whose
every selecting QSA query row passes is used. A row is *selecting* when `num_complete_blocks > block_topk` (block_topk = budget/ratio = 2), in every QSA layer of
the main forward **and** in MTP step 0 (both are required when MTP is present; main-only is accepted only with `--no-mtp`). A row passes when the k-th selected
block score is `> 0` and `kth - (k+1)th >= max(0.05, 0.05 * kth)`. The scores are the exact pre-`topk` tensors of the pinned HF `Qwen4ExpTextQSAIndexer.forward`
(relu/sum/sqrt scores over the causally visible complete blocks), captured by wrapping `torch.Tensor.topk` for calls made from that function: nothing is re-implemented.
The search is bounded (`--screen-qsa-max-seeds`, default 4096); exhausting it, or a geometry with no selecting row (e.g. `--layers 1`), is an error, never a fallback seed.
Variants are screened independently (a `--no-ple` or different-RoPE run chooses its own seed). The written run's scores are re-captured and must equal the screened run's.

Manifest (screened runs only): `seed = {requested: <start>, effective, init}` and `qsa_tie_screen = {rule: "tie-screened seed", start_seed, effective_seed,
rejected_count, rejected: [{seed, failing_rows, first_failing_row}], predicate, score_source, rows: [{source: "main"|"mtp", layer, position, complete_blocks,
block_topk, block_scores, kth, next, required_margin, observed_margin, passed}]}` (MTP layer is the MTP head's own layer index 0).

```bash
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/tiny-dense-dev-gpu-screened \
    --seed 3801 --rope-theta 10000000 --max-position-embeddings 1024 --screen-qsa-ties
$PY -I crates/hipfire-arch-qwen4/reference_oracle/generate_fixtures.py --tiny-profile tiny-dense-dev --out /tmp/tiny-dense-dev-gpu-nople-screened \
    --seed 3801 --rope-theta 10000000 --max-position-embeddings 1024 --no-ple --screen-qsa-ties
```

## Source / dense boundary

- Sources: the three Transformers files and three vLLM files are SHA-256 verified against
  `upstream.UPSTREAM_SOURCE_SPECS` **before use**; the verified bytes (not whatever import finds) are
  `exec`-ed under the real module names, and an already-imported unverified module is refused. A matching
  version string is never accepted alone (`transformers 5.16.1`, commit `93c8b7b4...`; vLLM `e126687a...`).
- `tiny-moe`: the real `Qwen4ExpForConditionalGeneration`, BF16, eager. Dumps are forward-hook / helper-tap captures of
  GDN conv+recurrence, PLE n-gram ids/embedding/conv, HC mix/inject, QSA indexer mask/pooling norms/attention, router/experts/shared expert, final mixer, logits.
- `tiny-dense-dev`: pinned decoder layers with the MLP slot replaced by
  `equations.dense_ffn_non_upstream` (SiLU gate*up -> down, BF16 rounding after every projection/activation/product/output)
  at the declared `intermediate_size`. The pinned `Qwen4ExpTextSparseMoeBlock` is swapped before construction, so no router/expert/shared
  tensor exists. Pinned HF has **no** dense decoder (`num_experts <= 0` rejected; layer always builds the sparse block);
  the dense profile is hipfire's hypothesis, labelled `non-upstream-dense`. It matched `Qwen4ExpTextMLP` bit-for-bit on the tiny case (reusable math only).
- MTP (`mtp_step0.npz`, inventory `mtp.*`): pinned HF ignores `^mtp.*` and has no MTP model. The branch executes the real vLLM
  `Qwen4ExpMultiTokenPredictor.forward` body from `amd/mtp.py` (AST-compiled, `self.*` set and key statements asserted against drift), with the pinned HF decoder
  layer (QSA + FFN + both HC modules) and HC final mixer behind vLLM's `(hidden, mlp_out, injection)` / `combine_and_mix` contracts. Label:
  `pinned-vllm-mtp-forward+hf-operators`. Not covered: later draft steps (`skip_topk` QSA index reuse, draft paged KV), vLLM Triton HC/QSA
  kernels (equivalence rests on vLLM `common/hyperconnection.py` torch equations matching HF `GatedResidual`, asserted). Embedding and LM head are shared with the main model.

## Negative controls (each must change named dumps; identity must change none)

`identity_rebuild`, `gdn_output_gate` (sigmoid<->silu), `ple_eos_history`, `ple_cached_history` (same suffix, different cached prefix),
MoE: `moe_router_gate`, `moe_norm_topk_prob`, `bf16_vs_f32_rounding`; dense: `dense_ffn_rounding`, `dense_gate_projection`.
Controls for an absent component are listed with `status: skipped, passed: null` and a reason (`--no-ple`: `ple_eos_history`, `ple_cached_history`); `--no-mtp` has no MTP control to skip.
The exact requested seed is used (no seed search) unless `--screen-qsa-ties` is given: tiny-moe and tiny-dense-dev at `--seed 3800` pass every control on torch 2.9.1+cpu, 1 thread.
Dumps are bit-stable on one torch build/thread count; cross-build comparison should use BF16 unit roundoff (2^-7).

## Exact KL quality method (`quality.py`)

Teacher = the pinned **source checkpoint** (unquantized routed experts) executed by the layer-major source engine; metric =
teacher-to-candidate full-vocabulary KL, nats per scored next-token position, temperature 1, no top-k/top-p/penalties. The
canonical 17-token top-32 path (`--source-reference` / `--compare`, `hipfire.qwen4.quality.v1`) is unchanged and remains the
regression; its top-32 uniform-tail `compare` is **not** the exact estimator. Do not subtract a floor KL from a candidate KL.

### Corpus manifest `hipfire.qwen4.kl_corpus.v1`

```json
{"schema":"hipfire.qwen4.kl_corpus.v1","name":"pilot-v0","seed":1234,
 "sequences":[
  {"id":"text-000","stratum":"text","tokens":[...],"score_positions":[0,1,2],"reset":true,"meta":{"source":"..","half":0}},
  {"id":"asst-000","stratum":"assistant","tokens":[...],"role_mask":[0,0,1,1,1],"score_range":[2,4]}]}
```

- `stratum` is `text` or `assistant`; every sequence is independent: all attention/conv/recurrent/PLE state starts at position 0 and
  nothing crosses sequences (`reset` must be true/omitted; intra-sequence resets are not supported).
- Scored items are **input positions** `p in [0, n-2]` predicting `tokens[p+1]`; the last token never has a target (rejected). Give
  exactly one of `score_positions`, `score_mask` (n x {0,1}) or `score_range` `[start,end)`. Optional `targets` must equal `tokens[p+1]`.
- `assistant` requires `role_mask` (1 = assistant content) and every scored target must be assistant content. `text` ignores it.
- Large corpora may use `token_payload {path (relative), sha256}` (u32le) with per-sequence `offset`/`length` instead of `tokens`.
- Optional `meta.half` (0/1) declares the predeclared split halves (default: alternating documents within a stratum).
- Identity: per-sequence token / score-position / role-mask sha256 plus a `corpus_sha256` over them (names and `meta` excluded).

### Commands

Runtime limits (CPU mode, the default): `--threads <= 16`, `RLIMIT_AS <= 64 GiB` (`--memory-limit-gib`, default 64), niceness >= 10,
CUDA/HIP hidden. Run pilots as `nice -n10 python quality.py ...` (the CLI also self-nices). `--device cuda[:N]` (ROCm uses `cuda`)
disables the AS limit unless `--memory-limit-gib N>0` is given and records peak host RSS plus per-layer device peak instead; one decoder layer's
weights and expert slices live on the device at a time. Always pass `--local-dir <source safetensors dir>` for a no-network run.

```bash
cd <worktree>/crates/hipfire-arch-qwen4/reference_oracle
# CPU pilot: native + reverse-experts in lockstep over ONE weight traversal, then floor compare (+ native-vs-native zero check)
nice -n10 $PY quality.py --exact-floor --corpus pilot.json --out-dir out/pilot --local-dir /data/qwen4-src --threads 16 --memory-limit-gib 64
# GPU rental
$PY quality.py --paired-floor --corpus pilot.json --out-dir out/pilot --local-dir /data/qwen4-src --device cuda --logits-storage bf16 --memory-limit-gib 0
# single teacher export, then compare with any export of the same corpus (e.g. a candidate in the same layout)
$PY quality.py --exact-export --corpus pilot.json --out-dir out/teacher --reduction-order native --local-dir /data/qwen4-src
$PY quality.py --exact-compare --reference out/teacher/native --candidate out/cand --out out/ordinary.json --bootstrap 2000 --seed 20260101
# paired per-position difference (e.g. ordinary minus teacher-ID-forced) from two compare outputs
$PY quality.py --exact-paired --positions-a out/ordinary.positions.npz --positions-b out/forced.positions.npz --reference out/teacher/native --out out/paired.json
$PY -I -m unittest test_quality_exact   # CPU-only unit tests (no checkpoint, no network)
```

Useful knobs: `--pass-tokens N` (default 0 = whole corpus in a single layer-major pass; every extra pass re-reads all weights),
`--head-rows`, `--expert-cache-gib` (per-layer expert RAM/device cache cap, default 8), `--max-layers K` (timing/artifact probe only: the export is
labelled `truncated_layers`, `--exact-floor` returns status `truncated_timing_probe_no_quality` and runs no floor/self comparison, and
`--exact-compare` raises on any truncated export, so a partial layer prefix can never be imported as BF16 teacher quality), `--export-hidden`, `--no-hash-files`, `--remote-byte-budget-gb` /
`--stream-disk-gb` (forwarded to the shared checkpoint reader), `--corpus canonical` (the 17-token sequence through the exact path).

### Outputs (`out-dir/<reduction-order>/`)

| file | content |
|---|---|
| `score_index.u32le` | `[N,3]` `(sequence, input position, target token)` per scored row, corpus order |
| `logits.f32le` / `logits.bf16le` | `[N,V]` actual forward logits; bf16 stores the forward's own bf16 words (accepted only if the head matmul dtype is bfloat16 and the F32 widening round-trips bit-exactly; the scorer widens exactly) |
| `route_ids.u16le`, `route_scores.f32le` | `[L,P,K]` layer-major tapes over **all** input positions (`P` = corpus input tokens, sequences concatenated in corpus order): ordered expert IDs and F32 router scores captured once per routed forward through the `route_sink` |
| `final_hidden.<dtype>le` | optional `[N,H]` final hidden rows (`--export-hidden`) |
| `export.json` | `identity` (corpus hashes, source/operator provenance incl. variant AST + diff hash, device/torch/HIP versions, file shapes + sha256) with `identity_sha256`, and `runtime` (see below) |

`--exact-floor` also writes `floor_native_vs_reverse.json` / `.positions.npz` and `floor_self_native.json` (zero sanity). Compare reports
hold, per `text`, `assistant` and the equal-weight `mixture` (0.5 text + 0.5 assistant): mean KL with a **document-bootstrap** 95% CI
(independent fixed-seed generator per stratum, >= 100 replicates, 2000 default), p50/p95/p99, teacher/candidate NLL + PPL, NLL delta, top-1 agreement,
teacher top-1 = target, routing agreement (exact ordered / set overlap), context-length strata, split-half agreement and the **stability
gate** (CI half-width <= 0.002 nat, and <= 5% of the mean once the mean >= 0.02). Per-position values (KL, NLL, top-1, route agreement) are
persisted in the `.positions.npz`.

### Exactness

KL is computed in float64 from the actual logits with one streamed online pass over vocabulary blocks: `KL = W/S - (lse_T - lse_C)` with
`W = sum exp(a-m_T)(a-b)`, `S = sum exp(a-m_T)`; nothing is rounded or clipped, non-finite logits fail, and a KL below `-1e-9` raises rather
than being clamped. Identical logits give exactly 0; `test_quality_exact.py` checks the toy distribution, extremes, block sizes, bf16 storage
and NLL against independent recomputations.

### Runtime artifacts

`export.json:runtime.timing` has per-layer wall, I/O (fetch + decode + host-to-device) and arithmetic seconds, input tokens/s per variant,
bytes read and expert hits/misses, current and peak RSS, per-layer device peak/allocated/reserved bytes (GPU), head seconds, the checkpoint reader's
`transfer_stats`/`coverage_report`, and `projection_327680` (I/O charged per pass, arithmetic per forwarded token per variant; an estimate that
ignores superlinear attention growth). Lockstep variants share one set of timings. CUDA timings synchronize before every clock read.
Not implemented here: teacher-ID-forced replay (`--exact-paired` only compares existing per-position outputs) and MTP.
