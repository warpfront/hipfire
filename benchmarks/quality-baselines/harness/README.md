# Harness — quant-quality eval scripts

This directory holds the tools that build / consume / aggregate the
KLD references and produce the result tables.

## Two reference generations — do not mix them

| | Legacy harness refs (this directory's `manifest.json`) | Current Qwen3.8-27B quality pins |
|---|---|---|
| Models | Qwen3.5-0.8B, Qwen3.5-9B, Qwen3.6-27B (BF16 teacher) | Qwen3.8-27B (`qwen3.8-27b.mq4-xts`) |
| Where published | HF **dataset** repo `hipfire-models/qwen-kldref` (the `hf_repo` field in `manifest.json`); fetched by `scripts/fetch-eval-refs.sh` | `kld/` in the **model** repo `hipfire-models/qwen3.8-27b` (not in `manifest.json`, not fetched by `fetch-eval-refs.sh`) |
| Corpus / size | `slice/wikitext2-1024s-2048ctx.txt`, 1175 chunks of 2048 tokens (~2.5 GB each) | WT2 `qwen3.8-27b.ref_wt2.bin` and code24 `heldout-code24.gptqsym.kldref.bin`, 24 chunks of 2048 tokens (50.7 MB / 12.6 MB) |
| Producer | `build_kld_ref` from a BF16 GGUF via llama.cpp (see `manifest.json` `producer_cmd`) | Published ready-made; self-contained (tokens embedded, no corpus or tokenizer download) |
| Use | Quant-ladder / Pareto comparisons across variants and GGUF anchors | Exact, bit-reproducible per-arch pins used by `AGENTS.md` |

`hipfire-models/eval-refs` is not a published repo; legacy refs live in
`hipfire-models/qwen-kldref`, and `scripts/fetch-eval-refs.sh` is named for
the *role* (eval references), not for an HF repo.

The current pins, their expected KLD and `.kldseq` SHA-256 per arch/route,
and the exact reproduction recipe are authoritative in two places; read
those instead of copying numbers from here:

- [`AGENTS.md` §5 → "Pinned Hugging Face bench fixture" → "Quality pins"](../../../AGENTS.md#pinned-hugging-face-bench-fixture)
- [`kld/VERIFY.md` in `hipfire-models/qwen3.8-27b`](https://huggingface.co/hipfire-models/qwen3.8-27b/blob/77ff3bea2/kld/VERIFY.md)
  (one `hf download`, one pinned `cargo build`, one copy-paste block; pinned
  at HF commit `77ff3bea2`)

Reference SHA-256s for the current pins (same in both places):

| Reference | SHA-256 |
|---|---|
| WT2 `kld/qwen3.8-27b.ref_wt2.bin` | `8c545178fb43647499c4c7bdd33178c24fd402615c641add9ed51a0c9a43234a` |
| code24 `kld/heldout-code24.gptqsym.kldref.bin` | `43ab7591f2acf6b5b4058b77253109230d96591be02469ddef55235102ec7582` |

## Files

| File | Purpose | Status |
|---|---|---|
| `manifest.json`         | SHA-pinned index of the legacy BF16 reference dumps (Qwen3.5-0.8B, Qwen3.5-9B, Qwen3.6-27B); does **not** list the Qwen3.8-27B pins | 9B and 27B entries uploaded to `hipfire-models/qwen-kldref` (dataset); 0.8B entry has `hf_repo: null` (local-only, not uploaded) |
| `kldref_format.py`      | Reader/writer for the hipfire HFKLDR ref format + HFKSEQ per-sequence sidecar (v2: adds mean_nll for PPL) | done |
| `kld_reduce.py`         | Bootstrap CI + result-table emitter (incl. PPL column) | done |
| `kld_diff.py`           | Four-metric A/B comparison of two HFKSEQ files | done |
| `tokenizer_parity.py`   | Tokenizer-parity check (hipfire vs llama.cpp BPE) | done; ran 2026-05-08 — see plan §4.5 "Tokenization (Step 1.5 verdict)" |
| `hfq_provenance.py`     | Prints the build provenance stamped into an `.hfq` | done |
| `spe_ablation.sh`       | SP-E ternary/binary ablation runner | done |
| `canary.md`             | 11-sequence harness-output reproducibility fixture | sequences populated; expected per-sequence KLDs are still `TBD` |

The corresponding Rust binaries live at
`crates/hipfire-runtime/examples/{build_kld_ref,eval_hipfire,eval_gguf}.rs`
(and `crates/saddle-lab/examples/tokenize_slice.rs` for the CPU-only
tokenizer helper). `eval_hipfire` has `required-features = ["deltanet"]`;
`build_kld_ref`, `eval_gguf` and `tokenize_slice` have none. Build with
`cargo build --release --features deltanet --example <name>`
(`--features deltanet` is only needed for `eval_hipfire`).

## File format

Authoritative source: `kldref_format.py` (spec in
`docs/plans/issue-113-quant-quality-eval.md` §4.2).

**HFKLDR v1 reference** (little-endian):

- 32-byte header: magic `HFKLDR\0\0` (8 bytes), `u32` version (1), `u32`
  `n_ctx`, `u32` `n_vocab`, `u32` `n_chunk`, `u16` `top_k`, `u16` flags
  (0), `u32` reserved (0).
- `n_ctx × n_chunk` `u32` token ids (embedded).
- Per chunk, per scored token (`n_ctx − 1 − n_ctx/2` tokens per chunk, i.e.
  1023 at `n_ctx = 2048`): `u32 top_indices[top_k]`, `f32 top_log_probs[top_k]`
  (descending log-prob), `f32 sum_p_residual` (probability mass outside the
  top-K), `f32` zero pad. `8 + 8·top_k` bytes per token (2056 B at
  `top_k = 256`).

Size check: `32 + 2048·24·4 + 24·1023·2056 = 50,675,552` bytes, matching the
published WT2 reference (`top_k = 256`, vocab 248320); the legacy refs
(`n_chunk = 1175`) are 2,480,989,032 bytes.

**HFKSEQ per-sequence sidecar** (the `.kldseq` evaluator output): magic
`HFKSEQ\0\0`, version 2 adds `mean_nll` per sequence (PPL = `exp(mean_nll)`);
the reader accepts v1 and returns NaN for `mean_nll`. The evaluator's
`.kldseq` SHA-256 is what the quality pins compare byte for byte.

## Reference fetch

Legacy refs: `scripts/fetch-eval-refs.sh` (at repo root) reads
`manifest.json` and either verifies (if locally present) or downloads (if
`.hf_repo` is set) each ref into `benchmarks/quality-baselines/refs/<name>`.
Entries with `hf_repo: null` (the 0.8B ref) are skipped with a warning when
absent locally.

Current Qwen3.8-27B pins: `hf download hipfire-models/qwen3.8-27b
qwen3.8-27b.mq4-xts kld/qwen3.8-27b.ref_wt2.bin
kld/heldout-code24.gptqsym.kldref.bin --local-dir <dir>` and verify the
SHA-256s above (the `VERIFY.md` block does both).

## Quality-pin protocol (current)

For anything compared against the Qwen3.8-27B pins, follow `VERIFY.md` /
`AGENTS.md` exactly:

- `eval_hipfire --scoring-mode prefill --max-chunks 24` — teacher-forced
  prefill over the 24 reference chunks; always pass `--kv-mode`, `--kv-v`
  and `--scoring-mode` explicitly instead of relying on defaults.
- KV flags per arch: gfx1100 and gfx1151 `--kv-mode q8 --kv-v q8`; gfx1201
  `--kv-mode fp8 --kv-v q8` (fp8 is admitted by the evaluator only for dense
  Qwen on gfx1201). Do **not** use `asym3` for pins (see below).
- Environment: `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0`, a fresh `HOME`
  (and `HIPFIRE_HOME`) and a fresh kernel cache (`HIPFIRE_KERNEL_CACHE`) per
  process; gfx1151 pins additionally require a *warm* shared kernel cache
  populated by one unscored run.
- Pins are exact only for the measured toolchain (ROCm 10.1 / HIP 7.16 /
  clang 24), arch and flags; a different compiler, arch, build or flags is
  not expected to reproduce the `.kldseq` hash.

## How to add a new quant variant

1. Make sure the BF16 reference for the model exists. If not, run
   `build_kld_ref` (see plan §4.3 "Producer: `build_kld_ref`") on a host
   with enough RAM for the BF16 weights (27B BF16 needs ~50 GB resident).
   Upload to the HF dataset repo `hipfire-models/qwen-kldref` and add an
   entry to `manifest.json` with `sha256`, `hf_repo`, `hf_repo_type`,
   `producer_cmd`, `llamacpp_commit`, `slice_md5`, and shape metadata.
   (A new current-generation pin set is instead published under the
   model repo's `kld/` with a `VERIFY.md`, and recorded in `AGENTS.md`.)

2. Run the candidate against the cached reference. Paths are relative to
   the repo root; adjust `--kv-mode` per arch (see protocol above). Full
   legacy-slice runs omit `--max-chunks`:

   - hipfire variants:
     ```
     HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 \
     cargo run --release -p hipfire-runtime --example eval_hipfire \
       --features deltanet -- \
       --model <path-to-hfq> \
       --ref benchmarks/quality-baselines/refs/<model>-bf16.kldref.bin \
       --output benchmarks/quality-baselines/results/<date>/per-seq/<variant>__<arch>.kldseq \
       --kv-mode q8 --kv-v q8 --scoring-mode prefill
     ```

     For a compressed-KV measurement spell the supported FWHT mode
     (`--kv-mode fwht3`, or `fwht2` / `fwht4`). The `asym2` / `asym3` /
     `asym4` (and `turbo*`) spellings are **deprecated since 0.4.0 and
     scheduled for removal in 0.5.0**; `eval_hipfire` builds the legacy
     Givens-rotation asym KV cache directly for them, and `asym3` is still the
     evaluator's built-in default `--kv-mode`, so never omit the flag. See
     [`docs/QUANTIZATION.md`](../../../docs/QUANTIZATION.md) for the KV-mode
     table. Older result tables under `results/` that mention `asym3`
     predate this and are historical context, not a current recipe.

   - GGUF anchor variants:
     ```
     cargo run --release -p hipfire-runtime --example eval_gguf -- \
       --candidate-gguf <path-to-cand.gguf> \
       --ref benchmarks/quality-baselines/refs/<model>-bf16.kldref.bin \
       --slice benchmarks/quality-baselines/slice/wikitext2-1024s-2048ctx.txt \
       --output benchmarks/quality-baselines/results/<date>/per-seq/<variant>__<arch>.kldseq \
       --llama-perplexity-bin <path-to-llama-perplexity>
     ```

   Output filename convention: `<variant>__<arch>.kldseq` —
   `kld_reduce.py` parses `rsplit("__", 1)`.

3. Aggregate:

   ```
   python3 benchmarks/quality-baselines/harness/kld_reduce.py \
     --result-dir benchmarks/quality-baselines/results/<date>/per-seq/ \
     --out-md   benchmarks/quality-baselines/results/<date>/result-table.md \
     --out-json benchmarks/quality-baselines/results/<date>/result-data.json
   ```

4. Eyeball the markdown table; commit alongside the run's
   `2026-MM-DD-quant-pareto.md` write-up.

## Plan reference

`docs/plans/issue-113-quant-quality-eval.md` is the PRD for the legacy
harness — binary format, eval matrix, scoring modes, validation
methodology, and pivot decisions. The parent
[`../README.md`](../README.md) has the legacy-ref download quick-start. For
the current Qwen3.8-27B pins, `AGENTS.md` and `kld/VERIFY.md` take
precedence over the plan.

## Pinned llama.cpp commit

`9dcf83552887bb898b4a98a5761361e504e31fc3` (master, 2026-05-08).

Applies to `build_kld_ref` and `eval_gguf` (legacy refs and GGUF anchors);
the current Qwen3.8-27B pin runs through `eval_hipfire` alone and needs no
llama.cpp. Both tools invoke `<bin> --version` before spawning, parse the
parenthesized short hash, and assert it's a prefix of the pinned 40-char
commit. If the user's llama.cpp build is from a different commit, the
format may have drifted; bail loudly.
