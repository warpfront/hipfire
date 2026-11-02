# Qwen4 static YaRN: long-context retrieval (research, not product)

These multi-needle fixtures and the runner are prepared but **not run** here. They exercise the
engine-level YaRN override `attention.yarn_factor` / `HIPFIRE_YARN_FACTOR` (#856); its own evidence is
a single passphrase needle in 295K- and 499K-token prompts on gfx1151. Runs above 57K on the Halo
need Main's explicit OK. `niah.sh` refuses them unless `MAIN_OK_LONGCTX=1` is set.

- `gen_niah_long.py TOKENIZER_JSON OUT_DIR` regenerates the three fixtures byte-identically. They are not
  committed (4.4 MB). `manifest.json` pins each fixture's sha256, token count and needle offsets, plus the
  tokenizer sha256. It uses the format of `benchmarks/longctx/niah/generate_niah_multi.py` (consumed by
  `scripts/serve_harness.py --niah-file`), calibrated with the Qwen3.x tokenizer
  (`crates/hipfire-runtime/tests/fixtures/hf_tokenizer_corpus/tokenizer.json`).
  Each fixture has four needles:

  | Fixture | Filler tokens | Needle token offsets | Needles past 262,144 |
  |---|---:|---|---:|
  | 48K (in-window control) | 47,914 | 4,809 / 19,144 / 33,579 / 45,517 | 0 |
  | 300K | 299,820 | 29,985 / 149,923 / 269,662 / 290,792 | 2 |
  | 500K | 499,724 | 49,937 / 224,855 / 349,957 / 474,876 | 2 |

- `niah.sh FIXTURE:FACTOR:MAXSEQ:KV ...` runs one fresh `hipfire serve` per arm under a fabric
  Halo hold (greedy, thinking off, MTP off, session cache off). For example: `48k:native:57344:auto`,
  `48k:2:57344:auto`, `500k:2:524288:q8`.

## 512K memory plan on Strix Halo (gfx1151, FN XTS)

This plan is scaled from a measured load: 33,586 MiB free after load. In that load a 262,144-token
context committed 15,392 MiB of QSA context, 512 MiB of gather workspace and an 860 MiB MTP request.

- **Default `kv=bf16` (F32 QSA state, about 61.6 KB per token):** 512K needs about 30 GiB of QSA,
  plus about 1 GiB of gather workspace, plus MTP and 2 GiB of headroom. That does not fit. The card
  cap lands near 493K tokens, or about 520K with MTP off.
- **Plan: `memory.kv_cache=q8`:**
  - Q8_0 K/V with BF16 index arenas costs 1,408 B per token per QSA layer (12 layers).
  - The MTP head stays F32 at 4,736 B per token.
  - That totals about 21.6 KB per token, or about 10.6 GiB at 524,288 tokens.
  - With gather workspace and headroom the margin is about 18 GiB.
  - Run with session cache off, MTP off and `--max-seq 524288`.
- **Expected time:** about 6–7 minutes of prefill for the 500K fixture, extrapolated from the stock
  Halo rate of 1,537 tok/s at 212K. One 30-minute hold covers load, prefill and decode.
