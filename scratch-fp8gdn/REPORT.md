# fp8v2 GDN chunk-scan admission — results (Card-B, gfx1201)

- Date: 2026-09-21. Worktree `/home/kaden/ClaudeCode/warpfront/wt-fp8sym`, branch
  `gfx1201-fp8-gdn-chunk`.
- Code commit: `60f3add5a42d69d131f975d2c5dcbc02a1f4c1dd` (6 insertions, 1 deletion in
  `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`, admission block ~:11614-11620;
  parent = merged tip `f3afdc1f6`).
- Model: `qwen3.8:27b-mq4-xt` = `/home/kaden/qcal/qat/v25/qwen3.8-27b.mq4v2.xt.sym-a035.hfq`
  (symmetric V2.5 XT, uniform MQ4G256V2), `--kv-mode fp8`.
- Card-B env: `HOME=/home/kaden/.hipfire-homes/ab1
  ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397
  HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
  HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_GRAPH=1
  HIPFIRE_DAEMON_BIN=<wt>/target/release/daemon`; fp8v2 arm adds `HIPFIRE_IU4_PREFILL=0`.
- Base daemon: `/home/kaden/ClaudeCode/warpfront/wt-lloyd/target/release/daemon`
  (mq4-lloyd @ f3afdc1f6, built 2026-09-21).
- Change: `gdn_chunk_scan_common_admitted` projection term was
  `gpu.flags.iu4_prefill_enabled()`; now
  `iu4_prefill_enabled() || (arch == "gfx1201" && gfx12_mq4v2_fp8_qkvza)`.
  gfx1100/gfx1151 reduce to the old term (second disjunct is false off gfx1201).

## Buffer-compatibility check (no stop condition)

All `batch_chunk_delta_net_input_projection` qkvza arms — iu4-prepared
(`gemm_qkvza_mq4g256v2_wmma_iu4_prepared`), uniform fp8-prepared
(`gemm_qkvza_hfq4g256_wmma_gfx12_mq4v2_fp8_prepared`), Lloyd fp8-prepared, and the
standalone-pack fp8 fallback via `fused_qkvza_key_for` — write the same four f32
batched scratch buffers (`pbs.dn_qkv_batch`, `dn_z_batch`, `dn_beta_batch`,
`dn_alpha_batch`) with identical m/k/n dims. The chunk scan consumes only
post-GEMM f32 (`gdn_chunk_prep` over `dn_qkv_batch` + conv/alpha/beta/a_log;
`require_bytes` checks `n*(2k+v)*4` f32) plus Q8/EF state — no iu4-specific
input. Per-layer admission (uniform MQ4G256V2 ×4) holds for this model. No other
iu4 gates exist on the scan path (`gdn_chunk_prepare/prep/scan_segment` take no
iu4 prep; kernels are shared across gfx1100/gfx1151/gfx1201).

## 1. KLD (fp8v2 arm, GRAPH=0)

`HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_IU4_PREFILL=0
./target/release/examples/eval_hipfire --model <resolved> --ref
/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin --kv-mode fp8 --kv-v q8 --scoring-mode
prefill --max-chunks 24 --output scratch-fp8gdn/c24.bin`
(example built with `cargo build --release -p hipfire-runtime --example eval_hipfire
--features deltanet`)

- c24 slice-mean KLD = **0.049387** (mean NLL 1.862507, PPL 6.4399, 24552 tokens,
  72.3 s) vs base 0.049063 → **PASS** (≤ 0.050, +0.000324).

## 2. Paired daemon bench (fp8v2 arm)

`hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,8192 --ctx 128 --tg 128
--spec off --runs 3 --warmups 1 --kv-mode fp8 --json`, medians of 3, tok/s:

| run | pp512 | pp8192 | decode c128/tg128 |
|---|---|---|---|
| base-1 | 1894.4 | 2028.3 | 28.70 (cold first run of day) |
| new-1 | 2065.7 | 2231.3 | 36.45 |
| base-2 | 1866.5 | 1999.2 | 36.46 |
| new-2 | 2052.7 | 2225.9 | 36.49 |

- pp8192: +10.0% (pair 1), +11.3% (pair 2) — the expected ~52 us/token GDN win
  (490 → ~448 us/token).
- pp512: +9.0%, +10.0%.
- decode: warmed runs agree within 0.1% (36.45/36.46/36.49); base-1 decode 28.70
  is a cold-clock outlier, unrelated to the change (decode path untouched).

TTFT (`--ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2
--json`, 5909 prompt tokens, medians of 8):

| arm | TTFT | pp |
|---|---|---|
| base (fp8v2) | 2986.55 ms | 1978.5 tok/s |
| new (fp8v2) | 2690.20 ms | 2196.5 tok/s (−9.9% TTFT, +11.0% pp) |

## 3. Battery (fp8v2 arm)

`python3 scripts/serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt
--thinking off` → **5/5** (`turns=5 runaway=0 empty=0 attractor=0
retrieval_miss=0`, avg prefill 703.0 tok/s, decode 36.5 tok/s).

## 4. iu4 arm TTFT (flag-free, new daemon unless noted)

| daemon | TTFT | pp |
|---|---|---|
| new | 1671.19 ms | 3535.8 tok/s |
| base | 1678.17 ms | 3521.1 tok/s (−0.4%, within noise) |

iu4 path unchanged by construction (first disjunct is the old term); still ~1.6×
faster than the fp8v2 arm on this prompt.

## Raw artifacts

`c24.bin`, `matrix-{base,new}{1,2}.json`, `ttft-{base,new,iu4,iu4-base}.json`
(+ `.stderr` logs) alongside this report in `scratch-fp8gdn/`.
