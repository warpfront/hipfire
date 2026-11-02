# gfx1100 decode campaign — independent re-verification and v1-only prune

- **Date:** 2026-10-09
- **Lifecycle:** historical
- **Branch under test:** `perf/gfx1100-decode-campaign` (pruned tip;
  binary md5 `a862264b04e215a5c2ce96224002ea91`,
  daemon md5 `54120d4936dc9652682bbef5b908014e`)
- **Baseline:** `master` @ `0f999cb4dc` (binary md5
  `879d8643ae946c2be6dc05e4e2d0b408`, daemon md5
  `022719b51d0a62f04d3b4bbe000fbedf`)
- **GPU:** gfx1100 (Radeon RX 7900 XTX, 24 GiB), exclusive.
- **Env:** containerized, `buun-llama:git-9be465c6c-rocm10.0.0`
  (ROCm 7.15 / HIP 7.15), shared `HIPFIRE_KERNEL_CACHE` volume.
- **Method:** `hipfire bench <model> --matrix --json --kv-mode q8 --spec off`
  (native daemon protocol), 5 runs / 3 warmups per cell, fresh process per arm,
  arms interleaved within a window. Identical image, model bytes and env for
  both arms; only the binary differs.

> This record covers two things: an independent re-verification of the 2026-08
> campaign against the *current* master, and the subsequent pruning of the
> campaign's v1-only surface. It is not a re-measurement of the campaign's
> original base-to-tip delta and must not be read as one.

## Fixtures

| Artifact | Identity | Notes |
|---|---|---|
| `qwen3.5-4b.mq4` (2,588,006,400 B) | md5 `712b69f8cf1016081cfa507c4d50e33d` | matches the campaign's fixture exactly |
| `qwen3.5-4b.mq3` / `-mq6` | HF `hipfire-models/qwen3.5-4b` | Magnum V2 family, 4B geometry |
| `qwen3.8-27b.mq4-xt` (14,987,185,152 B) | sha256 `80e7c624424fd1d363ba86681d3dc1e5ac5534e0e064306a32be204c4843d0f3` | byte-matches the live HF object AND the AGENTS.md §5 historical entry (the 2026-09-15 re-issue with AWQ sidecars). The campaign-era records' `9f91556f…` / 14,980,361,216 B pin is the documented pre-reissue upload — a superseded artifact, not stale documentation |
| `qwen3.8-27b.mq4` (15,662,615,552 B) | md5 `d1292b4d5bd6046693604201a6ca8074`, sha256 `5bb556a6…e507` | same size as the campaign's `2fb2edc2…` fixture, different bytes: HF re-uploaded this one too |
| `qwen3.8-27b.mq4-pro` | HF `hipfire-models/qwen3.8-27b` | "pro" tier, MQ4G256V2 |

## Retained gains (pruned branch vs master)

4B MQ4 product path, interleaved windows × 5 runs:

| artifact | decode @ctx64 | decode @ctx2048 | pp64 prefill | pp2048 |
|---|---|---|---|---|
| `qwen3.5-4b.mq4` (legacy) | **+6.0%** (202.06 → 214.21) | +6.2% | −0.0% | −0.2% |
| `qwen3.5-4b.mq3` (MQ3G256V2) | **+3.1%** (209.25 → 215.82) | — | −0.1% | — |
| `qwen3.5-4b.mq6` (MQ6G256V2) | **+2.6%** (157.85 → 161.99) | — | −1.0% | — |

Retained opt-in levers, re-measured against the campaign records:

| Lever | Record | Re-verified |
|---|---|---|
| `HIPFIRE_FA_KVWRITE_FOLD=1` | +0.5–1% decode | **+0.55%** (25 samples/arm; t=2.88, df=40.7, p=4.0e-3, Cohen's d=0.81) |
| `HIPFIRE_LM_HEAD_HFQ3=1` | +~10% decode (research-only, lossy) | **+9.8%** legacy mq4, **+11.0%** mq3, **+7.8%** mq6 |
| `HIPFIRE_QKVZA_FUSEDNORM=1` | −26% decode (negative oracle) | −25.6% — oracle since **removed** (v1-only) |

Correctness / bit-exactness (branch binary):

- `test_kernels`: **17 passed / 0 failed / 1 skipped** — identical to master.
- `probe_fa_kvwrite`: **every byte** of both Q8_0 cache rows plus
  `fa_q`/`fa_gate`/`fa_k` bitwise identical to the legacy pair-writer.
- `probe_fa_prep`: `fa_q`/`fa_gate`/`fa_k` IDENTICAL.
- End-to-end (native serve, temp 0, `enable_thinking=false`, 64 tokens):
  **master ≡ branch**, **branch ≡ branch+`FA_KVWRITE_FOLD`**, byte-for-byte.
- `cargo test --lib --workspace --locked`: 4582 passed, 0 failed.
- Serve battery (`--thinking off`, seed 42): branch 202.9 avg decode vs master
  190.0 (+6.8%), 5/5 coherent each (runaway=1 = reasoning turn at the length cap).

## v1-only surface removed (2026-10-09)

The 2026-08 campaign was measured on the legacy quant family
(`HFQ4G256` / `MQ4G256`). `hipfire quantize --format mq4` now emits
`MQ4G256V2` (`crates/hipfire-quantize/src/pipeline.rs`), i.e. v1 MQ is no
longer the product output, so everything below was discarded:

| Removed | Why (evidence) |
|---|---|
| BT2 prefill family: 6 kernels (`gemm_{gate_up,qkvza,qkv}_hfq4g256_wmma_bt*`, residual `ksplit_det_bt2` + `_ks2` + `finalize_ks2`), its 4 dispatch sites, `HIPFIRE_BT2_DISABLE`, the per-GEMM `*_BT2_FORCE`/`_KS2` arms, `HIPFIRE_GATE_UP_VARIANT=bt2/bt4`, and `scripts/deep_ab_bt2.sh` + `deep_ab_isolation.sh` | the kernels are `..._hfq4g256_...`: prefill profiles show them on the legacy 4B (`gemm_*_hfq4g256_wmma_bt2`) and show `gemm_*_mq4g256v2_wmma` with **no bt2** on every V2 artifact. Kill switch measured inert on mq3/mq6/27B-V2 (+0.2%/−0.7%/−0.1%, inside noise). The V2 family already ships its own batch-tiled arms upstream (`gemm_*_mq4g256v2_wmma_gfx11_bt`, `..._gfx12_bt`) |
| Fusednorm negative oracles: 3 kernels + launchers, the three `HIPFIRE_{QKVZA,QKV,GATE_UP}_FUSEDNORM` gates and their dispatch arm, `probe_qkvza_fusednorm` | gates were `matches!(dtype, MQ4G256 \| HFQ4G256)` — unreachable from any V2 dtype; measured −25.6% |
| Residual-family oracles: `HIPFIRE_RESIDUAL_LUT`, `HIPFIRE_RESIDUAL_PERSIST_R2` (+`_SCOPE`), `HIPFIRE_RESIDUAL_DUALROW` + 2 kernels | they lived inside `gemv_hfq4g256_residual` (legacy); V2 dtypes take `gemv_hfq4g256_residual_mq4v2` / `_mq4v2_lloyd` |
| 4B shape arms (9216/9216/2560) added to the seven `GFX1100_DENSE_GATE_UP_*` variant gates | those variants feed `gemm_gate_up_hfq4g256_*` — legacy-only GEMM family |
| The invalid LDS-staged probe (`HIPFIRE_XLDS_R`) | kernel paired nibbles 4..7 of each lane's word with x offsets 0..3 (needed a second `float4` at +4) and its launch requested 0 bytes of dynamic LDS for a kernel staging `k` floats in `extern __shared__`; the engaged arm degenerated into a repeating-token attractor while the record claimed "BITEXACT" |

Removed records: the BT2 27B checkpoint, the qkvza-fusednorm gfx1101 oracle
checkpoint, and the 2026-08 v1 campaign narrative + checkpoint chain.

Claim retractions that follow:

- **`+29%` BT2 pp32 prefill on 27B does not reproduce.** In its own harness
  (`bench_qwen35_mq4 --prefill 32`, 3 interleaved windows) master read
  460.5 and the branch 460.5 tok/s (BT2 disabled: 458.9) — nowhere near the
  recorded 583.95. Present-day `qwen3.8-27b.mq4` bytes differ from the
  recording's `2fb2edc2…` fixture, so that artifact is not retrievable by
  digest.
- **`+22.3%` pp64 prefill on 4B was BT2**, and goes away with it (pp64 is now
  −0.0% vs master). The decode deltas stand.
- The 4B speed-gate floor raise (`4b_mq4_pp32_prefill_tok_s` 1014.0 → 1843.5)
  is withdrawn: a clean ROCm 7.15 container cannot reach 1843.5 (branch read
  1663–1797, master 1601.0 — the branch was +3.9% over master, not regressed),
  so the gate and its pre-commit hook failed on the branch itself. Floors are
  restored to the merge-base rows and the gate passes (1657.1 pp32 / 197.8 gen).

## Gate status after the prune

- `scripts/leanup-ratchets.sh` — **OK** (21 assertions, 0 violations);
  `bypass_total` is back to **307** (the campaign's +3 raise is withdrawn, so
  neither the `RATCHET-RAISE:` trailer nor the `ratchet-raise` label is needed).
- `scripts/ratchet-diff.sh origin/master` — **OK**, no expectation weakened.
- `check-crate-maps.py --check`, `check-scratch-growth.py`,
  `check-changelog.py` — **OK**.
- `scripts/speed-gate.sh --fast` — **2 metrics passed**.
