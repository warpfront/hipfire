# S6-prefill receipt — route-N stage-b in the production prefill path

Worktree: `/home/kaden/ClaudeCode/warpfront/wt-fa2a`, branch `gfx1201-fa2-fragorder`.
Commit: `7b33e5ef8` (S6-prefill; parent `e7a43c418` = B2; StageB1's uncommitted
B3 kernel work present in tree, untouched by this change).
GPU: ordinal 2 (`HOME=/home/kaden/.hipfire-homes/ab2`, `ROCR_VISIBLE_DEVICES=2`,
`HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models`,
`HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels`).
Binaries (rebuilt post-commit): hipfire `24bc93ee5595809715cefee0b9221592`,
daemon `5440c724db37c9fc890038681fa575ca`,
eval_hipfire `3f57a0413b4006f974246c689f95c9eb`.

## Change (2 files, kernel untouched)

- `crates/hipfire-dispatch/src/families/attention.rs`
  - `AttnFp8E4m3KvBatchedMasked` FA2 arm: with `gfx12_fa2_fp8_enabled()`
    (exact gfx1200/1201 + `HIPFIRE_GFX12_FA2_FP8=1`) the FA2-eligible shape
    calls `attention_fp8_e4m3_fa2_gqa_fp8_gfx1201` (route N: S3 launcher,
    S4 `ensure_fa2_fp8_q_scratch` codes+sq, 32768 LDS); flag off keeps the
    Q0 f16 launcher bit-identically (same predicates, same args).
  - fwht3 arm comment: stale U0 twin note replaced (no stage-b fwht3 twin;
    launcher keeps f16, flag ignored + logged).
- `crates/rdna-compute/src/attention.rs`
  - `attention_q8_0_flash_prefill_wmma` FA2 ingress: flag on + FA2-eligible
    q8 shape fails loud (route Q needs B4; the KMODE=0+FP8 `_fp8_` entry
    still runs the f16 body — selecting it would be a silent wrong-mode run).
  - `attention_q8_0_fa2_gqa_gfx1201`: U0 symbol swap deleted (S5 intent —
    one meaning per symbol); flag on fails loud with the same route-Q text.
    Stale U0 LDS comments fixed (f16 launcher is 65536-only now).
- No partial/merge wiring: the production prefill FA2 path is direct-only
  (ctx ≤ 32768 in one launch); split-KV partial/merge exists only in the
  bench-only `attention_q8_0_fa2_gqa_split_gfx1201_bench` (q8, untouched).

## Gate 1 — flags-off exactness: PASS

`HIPFIRE_GFX12_FA2_FP8` unset, `HIPFIRE_LLOYD_GFX12=1 HIPFIRE_GFX12_FA2_PREFILL=1`,
model `qwen3.8-27b.mq4-xt`, ref `/home/kaden/kldrefs/qwen3.8-27b.ref_wt2.bin`.

- fp8 1-chunk (`g1-fp8-1.log`): slice-mean **0.031193**, NLL 2.202592,
  PPL 9.0484; `.kldseq` md5 **`a8208c16f20b078b6d1e744c45e1dea5`** ✓
- q8 1-chunk (`g1-q8-1.log`): slice-mean **0.029580**, NLL 2.197071,
  PPL 8.9986; `.kldseq` md5 **`e091d74e5f290efb9bc343f107a716b4`** ✓

## Gate 2 — flag-on fp8 route proof + completion: PASS

- Route proof: the run's new JIT entries are exactly the stage-b route-N
  module `attention_fp8_e4m3_fa2_gqa_gfx1201.5ee928a32820a36c`
  (`.hash/.hip/.hsaco/.radiowave.json`); no new Q0 (`_f16_`) module.
  hsaco strings export the bare stage-b body + partial + merge + fp8
  preconvert; the Q0 body symbol is absent. (The cached `.hip` source text
  contains both `#if` branches — it is the shared source file, not the
  compiled TU; the hsaco is the proof.)
- 1-chunk (`g2-fp8-1.log`): completion, finite slice-mean **0.030887**
  (NLL 2.201170, PPL 9.0356), md5 `59d6e83a4eaf289aa3c2b7ba8c322d48`.
  Near-but-not-equal 0.031193, as expected for B2 numerics (NOT gated;
  B3 lands the final arithmetic).
- 2-chunk (`g2-fp8-2.log`): completion, finite slice-mean **0.036857**
  (NLL 2.250811, PPL 9.4954), md5 `7bd0ef53f66e733af8544622290bc988`;
  no further cache entries (module reuse). Q0 2-chunk pin is 0.037398
  (decode receipt) — near-but-not-equal, same B2 expectation.

## Gate 3 — flag-on q8 fail-loud: PASS

`HIPFIRE_GFX12_FA2_FP8=1 --kv-mode q8 --max-chunks 1` exits 101 before
scoring with (`g3-q8-1.log`):

    forward_prefill_batch prefix: HipError { code: 0, message: "HIP error:
    attention AttnQ8_0KvBatchedMasked: HIP error: HipError(0):
    attention_q8_0_flash_prefill_wmma: HIPFIRE_GFX12_FA2_FP8=1 requests
    stage-b route-Q fp8 arithmetic on the q8 cache, which is not implemented
    (B4 pending); unset HIPFIRE_GFX12_FA2_FP8 or use --kv-mode fp8 (route N)
    (hipError=0) (hipError=0)", context: None }
## Bench (flag on, fp8) — daemon path route proof: PASS

Full matrix (`bench-fp8.json`, exit 0; informational until B3):

| row | median tok/s | samples |
|---|---|---|
| pp512 | 1486.3 | 1490.8, 1483.4, 1486.3 |
| pp2048 | 1462.8 | 1462.8, 1463.3, 1462.0 |
| pp8192 | 1372.9 | 1375.0, 1372.8, 1372.9 |
| pp32768 | 1106.4 | 1107.0, 1106.4, 1105.9 |
| tg64@128 (decode) | 36.44 | 36.445, 36.444, 36.439 |

Daemon-path route proof (JIT cache, ordinal-2 home): the full matrix ran
with a warm cache (no new entries — module already built by the eval runs).
Evicted the 4 stage-b files
(`attention_fp8_e4m3_fa2_gqa_gfx1201.5ee928a32820a36c.*`, count 0) and ran a
single-row daemon bench (`bench-fp8-pp512.json`, exit 0, pp512 1493.3 tok/s):
exactly those 4 files were re-created, same hash (deterministic source).
JIT is on-demand per requested symbol (`ensure_kernel`), so the daemon bench
launching the route-N stage-b module is the only way it gets built — the
daemon prefill path takes the route. No Q0 module was (re)built.

## Gate 4 — scoped check: PASS

`cargo check -q -p hipfire-dispatch -p hipfire-arch-qwen35 -p hipfire-runtime
-p hipfire-daemon` → exit 0 (only pre-existing dead-code warnings in
`hipfire-daemon/src/slots.rs`, untouched by this change).
