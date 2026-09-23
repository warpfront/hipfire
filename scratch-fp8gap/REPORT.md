# fp8v2 vs iu4 prefill gap — per-kernel attribution (Card-B, pp8192)

- Date: 2026-09-21. Worktree `/home/kaden/ClaudeCode/warpfront/wt-fp8sym` @ `555c477cb` (binary as-built, no edits).
- Model: `qwen3.8-27b.mq4-xt` (symmetric V2.5 XT), `--kv-mode fp8`, `HIPFIRE_GRAPH=1`.
- Card-B env: `HOME=/home/kaden/.hipfire-homes/ab1 ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397
  HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
  HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models HIPFIRE_GRAPH=1
  HIPFIRE_DAEMON_BIN=<wt>/target/release/daemon`; fp8v2 arm adds `HIPFIRE_IU4_PREFILL=0`.

## Method (in-daemon rocprof)

Wrapping `hipfire bench ... --matrix` directly in `rocprofv3 --kernel-trace` yields
empty output: the CLI `SIGKILL`s its daemon child on `Engine` drop
(`crates/hipfire-client/src/lib.rs:1196-1205`), so the profiler in the daemon
never finalizes/flushes (verified: daemon tool-init present, no finalization,
zero-byte out-dir). Instead:

1. Captured the exact daemon protocol by running the bench once per arm with
   `HIPFIRE_DAEMON_BIN=/tmp/fp8gap-daemon-tee.sh` (stdin tee + exec real daemon).
   Both arms send the identical 9-line sequence:
   `configure, ping, diag, load, diag, bench_prefill(8192), bench_prefill(8192),
   bench_decode(128,1), bench_decode(128,1)` → see `stdin-{iu4,fp8v2}.jsonl`.
2. Replayed the prefill-only prefix (first 7 lines) straight into the daemon under
   rocprof, `serve_runner.py`-style (daemon exits on stdin EOF → clean flush):
   `rocprofv3 --kernel-trace --stats -f csv -d prof-<arm> -o bench -- bash -c
   'exec <wt>/target/release/daemon < replay-<arm>.jsonl > /dev/null'`
   with the Card-B env above (± `HIPFIRE_IU4_PREFILL=0`).
3. Normalized with the repo's per-kernel recipe: `bench_kernel_stats.csv`
   `TotalDurationNs` per kernel ÷ tokens.

**Dispatch sets: 2 complete pp8192 passes per arm** (bench issues `bench_prefill`
twice; confirmed in-trace: `attention_…_packet` 32 calls = 16 FA layers × 2,
`gemv_…_multirow_r2` 2 calls, `embedding_q8_batched` 2 calls). Normalizer =
2 × 8192 = **16384 tokens**. Decode excluded from the trace by construction
(replay contains no `bench_decode`).

## Wall vs kernel sum (us/token)

| arm | bench wall tok/s | wall us/token (10⁶/tok_s) | kernel-sum us/token | gap (wall − sum) |
|---|---|---|---|---|
| iu4 (default) | 3661.1 (`bench-iu4.json`) | 273.14 | 274.69 (`prof-iu4/bench_kernel_stats.csv` ÷ 16384) | −1.55 (−0.6%) |
| fp8v2 (`HIPFIRE_IU4_PREFILL=0`) | 2042.0 (`bench-fp8v2.json`) | 489.72 | 493.38 (`prof-fp8v2/bench_kernel_stats.csv` ÷ 16384) | −3.66 (−0.7%) |

Both totals reconcile to within 1% of the bench wall (acceptance: 5%).
The gap is ~zero/negative — there is **no host/launch gap**: both arms keep the
device ~fully busy (sums slightly exceed wall via concurrent graph work, e.g.
`fillBuffer` overlapping compute). The entire 216.6 us/token wall delta
(489.72 − 273.14) is device-kernel time: kernel-sum delta = **218.69 us/token**.

Group subtotals (us/token, fp8v2 − iu4):

| group | iu4 | fp8v2 | delta |
|---|---|---|---|
| GEMM | 197.91 | 336.65 | **+138.74** |
| GDN (deltanet path) | 16.70 | 69.02 | **+52.32** |
| producers / norm / quant | 33.79 | 62.25 | **+28.46** |
| attention (FA2 packet, same kernel both arms) | 24.34 | 23.61 | −0.73 |
| other (rope, final norm, kv-write, gemv, emb, copies) | 1.95 | 1.90 | −0.05 |

Non-GEMM delta = 218.69 − 138.74 = **+79.95 ≈ the ~80 us**.

## Two-column table (us/token, sorted by |delta|)

| kernel | iu4 | fp8v2 | delta (fp8v2−iu4) | calls iu4/fp8v2 |
|---|---:|---:|---:|:---:|
| gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold | 0.00 | 161.83 | +161.83 | 0/128 |
| gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold | 134.21 | 0.00 | −134.21 | 736/0 |
| gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold | 0.00 | 102.93 | +102.93 | 0/256 |
| gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold | 63.70 | 0.00 | −63.70 | 256/0 |
| gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold | 0.00 | 55.47 | +55.47 | 0/96 |
| gated_delta_net_q8_fast | 0.00 | 53.37 | +53.37 | 0/1536 |
| fused_silu_mul_mq_rotate_awq | 0.00 | 22.07 | +22.07 | 0/128 |
| fused_silu_mul_mq_rotate_awq_i4_gfx12 | 16.54 | 0.00 | −16.54 | 128/0 |
| gemm_qkv_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold | 0.00 | 16.42 | +16.42 | 0/32 |
| gdn_pre_batched_gfx1201 | 0.00 | 15.65 | +15.65 | 0/96 |
| pack_f32_to_fp8_mq4v2_gfx12 | 0.00 | 14.30 | +14.30 | 0/256 |
| fused_rmsnorm_mq_rotate_awq_mq4v2_fp8_gfx12 | 0.00 | 10.61 | +10.61 | 0/256 |
| gdn_chunk_scan | 9.30 | 0.00 | −9.30 | 1536/0 |
| fused_rmsnorm_mq_rotate_awq_i4_gfx12 | 7.27 | 0.00 | −7.27 | 256/0 |
| gated_norm_mq_rotate_awq_i4_gfx12 | 5.81 | 0.00 | −5.81 | 96/0 |
| gated_norm_f32 | 0.00 | 5.76 | +5.76 | 0/96 |
| gdn_chunk_prep | 5.28 | 0.00 | −5.28 | 96/0 |
| rotate_x_mq_awq | 0.00 | 4.90 | +4.90 | 0/128 |
| gdn_chunk_kkt_solve | 2.12 | 0.00 | −2.12 | 1536/0 |
| sigmoid_mul_f32 | 0.00 | 1.90 | +1.90 | 0/32 |
| sigmoid_mul_rotate_x_mq_awq_i4_gfx12 | 1.44 | 0.00 | −1.44 | 32/0 |
| attention_fp8_e4m3_fa2_gqa_packet_gfx1201 | 24.34 | 23.61 | −0.73 | 32/32 |
| kv_cache_write_fp8_e4m3_batched | 0.37 | 0.30 | −0.08 | 64/64 |
| deinterleave_q_rmsnorm_f32_batched | 2.73 | 2.71 | −0.02 | 32/32 |
| rmsnorm_f32 | 0.44 | 0.43 | −0.01 | 34/34 |
| embedding_q8_batched | 0.03 | 0.04 | +0.01 | 2/2 |
| __amd_rocclr_fillBufferUnAligned | 0.22 | 0.22 | −0.01 | 1824/1824 |
| rope_partial_halfsplit_batched_f32 | 0.72 | 0.71 | −0.00 | 32/32 |
| __amd_rocclr_copyBuffer | 0.03 | 0.03 | −0.00 | 180/180 |
| gemv_mq4g256v2_multirow_r2 | 0.13 | 0.13 | −0.00 | 2/2 |
| mq_rotate_x | 0.00 | 0.00 | −0.00 | 2/2 |

## Top-5 deltas (by |delta|, kernel names)

1. `gemm_gate_up_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` **+161.83**
   (fp8v2-only; the iu4 counterpart row is #2).
2. `gemm_mq4g256v2_residual_mmq_iu4_full_set_symfold` **−134.21** (iu4-only).
3. `gemm_mq4g256v2_residual_wmma_fp8_v2_b128x128_gfx1201_symfold` **+102.93**.
4. `gemm_mq4g256v2_residual_mmq_iu4_full_add_symfold` **−63.70** (iu4-only).
5. `gemm_qkvza_mq4g256v2_wmma_fp8_v2_b128x128_gfx1201_symfold` **+55.47**.
6. (first non-GEMM) `gated_delta_net_q8_fast` **+53.37** — the head of the ~80 us.

## Attribution of the ~80 us non-GEMM gap

It is **not** a launch/host gap (wall−sum ≈ 0 on both arms) and **not** attention
(same FA2 packet kernel, −0.7). It is ~52 us of a **slower serial DeltaNet path**
plus ~28 us of **extra fp8-quant producer work**. The iu4 arm runs the chunked
GDN path (`gdn_chunk_scan` 9.30 + `gdn_chunk_prep` 5.28 + `gdn_chunk_kkt_solve`
2.12 = 16.70), while fp8v2 runs `gated_delta_net_q8_fast` (53.37, 1536 calls =
48 linear layers × 16 commits × 2 passes) **plus** a separate
`gdn_pre_batched_gfx1201` stage (15.65) that has no iu4 counterpart — together
69.02, i.e. +52.32 of the gap sits inside the GDN recurrence, not in GEMM.
The remaining +28.46 is the fp8 route materializing per-token FP8 in flight:
`pack_f32_to_fp8_mq4v2_gfx12` (+14.30, 256 calls feeding every fp8 GEMM) plus a
split SiLU/rotate/sigmoid trio (`fused_silu_mul_mq_rotate_awq` 22.07 +
`rotate_x_mq_awq` 4.90 + `sigmoid_mul_f32` 1.90 = 28.87) replacing iu4's fused
int4-emitting variants (`…_i4_gfx12` silu 16.54 + `sigmoid_mul_rotate_…_i4`
1.44 = 17.98), and the fp8-flavored rmsnorm (+10.61 vs −7.27). Net: the fp8v2
arm pays ~80 us/token because its DeltaNet recurrence is a serial q8 kernel
with an extra pre-stage, and its producers quantize f32→fp8 out-of-band instead
of emitting quantized tiles fused like the iu4 path.
