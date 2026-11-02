# Dense Qwen3.8-27B A4 error correction: concentration study (C1 kill gate)

> **Lifecycle: historical dark research record (2026-10-03); verdict KILL, not a shipped path.**
> The selective-A8 NPU correction route (C1) was killed by this study and never implemented in
> serving. All figures are measured on the dated capture below (one model, one prompt, one
> GPU) or are labelled estimates, and are not current performance or quality claims. This is
> unrelated to the 0.4.2 Flash-Next NPU trunk (BF16-direct only, calibration schema 4, input
> projections only; out-projection unwired); see [`npu-trunk.md`](npu-trunk.md) for the current route.

Dated 2026-10-03 (workspace branch `npu/a4-ec`, not in this repository). This is the §3 stage-1 experiment of `docs/npu/npu-uses-brainstorm.md` (C1/C2) on **real dense-27B activations**.

Labels: **M** = measured on the capture; **E** = estimate (model arithmetic on measured constants).

## 1. Capture (M)

| item | value |
|---|---|
| model | `qwen3.8-27b.mq4-xts`, sha256 `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae` (re-verified), local copy not in this repository |
| prompt | `tools/npu/a4-ec/prompt-wt2.txt` (WT2 excerpt), md5 `25508dd28676158c804a4881dc846a51`; 5416 tokens, first T = 4096 used, `tokens.u32` md5 `7ed43f55681553053f2a263858f127be` |
| device | local card E (device UUID elided), **gfx1201** (R9700), hipfire 0.4.1 beta snapshot f7960b4e8 (read-only), producers built with `-DIU4_A4_CANDIDATES=2` |
| layers | GDN 2, 21, 42, 61; FA 11, 31, 51, 63 |
| tags | `attn` (GDN wqkv/wz/w_beta/w_alpha; FA wq/wk/wv), `o` (wo), `mlp` (w_gate/w_up), `down` (w_down) |
| files | per (L, tag): `xrot.f32` = exact pre-quant f32 `x'` (`0.0625·s2⊙FWHT256(s1⊙x/awq)`) and `a4.bin` = device `block_i4_128` bytes (token-order, `HIPFIRE_A4_SLAB=0`); plus signs, tokens, `manifest.json` with every file's md5 |
| location | workspace-local `a4-ec/capture/` (4.8 GB, not in git); manifest md5 `becd31362ad3985d2f460c6e072c1488` |

The capture comes from the out-of-tree probe `tools/npu/a4-ec/probe` (path deps on the 0.4.1 beta snapshot, no hipfire edits, no HIP written). Its binary md5 is `1a91ae6fcf077783c0786466f907bef5`; the run took 34.7 s under the host's per-GPU lock wrapper (card E) and a host-memory lock.

hipfire has no GEMM-input or Xq dump hook, so the probe:
1. runs `forward_prefill_batch_with_pbs_opts(max_layer = L)` and `(L+1)` as one T = 4096 chunk;
2. calls the gfx1201 production producers on layer L's persisted buffers (`fused_rmsnorm_rotate_mq_i4_gfx12_batched`, `gated_norm_rotate_mq_i4_gfx12_batched`, `sigmoid_mul_rotate_x_mq_awq_i4_gfx12_batched`, F1-lite `fused_silu_hin_rotate_mq_i4_batched`).

Down `xrot` comes from `rotate_x_mq_i4_gfx12_batched(h)`. Its A4 bytes are identical to the production producer's in all 8 layers.

Self-check: every value is finite, `s == Σq`, and unsaturated `|x'−dq|/d ≤ 0.5`. A rerun reproduced the md5s for L2 and L11.

Caveats:
- `max_layer` uses the sequential GDN scan instead of chunk-scan, which differs at rounding level.
- `mlp` x_mid was read from a run with layer L's `w_down` bytes zeroed, so the ADD epilogue adds exactly 0.
- The arch is gfx1201, not gfx1151, so producer bytes may differ slightly from Halo. That is fine for an energy study.

## 2. Model (M)

`tools/npu/a4-ec` is a Rust CPU tool (faer + rayon) over the capture and the stored MQ4 weights `W'[n][k] = sc_e·(u−8)` (qt 44).

Definitions:
- `r' = x' − d·q`, taken from the device bytes.
- CPU c2 re-quant parity vs the device is **1.000000** for both codes and d, on every (layer, tag).
- Error `e_t = W'r'_t`; NSR = `Σ‖e‖²/Σ‖W'x'‖²`.
- Static sets are selected on tokens 0–2047 and evaluated on 2048–4095.
- `f` = true recovery `1 − ‖W'(r' with set corrected)‖²/‖W'r'‖²`.

Checks:
- `Σ_b E_b / E` = 0.963–1.007: per-block errors are nearly uncorrelated.
- `W'r' = W'_o r_o` holds to a max relative error of 9.2e-7.
- Randomized SVD at 8 vs 24 iterations differs by at most 1.0e-3.
- 7 unit tests.

Run: 229 s on 64 threads. Binary md5 `c5763fac6c34cdeafe8d4f07eda64598`. Outputs `results.json` (md5 `deb530200ccfdedba717d8fa9cfa1df4`) and `tables.md` (md5 `0ffe1190dae01a8fd8a1140a7a303fb0`). Those two files hold the full per layer × tag × weight tables.

### Concentration: static top-20% blocks, exact correction, held-out (M)

| L | type | wqkv/wq | wz/wk | wv | wo | w_gate | w_up | w_down |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | GDN | 0.291 | 0.298 | — | 0.399 | 0.222 | 0.220 | 0.256 |
| 11 | FA | 0.351 | 0.326 | 0.323 | 0.410 | 0.233 | 0.231 | 0.234 |
| 21 | GDN | 0.268 | 0.275 | — | 0.300 | 0.231 | 0.225 | 0.234 |
| 31 | FA | 0.294 | 0.304 | 0.300 | 0.403 | 0.256 | 0.248 | 0.216 |
| 42 | GDN | 0.289 | 0.295 | — | 0.374 | 0.255 | 0.243 | 0.225 |
| 51 | FA | 0.272 | 0.289 | 0.279 | 0.343 | 0.273 | 0.261 | 0.240 |
| 61 | GDN | 0.262 | 0.274 | — | 0.471 | 0.266 | 0.253 | 0.247 |
| 63 | FA | 0.265 | 0.286 | 0.277 | 0.361 | 0.253 | 0.245 | 0.303 |

For GDN layers, w_beta and w_alpha (N = 48) are 0.30–0.38. A uniform (white) error would give 0.20. **No case of 60 reaches 0.5**; the maximum is 0.471.

### Recovery per method, mean over captured layers (M)

| method | qkv | zba | FA q | FA k/v | o_proj | gate_up | down | wmean (calls·NSR) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static top-20%, exact | 0.278 | 0.286 | 0.296 | 0.297 | 0.383 | 0.245 | 0.244 | **0.293** |
| static top-20%, A8 plane | 0.273 | 0.282 | 0.291 | 0.292 | 0.377 | 0.241 | 0.240 | 0.289 |
| static top-40%, exact | 0.487 | 0.492 | 0.504 | 0.520 | 0.596 | 0.458 | 0.458 | 0.506 |
| dynamic oracle 20%/token, exact | 0.335 | 0.344 | 0.354 | 0.354 | 0.556 | 0.310 | 0.387 | 0.404 |
| dynamic proxy `‖r'_b‖²‖W'_b‖²` 20%, A8 | 0.329 | 0.336 | 0.346 | 0.344 | 0.546 | 0.304 | 0.380 | 0.396 |
| full-K A8 plane (ceiling) | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 |
| outlier channels 20% K, rotated | 0.287 | 0.298 | 0.309 | 0.309 | 0.378 | 0.255 | 0.252 | 0.299 |
| outlier channels 5% K, original domain | 0.122 | 0.140 | 0.155 | 0.186 | 0.155 | 0.085 | 0.102 | 0.125 |
| outlier channels 20% K, original domain | 0.319 | 0.331 | 0.357 | 0.419 | 0.411 | 0.261 | 0.301 | 0.336 |
| low-rank r=16, data-free | 0.043 | 0.085 | 0.109 | 0.096 | 0.054 | 0.036 | 0.029 | 0.049 |
| low-rank r=32, data-free | 0.064 | 0.113 | 0.144 | 0.155 | 0.084 | 0.052 | 0.044 | 0.074 |
| low-rank r=64, data-free | 0.097 | 0.152 | 0.196 | 0.250 | 0.130 | 0.076 | 0.070 | 0.112 |
| low-rank r=64, data-aware (fit on calib half) | 0.072 | 0.127 | 0.172 | 0.222 | 0.104 | 0.055 | 0.043 | 0.087 |

Data-aware low-rank loses to data-free on held-out tokens because `r'` is close to white: the error covariance is ≈ `σ²W'W'ᵀ`, and 2048 samples estimate it worse than the weight SVD does. The A8 second plane keeps about 98% of exact correction.

## 3. NPU cost (E)

Constants:
- int8 GEMM 26.9 TOPS busy × β 0.853 concurrent;
- port 55 GB/s;
- V8 input traffic 2.05 B/kop;
- 0.215 ms submit + 0.02 ms ring;
- M = 8192 per pp8192 step;
- the Y_corr bf16 write is 2 B/output.

The fold of the int32 partials is **not** charged, which is optimistic. The full per family × method table is in `tables.md` §3.

Static selective A8 is **bandwidth-bound by the Y_corr write**. For GDN qkv, top-20% costs 9.7 ms/call and 465 ms/step on the NPU, against 110 ms/step for the GPU-alone selective A8. Dynamic sets force a full-K plane: 37.7 ms/call on qkv, more than 3× the GPU GEMM.

## 4. KLD per NPU budget (E)

Assumptions:
- Gap 0.027 = A4 0.069 − fp8 0.042 (WT2), all attributed to A4 activation error.
- The gap is split across families ∝ calls × mean NSR (first order, uniform sensitivity): o_proj 0.272, down 0.236, gate_up 0.228, qkv 0.107, FA k/v 0.078, zba 0.059, FA q 0.020.
- Greedy allocation by ΔKLD per NPU-ms over all methods. This is not selective-A8-only: it includes full-K A8 on the small FA k/v.

| NPU ms/step | ΔKLD recovered | KLD after |
|---:|---:|---:|
| 100 | 0.0012 | 0.0678 |
| 250 | 0.0031 | 0.0659 |
| 500 | 0.0051 | 0.0639 |
| 1000 | 0.0075 | 0.0615 |
| 2000 | 0.0109 | 0.0582 |

Closing the full gap needs about 17.5 s of NPU time per step, against a 6.8 s step.

## 5. Verdict

**KILL the selective A8 NPU route (C1).**
- f_static(20%) = 0.293, NSR-weighted (0.266 weighted by absolute energy), below the 0.5 kill threshold. Every one of 60 cases is below 0.5, and the dense-specific brainstorm bar was 0.6 on qkv/gate_up, where the measured values are 0.278 and 0.245.
- After FWHT-256 and per-block MSE-clip, the A4 residual is close to white across 128-blocks.

The other forms are not cheap enough either:
- Dynamic per-token selection reaches only 0.40 and needs a full-K plane.
- Outlier channels track the block result (0.30–0.34 at 20%).
- Low-rank captures ≤ 0.11 at rank 64.

The only near-complete correction is a full-K A8 plane (0.984), which is a full second GEMM. That is GPU W4A8 territory (`A8Split128`), not an NPU side job. §4 bounds any NPU quality purchase at ≈ 0.003 KLD per 250 NPU-ms per step (E).

## 6. Reproduce

```
export CARGO_TARGET_DIR=<scratch-target-dir>   # probe
cd tools/npu/a4-ec/probe && cargo build --release --offline
# optional: serialize with other jobs using your site's host-memory / per-GPU lock wrapper
env HIPFIRE_LOCK_DIR=$(mktemp -d) \
  env HIPFIRE_DEVICES=<gpu-uuid> ROCR_VISIBLE_DEVICES=<gpu-uuid> \
      HIPFIRE_A4_SLAB=0 HIPFIRE_PREFILL_MAX_BATCH=4096 \
  $CARGO_TARGET_DIR/release/a4-ec-probe MODEL tools/npu/a4-ec/prompt-wt2.txt OUTDIR --t 4096 \
  --model-sha256 3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae --card E \
  --card-uuid <gpu-uuid> --commit f7960b4e846f5206b03aec6ddc177a82baec6fec
export CARGO_TARGET_DIR=<scratch-target-dir>      # CPU model
cd tools/npu/a4-ec && cargo test --release --offline
cargo run --release --offline -- run --capture OUTDIR --model MODEL --out <results-dir>
```
