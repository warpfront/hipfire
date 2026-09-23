# gfx1201 fp8v2 fused producers

- Date: 2026-09-21
- Base: `mq4-lloyd` / `cd9b75b27c9e9be0dfb4c80a7fe6d4cab62c80e8`
- Branch: `gfx1201-fp8-producers` / `6df5c84c5742eaa49721ff9bedcb50caa13ad574`
- Device: Card B, gfx1201, HIP 7.15
- Model: `qwen3.8:27b-mq4-xt`, `--kv-mode fp8`, `HIPFIRE_IU4_PREFILL=0`, `HIPFIRE_GRAPH=1` unless noted

## Implementation

The residual fp8v2 projections now consume producer-emitted FP8 E4M3 bytes, per-token power-of-two scales, and half-row sums directly:

- SwiGLU + optional AWQ + FWHT + FP8 pack (`fused_silu_mul_mq_rotate*_mq4v2_fp8_gfx12`)
- Gated RMSNorm + optional AWQ + FWHT + FP8 pack (`gated_norm_mq_rotate*_mq4v2_fp8_gfx12`)
- Sigmoid gate + optional AWQ + FWHT + FP8 pack (`sigmoid_mul_rotate_x_mq*_mq4v2_fp8_gfx12`)
- A plain rotate(+AWQ)+FP8 variant is provided by the same rotate kernel/launcher.

The existing gfx1201 RMSNorm + rotate + FP8 producer was already present at the base commit and remains the producer for QKVZA/QKV/gate-up inputs. The new prepared uniform-MQ4v2 residual launcher preserves the incumbent FP8 GEMM geometry and only skips its standalone activation pack.

Commits (one producer per commit, followed by the uniform consumer cutover):

1. `8f2352265` — `gfx1201: fuse SwiGLU producer into fp8 stream`
2. `738e7b20e` — `gfx1201: fuse gated norm producer into fp8 stream`
3. `7408b24e6` — `gfx1201: fuse sigmoid rotate producer into fp8 stream`
4. `6df5c84c5` — `gfx1201: consume fused fp8 producers on uniform mq4v2`

## Exactness

Eval was built with `cargo build --release -p hipfire-runtime --example eval_hipfire --features deltanet` and run with graph/normalization disabled, fp8/q8 KV, prefill scoring, and `--max-chunks 2`.

| output | MD5 |
|---|---|
| base (`c2-base.bin`) | `792542a0a9cecd3610b4742ba3f43480` |
| fused (`c2-new.bin`) | `792542a0a9cecd3610b4742ba3f43480` |

The c2 artifacts are byte-identical.

## Producer profile

Method matches `wt-fp8sym/scratch-fp8gap/REPORT.md`: replay the captured prefill-only daemon protocol under `rocprofv3 --kernel-trace --stats`, then divide each kernel's `TotalDurationNs` by 16,384 tokens (two complete pp8192 passes). Raw data is in `prof-base/bench_kernel_stats.csv` and `prof-fused/bench_kernel_stats.csv`.

For the old split paths, the pack trace has two non-overlapping duration clusters: 128 FFN-down calls and 128 attention-output calls. The latter is split 96:32 between GDN and sigmoid, matching the producer/rotate call counts. Thus each row below compares the complete producer chain that feeds one projection.

| producer chain | base us/token | fused us/token | delta us/token | delta |
|---|---:|---:|---:|---:|
| SwiGLU + AWQ/FWHT + pack | 32.432 | 25.160 | **-7.272** | -22.42% |
| gated RMSNorm + AWQ/FWHT + pack | 12.496 | 6.945 | **-5.551** | -44.42% |
| sigmoid + AWQ/FWHT + pack | 4.114 | 2.157 | **-1.956** | -47.56% |
| subtotal, newly fused residual producers | 49.041 | 34.262 | **-14.780** | -30.14% |
| pre-existing RMSNorm fused producer | 10.624 | 10.963 | +0.339 | profile variation |
| standalone `pack_f32_to_fp8_mq4v2_gfx12` | 14.331 (256 calls) | **0.000 (0 calls)** | **-14.331** | eliminated |

Total traced kernel time moved from 443.589 to 432.496 us/token (-11.092 us/token); unrelated GEMM timing variation accounts for the difference from the producer-only subtotal.

## Paired daemon matrix

Command shape: `hipfire bench qwen3.8:27b-mq4-xt --matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`.

| pair | arm | pp512 tok/s | pp8192 tok/s | tg128@128 tok/s |
|---|---|---:|---:|---:|
| 1 | base | 2101.2 | 2267.2 | 36.5586 |
| 1 | fused | 2233.4 | 2314.8 | 36.4919 |
| 1 | delta | **+6.292%** | **+2.100%** | -0.183% |
| 2 | base | 2066.6 | 2231.6 | 36.4389 |
| 2 | fused | 2211.2 | 2295.1 | 36.4392 |
| 2 | delta | **+6.997%** | **+2.845%** | +0.001% |

The arithmetic mean of paired pp8192 deltas is **+2.47%**. Both pairs exceed the +1.5% target; decode remains within 1%.

## TTFT, 5909 tokens

Command shape: `hipfire bench ... --ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --kv-mode fp8 --json`.

| pair | arm | median TTFT ms | pp tok/s |
|---|---|---:|---:|
| 1 | base | 2692.478 | 2194.636 |
| 1 | fused | 2618.260 | 2256.844 |
| 1 | delta | **-2.757%** | **+2.835%** |
| 2 | base | 2702.268 | 2186.683 |
| 2 | fused | 2622.621 | 2253.091 |
| 2 | delta | **-2.947%** | **+3.037%** |

Mean paired TTFT improvement is **2.85% lower latency**.

## Guardrails

- Flag-free IU4 TTFT, one measured run after one warmup: base 1672.544 ms / 3532.942 tok/s; fused build 1673.526 ms / 3530.868 tok/s (**+0.059% latency, -0.059% throughput**).
- `python3 scripts/serve_harness.py --mode battery --model qwen3.8:27b-mq4-xt --thinking off`: **5/5 turns completed**, runaway=0, empty=0, attractor=0, retrieval_miss=0.
- Release eval and daemon builds completed successfully.
