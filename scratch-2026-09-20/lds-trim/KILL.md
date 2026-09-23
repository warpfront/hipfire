# KILL: gfx1201 IU4 LDS 20 KiB -> 16 KiB

Branch: `gfx1201-iu4-lds-trim` at base `390d8228ff10f18ec7350bb3eaa7bf577d4148cf`

Verdict: **KILL / do not land.** The candidate is bit-identical and removes 4 KiB of LDS, but it remains at 3 blocks/CU and regresses pp8192 by about 12.09% in both paired measurements. TTFT latency regresses by 13.63-13.74%. No commit was made.

## Change

Removed the double-buffered Xq `(d,s)` and packed weight `(scale,zp)` metadata planes from LDS. The fold reads immutable metadata directly from global memory while retaining the existing conversions and `IU4_FOLD_RN` arithmetic DAG. A/W staging remains double-buffered.

Host launch LDS, oracle default/OCC query, and the timing rig were changed from 20480 to 16384 bytes.

### LDS map before: 20480 bytes

| Plane | Byte range | Bytes |
|---|---:|---:|
| A0 | 0-4095 | 4096 |
| W0 | 4096-8191 | 4096 |
| W1 | 8192-12287 | 4096 |
| DS0 | 12288-13311 | 1024 |
| DS1 | 13312-14335 | 1024 |
| SZ0 | 14336-15359 | 1024 |
| SZ1 | 15360-16383 | 1024 |
| A1 | 16384-20479 | 4096 |

### LDS map after: 16384 bytes

| Plane | Byte range | Bytes |
|---|---:|---:|
| A0 | 0-4095 | 4096 |
| W0 | 4096-8191 | 4096 |
| A1 | 8192-12287 | 4096 |
| W1 | 12288-16383 | 4096 |

Store-transpose slots continue to alias the base of LDS after staging is dead.

## Correctness oracle

Command: `cargo run --release -p hipfire-runtime --example tmp_iu4_gfx12_oracle --features lab -- /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt`

Result: `GFX12-IU4 ORACLE PASS`. All nine cases passed with `cpu_bitwise=OK (mism=0)` and `repeat_identical=OK (mism=0)`:

- gate set/add
- down set/add
- m48tail
- n80cols
- k256 set/add
- m100n100

## Resources and occupancy

Radiowave report: `/home/kaden/.hipfire-homes/ab1/.hipfire_kernels/gfx1201/tmp_iu4_gfx12_oracle.263edd255dfc920f.radiowave.json`

| Kernel | Baseline registers | Candidate registers | Candidate spills/scratch |
|---|---:|---:|---:|
| full_set | 202 VGPR / 40 SGPR | 219 VGPR / 56 SGPR | 0 / 0 B |
| full_add | 202 VGPR / 40 SGPR | 219 VGPR / 56 SGPR | 0 / 0 B |

`hipOccupancyMaxActiveBlocksPerMultiprocessor`, block 256, dynamic LDS 16384:

- `full_set`: 3 blocks/CU
- `full_add`: 3 blocks/CU

Therefore reducing LDS alone does not expose a fourth resident block; the resulting register allocation is the limiter. Two alternate metadata forms were also compiled: repeated per-pair loads reached 213 VGPR but still reported 3 blocks/CU, while shuffle-distributed headers rose to 242 VGPR and 2 blocks/CU. Neither justified serving measurement.

## Paired Card-B serving measurements

Card: `GPU-e475645fe0200397`. Each arm was warmed first; warm results were discarded. Matrix command used `--matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`. Candidate and baseline daemon binaries were pinned explicitly per arm.

| Pair | Arm | pp512 tok/s | pp8192 tok/s | tg128@128 tok/s |
|---:|---|---:|---:|---:|
| 1 | baseline | 2877.1 | 2931.0 | 36.47657 |
| 1 | candidate | 2497.9 | 2576.5 | 36.47641 |
| 1 | delta | -13.18% | **-12.09%** | -0.0004% |
| 2 | baseline | 2857.7 | 2915.7 | 36.47774 |
| 2 | candidate | 2490.9 | 2563.3 | 36.46306 |
| 2 | delta | -12.84% | **-12.09%** | -0.0402% |

Gate: pp8192 required at least +1.5%; both pairs fail decisively.

Raw rows: `matrix-base-1.json`, `matrix-cand-1.json`, `matrix-base-2.json`, `matrix-cand-2.json` in this directory.

## Paired TTFT

Command used the 5909-token prompt with `--ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --json`.

| Pair | Baseline median | Candidate median | Latency delta | Throughput-equivalent delta |
|---:|---:|---:|---:|---:|
| 1 | 2082.462 ms | 2368.679 ms | +13.74% | **-12.08%** |
| 2 | 2090.980 ms | 2376.030 ms | +13.63% | **-12.00%** |

Gate: TTFT performance required no worse than -1%; both pairs fail.

Raw rows: `ttft-base-1.json`, `ttft-cand-1.json`, `ttft-base-2.json`, `ttft-cand-2.json` in this directory.

## Conclusion

The metadata planes are not dead weight: replacing their LDS reuse with global loads increases register pressure and repeats metadata traffic in the hot fold. The candidate meets the 16 KiB and bit-identity requirements but misses the actual residency target and severely regresses prefill, so it must not be committed or merged.
