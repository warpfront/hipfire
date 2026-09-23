# ATiledProducers — KILL

## Verdict

Kill the producer-emitted `At + Ds` activation layout and direct-global-A
gfx1201 IU4 GEMM.

The previously missing decisive measurement is now complete on Card-C, in the
real daemon, after rebasing the implementation onto `5729b526f`. The candidate
is oracle-exact, uses the upstream four-candidate producer quantizer, reduces
GEMM LDS from 20,480 to 12,288 bytes, and reduces the traced GEMM allocation
from 208 to 200 VGPR. None of that raises occupancy: both kernels admit three
blocks/CU. At pp8192 the candidate loses both required pairs by 5.201% and
4.807%, versus a required gain of at least 1.5%.

In-daemon rocprof attributes the loss to the candidate GEMM, not launch count:
the pp8192 `set` sum grows 5.767% and `add` grows 10.846%. The candidate
producer sums also grow 2.545% in aggregate.

The old isolated KX3 result used an M=34816 gate projection and was not
decisive because the live daemon splits that work into two M=17408 GEMMs. This
report supersedes that synthetic kill with the real daemon result.

No product commit was made.

## Pin

- Branch: `gfx1201-atiled-iu4`
- Base/HEAD: `5729b526f50f114955df3303a0e994ce882283fc`
- Device: Card-C, `GPU-085289909a86cc63`, gfx1201 Radeon AI PRO R9700
- Home: `/home/kaden/.hipfire-homes/ab2`
- Model: `qwen3.8:27b-mq4-xt`
- Fixed runtime: `HIPFIRE_GRAPH=1`, speculation off, FP8 KV
- Arm switch: `kernel.gfx12_iu4_atiled` through
  `HIPFIRE_GFX12_IU4_ATILED={0,1}`
- Normal gate daemon twins:
  `daemon-gates/daemon-{off,on}`, identical md5
  `7a1d83d1bb27ec0b83aa484868a834f3`

The only difference between normal OFF and ON processes was the resolved
candidate flag. Every arm was a fresh daemon. OFF and ON were warmed before
the measured pairs.

## Base merge and producer contract

The uncommitted oracle-exact implementation was carried forward to
`5729b526f`. The merge kept the upstream gfx1201 producer quantizer's
four-candidate subset `{0,2,5,7}` and made both row-major and `At + Ds`
production emitters instantiate that same four-candidate recipe. Standalone
oracle/layout helpers retain their non-production recipe where required.

The real A-tiled rotate producer was compared byte-for-byte with the real
row-major gfx1201 rotate producer:

```text
PRODUCER BYTES PASS rotate K=5120 N=512 bytes=1474560
```

Thus the production `At + Ds` path is on the upstream four-candidate producers,
and its emitted block values are byte-identical to the corresponding upstream
row-major producer.

## Oracle exactness

Command:

```text
cargo build --release -p hipfire-runtime \
  --example tmp_iu4_gfx12_oracle --features lab
./target/release/examples/tmp_iu4_gfx12_oracle \
  /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt
```

The oracle passed every shipped-vs-candidate bit-identity case:

- gate set/add
- down set/add
- M=48 tail
- N=80 partial tile
- K=256 set/add
- M=N=100
- repeated launch identity
- NT twin identity
- pinned CPU fold DAG identity

Final result:

```text
GFX12-IU4 ORACLE PASS
```

## Required daemon matrix

Core command for every row:

```text
hipfire bench qwen3.8:27b-mq4-xt --matrix \
  --pp 512,8192 --ctx 128 --tg 128 --spec off \
  --runs 3 --warmups 1 --kv-mode fp8 --json
```

`Speedup` is `ON/OFF - 1` for throughput.

| Pair | Metric | OFF | ON | Speedup |
|---|---|---:|---:|---:|
| 1 | pp512 tok/s | 2901.5 | 2708.4 | -6.655% |
| 1 | pp8192 tok/s | 2949.6 | 2796.2 | **-5.201%** |
| 1 | decode tok/s | 36.3674 | 36.3654 | -0.005% |
| 2 | pp512 tok/s | 2869.8 | 2702.0 | -5.847% |
| 2 | pp8192 tok/s | 2927.0 | 2786.3 | **-4.807%** |
| 2 | decode tok/s | 36.2385 | 34.3629 | -5.176% |

The second decode pair entered a visibly slower thermal/clock state. Decode
does not admit the candidate at batch 1, so it is a control measurement, not
candidate kernel work. The pp8192 result is stable across both pairs and
decisively fails the required `>= +1.5%` gate.

Raw matrix samples:

```text
pair1 OFF pp512 [2879.8,2901.5,2906.7]
          pp8192 [2944.6,2955.1,2949.6]
          decode [36.360906,36.368870,36.367399]
pair1 ON  pp512 [2685.9,2719.7,2708.4]
          pp8192 [2793.4,2801.4,2796.2]
          decode [36.364565,36.366583,36.365427]
pair2 OFF pp512 [2860.6,2869.8,2879.2]
          pp8192 [2924.8,2933.2,2927.0]
          decode [36.355933,36.238492,35.319833]
pair2 ON  pp512 [2685.3,2707.6,2702.0]
          pp8192 [2783.1,2790.4,2786.3]
          decode [36.202498,34.362905,34.074879]
```

## Required TTFT

Command:

```text
hipfire bench qwen3.8:27b-mq4-xt --ttft \
  --prompt-file benchmarks/prompts/ttft_5900.txt \
  --runs 8 --warmups 2 --spec off --kv-mode fp8 --json
```

Prompt: 5909 tokens, md5 `ed720348b81a19fab64d4783c75c1ae3`.
`TTFT speedup` is `OFF/ON - 1`, so a positive value is faster and the required
floor is -1%.

| Pair | OFF median TTFT | ON median TTFT | TTFT speedup | OFF pp tok/s | ON pp tok/s |
|---|---:|---:|---:|---:|---:|
| 1 | 2050.160 ms | 2046.537 ms | +0.177% | 2882.215 | 2887.318 |
| 2 | 2043.404 ms | 2079.582 ms | **-1.740%** | 2891.744 | 2841.437 |

The 5909-token shape is not divisible by 128, so the A-tiled admission predicate
keeps the candidate inactive for TTFT. The pair spread is therefore a daemon
control/drift result; pair 2 nevertheless misses the literal -1% floor. The
matrix kill is independent and decisive.

TTFT sample milliseconds:

```text
pair1 OFF [2043.443568,2045.227330,2047.007486,2048.676965,
           2051.643849,2053.667044,2054.738025,2056.065716]
pair1 ON  [2042.039041,2042.652290,2044.768886,2045.395735,
           2047.677882,2048.521530,2051.055839,2052.716713]
pair2 OFF [2037.639417,2039.388587,2040.738210,2042.592429,
           2044.214876,2045.711494,2047.463878,2048.650535]
pair2 ON  [2074.987777,2075.539202,2077.119425,2078.516211,
           2080.648030,2083.044390,2084.861491,2086.977477]
```

## In-daemon rocprof attribution

The daemon was traced for one real pp8192 pass plus the decode control. The
table below selects only pp8192 dispatches by live grid dimensions, excluding
the ctx128 control. Calls are identical between corresponding arms.

| Kernel | Calls | OFF total | ON total | Growth |
|---|---:|---:|---:|---:|
| IU4 GEMM set | 736 | 2675.175 ms | 2829.448 ms | **+5.767%** |
| IU4 GEMM add | 256 | 1271.868 ms | 1409.817 ms | **+10.846%** |
| RMS/rotate producer | 256 | 217.292 ms | 228.278 ms | +5.056% |
| gated-norm producer | 96 | 165.301 ms | 168.224 ms | +1.768% |
| SiLU/rotate producer | 128 | 331.618 ms | 335.893 ms | +1.289% |
| sigmoid/rotate producer | 32 | 26.520 ms | 27.188 ms | +2.518% |

Category sums:

| Category | OFF | ON | Growth | Added time |
|---|---:|---:|---:|---:|
| IU4 GEMMs | 3947.043 ms | 4239.265 ms | **+7.404%** | +292.223 ms |
| four producers | 740.732 ms | 759.582 ms | +2.545% | +18.851 ms |
| combined | 4687.774 ms | 4998.848 ms | **+6.636%** | +311.073 ms |

The kernel that grows most in relative terms is
`gemm_mq4g256v2_residual_mmq_iu4_atiled_add` (+10.846%). The largest absolute
growth is `..._atiled_set` (+154.273 ms); `..._atiled_add` adds 137.949 ms.
Together those two GEMMs explain 94% of the measured combined growth.

Correct trace artifacts:

- `daemon-gates/trace-correct-off/profile_kernel_stats.csv`
- `daemon-gates/trace-correct-off/profile_kernel_trace.csv`
- `daemon-gates/trace-correct-on/profile_kernel_stats.csv`
- `daemon-gates/trace-correct-on/profile_kernel_trace.csv`

The profiling-only daemon twins are byte-identical (md5
`fe4051d12f9c14183a342b81ca54e959`) and differ from the normal twins only by a
temporary `shutdown` command used to let rocprof drain. The profiling client
used the current config schema plus the matching graceful shutdown hook. Both
temporary source hooks were removed after collection.

## In-situ resources and occupancy

rocprof's live dispatch metadata on Card-C:

| Kernel family | LDS | VGPR | Max active blocks/CU |
|---|---:|---:|---:|
| incumbent IU4 full set/add | 20,480 B | 208 | 3 |
| candidate IU4 A-tiled set/add | 12,288 B | 200 | 3 |

The occupancy values were queried against the live compiled functions with
256-thread blocks on Card-C. The LDS reduction does not cross an occupancy
step.

Producer VGPR allocations from the same daemon trace:

| Producer | OFF | A-tiled |
|---|---:|---:|
| RMS/rotate | 96 | 104 |
| gated-norm | 48 | 48 |
| SiLU/rotate | 56 | 56 |
| sigmoid/rotate | 88 | 88 |

The candidate saves eight GEMM VGPRs but adds eight to RMS/rotate, still with no
GEMM occupancy gain.

## Final disposition

- Correctness: PASS
- Four-candidate producer byte identity: PASS
- pp8192 daemon gate: **FAIL**
- TTFT literal two-pair floor: **FAIL** (candidate inactive at N=5909)
- Kernel attribution: candidate GEMM sums grow 7.404%
- Occupancy rationale: 12 KiB LDS still admits only three blocks/CU
- Action: KILL; do not commit or promote
