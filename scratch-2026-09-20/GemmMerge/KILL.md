# KILL: dense gate/up IU4 launch merge

Do not ship or re-propose either variant from this experiment. Both preserve output bits, but neither clears the required pp8192 `>= +1.5%` daemon gate on Card-C (`GPU-085289909a86cc63`).

## Variants

- `c516bdaba`: merge the two dense gate/up launches into one logical `M=34816` launch, direct raster.
- `d32b1dcda`: apply G8T0 rastering only to that merged launch.

The scratch allocation is one contiguous owner with gate/up planar aliases. Existing non-merged launch paths are unchanged.

## Correctness

Both variants pass the GFX12 IU4 oracle, including bit-identical gate and up planes (`mism=0`). Both produce the same exact-base c2 KLD output:

- KLD: `0.063152`
- exact-base / direct / G8T0 c2 md5: `64b41c68a012ea7b9ce1ab9375722a1c`
- c2 files: `a/exact-base-c2.bin`, `a/c2.bin`, `b/c2.bin`

## Direct-raster merged launch (`c516bdaba`)

Paired daemon matrix medians (`runs=3`, `warmups=1`):

| Pair | Base pp8192 | Candidate pp8192 | Delta |
|---|---:|---:|---:|
| 1 | 2945.3 tok/s | 2874.0 tok/s | -2.42% |
| 2 | 2958.4 tok/s | 2874.2 tok/s | -2.85% |

Paired 5,909-token client TTFT medians (`runs=8`, `warmups=2`):

| Pair | Base | Candidate | Latency delta |
|---|---:|---:|---:|
| 1 | 2047.801 ms | 2108.021 ms | +2.94% |
| 2 | 2071.416 ms | 2127.890 ms | +2.73% |

In-daemon rocprof (one warm plus one measured pp8192, with the same decode tail):

- all IU4 kernels: base `4086.715 ms`, candidate `4203.468 ms` (`+2.857%`)
- merged gate/up symbol: `320` calls, `1955.831 ms`
- residual `full_set`: `1200` calls, `909.168 ms`
- all traced kernels: base `5684.303 ms`, candidate `5821.585 ms`
- raw candidate trace: `a/rocprof-direct-graceful/`
- raw base trace: `b/rocprof-base-graceful/`

The direct merged kernel is slower than the two established launches, overwhelming the saved launch overhead.

## Merged launch with G8T0 (`d32b1dcda`)

Paired daemon matrix medians (`runs=3`, `warmups=1`):

| Pair | Base pp8192 | Candidate pp8192 | Delta |
|---|---:|---:|---:|
| 1 | 2932.2 tok/s | 2941.5 tok/s | +0.32% |
| 2 | 2930.0 tok/s | 2958.1 tok/s | +0.96% |

The mean paired uplift is approximately `+0.64%`, below the required `+1.5%`. Decode rows were unstable across fresh daemon processes (one pair favored base heavily, the other favored candidate); the pp8192 gate already rejects the variant.

Paired 5,909-token client TTFT medians (`runs=8`, `warmups=2`):

| Pair | Base | Candidate | Latency delta |
|---|---:|---:|---:|
| 1 | 2042.314 ms | 2023.632 ms | -0.91% |
| 2 | 2033.484 ms | 2023.738 ms | -0.48% |

TTFT is safe, but insufficient to override the pp8192 threshold.

In-daemon rocprof:

- all IU4 kernels: base `4086.715 ms`, candidate `4069.502 ms` (`-0.421%`)
- merged gate/up symbol: `320` calls, `1840.204 ms`
- residual `full_set`: `1200` calls, `904.711 ms`
- all traced kernels: base `5684.303 ms`, candidate `5672.498 ms` (`-0.208%`)
- raw candidate trace: `b/rocprof-candidate-graceful/`
- raw base trace: `b/rocprof-base-graceful/`

G8T0 recovers the direct-raster regression and saves about `17.2 ms` of IU4 kernel time over the traced workload, but the request-level ceiling is too small to clear the shipping gate.
