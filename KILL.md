# gfx1201 separate FP8 gate/up intermediate — KILLED

## Verdict

Do not enable `HIPFIRE_GFX12_FP8_GATEUP_OUT` by default. The implementation is numerically sound and satisfies the resource ceiling, but it fails the load-bearing daemon performance gates on the assigned gfx1201 card. The flag remains default-off.

The candidate replaces only the terminal f32 stores of the two IU4 gate/up GEMMs with separate unscaled OCP E4M3FN byte planes. The following incumbent AWQ/FWHT/MSE-int4 producer decodes those planes and otherwise retains the same arithmetic and int4 sidecar layout.

A fused-product variant was not attempted: the separate merged gate/up launch remained on another experimental branch after its own pp8192 regression and was not available in the base used here.

## Fixed fixture

- Base: `5729b526f`
- Branch: `gfx1201-fp8-gateup`
- GPU UUID: `GPU-05f92432f2312a0e`
- Home: `/home/kaden/.hipfire-homes/ab4`
- Model: `qwen3.8:27b-mq4-xt`
- Graph mode for daemon gates: enabled
- Matrix protocol: `--matrix --pp 512,8192 --ctx 128 --tg 128 --spec off --runs 3 --warmups 1 --kv-mode fp8 --json`
- TTFT protocol: `--ttft --prompt-file benchmarks/prompts/ttft_5900.txt --runs 8 --warmups 2 --json`

## Correctness and resources

PASS:

- GEMM FP8 epilogue oracle: every device byte matched host conversion of the incumbent f32 GEMM output with OCP E4M3FN round-to-nearest-even.
- Producer oracle, no AWQ: candidate int4 sidecar matched the incumbent producer byte-for-byte on the same FP8-decoded inputs.
- Producer oracle, AWQ: candidate int4 sidecar matched byte-for-byte.
- Existing IU4 GEMM CPU oracle and repeat identity also passed in the exercised edge case.
- GEMM candidate resource usage: 200 VGPR, 40 SGPR, zero scratch, zero spills (ceiling: 202 VGPR).
- FP8 producer resource usage: 84 VGPR, 18 SGPR, zero scratch, zero spills.

Oracle command:

```text
cargo build --release -p hipfire-runtime --example tmp_iu4_gfx12_oracle --features lab
./target/release/examples/tmp_iu4_gfx12_oracle /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt --fp8-only
```

Result: `GFX12-FP8-GATEUP ORACLE PASS`.

## KLD gates

Both required c24 gates passed:

| Route | Candidate flag | Extra route setting | Slice-mean KLD | Limit | Verdict |
|---|---:|---|---:|---:|---|
| IU4 | 1 | default IU4 | 0.081676 | 0.10 | PASS |
| FP8v2 control | 1 | `HIPFIRE_IU4_PREFILL=0` | 0.049236 | 0.05 | PASS |

The FP8v2 control does not admit this IU4-only candidate, as intended; it verifies the required alternate route under the same build.

## Paired daemon gates

### Matrix

Higher is better. The pp8192 acceptance threshold was at least +1.5% in both pairs.

| Pair | Metric | Baseline | Candidate | Delta | Verdict |
|---:|---|---:|---:|---:|---|
| 1 | pp512 tok/s | 2909.4 | 2870.6 | -1.334% | — |
| 1 | pp8192 tok/s | 2965.9 | 2938.6 | -0.920% | FAIL |
| 1 | tg128@128 tok/s | 36.4501 | 36.4718 | +0.059% | PASS |
| 2 | pp512 tok/s | 2897.5 | 2883.5 | -0.483% | — |
| 2 | pp8192 tok/s | 2970.0 | 2946.9 | -0.778% | FAIL |
| 2 | tg128@128 tok/s | 36.1745 | 36.4693 | +0.815% | PASS |

### Client TTFT, 5,909 prompt tokens

Lower is better. The acceptance threshold allowed at most 1% regression.

| Pair | Baseline median | Candidate median | Delta | Verdict |
|---:|---:|---:|---:|---|
| 1 | 2039.170 ms | 2065.106 ms | +1.272% | FAIL |
| 2 | 2035.466 ms | 2055.041 ms | +0.962% | PASS |

Raw JSON is under `scratch-fp8-gateup/{matrix,ttft}-{base,candidate}-{1,2}.json` in this worktree.

## In-daemon rocprof attribution

`rocprofv3 --kernel-trace --stats` wrapped the daemon itself for matching warmed pp8192 runs. The CSV traces are under `scratch-fp8-gateup/rocprof-{base,candidate}/`.

| Shape / kernel | Baseline avg | Candidate avg | Delta | Gate |
|---|---:|---:|---:|---|
| N=8192 gate/up IU4 SET GEMM | 7.117897 ms | 7.462691 ms | +4.844% | FAIL (must be no worse than +1%) |
| N=128 gate/up IU4 SET GEMM | 139.634 us | 141.786 us | +1.541% | FAIL |
| N=8192 AWQ SwiGLU/FWHT/int4 producer | 2.575301 ms | 2.553901 ms | -0.831% | PASS, but too small to offset GEMM |
| N=128 AWQ SwiGLU/FWHT/int4 producer | 42.893 us | 42.190 us | -1.638% | PASS |

The f32 gate/up GEMM used grid `(34816,64)` for N=8192 and `(34816,1)` for N=128; the candidate FP8 symbol used the same grids and call counts. At N=8192 each arm recorded 256 gate/up GEMM calls and 128 producer calls, so the sums are directly paired.

The root cause is therefore localized: reducing gate/up output traffic saves less than 1% in the producer, while native FP8 packing makes each gate/up GEMM about 4.8% slower at the load-bearing shape. End-to-end pp8192 regresses in both pairs instead of reaching the required gain.

## Final action

- Keep `HIPFIRE_GFX12_FP8_GATEUP_OUT` default-off.
- Do not run the 5/5 serve battery: that gate is required only for a candidate selected to ship, and this candidate is already rejected by both pp8192 pairs, one TTFT pair, and the GEMM kernel ceiling.
