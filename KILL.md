# Producer load-ahead / block-coarsening — KILL

Base: `mq4-lloyd` `96ec01887`  
Branch: `gfx1201-producer-pipeline`  
Device: Card-C (`GPU-085289909a86cc63`, ab2), gfx1201

## Decision

KILL. The largest producer, `fused_silu_mul_mq_rotate_awq_i4_gfx12`, failed its in-daemon gate in both evaluated two-group implementations. The smaller RMSNorm and gated-norm producers were therefore not changed, per the required SiLU-first gate order. All experimental production changes were reverted; this branch retains only this report.

## Ownership and launch geometry

Baseline launches `grid=[K/256, N]`, `block=[32]`. One wave/workgroup owns one 256-element MQ group (two 128-element IU4 blocks). At the hot `K=17408`, that is 68 workgroups per token.

Both candidates launched `grid=[ceil((K/256)/2), N]`, `block=[32]`. One wave/workgroup owned two adjacent 256-element groups, so the hot launch was 34 workgroups per token. Each lane loaded eight gate, eight up, and eight AWQ floats for group `g+1` before performing group `g`'s SiLU, FWHT, RTN, and pack. Arithmetic, butterfly order, reduction order, output indexing, and producer sidecar calls were unchanged.

Two code shapes were measured:

1. **Straight-line pair:** load both groups, then execute two inlined bodies.
2. **Rolled pair:** keep current/next `float4` register buffers and run a non-unrolled two-iteration loop, reducing instruction footprint.

## ISA/resource gate

Compiled with `hipcc -O3 --offload-arch=gfx1201 -S -Rpass-analysis=kernel-resource-usage`.

| Variant | SGPR | VGPR | Scratch/spills | Occupancy | Code bytes | First source-use wait |
|---|---:|---:|---:|---:|---:|---|
| baseline | 18 | 55 | 0 / 0 | 16 waves/SIMD | — | `s_wait_loadcnt 0x3` |
| straight-line pair | 22 | 80 | 0 / 0 | 16 waves/SIMD | 13,060 | `s_wait_loadcnt 0x4` after 12 group loads |
| rolled pair | 33 | 93 | 0 / 0 | 16 waves/SIMD | 7,212 | `s_wait_loadcnt 0x9`, then partial waits `0x7`, `0x5`, `0x3`, `0x2` |

The load-ahead was real in both candidates: neither waited to zero before first use. The rolled form left nine loads outstanding across initial arithmetic, but its additional live registers and loop control did not improve elapsed time.

## Real-input byte identity

A disposable extension of the existing real-input oracle ran a layer-0 N=128 prefill, captured the live `pbs.gate_ffn_batch` / `pbs.up_batch`, and emitted the full prepared SiLU producer stream for baseline and candidate on Card-C. The compared stream was 1,253,376 bytes.

- baseline MD5: `4be48f3449a52df0d111733e8637ac53`
- candidate MD5: `4be48f3449a52df0d111733e8637ac53`
- `cmp`: clean

A diagnostic F32-store run also compared every rotated F32 bit and found no difference. The repository's standalone-chain oracle currently differs from both baseline and candidate because the gfx1201 producer uses the shipped RTN candidate set while the standalone quantizer retains its portable eight-candidate path; direct baseline-vs-candidate bytes are the relevant identity gate.

## In-daemon rocprof gate

Recipe: pinned daemon, `rocprofv3 --kernel-trace`, matrix `--pp 8192 --ctx 128 --tg 128 --spec off --runs 1 --warmups 1 --kv-mode fp8 --json`. Hot producer rows were selected by N=8192 geometry and summed over warmup plus measured pass, then divided by `8192 * 2`.

| Producer | Baseline us/token | Straight-line pair us/token | Delta | Rolled pair us/token | Delta |
|---|---:|---:|---:|---:|---:|
| SiLU | 16.7231 | 16.8813 | +0.946% | 16.9838 | +1.559% |
| RMSNorm (unchanged) | 7.3996 | 7.4352 | +0.481% noise | 7.4136 | +0.189% noise |
| gated norm (unchanged) | 5.9715 | 6.0141 | +0.712% noise | 5.9549 | -0.278% noise |
| **producer sum** | **30.0943** | **30.3305** | **+0.785%** | **30.3523** | **+0.857%** |

The corresponding profiled pp8192 rows were 3613.8 tok/s baseline, 3598.0 tok/s straight-line, and 3589.6 tok/s rolled. These one-run wall rows are supporting observations, not the paired shipment gate.

## Downstream gates

Candidate ATT, paired TTFT/matrix, c2, and battery were not run because SiLU regressed at the mandatory in-daemon producer gate. RMSNorm and gated-norm load-ahead were not attempted because the assignment explicitly conditioned them on SiLU clearing that gate. No candidate is shipped.
