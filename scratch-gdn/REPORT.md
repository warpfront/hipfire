# gfx11 GDN prefill verdict

**Keep** exact C64 row-major prep on gfx1100/gfx1151 and exact value-head KKT on gfx1100; **kill** Halo C32 KKT/scan and its prefix-reset prep. The production branch contains no C32 kernel, switch, or dispatch. The exact routes default on only on their named GPU, with `HIPFIRE_GDN_PREP_GFX11=0` and `HIPFIRE_GDN_KKT_GFX1100=0` restoring the incumbents. Other architectures use the original kernels.

## End-to-end quality on the integrated stack

`gfx11-gdnprefill-stack`, based on `stack-1@1fcb53d4d`, Qwen3.8-27B MQ4V2 XT QAT, q8/q8, WT2 prefill scoring, 24 chunks, each card isolated. The default and the same stack with both switches set to `0` produced byte-identical complete KLD sequences:

| Card | Baseline / default KLD | SHA-256 of both outputs | Delta against landed WT2 |
| --- | ---: | --- | ---: |
| gfx1100 | 0.077271 / 0.077271 | `fbbaa065f792681ec41499c5b8fdebab77f1062b308713240c4e39ae1ec62143` | +0.000392 vs 0.076879 |
| gfx1151 | 0.075973 / 0.075973 | `79bbff9ef0d2b6df0f03bd50f163710c51e6d182a8f5b51d15629f0cb7b67774` | -0.000928 vs 0.076901 |

The nonzero differences from landed WT2 come from the upstream beta/alpha fold in this combined stack, not these byte-exact GDN changes. Reproduction: `scratch-gdn/run_wt2.py <gfx1100|gfx1151> <baseline|default>` on hipx. Output/logs: `/home/kaden/hipfire-gdnstack/scratch-gdn/quality/<card>/<tag>/`.

## Integrated pp8192 ROCprof trace

Fresh Q8 VMM serve process per route, warm request then one uncached 8192-token request, one card at a time, both exact levers off versus default on. Both prompt-token counts were 8192 and cached counts were zero, with device and Q8 VMM assertions in each serve log. Times in the kernel columns sum the warm and traced request, so compare paired rows rather than reading them as single-request latency.

| Card | Route | Request wall ms | All GPU kernels ms | GDN prep ms | GDN KKT ms | GDN scan ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gfx1100 | baseline | 3183.626 | 6268.068 | 79.728 | 48.302 | 161.476 |
| gfx1100 | default | 3160.997 | 6259.640 | 73.333 | 30.497 | 161.107 |
| gfx1151 | baseline | 7790.814 | 15545.545 | 311.341 | 73.976 | 482.159 |
| gfx1151 | default | 7748.976 | 15501.310 | 269.390 | 78.379 | 480.559 |

Wall improved 22.629 ms on gfx1100 and 41.838 ms on gfx1151 in these single pairs; gfx1100 targeted prep+KKT kernel sums improved 24.200 ms, Halo prep improved 41.951 ms. The unchanged Halo KKT fluctuation is not a claimed benefit. Full traces and process/arch/KV assertions: `/home/kaden/hipfire-gdnstack/scratch-gdn/trace/<card>-8192-{baseline,default}/`. These are not ABBA throughput gates; Stacker runs those after merging.

## Production prefill trace on the isolated lever branch

Uncached pp8192 on Qwen3.8-27B MQ4V2 XT QAT with fresh Q8 VMM daemon and ROCprof, one warm request before the trace request, single stream. On the pre-stack branch, gfx1100 both exact levers off vs both on: request wall 3215.874 → 3192.966 ms; total warm+trace GPU kernel time 6388.84 → 6363.12 ms. The GDN prep kernel sum fell 72.060 → 69.728 ms and KKT fell 48.487 → 31.162 ms; scan changed 161.637 → 160.754 ms. On gfx1151, exact prep off vs exact C64 prep on (C32 forced off): request wall 8269.532 → 8198.379 ms; warm+trace GPU kernel time 16544.88 → 16410.85 ms; prep fell 307.420 → 267.312 ms; KKT 75.010 → 73.200 ms; scan 481.402 → 480.228 ms. ROCprof CSVs, serve logs, request token counts, arch/KV assertions, and pre/post process snapshots are in `/home/kaden/hipfire-gdnprefill/scratch-gdn/trace/{gfx1100-8192,gfx1151-8192}-{baseline,default}/` (Halo exact trace is named `gfx1151-8192-default` on the original C32 opt-in commit). These timings do not assert the integrated stack's ABBA result; Stacker owns that gate.

## Rejected C32: real-layer parity, not just average KL

One genuine WT2 prompt chunk (2047 prefill tokens) through the full 64-layer model, Halo, same stack-port binary and model, same input/reset state, GDN layer 0 final state after the scored prefill. A temporary evaluator probe downloaded Q8 codes, F32 per-row scale and F16 EF residual; the probe source was removed from the branch. Reconstructing each state entry as `signed_code*row_scale+EF`, C64 vs C32 on 786,432 entries: max |C64| = 45.89617, max |C32-C64| = 0.299700, **max range-relative error 0.6530%** (`max_abs_diff / max_abs_C64`), and **relative RMS error 0.2794%** (`sqrt(sum(diff²)/sum(C64²))`). The maximal elementwise relative error with a denominator floor at 0.0001 × global max |C64| is 166.1% near zero; raw elementwise relative error without a floor is not meaningful near zero. Codes differ in 15,283/786,432 entries; row scales differ in 4,231/6,144; EF differs in 544,014/786,432. The same single-chunk prefill-scored KL worsened 0.048628 → 0.055033 (+0.006405) despite the isolated 24-chunk average initially improving 0.076901 → 0.075689. Accordingly C32 is killed rather than shipped based on its average. Raw state snapshots: `/home/kaden/hipfire-gdnstack/scratch-gdn/state-probe/{c64,c32}/{codes,scales,ef}.bin`; full evaluator logs were observed for both paths.

The standalone HIP C32 prototype reduced Halo scan from 5.759 to 3.757 ms per 512-token GDN layer in a synthetic input but changed outputs and persistent state; its speed does not override the real-layer quality failure. The KKT value-head prototype regressed on Halo, so the original KKT remains there. Exact gfx11 prep and gfx1100 KKT preserve the existing FP32 prefix and scan arithmetic order.
