# Flash-Next EP2 serve throughput, dual R9700 (cp20, pre-registered, 12 loads)

**Lifecycle:** historical (dated measurement; not a floor, default or admission).
**Date:** 2026-11-10/11 (UTC). **Build:** beta `13a4ba3c66166424d660868fea003219c55c9ee3`
(`cargo build --release -p hipfire-cli -p hipfire-daemon`; hipfire sha256
`a064876000ab35742a176bd4d2eca8d38237100fb672916730d996426c0ce97f`, daemon sha256
`e3646c242a5be13271bb9ef0caccc772a8ce064ea78d7c195cd3167adf57a592`). The release head
`7f7cf7f50f` adds 8 commits over it, all PeaceMaker/RIP/CDNA tooling outside the gfx1201 serve path.

## Fixture

| item | value |
|---|---|
| Host / cards | one host, 2x Radeon AI PRO R9700 (gfx1201), EP2 |
| Model | `qwen3.8-flash-next-base.mq4-xts-symtrunk` sha256 `27dfff6692eba3385655959d134771fcff154c6279ab0a6c43cb8c7d81a7fa31`; `.hc8` `2adf32e564f09efbe185c7ebf57483dc58103f21341569e716173098f0e2ee67`; `.ep.hot-units` `a50317cf2014f88d306d88366131212757d875c4c6e65572d659d361ec17db13` |
| Load | `hipfire serve`, C concurrent streams (C = 1, 4, 8), one story prompt and one code prompt (HumanEval `has_close_elements`), T=1, top_p 0.95, top_k 20, max_tokens 512, thinking off, EOS on; warm request excluded |
| Protocol | pre-registered (frozen PREREG md5 `da02be666456734f26827e7398c8ed72`), 3 jobs x 4 fresh serves = 12 loads, host-load and foreign-GPU checks per job (all 3 clean), no load dropped or re-run |
| Statistic | aggregate EOS-on tok/s, pooled median over 12 loads; 95% bootstrap CI of the median ratio (20000 draws) |

## Result

Reference: a published dual-R9700 Flash-Next result (story 189.2 / 467.2 / 664.5, code 262.5 / 618.5 / 879.9 tok/s
at C1 / C4 / C8).

| cell | pooled median tok/s | min-max | vs the published dual-R9700 result | 95% CI | TTFT s (median) | decode tok/s per stream |
|---|---:|---|---:|---|---:|---:|
| story C1 | 135.9 | 128.5-141.1 | 0.718x | [0.695, 0.735] | 0.065 | 140.4 |
| story C4 | 336.2 | 324.2-345.2 | 0.720x | [0.705, 0.729] | 0.119 | 92.6 |
| story C8 | 527.3 | 503.1-548.5 | 0.794x | [0.775, 0.806] | 0.177 | 73.6 |
| code C1 | 205.1 | 196.4-214.3 | 0.781x | [0.763, 0.789] | 0.112 | 243.7 |
| code C4 | 579.0 | 532.4-680.7 | 0.936x | [0.921, 0.970] | 0.179 | 192.0 |
| code C8 | **930.9** | 875.2-965.5 | **1.058x** | [1.032, 1.071] | 0.263 | 156.5 |

Only code C8 exceeds the reference (11 of 12 loads above 879.9). The other five cells are below it.
Against another open engine measured on the same host (3 reps per cell): 0.935x / 0.884x / 0.944x (story C1/C4/C8),
0.849x / 0.960x / 0.877x (code C1/C4/C8). Its TTFT is lower in every cell (0.043-0.127 s); hipfire's per-stream
decode is higher on code C4 (192.0 vs 174.3) and code C8 (156.5 vs 152.8) and lower elsewhere.

## Caveats

- **Fixture:** the code prompt asks for a function whose signature and docstring are in the prompt; about 60% of the
  reply copies prompt text, which favours speculative/n-gram drafting (tau 5.8 tokens per step on code vs 2.2 on story).
  The code numbers do not transfer to generic code generation.
- **Cross-lane overlap:** at T=1 several of the C streams return identical code replies (median distinct replies
  4 of 8 at code C8). The overlap check (no stream copying another stream's tokens mid-flight) passed in every
  cell, but identical replies still share acceptance behaviour; story replies are all distinct.
- Aggregate tok/s mixes TTFT and decode; the split is in the table. Jobs differ up to 7% on code C4; only pooled
  figures are quoted.

Run record: `release-0.4.5/stepshep/cp/cp20/RESULT.md` (per-load values, step-ms, lever log lines, host checks)
on the measurement host; not yet copied to a durable public location.
