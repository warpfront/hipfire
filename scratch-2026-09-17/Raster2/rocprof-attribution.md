# G8T0 serving-shape attribution (rocprofv3 kernel-trace, ONE pp8192 pass/arm)

## Arms (ordinal 2, ab2, HIPFIRE_GRAPH=1 HIPFIRE_LLOYD_GFX12=1)
- base: HEAD kernel (G0), hipfire-base 6ffeda3f366fd30dafc8aeeda07c2edf,
  daemon-base e495a2335ee2903c90e0e2add004febe
- g8t0: `g8t0.patch` applied (swizzle defines G8T0 + remap block, no cpol),
  hipfire-g8t0 116182ab914d546d33147142bbab985a,
  daemon-g8t0 dca631c9c3d3e375a9f538787d9f6e26
- Candidate verified live: cache hsaco
  `gemm_mq4g256v2_residual_mmq_iu4_gfx12.b32ed77ca4013916` contains
  `#define IU4_SWIZZLE_G 8` (compiled in g8t0 warmup, reused by trace).
- Bench: `bench qwen3.8:27b-mq4-xt --matrix --pp 8192 --ctx 128 --tg 64
  --spec off --runs 1 --kv-mode fp8 --json`; untraced `--warmups 1` warmup
  immediately before each traced `--warmups 0` pass.
- Serving rows from the traced runs: base pp8192 2430.1 tok/s,
  g8t0 pp8192 2431.7 tok/s (+0.07%); warmups 2449.4 vs 2437.2 (−0.50%).
- Traces: `rocprof-base/trace-base_kernel_trace.csv`,
  `rocprof-g8t0/trace-g8t0_kernel_trace.csv` (device ns timestamps).

## iu4 per-shape: median / sum (device time; y=64 = pp8192 prefill, y=1 = tg64 decode)

| kernel | grid (x,y,z) | n | base med (us) | g8t0 med (us) | med Δ | base sum (ms) | g8t0 sum (ms) | sum Δ |
|---|---|---|---|---|---|---|---|---|
| full_set | 34816,64,1 | 256 | 6832.49 | 6814.18 | −0.27% | 1769.24 | 1761.98 | −0.41% |
| full_set | 20480,64,1 | 96 | 4237.45 | 4257.10 | +0.46% | 405.34 | 407.21 | +0.46% |
| full_set | 24576,64,1 | 32 | 5084.74 | 5122.28 | +0.74% | 162.33 | 163.65 | +0.81% |
| full_set | 12288,64,1 | 96 | 2549.91 | 2551.84 | +0.08% | 243.96 | 244.08 | +0.05% |
| full_set | 2048,64,1 | 64 | 434.17 | 437.57 | +0.78% | 27.72 | 27.96 | +0.87% |
| full_set | 256,64,1 | 192 | 67.16 | 66.80 | −0.54% | 12.84 | 12.77 | −0.55% |
| full_add | 10240,64,1 | 256 | 6633.12 | 6821.50 | +2.84% | 1246.62 | 1244.30 | −0.19% |
| full_set | 34816,1,1 | 256 | 139.72 | 140.96 | +0.89% | 35.63 | 35.99 | +1.01% |
| full_set | 20480,1,1 | 96 | 84.92 | 85.68 | +0.89% | 8.19 | 8.27 | +0.98% |
| full_set | 12288,1,1 | 96 | 61.68 | 62.36 | +1.10% | 5.93 | 6.00 | +1.18% |
| full_set | 24576,1,1 | 32 | 89.32 | 89.96 | +0.72% | 2.84 | 2.86 | +0.70% |
| full_set | 2048,1,1 | 64 | 43.92 | 44.44 | +1.18% | 2.80 | 2.83 | +1.07% |
| full_set | 256,1,1 | 192 | 43.12 | 43.60 | +1.11% | 8.24 | 8.33 | +1.09% |
| full_add | 10240,1,1 | 256 | 185.32 | 188.60 | +1.77% | 35.23 | 35.64 | +1.16% |

Note: no (17408,·) grid exists — the daemon launches gate/up at full
N=34816 (plus qkvza/qkv head-group splits at 20480/24576/12288/2048/256),
not two N=17408 halves. The full_add median/sum split on (10240,64,1)
(+2.8% med vs −0.2% sum, n=256) is distribution skew in an untimed-warmup
single pass, not a real shift — the sum agrees with serving (+0.07%).

## Totals (device-time sums over the traced pass incl. tg64 decode)

- iu4_full_set: 2685.06 → 2681.93 ms (−0.12%)
- iu4_full_add: 1281.85 → 1279.94 ms (−0.15%)
- iu4 combined (n=1984): 3966.90 → 3961.86 ms (−0.13%)
- everything-else (n=121354): 6149.12 → 6170.10 ms (+0.34%)
- total: 10116.03 → 10131.96 ms (+0.16%)

## Verdict
GEMM sum drops 0.13% — far below the 5% bar. No grid shape gains beyond
noise; the rig's −10.7% on the synthetic N=34816/M=4096 gate shape does not
transfer to any serving shape (largest serving GEMM, full_set 34816×64,
−0.41% sum). The serving pairs were not noise — G8T0 is flat in serving.
No re-pair warranted.

## Method note (for reuse)
The slice2 background-shim recipe (`rocprofv3 ... &` + `wait` under
`#!/bin/sh`→dash) is broken: dash gives background jobs /dev/null stdin,
so the daemon gets instant EOF. And a foreground wrapper gets SIGKILLed
with the CLI's Engine drop before rocprofv3 finalizes (no CSV).
Working recipe used here: python double-fork launcher (`bin/det-launch.py`)
+ minimal fd-passthrough rocprof relay (`bin/minpy-rocprof.py`) + per-arm
shims (`daemon-det-*.sh`). The detached daemon survives the Engine-drop
kill, exits on CLI-exit EOF, rocprofv3 finalizes. (Do NOT use a threaded
python relay: threads-only pump wedges the handshake pre-first-send on
this box — mechanism unidentified; fd-passthrough is unaffected.)
