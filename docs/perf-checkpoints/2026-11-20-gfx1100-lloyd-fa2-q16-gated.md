# 2026-11-20 — gfx1100 lloyd-V FA2 Q16/FILL gated A/B

**Lifecycle:** `historical` (measurement evidence only; not a current default,
not an admission decision). **Disposition:** candidate perf fix for upstream
`beta`; typed flag default OFF; not promoted to the daily driver.

## Fixture and method

- Host: AMD RX 7900 XTX, **gfx1100**, 48 CU, 23.98 GB; HIP 7.16 / clang 24,
  ROCm 10.1 (`therock-dist-windows-gfx110X-all-10.1.0`), Windows x86_64,
  **legacy KV** (Windows VMM/graph semantics uncertified).
- Kernel cache: `C:\Users\Jan\.hipfire_kernels\gfx1100`, warm (7,734 files at
  first run); a cold-JIT arm can eat the whole effect on this host.
- Model: `C:\Users\Jan\Swift-1.5-Qwen3.8-27b.mq4l-xt`, 14,987,262,016 B,
  md5 `d0e167dca0c63841b9332b2e1e6545f5`.
- Prompt: `benchmarks/prompts/qwen38_issue693_longcode_20676.txt`,
  md5 `b4d0b63cddcac872648ddf3cdd92cac2`, 21,582 tokens.
- Bench shape (every arm): `hipfire bench <model> --prompt-file <above>
  --runs 2 --warmups 1 --max-tokens 8 --backend noslots --workload stateless
  --spec off --kv-k fwht4 --kv-v lloyd4 --json`, with
  `HIPFIRE_LLOYD_MMQ_A4=1` and `HIPFIRE_DAEMON_BIN` selecting the matching
  daemon. One throwaway load per binary, then ABBA, 3 s between loads.
- Bootstrap: 20,000 resamples of the mean of 4 loads per arm, CI at the
  500th / 19,500th sorted difference.

## Binaries

| Arm | Source | hipfire md5 | daemon md5 |
|---|---|---|---|
| INST daily driver | `ab0d89deb` | `abf0834fa904b83b2a350518adc98eab` | `bef146ba58567bea00d833a8c1d45085` |
| bare beta | `9e410448e` | `16c0409a06f4be3a358783513fa9f0eb` | `e83a380b62d30290fed81f8ff768ccb8` |
| gated, flag ON | `9e410448e` + `fa2_lloydv_q16` | `f3899eba5edbac1ec94a4f45687bba36` | `b4ea4d0ee6ef56b2cd2ed9bca9e48584` |

## Measured prefill (tok/s)

| Arm | 4 interleaved loads |
|---|---|
| INST | 1203.8, 1204.4, 1204.0, 1201.9 |
| gated flag ON | 1256.2, 1256.2, 1254.2, 1254.5 |

**gated − INST = +51.75 tok/s, 95% bootstrap CI [50.54, 53.41]** (+4.3%).

Decode gate (`--spec off`, `--max-tokens 64`, same fixture, 4 loads/arm):
gated − INST = +0.05 tok/s, CI [−0.15, 0.28] — no regression.

Flag OFF on the gated binary measured 1146.35 tok/s, matching bare beta's
1146.1–1146.6 band — the default path is unchanged.

## Correctness

Greedy `hipfire run -j`, `--temp 0`, `--kv-k fwht4 --kv-v lloyd4`:
- short prompt (48 tokens): bare beta == gated OFF == gated ON,
  md5 `07cb2532f0dc2a4d391ce40238dedc41`.
- 6,000-char slice of the committed code prompt (64 tokens): bare beta ==
  gated ON, md5 `c106a243bb3f9c505c5e1404cd4200e1`.

The installed build differs from both on the long prompt
(md5 `5ff5e882e3134eb50335bf8ec6bc07e0`) — a pre-existing installed-vs-upstream
divergence unrelated to this change. Byte-identity is claimed only against the
unflagged upstream build.

## Root cause and caveat

Upstream's lloyd-V FA2 split (`e1dbeef40`, carry of #1089) forbids Q16 and FILL
for VMODE 4/3 and reverted the launcher to Q8 (`div_ceil(8)`, 128 threads).
`HIPFIRE_GFX11_FA2_PREFILL=0` collapses the installed-vs-beta gap to zero;
`HIPFIRE_FA2_FILL=0` costs installed 32 tok/s. The gated path restores Q16
(16 query rows/workgroup, 256 threads) and the warp-specialized lloyd-V FILL.
An earlier attempt that changed `div_ceil(8)`→`div_ceil(16)` but kept 128 threads
measured +160 tok/s and emitted 64 `!` tokens — rejected after an output check.

Fixture-bound to gfx1100 / legacy KV / this 21K Lloyd cell. No gfx12, VMM-KV,
other-quant or general prefill claim follows from this record.
