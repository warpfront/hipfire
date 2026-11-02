# Offload pass-back: mode A/B/C on current HEAD — gfx1201 — 2026-10-02

**Lifecycle:** `historical`

**Disposition:** the current-build end-to-end `memory.offload_exec` A/B, replacing
the split record's § 1 / § 8.1 headline for **current-HEAD** figures. The two
passback numbers that were live before this each described a different build:
[`2026-10-01-offload-passback-split-gfx1201.md`](2026-10-01-offload-passback-split-gfx1201.md)
(§ 1: passback +11.7 % over `cpu` at 8 of 32 spilled) was built at `f9079882c`,
before `fc5c0723d` (FFN-down co-inference, +3.8 % / +2.6 %), `9c722823c` and
`ce8f3e5ef`; and `fc5c0723d`'s own +17.6 % at 16 spilled lived only in its commit
message. Neither described the shipped build. The 2026-10-01 record stands under
its own fixture; it is not amended.

## Fixture (measured)

| | |
|---|---|
| Host | 1 × Radeon RX 9070 XT `gfx1201`, PCIe 4.0 ×16, HIP 7.2; Ryzen 7 7800X3D, 16 cores |
| **Load (1-min)** | **3.78 at start, 8.42 at end** (`/proc/loadavg`), uncontrolled |
| Source HEAD | `744340166e5fd4aded8a860927d1d57b7cfa2add` |
| `target/release/daemon` | md5 `b35c25f032f80882d757512d0f8ee98a` |
| `target/release/hipfire` | md5 `ee24d2671dc3480ac8949a43df71b156` |
| Model | `~/.hipfire/models/qwen3.5-9b.mq4`, md5 `296092bf1e6a45d78c1acf815eb93366` |
| Prompt | `benchmarks/prompts/gpu_offload_probe.txt`, md5 `5835c71e471849b4a72e1dc8e39695e7` |
| Budget | 24 and 16 resident of 32 → **8 and 16 spilled** |

`--spec off --backend noslots --workload stateless`, greedy, 128 tokens, one fresh
process per run (`pkill daemon` before each).

## Method

`cpu` / `passback` (`share auto`) / `pcie`, 4 interleaved rounds per budget,
**arm order rotated by round** (`round % 3`) so no arm systematically takes the
post-kill cold slot. Medians over the 4 rounds.

## Result (measured)

Median decode tok/s, [4 rounds]:

| spilled | `cpu` | `passback` | `pcie` | passback vs cpu | passback vs pcie | pcie vs cpu |
|---:|---:|---:|---:|---:|---:|---:|
| 8 / 32 | 28.65 `[28.4–29.0]` | **34.10** `[30.5–34.6]` | 23.90 `[23.5–24.0]` | **+19.0 %** | **+42.7 %** | −16.6 % |
| 16 / 32 | 16.65 `[16.6–16.8]` | **19.60** `[19.0–20.5]` | 12.95 `[12.8–13.3]` | **+17.7 %** | **+51.4 %** | −22.2 % |

Wall tok/s tracked decode (cpu 26.9 / passback 31.6 / pcie 22.7 at 8 spilled).

## Reading

- **On this build passback is +19.0 % / +17.7 % over `cpu`** at 8 / 16 of 32
  spilled, and `pcie` remains the loser at both points (−16.6 % / −22.2 % vs
  `cpu`). The passback arm is the only one whose spread is nontrivial
  (`[30.5–34.6]` at 8 spilled, a low first round), which is expected: it is the
  only arm that uses both engines and so is the most contention-sensitive.
- **Cross-session comparison, stated as such.** The 2026-10-01 record measured
  passback +11.7 % / +13.9 % at the same two points on `f9079882c`. The gain on
  the current build is ~7 point / ~4 point wider, consistent with `fc5c0723d`'s
  +3.8 % / +2.6 % FFN-down co-inference landing on top of it, plus the latch work
  (`ce8f3e5ef`, which the separate band A/B shows is neutral). The host load
  differs between the sessions (here 3.78 → 8.42; there 3.3–5.9), and absolutes
  in this run are depressed for every arm, so the **relative** deltas are the
  readable part and the absolute tok/s are not a quiet-host figure.

## Not measured / out of scope

- **One host, one model, one prompt, 128 tokens.** Load uncontrolled and rising
  through the run (24 sequential 5.3 GB model loads plus host background), so
  these are contended-host numbers; a quiet-host re-run would move the absolutes.
- **27B / 64-layer, other arches, other quant formats.** The 2026-10-01 § 8.2 27B
  point is untouched and still n=1.
- **Serve / slots path, long context, prefill, spec decode, MoE.** Untouched by
  this record.
- **Correctness.** Mode output bit-identity was established in the 2026-10-01
  record (§ 6) and is not re-run here; this record is throughput only.
