# `offload_exec=cpu` MoE expert splice is decode-only — prefill collapse removed

**Lifecycle:** historical. Date: 2026-10-04. Branch:
`feature/qwen35-moe-offload`. Measurement binary: daemon md5
`8330bca9a7b02d6a658d52f0adfe1e79`, CLI md5 `54299dadb260fe2ee106e1a47b00a4df`
(`cargo build --release -p hipfire-daemon -p hipfire-cli` of this branch).
Fixture: `ornith-1.5-35b-a3b.mq4` (trunk md5 `f6cb95300d29…`) +
`ornith-1.5-35b-a3b.mtp` (head md5 `0cf41c5a3643…`). Host: 1× RX 9070 XT
`gfx1201` (16 GiB), ROCm/HIP 7.2, Ryzen 7 7800X3D, 28 GB RAM. All arms
`HIPFIRE_MOE_EXPERT_BUDGET=auto`, `HIPFIRE_GRAPH=0`, one fresh daemon per arm.

**What changed.** `memory.offload_exec=cpu` previously recomputed the routed
expert FFN on the CPU on **every** forward — decode *and* prompt prefill *and*
the MTP verify batch. This record measures the change that routes it to
single-token decode only (batched forwards keep the real expert tables and the
GPU grouped PCIe read), i.e. llama.cpp's per-op batch-routing shape.

## AR, 511-token prompt (`benchmarks/prompts/ttft_511.txt`, md5 `7423e8940920082c6fa11576d23bc9a2`)

`hipfire bench --spec off --runs 3 --warmups 2 --max-tokens 32 --backend noslots
--workload stateless --kv-mode q8`; medians of 3.

| arm | prefill tok/s | decode tok/s | host expert layers |
|---|---|---|---|
| `cpu` **before** (pre-change build) | 61.5 | 34.0 | 17 |
| `cpu` **after** (this binary) | **962.8** | 33.4 | 17 |
| `pcie` after (this binary) | 947.1 | 62.7 | 17 |

The prefill collapse (15.7×) is removed; cpu prefill is at pcie parity. Decode
is unchanged — the single-token decode splice is deliberately kept, and on this
host it is ~1.9× slower than the PCIe grouped read (33.4 vs 62.7). That residual
is the known open item, not a regression this record claims to fix.

`pcie` pre-change, for reference: 962.7 prefill / 64.2 decode (graph off);
881.6 / 41.1 with capture on. The pre-change binary's md5 was not captured
before it was rebuilt — the before/after arms are therefore one variable apart
(this diff) but not binary-digest-pinned.

## MTP, `merge_sort` prompt (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`)

`--spec mtp --max-tokens 128`; 20 host expert layers in both arms (the head's
device reserve makes `auto` spill 3 more than AR's 17 — placement differs from
the AR arms, so only the cpu-vs-pcie pair is comparable).

| arm | decode tok/s |
|---|---|
| `cpu` before (pre-change build) | 21.5 |
| `cpu` after | 23.7 |
| `pcie` after | 23.6 |

cpu-MTP now equals pcie-MTP: the cpu-specific penalty is gone. MTP is still
slower than AR on this fixture (23.7 vs 33.4) — that is the pre-existing
verify-over-offload cost, and `auto` already degrades MTP to AR over
host-mapped experts.

## Correctness

Greedy decode of `The capital of France is` (24 tokens) is byte-identical
between the `cpu` and `pcie` arms: `The capital of France is **Paris`. The
sink-twin selection is pinned by
`pipeline::sealed_moe::tests::live_validation_accepts_the_sink_twin_and_rejects_a_third_table`
(the real tables and the load-time twins validate; a third same-shape table is
rejected). Test suites: `hipfire-dispatch` 306, `hipfire-arch-qwen35` 241,
`hipfire-loader` 65 — all green.

## Caveats

n=3 per prompt per AR arm; MTP arms are a single bench run each (the pre-change
21.5 was also n=1, from the same `--spec mtp` command). Raw logs:
`results-offload/{repro-,fix-,final-}*.json`. Not an admission and not a
`docs/BENCHMARKS.md` claim; the decode-splice residual is unresolved and the
`HIPFIRE_MTP_INCREMENTAL=1` interleaved verify route (single-token rows) is not
covered by this record.
