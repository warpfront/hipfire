# MoE system-RAM offload — AR vs MTP, four arms (post decode-only-routing fix)

**Lifecycle:** historical. Date: 2026-10-04. Branch
`feature/qwen35-moe-offload`. Binary: `hipfire` md5
`9b4b2379a1c1ac138da104f3e8c48ffb`, daemon md5
`8dc96a9b135800784a70c108a4dc7fb4`. Host: 1× RX 9070 XT `gfx1201` (16 GiB),
HIP 7.2, Ryzen 7 7800X3D, 28 GB RAM, DDR5-6000 (peak 96 GB/s theoretical,
~56 GB/s measured achievable — see the int8 record). Load ~1.3–5.0 during the
run.

This is the record the `2026-10-04-mtp-over-trunk-expert-spill` checkpoint asked
for, after the two fixes it led to:
- `fe91b2d87` — the MoE CPU expert splice runs on single-token decode only;
  prompt prefill and the MTP verify batch keep the GPU grouped PCIe read
  (before this, `offload_exec=cpu` collapsed prefill ~15×: 61.5 vs 962 tok/s).
- `58b30ed93` — the decode splice batches all experts into one rayon region per
  projection (cpu MoE decode 31.4 → 49.2 tok/s).

## Fixture and method

`ornith-1.5-35b-a3b.mq4` (trunk md5 `f6cb95300d29c0c4c10d017180c12ab8`) +
`ornith-1.5-35b-a3b.mtp` (head md5 `0cf41c5a36430c3570c509643db102b5`).

```bash
HIPFIRE_OFFLOAD_EXEC={pcie,cpu} HIPFIRE_MOE_EXPERT_BUDGET=auto HIPFIRE_GRAPH=0 \
  hipfire bench ~/.hipfire/models/ornith-1.5-35b-a3b.mq4 --spec {off,mtp} \
  --runs 3 --warmups 2 --max-tokens 128 --backend noslots --workload stateless \
  --kv-mode q8 --prompt-file benchmarks/prompts/merge_sort_thinking_off.txt --json
```

Prompt `merge_sort_thinking_off.txt`, md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`
(38 tokens). `--spec mtp` is explicit: `auto` degrades MTP to AR over
host-mapped experts.

## Result — medians of 3

| spec | exec | decode tok/s | samples | prefill tok/s | host expert layers | τ |
|---|---|---|---:|---:|---|---|---:|
| off (AR) | pcie | **64.2** | 64.1 / 64.2 / 64.3 | 132.1 | 17 | — |
| off (AR) | cpu | **50.5** | 49.3 / 50.5 / 50.6 | 132.2 | 17 | — |
| mtp | pcie | **23.4** | 23.6 / 23.4 / 23.4 | 97.0 | 20 | 1.35 |
| mtp | cpu | **23.5** | 23.5 / 23.5 / 23.4 | 96.4 | 20 | 1.35 |

Read this as two facts:

1. **The cpu-specific MTP penalty is gone.** MTP cpu (23.5) now equals MTP pcie
   (23.4). Before the decode-only-routing fix the same pairing was 21.5 (cpu)
   against a pcie MTP that was itself worse than its AR — the fix removed the
   splice from the verify batch, so the two exec modes run the same code there.
2. **MTP is still slower than AR on this fixture** (23.5 vs 50.5 cpu, 23.4 vs
   64.2 pcie), because verify-over-offload re-reads host-mapped experts for
   `(K+1)/τ` rows per emitted token at τ=1.35. That is the structural
   `auto`-degrades result the gate already encodes, not a regression.

**Confound, stated:** the MTP arms resolve **20** host expert layers vs the AR
arms' **17** — the MTP head's device reserve makes `auto` spill three more
layers. So MTP-vs-AR is *not* a clean one-variable comparison; only the
cpu-vs-pcie pair within a spec is.

**Prefill column is not throughput:** the prompt is 38 tokens, so `prefill_tok_s`
measures launch overhead (the harness warns about this). The meaningful prefill
result is on a ≥256-token prompt and is recorded in
`2026-10-04-moe-cpu-exec-decode-only-routing.md` (cpu 962.8 vs pcie 947.1 — the
15× collapse removed).

## Correctness

Greedy decode of `The capital of France is` (24 tokens) is byte-identical
between the `cpu` and `pcie` arms: `The capital of France is **Paris`.

## Open

cpu MoE decode is 50.5 vs pcie 64.2 (1.27×) — the residual after the fix. It is
localized to the small-`m` CPU GEMV's per-call/per-row overhead, not memory
class, ISA, or arithmetic; see the batched-GEMV and int8 records. Raw logs:
`results-offload/doc-{off,mtp}-{pcie,cpu}.json`.
