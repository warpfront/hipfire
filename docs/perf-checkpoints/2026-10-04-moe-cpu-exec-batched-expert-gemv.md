# Amendment: batched multi-expert GEMV for the `offload_exec=cpu` decode splice

**Lifecycle:** historical. Date: 2026-10-04. Amends
[`2026-10-04-moe-cpu-exec-decode-only-routing.md`](2026-10-04-moe-cpu-exec-decode-only-routing.md)
(unchanged): that record removed the prefill collapse and left cpu decode at
33.4 tok/s vs pcie 62.7. This record attributes and closes most of that
remaining decode gap.

Binary: `hipfire` md5 `13f74c6f385dfee37d64953f59305f59`, daemon md5
`1b21a38e8915da2172e8ff14f9b57cd1`. Same fixture, host, prompt
(`merge_sort_thinking_off.txt`, md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`),
`HIPFIRE_GRAPH=0`, `--backend noslots --workload stateless --kv-mode q8`,
medians of 3, 17 host expert layers.

## The finding

`HIPFIRE_CPU_EXEC_TRACE=1` was extended to the decode splice (it had no
attribution; the earlier record flagged this as the open instrument gap). Per
layer per decode token, 1.29 ms was in the GEMV loop and 0.23 ms in the D2H/H2D
copies — 81% of a 31.4 tok/s token. The loop issued 2k small per-expert `gemv`
calls (k=8 experts × gate_up/down), each opening its own rayon region; rayon
scaling from 1 to 16 threads was only 1.9×, i.e. region entry dominated.

## The fix

`hipfire_cpu::gemv::gemv_experts(q, m, k, pairs, out)` — one rayon region over
every `(expert, row)` output element. `moe_cpu_experts` now issues two such
regions per layer (all experts' gate_up, then all experts' down) instead of 2k
`gemv` calls. Each output element is the same `dot_row_simd` call, and the
residual is still accumulated in rank order, so the result is bit-identical to
the per-expert loop — pinned by
`gemv::test::gemv_experts_matches_per_expert_gemv`.

## Result (AR decode, 64-token prompt)

| arm | decode tok/s | splice gemv ms/layer | splice d2h ms/layer |
|---|---|---|---|
| cpu, before (per-expert `gemv`) | 31.4–33.4 | 1.29 | 0.18 |
| cpu, after (`gemv_experts`) | **49.2** | 0.52 | 0.18 |
| pcie (control, same binary) | 64.3 | — | — |

cpu MoE decode +47%; the gap to pcie narrows from ~1.9× to ~1.3×. Greedy text
unchanged and equal to pcie (`The capital of France is **Paris`).

## Residual (open)

The batched GEMV scales only 4.35× from 1 to 16 rayon threads (2.26 → 0.52
ms/layer), so it is no longer region-overhead-bound but not compute-bound
either; the 0.18 ms/layer of blocking D2H (four `hipMemcpy` syncs; async needs
pinned host staging, which this path does not use) is now 16% of a cpu token.
cpu does not yet beat pcie on this host. Levers not tried here: collapsing the
per-layer D2H syncs, and an AVX-512 decode path (this host is Zen 4, the
kernels are AVX2 for fleet portability). Raw logs: `results-offload/batch-*.json`.
