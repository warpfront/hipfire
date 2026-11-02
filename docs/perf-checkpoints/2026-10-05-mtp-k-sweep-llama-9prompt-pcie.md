# MTP K-sweep on llama-9 prompts, pcie offload — 2026-10-05

**Lifecycle:** `historical`
**Fixture:** `qwen3.6-35b-a3b.mq4p` (md5 `f9b5b13eb24ecbe4b39f3a3f4a297645`)
+ `.mtp` (md5 `51076bfb5b489832f2f6c4191b9799b0`), gfx1201, KV q8,
`max_seq=4096`, `max_tokens=256`, greedy, thinking off.
**Binaries:** hipfire `b721870ee59ec7848af0667c54faed33`,
daemon `f0299fa706bece6f75dee455fb213e84` (= handoff candidate build:
coalesced grouped narrow verify + narrow-verify Path-1 gate).
**Live route only:** `serve_engine` → `mtp_draft_phase_inner` +
`mtp_batched_verify_accept_from_batch` via `Qwen35MtpDrafter`
(`spec_step_mtp_compressed_serial_with_k`); native `.mtp` sidecar, `--mtp on`.
`mtp_probe_step`, `spec_step_mtp{,_trunk_spine,_compressed}`,
`spec_step_dflash_mtp{,_tree}` have no serve caller for qwen35 MoE
(examples/tests only) and were not benched.
**Env (all cells):** `HIPFIRE_MOE_EXPERT_BUDGET=auto`,
`HIPFIRE_OFFLOAD_EXEC=pcie`, `HIPFIRE_GRAPH=0`, `HIPFIRE_DPM_WARMUP_SECS=10`,
one discarded 16-token same-prompt warmup, one fresh serve process per cell,
`flock /tmp/hipfire-gpu.lock`.
**Prompts:** byte-identical to `mtp-bench.py` PROMPTS (prompt_md5 equal in
every cell). Raw cells: `.codeinsight+research/mtp-ksweep/`
(`ar-pcie.json`, `mtp-k{1..5}-pcie.json` + `.log`/`.stdout` each).
**No error bar:** one rep per cell. Do not quote a ratio past this fixture.

## Decode tok/s (per prompt; AR = `--mtp off` same backend)

| prompt | AR | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---:|---:|---:|---:|---:|
| code_python | 62.9 | 36.0 | 53.1 | 55.2 | 38.4 | 34.0 |
| code_cpp | 64.0 | 44.7 | 51.3 | 53.2 | 38.2 | 33.2 |
| explain_concept | 59.5 | 36.5 | 38.6 | 39.9 | 25.2 | 20.3 |
| summarize | 60.7 | 37.2 | 42.7 | 57.1 | 32.5 | 26.9 |
| qa_factual | 62.3 | 39.3 | 42.3 | 55.9 | 27.3 | 23.6 |
| translation | 62.7 | 36.6 | 35.4 | 43.1 | 21.6 | 17.3 |
| creative_short | 61.3 | 32.2 | 31.5 | 40.9 | 20.2 | 16.9 |
| stepwise_math | 62.0 | 41.4 | 46.1 | 66.5 | 31.2 | 28.6 |
| long_code_review | 58.9 | 34.3 | 34.5 | 42.8 | 21.2 | 16.9 |
| **median** | **62.0** | **36.6** | **42.3** | **53.2** | **27.3** | **23.6** |

## τ per prompt (accept rate = τ/K for llama comparison)

| prompt | K=1 τ (acc) | K=2 τ (acc) | K=3 τ (acc) | K=4 τ (acc) | K=5 τ (acc) |
|---|---|---|---|---|---|
| code_python | 0.92 (0.920) | 1.82 (0.910) | 2.39 (0.797) | 2.97 (0.743) | 3.29 (0.658) |
| code_cpp | 0.96 (0.960) | 1.84 (0.920) | 2.38 (0.793) | 2.60 (0.650) | 2.73 (0.546) |
| explain_concept | 0.66 (0.660) | 1.07 (0.535) | 1.34 (0.447) | 1.43 (0.357) | 1.32 (0.264) |
| summarize | 0.78 (0.780) | 1.53 (0.765) | 1.94 (0.647) | 2.36 (0.590) | 2.57 (0.514) |
| qa_factual | 0.74 (0.740) | 1.27 (0.635) | 1.52 (0.507) | 1.70 (0.425) | 1.71 (0.342) |
| translation | 0.60 (0.600) | 1.13 (0.565) | 1.25 (0.417) | 1.38 (0.345) | 1.38 (0.276) |
| creative_short | 0.50 (0.500) | 0.73 (0.365) | 0.95 (0.317) | 1.11 (0.278) | 1.11 (0.222) |
| stepwise_math | 0.85 (0.850) | 1.48 (0.740) | 2.04 (0.680) | 2.19 (0.547) | 2.49 (0.498) |
| long_code_review | 0.58 (0.580) | 0.90 (0.450) | 1.04 (0.347) | 1.18 (0.295) | 1.02 (0.204) |
| **median τ (acc)** | **0.74 (0.740)** | **1.27 (0.635)** | **1.52 (0.507)** | **1.70 (0.425)** | **1.71 (0.342)** |

Median ratio-to-own-AR: **K=1 0.61×, K=2 0.68×, K=3 0.83×, K=4 0.44×, K=5
0.38×.** One prompt beats AR (stepwise_math K=3, 1.07×); everything else
trails at every K.

## Against llama-server (same 9 prompts, Qwen3.6-35B-A3B UD-Q4_K_M GGUF)

llama accept (draft fraction): K=1 .812 / K=2 .714 / K=3 .630; tok/s ~50/~56/~52
vs AR ~44.5 (**1.12×/1.26×/1.17×**). Ours (median accept): K=1 .740 / K=2 .635 /
K=3 .507 — same decline-with-K slope shape, lower intercept — while tok/s stays
<AR at every K. The gap is verify row cost, not the slope: K=1 isolates it
(accept .74 vs llama .812, yet 0.61× vs ~1.1×).

## cpu backend: refused at load, no cells

`HIPFIRE_OFFLOAD_EXEC=cpu` fails closed on this fixture
(`qwen35/load.rs:6163`): graded mixed-dtype host-placed experts + cpu exec =
"cold-tier experts would silently mis-decode". The CPU splice only covers
uniform-dtype non-AWQ experts; fully-resident is unrunnable (19.8 GB weights
vs 16 GB VRAM). No cpu arm exists for this fixture; pcie (GPU reads
host-mapped weights over PCIe, CPU idle) is the only offload backend.
Placement confound: AR spills 14 routed-expert layers, MTP 16 (6.41 GiB host),
so MTP pays 2 extra layers of PCIe-read verify cost before accepting a token.

## Correctness notes (not a parity claim)

All MTP cells coherent, no attractor/empty output. MTP≢AR at token level:
code_python K=1 same gen count (238) but whitespace diverges early (AR
indented `        F(0) = 0`, MTP `    F(0) = 0`). Token-id parity needs serve
transcripts (`bench` swallows `committed`); unrun here.
