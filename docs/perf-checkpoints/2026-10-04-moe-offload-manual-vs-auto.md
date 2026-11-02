# MoE offload manual-vs-auto + llama.cpp anchor — 2026-10-04

**Lifecycle:** `historical`

**Disposition:** exploratory measurement for
[`docs/methodology/cpu-offload-head-to-head-plan.md`](../methodology/cpu-offload-head-to-head-plan.md).
Not an admission, not a `docs/BENCHMARKS.md` claim, not comparable across
host/model/quant/GPU/prompt/method. Raw per-point `--json` (full `samples`
arrays) plus per-point `.err` files carrying the `partial offload:` load lines
quoted below are committed at
`data-2026-10-04-moe-offload-manual-vs-auto/` next to this record.

## Fixture identity (measured)

| artifact | digest |
|---|---|
| source tree (`git rev-parse HEAD`) | `abde7a189e8e44b0967ade01d5cba91840776d93` |
| `target/release/hipfire` (md5) | `f4665016fcbba9bf64e1fbd2401f5c14` |
| `target/release/daemon` (md5) | `b0fdb2aad9dd28f4160050591d96adcd` |
| hipfire trunk `~/.hipfire/models/qwen3.5-35b-a3b.mq4` | 19,663,624,448 B; uniform MQ4 (packable), 40L, 256 experts × 8/tok |
| llama GGUF `Qwen3.6-35B-A3B-Q4_K_M.gguf` | 21,155,768,832 B (`model_size`); `qwen35moe`, 40L, 256 experts, 8/tok, 733 tensors |
| prompt `benchmarks/prompts/gpu_offload_probe.txt` | md5 `5835c71e471849b4a72e1dc8e39695e7`, 215 B, 59 tokens |
| llama.cpp build | commit `11fe02151`, build 11382; `llama-bench` sha256 `41ef585fca9eb02348567b189857280840258e04338bb48407eeb08c996658fe` |

Host: 1× RX 9070 XT `gfx1201` (16304 MB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM, THP `[always]`. Shared desktop (loadavg 4–9 during
arms, lm-studio resident); MemAvailable drifted 15.5–20 GiB across arms —
see N4 note. Bench JSON: `kv_mode q8`, `kv_backend vmm`, `--spec off
--backend noslots --workload stateless`, 5 runs / 2 warmups / 64 gen tokens.
llama.cpp ran Vulkan/radv (`--device Vulkan0`), KV f16 default, `-t 8`,
synthetic `-p 512` prompt — **no byte-identical prompt parity** with hipfire
(llama-bench has no prompt-file flag); matched by spill count only.

**Base-model substitution (method delta).** The plan's Qwen3.6 hipfire trunk
(`qwen3.6-35b-a3b.mq4p`, graded per-expert tiering) refuses *every* spill:
`qwen35/load.rs:5678` — only MQ4/MQ4V2/MQ4C experts pack into the host blob,
so expert-spill (and whole-layer spill, same gate) fail closed at layer 0,
manual *and* `auto` (auto chose 14 host layers, 13083/13095 MiB, then hit
the same gate). Resident also impossible: 18827 MiB needed vs 13356
available even at `max_seq 2048`. The hipfire arm therefore uses the
packable uniform-MQ4 `qwen3.5-35b-a3b.mq4` while the llama anchor uses the
supplied Qwen3.6 Q4_K_M GGUF — different base model AND different quant
(MQ4 ~494 MB/layer file bytes vs Q4_K_M ~529 MB/layer). Cross-engine rows
are per-spilled-layer, never raw tok/s deltas.

## Hipfire manual frontier (qwen3.5-35B, max_seq 32768, kv q8)

N = `HIPFIRE_MOE_EXPERT_BUDGET` (layers keeping routed experts on GPU;
spilled = 40 − N), 408 MiB routed experts/layer. One fresh daemon per point,
serialized except where noted.

| N (kept) | spilled | `device weights X of Y` | pcie dec / pre (med, n=5) | cpu dec / pre (med, n=5) | coverage |
|---:|---:|---|---|---|---|
| 26 | 14 | 13025/13095 MiB | 63.3 (σ0.26) / 162.2 | 37.5 (σ0.75) / 73.5 | cpu 14/14, uncovered: none |
| 22 | 18 | 11393/13067 MiB | 53.6 (σ2.82) / 136.6 | 33.2 (σ0.64) / 60.8 | cpu 18/18, none |
| 16 | 24 | 8945/13067 MiB | 48.7 (σ0.12) / 110.5 | 27.3 (σ0.04) / 47.1 | cpu 24/24, none |
| 10 | 30 | 6497/13067 MiB | 42.0 (σ0.38) / 89.2 | 22.7 (σ0.24) / 37.6 | cpu 30/30, none |
| 4 | 36 | 4049/13095 MiB | — (see N4 note) | 20.0 (σ0.23) / 32.6 | cpu 36/36, none |

Refusals with numbers (no OOM): N34/40 refuse VRAM-side (N34: needs 16289
vs 13095, 3194 over; N28: 13904 vs 13095, 809 over); N0 refuses host-side
(16.0 GiB pinned + 4 GiB headroom vs MemAvailable 16.8 GiB).

**N4 is host-state-dependent, not structural.** Single-sample probe
(`q35_N4_probe`, 38.1 dec) loaded at higher MemAvailable; the 5-run
`manual_pcie_N4` refused at MemAvailable 15.5 GiB (14.3 GiB pinned + 4 GiB
headroom); the later 5-run `manual_cpu_N4` loaded at MemAvailable ~20 GiB
(20.0 dec). Report as: loadable when MemAvailable clears ~18.3 GiB, refused
below — the deep-spill boundary moves with host state.

**Contaminated, re-run clean:** first-pass N16/N10 ran concurrent with a
`llama-bench` probe (shared VRAM + 8c): N16 σ10.0, N10 samples
[24.9, 42.5, 20.1, 12.2, 40.7] with prefill/TTFT scatter to match. The
`_clean` rows above are serialized re-runs (σ0.12/σ0.38); the concurrent
cells are void and not tabulated.

**cpu-exec verdict on this trunk:** loses at every spill (pcie − cpu:
N26 +25.8, N22 +20.4, N16 +21.4, N10 +19.3 tok/s). MoE-internal reading only:
the spilled set is routed-expert FFN (attention/router/shared/KV stay resident),
so each spilled layer pays per-step D2H/GEMV/H2D serialization against only the
expert payload — on this trunk that trade loses at all five points. No dense
comparison is drawn (different spill unit, different execution path).

## Auto vs manual (the plan's question)

| exec | auto placement | auto rate (med, n=5) | matched manual (N26) | delta | on-frontier? |
|---|---|---|---|---|---|
| pcie | 14 host layers, 13025/13067 MiB (70 MiB idle, < 1 layer) | 62.9 / 162.3 | 63.3 / 162.2 | −0.4 (−0.6%) | **yes** |
| cpu | 14 host layers, 13025/13067 MiB | 38.5 / 78.1 | 37.5 / 73.5 | +1.0 (+2.7%) | **yes** |

`auto` lands on the smallest-spill-that-fits (N26 = 14 spilled) with
sub-layer idle capacity and reproduces the matched manual rate within noise
(the cpu +2.7% is one σ against the manual σ0.75 plus host drift, not a
placement difference — same split, same kernels). No over-spill, no
over-place, no refusal where a manual split loads. Auto matched the manual
frontier at 32768 on this trunk; other contexts unmeasured, so the plan's
cross-context verdict stays open.

On the graded `.mq4p` trunk, auto's arithmetic also reached the fit
frontier (14 host layers, 13083/13095 MiB, 12 MiB idle) but refused at the
packing gate — an execution refusal, not a placement miss. Recorded as
`auto_mq4p.err`; no throughput cell exists for that trunk by construction.

## llama.cpp anchor (Qwen3.6 Q4_K_M, `--device Vulkan0`, `-p 512 -n 64`)

| `--n-cpu-moe` (first-N layers to CPU) | tg tok/s (avg, r=2–3) | note |
|---:|---|---|
| 0 (default, `-ngl -1`) | 37.7 [37.8, 37.7, 37.7] | fit untouched; 20.7 GB model vs 13.5 GB free so fit placed it (split not audited from this arm — see fit-auto below), not resident |
| 14 | 38.4 [36.5, 40.9, 37.7] | matched-spill vs hipfire N26 |
| 18 | 40.3 [40.1, 40.6] | matched-spill vs hipfire N22 |
| 24 | 30.5 [31.8, 34.0, 25.8] | wide spread — host-state suspect, not a rate claim |
| 30 | 29.8 [28.1, 30.5, 30.8] | |

**llama fit-auto at `-c 32768`:** `common_params_fit_impl` projects 20710
MiB vs 13573 free, sheds 8161 MiB, and converges on
`(n_layer, n_part, overflow) = (41, 19, ATTN)` — 41/41 layers GPU-assigned
with partial (MoE-vs-dense fraction) overflow, i.e. sub-layer granularity
hipfire's whole-expert-layer units cannot express. Final load: all 41
layer slots on Vulkan0, MoE fraction spilled. (No fit-auto transcript is
retained — neither the committed data dir nor `llama_probe_ngl0.err`
covers this arm — so treat the `(41, 19, ATTN)` tuple as session
observation, not retained evidence.)

**Cross-engine reading (per-layer, not raw tok/s):** at matched 14-spill,
hipfire-pcie 63.3 vs llama 38.4; at 18-spill, 53.6 vs 40.3. The gap is
expected from the method deltas, not a finding: different base model (3.5
vs 3.6), different quant (MQ4 vs Q4_K_M), different GPU API (HIP vs
Vulkan/radv), different KV (q8 vs f16 — ~2× KV bytes against the same fit
budget), synthetic vs file prompt. MoE-internal shape only: llama's curve is
flat 37–40 across 0–18 spill while hipfire-pcie falls 63→54 over the same
range (every spilled byte re-crosses PCIe per token). No dense record is
invoked as corroboration.

## What was NOT measured

- slots/serve path (sequential daemon only, per runbook — cpu-exec never
  enters `execute_steps` on the batched path); prefill (GPU-side; the
  59-token probe prompt makes `prefill_tok_s` a launch-overhead number —
  bench warns as much); `fwht3` KV; graph-on; DFlash/MTP (spec off both
- engines); serve_harness coherence (rate-only record — no decoded text was
  eyeballed on any offload arm, so no product claim follows without a
  claim-scoped serve_harness pass per the plan's §Reporting).
- `auto` at other contexts (plan asks 2048/8192/32768): only 32768 run;
  the frontier answer is single-context.
- llama's own spill frontier untested below ncmoe 14 (ncmoe 10/12 never run):
  whether llama reaches a smaller spill than hipfire auto's 14 — the plan's
  over-spill direction cross-engine — is unanswered, not just thin.
- plan's 9B dense resident-fit zero-diff guard under `auto` (a fitting model
  must stay fully resident, byte-identical to no knobs): not run. No dense
  arms measured here at all.

## Reproduction

```bash
PROMPT=benchmarks/prompts/gpu_offload_probe.txt
COMMON=(--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots \
        --workload stateless --kv-mode q8 --prompt-file "$PROMPT" --json)
for N in 26 22 16 10 4; do
  HIPFIRE_MOE_EXPERT_BUDGET=$N HIPFIRE_OFFLOAD_EXEC=pcie HIPFIRE_GRAPH=0 \
    ./target/release/hipfire bench qwen3.5:35b-a3b "${COMMON[@]}"
done
# llama anchor (Vulkan-only build; needs LD_LIBRARY_PATH on build/bin):
llama-bench -m <Qwen3.6-Q4_K_M.gguf> --n-cpu-moe 18 -p 512 -n 64 -r 3 -o json --device Vulkan0
```
