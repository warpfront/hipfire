# CPU-offloaded MoE + GPU MTP extended 16-cell spread, post-gate-removal — 2026-10-11

**Lifecycle:** `historical`
**Fixture:** `qwen3.6-35b-a3b.mq4p` (md5 `f9b5b13eb24ecbe4b39f3a3f4a297645`, 19,757,996,288 B) + `.mtp` sidecar (md5 `51076bfb5b489832f2f6c4191b9799b0`), gfx1201 (RX 9070 XT, PCI `0000:03:00.0`, HIP 7.2), KV q8 + vmm, `max_seq=4096`, `max_tokens=256`, greedy, thinking off.
**Binaries:** hipfire `1420653877697b0baab7e726d387f92d`, daemon `fd635fbfe94ed68fb02815d6ace11a25` (= `hipfire 0.4.1+patch.1`, commit `f20a1f230cc8a13035f8f94555eb45fe687cd88b`, branch `feature/qwen35-moe-offload`; code commit `099193aa3` carries the same source).
**Source:** branch `feature/qwen35-moe-offload`, built from `f20a1f230cc8` + listed patches: loader CPU-MTP gate deletion (auto-decline + forced-on hard error), `mtp_spec.rs` runtime backstop deletion, `invalidate_fp16_cache()` after `run_cpu_moe_experts` success at `forward.rs:906` + `prefill.rs:5108` (narrowed fix; broad gfx12 launcher edit reverted, `gemm.rs` at original). Pre-existing dirty-tree CPU optimizations (async offload returns from pinned staging, six-bit AVX2 unpack, affine-sum reuse, nested Rayon) are part of the fixture. Docs/changelog carry the same patches. Temp dump hooks were present in diagnostic builds only; this build has none.
**Live route only:** `serve_engine` MTP compressed-serial K=3 via `Qwen35MtpDrafter`; native `.mtp` sidecar, `--spec mtp` / `--spec off`. MTP head device-resident in every MTP arm; what spills is the trunk's routed experts.
**Env (all cells):** `HIPFIRE_OFFLOAD_EXEC={pcie,cpu}` explicit per cell, `HIPFIRE_MOE_EXPERT_BUDGET=24` on P-set arms / unset on P-auto arms, `HIPFIRE_GPU_LAYER_BUDGET` always unset, `HIPFIRE_MTP_K=3` on MTP arms (unset on AR arms), `HIPFIRE_REPLAY_BACKEND=hip`, `HIPFIRE_GRAPH=0`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`, `HIPFIRE_HOME=/home/avery/.cache/serve_harness_mtp_ksweep`, `flock /tmp/hipfire-gpu.lock`.
**Policy:** one fresh process per sample (80 total: 16 cells × 5 runs); `--warmups 3` discarded; interleaved off/mtp order per prompt (thermal balance); merge prompt first, math second.
**Prompts (committed, byte-identical):** `benchmarks/prompts/merge_sort_thinking_off.txt` (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`, 140 chars / 38 tok) and `benchmarks/prompts/mtp-bench/stepwise_math.txt` (md5 `e527794265ce421b9cf4bc7cff631490`); `prompt_md5` asserted per run from bench.json.
**Harness:** native `hipfire bench --backend noslots --workload stateless --runs 1 --warmups 3 --max-tokens 256 --max-seq 4096 --kv-mode q8 --kv-backend vmm --spec {off,mtp} --prompt-file … --json`.
**Power / memory:** per-run `power` + `power_device` objects retained in every `*.bench.json`; per-cell 2 Hz hwmon+gpu_metrics telemetry in `*.power.jsonl` + `sys_before/after.json` snapshots. Host power limits unavailable from sysfs on this box (recorded as such).
**Raw cells:** `.codeinsight+research/cpu-mtp-extended-mtp-narrow-final/` (`*.bench.json` / `*.log` / `*.stdout` / `*.power.jsonl` / `*.sys_before|after.json` per run, `*.fixed-verify` per run, `VERIFY-FIXED-SUMMARY.txt`).
**Route verification:** post-hoc `.codeinsight+research/cpu-mtp-verify-fixed.sh` → **80/80 FIXED-PASS** (bench.json `spec_requested`/`spec_route`/`spec_tau`/`samples.decode`/`prompt_md5` + log placement/backend lines; the suite's inline `route_verify` false-failed every AR arm on a stale `decode (` stderr regex — verifier artifact, data unaffected; the suite's own audit basename bug likewise). MTP arms show `MTP head loaded (sidecar …)` + `qwen35 MTP speculator enabled (compressed-serial, K=3)` + `drafter=mtp` lines; AR arms show none of the three. CPU arms show `cpu exec: 16/16` (set) or `14/14` (autoAR) seam coverage; PCIe arms show `host-mapped` lines and no `cpu exec:` line.
**No error bar beyond the 5:** numbers below are per-cell medians over 5 fresh processes; do not quote a ratio past this fixture. This record is measurement, not admission; newest file ≠ current baseline.

Terminology: "P-set" pins `HIPFIRE_MOE_EXPERT_BUDGET=24` (24 expert layers resident, 16 spilled); "P-auto" leaves both budgets unset (auto-fit). Because the MTP head consumes VRAM, auto-fit spills the same 16 as P-set on MTP arms but only 14 on AR arms (see § Placement). "cpu/pcie" is `memory.offload_exec` — who multiplies spilled weights. KV stays in VRAM either way.

Acceptance rate: token acceptance % = τ / K × 100 with K=3 (τ is `spec_accepted / spec_cycles` per `crates/hipfire-generate/src/qwen.rs:3287` — mean accepted *draft* tokens per window; the seed/bonus token is not counted in `spec_accepted`). Reported per run from bench.json `spec_tau`, median + min–max like every other metric. Same framing as llama.cpp's acceptance rate for cross-harness comparison.

## Cell medians (decode tok/s; τ and accept% where applicable)

| Prompt | Cell               | dec med | pre med | wall med | ttft med | τ med | accept% med |
| ------ | ------------------ | ------: | ------: | -------: | -------: | ----: | ----------: |
| merge  | off-P-setAR-pcie   |    58.1 |   130.4 |     52.8 |    291.3 |     — |           — |
| merge  | off-P-setAR-cpu    |    60.7 |   125.8 |     54.7 |    302.0 |     — |           — |
| merge  | off-P-autoAR-pcie  |    61.1 |   141.1 |     55.6 |    269.4 |     — |           — |
| merge  | off-P-autoAR-cpu   |    62.9 |   137.1 |     56.9 |    277.2 |     — |           — |
| merge  | mtp-P-setMTP-pcie  |    71.1 |   113.2 |     62.4 |    335.8 |  2.46 |        82.0 |
| merge  | mtp-P-setMTP-cpu   |    87.3 |   111.9 |     74.4 |    339.5 |  2.39 |        79.7 |
| merge  | mtp-P-autoMTP-pcie |    72.2 |   113.3 |     63.0 |    335.2 |  2.46 |        82.0 |
| merge  | mtp-P-autoMTP-cpu  |    88.9 |   113.0 |     75.3 |    336.4 |  2.39 |        79.7 |
| math   | off-P-setAR-pcie   |    56.9 |   167.2 |     52.6 |    370.7 |     — |           — |
| math   | off-P-setAR-cpu    |    60.4 |   167.1 |     55.5 |    371.0 |     — |           — |
| math   | off-P-autoAR-pcie  |    63.4 |   208.2 |     59.0 |    297.7 |     — |           — |
| math   | off-P-autoAR-cpu   |    63.8 |   184.2 |     58.9 |    336.5 |     — |           — |
| math   | mtp-P-setMTP-pcie  |    64.8 |   151.9 |     58.8 |    408.1 |  2.04 |        68.0 |
| math   | mtp-P-setMTP-cpu   |    80.6 |   154.0 |     71.4 |    402.6 |  2.00 |        66.7 |
| math   | mtp-P-autoMTP-pcie |    70.1 |   173.2 |     63.8 |    358.1 |  2.04 |        68.0 |
| math   | mtp-P-autoMTP-cpu  |    81.5 |   160.2 |     72.5 |    387.0 |  2.00 |        66.7 |

τ is flat within each prompt×backend (all-5 identical): merge 2.46 pcie / 2.39 cpu, math 2.04 pcie / 2.00 cpu. Accept% inherits that flatness (merge 82.0/79.7, math 68.0/66.7). Within-cell decode spread is tight on PCIe arms (±0.1–1.4) and wider on CPU arms (e.g. merge-mtp-P-autoMTP-cpu: 85.6–89.3 — host scheduling noise; n=5 absorbs it).

## Spec change: off → MTP

Parity pair (same 16 spilled layers; isolates speculative gain):

| Prompt | Backend | AR (setAR) | MTP (setMTP) |     Δ |    Δ % |
| ------ | ------- | ---------: | -----------: | ----: | -----: |
| merge  | pcie    |       58.1 |         71.1 | +13.0 | +22.4% |
| merge  | cpu     |       60.7 |         87.3 | +26.6 | +43.8% |
| math   | pcie    |       56.9 |         64.8 |  +7.9 | +13.9% |
| math   | cpu     |       60.4 |         80.6 | +20.2 | +33.4% |

Honest pair (same config; placement drifts with head VRAM):

| Prompt | Backend | AR (autoAR) | MTP (autoMTP) |     Δ |    Δ % |
| ------ | ------- | ----------: | ------------: | ----: | -----: |
| merge  | pcie    |        61.1 |          72.2 | +11.1 | +18.2% |
| merge  | cpu     |        62.9 |          88.9 | +26.0 | +41.3% |
| math   | pcie    |        63.4 |          70.1 |  +6.7 | +10.6% |
| math   | cpu     |        63.8 |          81.5 | +17.7 | +27.7% |

MTP wins 8/8. τ: merge 82.0/79.7% accept, math 68.0/66.7% — backend moves acceptance by ~2.3 pts on merge, ~1.3 pts on math (see § Acceptance-rate note).

## Backend change: pcie → cpu (same spec, same placement intent)

| Prompt | Spec arm | Placement | pcie |  cpu | delta | delta % |
| ------ | -------- | --------- | ---: | ---: | ----: | ------: |
| merge  | off      | set       | 58.1 | 60.7 |  +2.6 |   +4.5% |
| merge  | off      | auto      | 61.1 | 62.9 |  +1.8 |   +2.9% |
| merge  | mtp      | set       | 71.1 | 87.3 | +16.2 |  +22.8% |
| merge  | mtp      | auto      | 72.2 | 88.9 | +16.7 |  +23.1% |
| math   | off      | set       | 56.9 | 60.4 |  +3.5 |   +6.2% |
| math   | off      | auto      | 63.4 | 63.8 |  +0.4 |   +0.6% |
| math   | mtp      | set       | 64.8 | 80.6 | +15.8 |  +24.4% |
| math   | mtp      | auto      | 70.1 | 81.5 | +11.4 |  +16.3% |

**CPU backend wins 8/8** — an inversion of the 2026-10-08 record (CPU lost 10–18% everywhere at full seam coverage). The Oct-08 build predates the current dirty-tree CPU optimization stack: async D2H/H2D returns from owned pinned staging, streamlined exact six-bit AVX2 unpacking, affine/codebook sum reuse, nested Rayon parallelism. Those — not the `invalidate_fp16_cache` correctness fix (which only refreshes a stale staging pointer on the CPU-deferred path and changes no arithmetic) — are the fixture-bound explanation for the inversion. Seam coverage is full (16/16 or 14/14, 0 steps left on GPU), so this is compute winning, not fallback. MTP amplifies the win (fewer AR steps per token amortize per-step cost): +22–24% on MTP arms vs +1–6% on AR arms. Prefill is unaffected (≤2 tok/s either way — wide prefill keeps the GPU grouped PCIe read even on cpu arms, by design).

## Placement change: set (pinned 24) → auto (auto-fit)

| Prompt | Spec | Backend | set  | auto | delta | delta % | Spilled (set to auto, from logs) |
| ------ | ---- | ------- | ---- | ---: | ----: | ------: | -------------------------------: |
| merge  | off  | pcie    | 58.1 | 61.1 |  +3.0 |   +5.2% |                         16 to 14 |
| merge  | off  | cpu     | 60.7 | 62.9 |  +2.2 |   +3.6% |                         16 to 14 |
| merge  | mtp  | pcie    | 71.1 | 72.2 |  +1.1 |   +1.5% |                         16 to 16 |
| merge  | mtp  | cpu     | 87.3 | 88.9 |  +1.6 |   +1.8% |                 16 to 16 (noise) |
| math   | off  | pcie    | 56.9 | 63.4 |  +6.5 | +11.4% |                         16 to 14 |
| math   | off  | cpu     | 60.4 | 63.8 |  +3.4 |   +5.6% |                         16 to 14 |
| math   | mtp  | pcie    | 64.8 | 70.1 |  +5.3 |   +8.2% |                  16 to 16 (logs) |
| math   | mtp  | cpu     | 80.6 | 81.5 |  +0.9 |   +1.1% |                 16 to 16 (noise) |

Without the head, auto-fit keeps 2 more expert layers resident (14 spilled vs 16) and runs faster; with the head on GPU, auto-fit spills the same 16 and the parity pair is near-exact on merge. Math-MTP-pcie auto (+8.2% at same 16 spilled) is unexplained variance on that cell — flagged, not modeled; math prefill spread is wider than merge throughout (±10).

## Prompt change: merge → math (same cell)

| Cell               | merge | math |    Δ |   Δ % | accept merge → math |
| ------------------ | ----- | ---: | ---: | ----: | ------------------- |
| off-P-setAR-pcie   | 58.1  | 56.9 | −1.2 | −2.1% | —                   |
| off-P-setAR-cpu    | 60.7  | 60.4 | −0.3 | −0.5% | —                   |
| off-P-autoAR-pcie  | 61.1  | 63.4 | +2.3 | +3.8% | —                   |
| off-P-autoAR-cpu   | 62.9  | 63.8 | +0.9 | +1.4% | —                   |
| mtp-P-setMTP-pcie  | 71.1  | 64.8 | −6.3 | −8.9% | 82.0 → 68.0         |
| mtp-P-setMTP-cpu   | 87.3  | 80.6 | −6.7 | −7.7% | 79.7 → 66.7         |
| mtp-P-autoMTP-pcie | 72.2  | 70.1 | −2.1 | −2.9% | 82.0 → 68.0         |
| mtp-P-autoMTP-cpu  | 88.9  | 81.5 | −7.4 | −8.3% | 79.7 → 66.7         |

AR barely moves with prompt; MTP drops −2 to −7.4, all of it acceptance (82→68% pcie, 79.7→66.7% cpu). Math acceptance is identical across backends-to-1.3 pts — backend and placement affect step cost, not acceptance.

## Acceptance-rate note (new vs 2026-10-08)

τ differs by backend at fixed K=3: merge 2.46 pcie vs 2.39 cpu (82.0 vs 79.7%), math 2.04 vs 2.00 (68.0 vs 66.7%) — perfectly consistent within prompt×backend (all-5 identical, zero overlap). τ *should* be backend-independent (same weights, same drafts); the ~2-pt gap is a fingerprint of host-vs-device arithmetic diverging at the 1-LSB level under the exact-argmax accept rule (`greedy_trunk_spine_accept`): a close second flipped by FP32-CPU vs GPU dot order turns an accept into a reject. The order-of-error difference is expected and tiny; the binary accept rule has zero tolerance for it. Whether to relax the criterion (e.g. near-argmax acceptance) is scoped as future work — any relaxation must re-prove the byte-identity bar the exact rule currently holds (PCIe MTP output byte-identical to AR over 166 tokens, verified this session; CPU MTP likewise byte-identical to CPU AR). Closing the 2-pt gap recovers ~3% on the CPU arm for free.

## Correctness / parity / serve (same build)

- **Product smoke:** CPU MTP merge prompt → `finish:stop`, 166 tokens, `drafter=mtp tau=2.39`, content **byte-identical** to CPU-AR reference on identical placement/prompt. PCIe MTP likewise byte-identical to AR. No refusal, no AR fallback, coherent decoded code. Raw: `.codeinsight+research/cpu-mtp-smoke-20261010-gatefree/narrowcheck/narrow-mtp.json/.stderr`.
- **Pre-fix failure (for the record):** before the `invalidate_fp16_cache` fix, the CPU MTP arm stopped at 6 tokens with τ=0.00 (all-reject, 2× repro) while AR completed 166 tokens. Root cause: stale FP16 activation staging in `gemm_qkv_q8_0_wmma_gfx12` (pointer-keyed `ensure_fp16_x` cache reused across layers/widths as the same `pbs.x_rot_batch` pointer changed contents — exposed by the CPU path omitting routed GPU stages). Fixed by invalidating the cache at the CPU-deferred expert handoff only; gfx12 launcher restored to original, no kernel/arithmetic/scheduling change. CPU-vs-GPU float results are otherwise not byte-equal (documented existing numerical contract); AR/MTP greedy parity holds exactly here.
- **Graded oracle:** `gpu_moe_cpu_parity graded_` 3/3 PASS (per-expert rel ~1e-7 vs 5e-3 tol; multirow splice residual rel 0.0). `hipfire-cpu` lib 37/37, `hipfire-dispatch` `cpu_exec` 5/5 PASS.
- **State oracle:** `mtp_cpu_offload_rollback_oracle` PASS (trials=6, per_forward=16; K1/K3 full/partial/zero accept; retained state within `CPU_REL_TOL` vs serial AR, byte-exact vs same-route truncated windows, reset byte-reproducible, real CPU splice counters, `graph_captures == 0`). Same-route byte-exactness vs serial-route tolerance are different contracts by design (see oracle header). Raw: `.codeinsight+research/cpu-oracle-20261010-narrow/` + `cpu-rollback-oracle-narrow.log`.
- **Serve:** battery exit 0 — 5 turns, empty=0 attractor=0, runaway=1 on the reason turn only (`finish=length` at the 256-token cap: a token-budget artifact on a long reasoning answer, not degenerate output; τ 0.93–2.35, coherent text). Chain exit 0 — 5/5 `stop`, runaway=0 empty=0 attractor=0, prefix cache `cached` 168→656, τ 1.08–2.25. Raw: `.codeinsight+research/serve-cpu-mtp-20261010-final/`.

## Caveats

- `hipfire bench` does not emit decoded text; output health is attested by τ/cycles/windows + route lines, plus the separate smoke/serve evidence above with eyeballed generations. No coherence claim beyond "spec route engaged with sane τ" from the bench cells alone.
- TTFT moves with spec (MTP pays head-prefill/state alloc once: merge ~270–300→~336–340, math ~298–371→~358–408) but not with backend.
- Wall tok/s tracks decode in every cell (ratio 0.85–0.90).
- Math prefill spread is wider than merge (±10 vs ±2); math-MTP-pcie-auto +8.2% at identical spill is unexplained cell variance.
- Prior related: `2026-10-08-cpu-mtp-extended-16cell-spread.md` (pre-gate-removal, pre-CPU-optimization-stack, pre-fix — CPU lost everywhere; superseded as a speed record, retained as method reference), `2026-10-05-mtp-k-sweep-llama-9prompt-pcie.md`, `2026-10-04-moe-offload-manual-vs-auto.md` + amendment.

## Bottom line

- CPU-offloaded MTP works: gate removed, stale-cache bug fixed narrowly, 166-token output byte-identical to AR, τ 2.39/2.00 with 79.7/66.7% acceptance.
- **CPU backend beats PCIe 8/8 on this fixture** (+1–6% AR, +16–24% MTP) — inversion vs Oct-08, attributed to the day's CPU optimization stack (async copies, SIMD, Rayon), not the correctness fix. On this discrete-GPU fixture the CPU arm is now speed *and* fit.
- MTP beats AR on every pairing: +14–22% pcie / +28–44% cpu at parity. Genre matters (merge > math via acceptance).
- Auto-fit behaves as modeled without the head; the τ/acceptance backend gap (~2 pts) is future acceptance-criteria work, not a defect.

## Raw data (80/80 runs; decode/prefill/wall tok/s; ttft ms)

| run                         | decode | prefill | wall | ttft  | τ    | acc% | spec | seam  |
| --------------------------- | ------ | ------- | ---- | ----- | ---- | ---: | ---- | ----- |
| math-mtp-P-autoMTP-cpu-r1   | 81.5   | 160.2   | 72.5 | 387.0 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r2   | 81.4   | 149.4   | 72.5 | 414.9 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r3   | 81.7   | 162.7   | 73.3 | 359.3 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r4   | 81.7   | 157.6   | 73.0 | 379.4 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r5   | 79.5   | 172.6   | 70.6 | 390.4 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-autoMTP-pcie-r1  | 70.1   | 173.2   | 63.8 | 358.1 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r2  | 70.1   | 171.7   | 63.8 | 358.9 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r3  | 70.1   | 173.4   | 63.9 | 357.5 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r4  | 70.1   | 172.7   | 63.8 | 361.1 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r5  | 70.1   | 173.3   | 63.8 | 357.9 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-setMTP-cpu-r1    | 80.0   | 158.0   | 71.3 | 402.6 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r2    | 80.6   | 152.9   | 71.4 | 392.4 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r3    | 81.7   | 154.0   | 72.2 | 404.8 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r4    | 81.3   | 149.9   | 72.1 | 404.9 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r5    | 80.3   | 156.6   | 71.4 | 413.5 | 2.00 | 66.7 | mtp  | 16/16 |
| math-mtp-P-setMTP-pcie-r1   | 64.9   | 151.6   | 58.7 | 408.1 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r2   | 64.8   | 151.9   | 58.8 | 408.3 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r3   | 65.0   | 163.4   | 59.2 | 379.4 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r4   | 64.8   | 149.3   | 58.8 | 408.6 | 2.04 | 68.0 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r5   | 64.8   | 152.3   | 58.7 | 415.2 | 2.04 | 68.0 | mtp  | —     |
| math-off-P-autoAR-cpu-r1    | 62.6   | 184.2   | 58.1 | 336.5 | —    | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r2    | 64.2   | 179.5   | 59.3 | 345.5 | —    | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r3    | 64.3   | 186.3   | 59.3 | 332.9 | —    | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r4    | 63.8   | 196.5   | 58.9 | 315.5 | —    | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r5    | 63.6   | 184.0   | 58.9 | 337.2 | —    | —    | off  | 14/14 |
| math-off-P-autoAR-pcie-r1   | 60.9   | 208.2   | 56.3 | 297.7 | —    | —    | off  | —     |
| math-off-P-autoAR-pcie-r2   | 63.4   | 180.5   | 59.0 | 343.5 | —    | —    | off  | —     |
| math-off-P-autoAR-pcie-r3   | 63.4   | 208.7   | 59.0 | 297.8 | —    | —    | off  | —     |
| math-off-P-autoAR-pcie-r4   | 63.3   | 209.9   | 59.0 | 295.4 | —    | —    | off  | —     |
| math-off-P-autoAR-pcie-r5   | 63.4   | 206.9   | 58.8 | 298.5 | —    | —    | off  | —     |
| math-off-P-setAR-cpu-r1     | 60.5   | 167.1   | 55.5 | 371.0 | —    | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r2     | 60.0   | 166.1   | 55.3 | 373.3 | —    | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r3     | 59.0   | 179.9   | 54.7 | 344.6 | —    | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r4     | 60.9   | 169.6   | 56.2 | 365.3 | —    | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r5     | 60.4   | 167.0   | 55.5 | 371.0 | —    | —    | off  | 16/16 |
| math-off-P-setAR-pcie-r1    | 55.9   | 167.2   | 51.8 | 370.7 | —    | —    | off  | —     |
| math-off-P-setAR-pcie-r2    | 56.9   | 166.6   | 52.6 | 372.4 | —    | —    | off  | —     |
| math-off-P-setAR-pcie-r3    | 56.9   | 165.4   | 52.6 | 374.8 | —    | —    | off  | —     |
| math-off-P-setAR-pcie-r4    | 56.8   | 171.2   | 52.6 | 367.0 | —    | —    | off  | —     |
| math-off-P-setAR-pcie-r5    | 57.0   | 179.3   | 52.7 | 345.8 | —    | —    | off  | —     |
| merge-mtp-P-autoMTP-cpu-r1  | 87.5   | 113.0   | 74.6 | 336.4 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r2  | 85.6   | 107.5   | 72.4 | 353.5 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r3  | 89.2   | 115.0   | 75.7 | 330.3 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r4  | 88.9   | 113.1   | 75.3 | 336.4 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r5  | 89.3   | 111.3   | 75.6 | 340.9 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-pcie-r1 | 73.9   | 111.3   | 64.3 | 341.5 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r2 | 72.8   | 113.4   | 63.3 | 337.0 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r3 | 72.1   | 113.3   | 62.8 | 337.0 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r4 | 71.1   | 112.6   | 62.0 | 339.7 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r5 | 72.2   | 114.0   | 63.0 | 333.2 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-setMTP-cpu-r1   | 88.7   | 111.5   | 75.0 | 340.9 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r2   | 86.8   | 111.0   | 73.8 | 342.5 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r3   | 87.3   | 112.3   | 74.4 | 339.5 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r4   | 85.9   | 115.3   | 73.2 | 329.6 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r5   | 88.5   | 111.9   | 73.9 | 342.0 | 2.39 | 79.7 | mtp  | 16/16 |
| merge-mtp-P-setMTP-pcie-r1  | 70.9   | 113.2   | 62.4 | 335.8 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r2  | 71.8   | 123.3   | 62.7 | 308.3 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r3  | 71.0   | 110.3   | 61.9 | 344.4 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r4  | 71.4   | 112.6   | 62.4 | 336.5 | 2.46 | 82.0 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r5  | 71.1   | 114.2   | 62.7 | 335.6 | 2.46 | 82.0 | mtp  | —     |
| merge-off-P-autoAR-cpu-r1   | 62.8   | 137.1   | 56.7 | 277.2 | —    | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r2   | 63.7   | 135.9   | 57.5 | 279.6 | —    | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r3   | 62.7   | 138.3   | 56.7 | 274.8 | —    | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r4   | 62.9   | 136.8   | 56.9 | 277.5 | —    | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r5   | 63.3   | 137.5   | 57.1 | 276.9 | —    | —    | off  | 14/14 |
| merge-off-P-autoAR-pcie-r1  | 61.1   | 141.1   | 55.6 | 269.4 | —    | —    | off  | —     |
| merge-off-P-autoAR-pcie-r2  | 61.0   | 139.4   | 55.5 | 272.5 | —    | —    | off  | —     |
| merge-off-P-autoAR-pcie-r3  | 61.1   | 142.0   | 55.6 | 267.2 | —    | —    | off  | —     |
| merge-off-P-autoAR-pcie-r4  | 60.8   | 140.1   | 55.5 | 272.5 | —    | —    | off  | —     |
| merge-off-P-autoAR-pcie-r5  | 62.2   | 152.6   | 56.7 | 249.0 | —    | —    | off  | —     |
| merge-off-P-setAR-cpu-r1    | 59.4   | 125.5   | 53.6 | 302.0 | —    | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r2    | 60.7   | 126.1   | 54.7 | 300.3 | —    | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r3    | 60.7   | 125.8   | 54.7 | 302.1 | —    | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r4    | 59.5   | 126.5   | 53.7 | 300.3 | —    | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r5    | 61.4   | 125.6   | 55.2 | 302.9 | —    | —    | off  | 16/16 |
| merge-off-P-setAR-pcie-r1   | 58.0   | 130.4   | 52.6 | 291.3 | —    | —    | off  | —     |
| merge-off-P-setAR-pcie-r2   | 58.1   | 125.9   | 52.8 | 301.9 | —    | —    | off  | —     |
| merge-off-P-setAR-pcie-r3   | 58.2   | 129.4   | 52.9 | 294.6 | —    | —    | off  | —     |
| merge-off-P-setAR-pcie-r4   | 58.2   | 130.9   | 52.9 | 293.4 | —    | —    | off  | —     |
| merge-off-P-setAR-pcie-r5   | 58.1   | 138.5   | 52.8 | 274.3 | —    | —    | off  | —     |

Per-run prompt_md5 / `spec_route` / power objects live in the set dir's `*.bench.json`; placement lines in `*.log`; telemetry in `*.power.jsonl`; verifier summary in `VERIFY-FIXED-SUMMARY.txt`.
