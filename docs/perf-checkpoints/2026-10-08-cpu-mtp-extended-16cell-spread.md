# CPU-offloaded MoE + GPU MTP extended 16-cell spread — 2026-10-08

**Lifecycle:** `historical`
**Fixture:** `qwen3.6-35b-a3b.mq4p` (md5 `f9b5b13eb24ecbe4b39f3a3f4a297645`, 19,757,996,288 B) + `.mtp` sidecar (md5 `51076bfb5b489832f2f6c4191b9799b0`), gfx1201 (RX 9070 XT, PCI `0000:03:00.0`, HIP 7.2), KV q8 + vmm, `max_seq=4096`, `max_tokens=256`, greedy, thinking off.
**Binaries:** hipfire `f611147eae670bd4ec35ea86cadeac37`, daemon `6a2a24229c766f48108a95f11dd620ba` (= `hipfire 0.4.1+patch.1`, commit `f5a4593b134563208456b759b972e9e669e39ec9`, branch `feature/qwen35-moe-offload`, clean tree).
**Source:** branch `feature/qwen35-moe-offload`, commit `f5a4593b134563208456b759b972e9e669e39ec9`; tree clean at manifest time.
**Live route only:** `serve_engine` MTP compressed-serial K=3 via `Qwen35MtpDrafter` (`mtp_draft_phase_inner` + `mtp_batched_verify_accept_from_batch`); native `.mtp` sidecar, `--spec mtp` / `--spec off`. MTP head device-resident in every MTP arm; what spills is the trunk's routed experts.
**Env (all cells):** `HIPFIRE_OFFLOAD_EXEC={pcie,cpu}` explicit per cell, `HIPFIRE_MOE_EXPERT_BUDGET=24` on P-set arms / unset on P-auto arms, `HIPFIRE_GPU_LAYER_BUDGET` always unset, `HIPFIRE_MTP_K=3` on MTP arms (unset on AR arms), `HIPFIRE_REPLAY_BACKEND=hip`, `HIPFIRE_GRAPH=0`, `HIPFIRE_DPM_WARMUP_SECS=10`, `HIP_VISIBLE_DEVICES=0`, `HIPFIRE_HOME=/home/avery/.cache/serve_harness_mtp_ksweep`, `flock /tmp/hipfire-gpu.lock`.
**Policy:** one fresh process per sample (80 total: 16 cells × 5 runs); `--warmups 3` discarded; interleaved off/mtp order per prompt (thermal balance); merge prompt first, math second.
**Prompts (committed, byte-identical):** `benchmarks/prompts/merge_sort_thinking_off.txt` (md5 `253c7ac50857fe6d0e10fb0d2c5e35c0`, 140 chars / 38 tok) and `benchmarks/prompts/mtp-bench/stepwise_math.txt` (md5 `e527794265ce421b9cf4bc7cff631490`); `prompt_md5` asserted per run from bench.json.
**Harness:** native `hipfire bench --backend noslots --workload stateless --runs 1 --warmups 3 --max-tokens 256 --max-seq 4096 --kv-mode q8 --kv-backend vmm --spec {off,mtp} --prompt-file … --json`.
**Power / memory:** per-run `power` + `power_device` objects retained in every `*.bench.json`; per-cell 2 Hz hwmon+gpu_metrics telemetry in `*.power.jsonl` (~70 samples/cell) + `sys_before/after.json` snapshots. Host power limits unavailable from sysfs on this box (recorded as such).
**Raw cells:** `.codeinsight+research/cpu-mtp-extended-mtp-ext-full-20261008-193815/` (`*.bench.json` / `*.log` / `*.stdout` / `*.power.jsonl` / `*.sys_before|after.json` per run, `*.fixed-verify` per run, `VERIFY-FIXED-SUMMARY.txt`).
**Route verification:** post-hoc `.codeinsight+research/cpu-mtp-verify-fixed.sh` → **80/80 FIXED-PASS** (bench.json `spec_requested`/`spec_route`/`spec_tau`/`samples.decode`/`prompt_md5` + log placement/backend lines; the suite's inline `route_verify` false-failed every AR arm on a stale `decode (` stderr regex — verifier artifact, data unaffected). MTP arms show `MTP head loaded (sidecar …)` + `qwen35 MTP speculator enabled (compressed-serial, K=3)` + `drafter=mtp` lines; AR arms show none of the three. CPU arms show `cpu exec: 16/16` (set) or `14/14` (autoAR) seam coverage; PCIe arms show `host-mapped` lines and no `cpu exec:` line.
**No error bar beyond the 5:** numbers below are per-cell medians over 5 fresh processes; do not quote a ratio past this fixture. This record is measurement, not admission; newest file ≠ current baseline.

Terminology: "P-set" pins `HIPFIRE_MOE_EXPERT_BUDGET=24` (24 expert layers resident, 16 spilled); "P-auto" leaves both budgets unset (auto-fit). Because the MTP head consumes VRAM, auto-fit spills the same 16 as P-set on MTP arms but only 14 on AR arms (see § Placement). "cpu/pcie" is `memory.offload_exec` — who multiplies spilled weights. KV stays in VRAM either way.

## Cell medians (decode tok/s; τ where applicable)

| Prompt | Cell               | dec med | pre med | wall med | ttft med | τ med |
| ------ | ------------------ | ------: | ------: | -------: | -------: | ----: |
| merge  | off-P-setAR-pcie   |    61.3 |   146.3 |     56.0 |    259.7 |     — |
| merge  | off-P-setAR-cpu    |    50.2 |   147.2 |     46.6 |    258.2 |     — |
| merge  | off-P-autoAR-pcie  |    64.7 |   159.9 |     59.2 |    237.6 |     — |
| merge  | off-P-autoAR-cpu   |    52.9 |   160.4 |     49.2 |    236.9 |     — |
| merge  | mtp-P-setMTP-pcie  |    77.5 |   127.3 |     68.0 |    298.6 |  2.46 |
| merge  | mtp-P-setMTP-cpu   |    67.4 |   127.3 |     60.1 |    298.6 |  2.41 |
| merge  | mtp-P-autoMTP-pcie |    77.5 |   127.7 |     68.1 |    297.7 |  2.46 |
| merge  | mtp-P-autoMTP-cpu  |    68.2 |   127.9 |     60.7 |    297.2 |  2.41 |
| math   | off-P-setAR-pcie   |    59.7 |   192.0 |     55.6 |    322.8 |     — |
| math   | off-P-setAR-cpu    |    49.9 |   191.8 |     46.9 |    323.2 |     — |
| math   | off-P-autoAR-pcie  |    63.2 |   207.7 |     58.8 |    298.5 |     — |
| math   | off-P-autoAR-cpu   |    52.6 |   207.1 |     49.6 |    299.3 |     — |
| math   | mtp-P-setMTP-pcie  |    70.0 |   170.3 |     63.7 |    364.2 |  2.04 |
| math   | mtp-P-setMTP-cpu   |    63.3 |   170.5 |     58.1 |    363.7 |  2.04 |
| math   | mtp-P-autoMTP-pcie |    70.0 |   170.8 |     63.7 |    363.0 |  2.04 |
| math   | mtp-P-autoMTP-cpu  |    63.0 |   170.9 |     57.9 |    362.8 |  2.04 |

Within-cell spread is tight on PCIe arms (all-5 identical to ±0.1) and wider on CPU arms (e.g. merge-mtp-P-setMTP-cpu: 64.2–68.9 — host scheduling noise; n=5 absorbs it).

## Spec change: off → MTP

Parity pair (same 16 spilled layers; isolates speculative gain):

| Prompt | Backend | AR (setAR) | MTP (setMTP) |     Δ |    Δ % |
| ------ | ------- | ---------: | -----------: | ----: | -----: |
| merge  | pcie    |       61.3 |         77.5 | +16.2 | +26.4% |
| merge  | cpu     |       50.2 |         67.4 | +17.2 | +34.3% |
| math   | pcie    |       59.7 |         70.0 | +10.3 | +17.3% |
| math   | cpu     |       49.9 |         63.3 | +13.4 | +26.9% |

Honest pair (same config; placement drifts with head VRAM):

| Prompt | Backend | AR (autoAR) | MTP (autoMTP) |     Δ |    Δ % |
| ------ | ------- | ----------: | ------------: | ----: | -----: |
| merge  | pcie    |        64.7 |          77.5 | +12.8 | +19.8% |
| merge  | cpu     |        52.9 |          68.2 | +15.3 | +28.9% |
| math   | pcie    |        63.2 |          70.0 |  +6.8 | +10.8% |
| math   | cpu     |        52.6 |          63.0 | +10.4 | +19.8% |

MTP wins 8/8. CPU-arm % gains are larger only because the CPU AR baseline is slower — absolute gains are comparable (±1 tok/s). τ: merge 2.41–2.46, math 2.04 flat across backends/placements.

## Backend change: pcie → cpu (same spec, same placement intent)

| Prompt | Spec arm | Placement | pcie |  cpu | delta | delta % |
| ------ | -------- | --------- | ---: | ---: | ----: | ------: |
| merge  | off      | set       | 61.3 | 50.2 | -11.1 |  -18.1% |
| merge  | off      | auto      | 64.7 | 52.9 | -11.8 |  -18.2% |
| merge  | mtp      | set       | 77.5 | 67.4 | -10.1 |  -13.0% |
| merge  | mtp      | auto      | 77.5 | 68.2 |  -9.3 |  -12.0% |
| math   | off      | set       | 59.7 | 49.9 |  -9.8 |  -16.4% |
| math   | off      | auto      | 63.2 | 52.6 | -10.6 |  -16.8% |
| math   | mtp      | set       | 70.0 | 63.3 |  -6.7 |   -9.6% |
| math   | mtp      | auto      | 70.0 | 63.0 |  -7.0 |  -10.0% |

CPU backend loses 8/8 at full seam coverage (16/16 or 14/14 per log) — the deficit is compute, not fallback: every spilled expert step ran on the CPU and still lost to the GPU grouped PCIe read. MTP narrows the gap (fewer AR steps per token amortize the slower expert FFN). Prefill is unaffected (≤1 tok/s either way — wide prefill keeps the GPU grouped PCIe read even on cpu arms, by design; cf. `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5097-5100`).

## Placement change: set (pinned 24) → auto (auto-fit)

| Prompt | Spec | Backend | set  | auto | delta | delta % | Spilled (set to auto, from logs) |
| ------ | ---- | ------- | ---- | ---: | ----: | ------: | -------------------------------: |
| merge  | off  | pcie    | 61.3 | 64.7 |  +3.4 |   +5.5% |                         16 to 14 |
| merge  | off  | cpu     | 50.2 | 52.9 |  +2.7 |   +5.4% |                         16 to 14 |
| merge  | mtp  | pcie    | 77.5 | 77.5 |   0.0 |    0.0% |                         16 to 16 |
| merge  | mtp  | cpu     | 67.4 | 68.2 |  +0.8 |   +1.2% |                 16 to 16 (noise) |
| math   | off  | pcie    | 59.7 | 63.2 |  +3.5 |   +5.9% |                         16 to 14 |
| math   | off  | cpu     | 49.9 | 52.6 |  +2.7 |   +5.4% |                         16 to 14 |
| math   | mtp  | pcie    | 70.0 | 70.0 |   0.0 |    0.0% |                         16 to 16 |
| math   | mtp  | cpu     | 63.3 | 63.0 |  -0.3 |   -0.5% |                 16 to 16 (noise) |

Without the head, auto-fit keeps 2 more expert layers resident (14 spilled vs 16, device weights 13083 vs 12263 MiB) and runs ~5–6% faster; with the head on GPU, auto-fit spills the same 16 and the parity pair is exact. AutoMTP ≈ setMTP is therefore a finding, not a missing effect.

## Prompt change: merge → math (same cell)

| Cell               | merge | math |    Δ |   Δ % | τ merge → math |
| ------------------ | ----- | ---: | ---: | ----: | -------------- |
| off-P-setAR-pcie   | 61.3  | 59.7 | −1.6 | −2.6% | —              |
| off-P-setAR-cpu    | 50.2  | 49.9 | −0.3 | −0.6% | —              |
| off-P-autoAR-pcie  | 64.7  | 63.2 | −1.5 | −2.3% | —              |
| off-P-autoAR-cpu   | 52.9  | 52.6 | −0.3 | −0.6% | —              |
| mtp-P-setMTP-pcie  | 77.5  | 70.0 | −7.5 | −9.7% | 2.46 → 2.04    |
| mtp-P-setMTP-cpu   | 67.4  | 63.3 | −4.1 | −6.1% | 2.41 → 2.04    |
| mtp-P-autoMTP-pcie | 77.5  | 70.0 | −7.5 | −9.7% | 2.46 → 2.04    |
| mtp-P-autoMTP-cpu  | 68.2  | 63.0 | −5.2 | −7.6% | 2.41 → 2.04    |

AR barely moves with prompt; MTP drops −4 to −7.5, all of it τ (2.46 → 2.04). Math τ is identical across all four MTP cells — backend and placement affect step cost, not acceptance.

## Caveats

- `hipfire bench` does not emit decoded text; output health is attested by τ/cycles/windows + route lines (e.g. `drafter=mtp … decode (166 tok, 48 windows)`), not eyeballed generations. No `serve_harness` battery/chain was run — no coherence claim beyond "spec route engaged with sane τ."
- TTFT moves with spec (MTP pays head-prefill/state alloc once: merge ~258→~298, math ~300→~363) but not with backend.
- Wall tok/s tracks decode in every cell (ratio 0.91–0.93).
- Prior related: `2026-10-04-mtp-over-trunk-expert-spill.md` (MTP regressed on older build/fixture — superseded here, retained), `2026-10-05-mtp-k-sweep-llama-9prompt-pcie.md` (K-sweep on this trunk), `2026-10-04-moe-offload-manual-vs-auto.md` + amendment (manual vs auto placement).

## Bottom line

- MTP beats AR on every pairing: +17–26% pcie / +27–34% cpu at parity, +11–20% / +20–29% honest. Genre matters (merge > math via τ).
- CPU backend loses to PCIe everywhere (−10 to −18%) at full seam coverage. On this discrete-GPU fixture the CPU arm's value is fit, not speed.
- Auto-fit behaves exactly as modeled: +5–6% without head (2 fewer spilled layers), identical with head.

## Raw data (80/80 runs; decode/prefill/wall tok/s; ttft ms)

| run                         | decode | prefill | wall | ttft  | τ    | spec | seam  |
| --------------------------- | ------ | ------- | ---- | ----- | ---- | ---- | ----- |
| math-mtp-P-autoMTP-cpu-r1   | 62.0   | 171.1   | 57.0 | 362.4 | 2.04 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r2   | 64.0   | 170.9   | 58.7 | 362.8 | 2.04 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r3   | 62.7   | 169.5   | 57.5 | 365.7 | 2.04 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r4   | 63.0   | 171.4   | 57.9 | 361.6 | 2.04 | mtp  | 16/16 |
| math-mtp-P-autoMTP-cpu-r5   | 63.2   | 170.7   | 58.0 | 363.2 | 2.04 | mtp  | 16/16 |
| math-mtp-P-autoMTP-pcie-r1  | 70.1   | 171.9   | 63.8 | 360.6 | 2.04 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r2  | 70.0   | 171.5   | 63.7 | 361.6 | 2.04 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r3  | 70.0   | 170.7   | 63.7 | 363.2 | 2.04 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r4  | 70.0   | 169.6   | 63.6 | 365.5 | 2.04 | mtp  | —     |
| math-mtp-P-autoMTP-pcie-r5  | 70.0   | 170.8   | 63.7 | 363.0 | 2.04 | mtp  | —     |
| math-mtp-P-setMTP-cpu-r1    | 64.1   | 170.8   | 58.7 | 363.1 | 2.04 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r2    | 62.6   | 169.0   | 57.5 | 366.9 | 2.04 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r3    | 63.0   | 171.7   | 57.9 | 361.0 | 2.04 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r4    | 63.5   | 170.5   | 58.2 | 363.7 | 2.04 | mtp  | 16/16 |
| math-mtp-P-setMTP-cpu-r5    | 63.3   | 170.4   | 58.1 | 363.9 | 2.04 | mtp  | 16/16 |
| math-mtp-P-setMTP-pcie-r1   | 70.0   | 171.2   | 63.7 | 362.2 | 2.04 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r2   | 70.0   | 171.4   | 63.7 | 361.7 | 2.04 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r3   | 70.1   | 170.0   | 63.7 | 364.8 | 2.04 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r4   | 70.0   | 170.3   | 63.7 | 364.2 | 2.04 | mtp  | —     |
| math-mtp-P-setMTP-pcie-r5   | 70.0   | 170.2   | 63.7 | 364.3 | 2.04 | mtp  | —     |
| math-off-P-autoAR-cpu-r1    | 52.5   | 205.7   | 49.5 | 301.5 | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r2    | 52.6   | 205.9   | 49.6 | 301.1 | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r3    | 52.2   | 207.3   | 49.2 | 299.1 | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r4    | 53.1   | 207.1   | 50.0 | 299.3 | —    | off  | 14/14 |
| math-off-P-autoAR-cpu-r5    | 52.6   | 208.0   | 49.6 | 298.1 | —    | off  | 14/14 |
| math-off-P-autoAR-pcie-r1   | 63.2   | 205.9   | 58.8 | 301.1 | —    | off  | —     |
| math-off-P-autoAR-pcie-r2   | 63.2   | 211.7   | 58.9 | 292.8 | —    | off  | —     |
| math-off-P-autoAR-pcie-r3   | 63.1   | 205.7   | 58.7 | 301.4 | —    | off  | —     |
| math-off-P-autoAR-pcie-r4   | 63.2   | 208.7   | 58.9 | 297.1 | —    | off  | —     |
| math-off-P-autoAR-pcie-r5   | 63.1   | 207.7   | 58.8 | 298.5 | —    | off  | —     |
| math-off-P-setAR-cpu-r1     | 50.0   | 192.3   | 47.0 | 322.5 | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r2     | 50.3   | 191.8   | 47.3 | 323.2 | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r3     | 48.6   | 191.5   | 45.8 | 323.8 | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r4     | 49.9   | 192.4   | 46.9 | 322.2 | —    | off  | 16/16 |
| math-off-P-setAR-cpu-r5     | 49.9   | 191.8   | 46.9 | 323.3 | —    | off  | 16/16 |
| math-off-P-setAR-pcie-r1    | 59.7   | 190.9   | 55.5 | 324.8 | —    | off  | —     |
| math-off-P-setAR-pcie-r2    | 59.7   | 192.0   | 55.5 | 322.8 | —    | off  | —     |
| math-off-P-setAR-pcie-r3    | 59.8   | 191.7   | 55.6 | 323.5 | —    | off  | —     |
| math-off-P-setAR-pcie-r4    | 59.7   | 192.5   | 55.6 | 322.0 | —    | off  | —     |
| math-off-P-setAR-pcie-r5    | 59.7   | 192.2   | 55.6 | 322.7 | —    | off  | —     |
| merge-mtp-P-autoMTP-cpu-r1  | 68.2   | 127.9   | 60.7 | 297.2 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r2  | 68.4   | 127.4   | 60.9 | 298.2 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r3  | 68.5   | 128.3   | 61.0 | 296.2 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r4  | 67.0   | 127.4   | 59.8 | 298.2 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-cpu-r5  | 67.8   | 128.2   | 60.5 | 296.5 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-autoMTP-pcie-r1 | 77.6   | 127.6   | 68.1 | 297.9 | 2.46 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r2 | 77.5   | 128.1   | 68.1 | 296.7 | 2.46 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r3 | 77.5   | 127.4   | 68.0 | 298.2 | 2.46 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r4 | 77.6   | 127.7   | 68.1 | 297.7 | 2.46 | mtp  | —     |
| merge-mtp-P-autoMTP-pcie-r5 | 77.5   | 128.1   | 68.1 | 296.7 | 2.46 | mtp  | —     |
| merge-mtp-P-setMTP-cpu-r1   | 66.0   | 128.0   | 59.0 | 296.8 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r2   | 68.9   | 127.5   | 61.3 | 298.1 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r3   | 68.1   | 127.2   | 60.6 | 298.7 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r4   | 67.4   | 127.3   | 60.1 | 298.6 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-setMTP-cpu-r5   | 64.2   | 127.1   | 57.6 | 298.9 | 2.41 | mtp  | 16/16 |
| merge-mtp-P-setMTP-pcie-r1  | 77.5   | 126.2   | 68.0 | 301.2 | 2.46 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r2  | 77.5   | 127.1   | 68.0 | 299.0 | 2.46 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r3  | 77.5   | 127.7   | 68.0 | 297.5 | 2.46 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r4  | 77.6   | 127.9   | 68.1 | 297.1 | 2.46 | mtp  | —     |
| merge-mtp-P-setMTP-pcie-r5  | 77.5   | 127.3   | 68.1 | 298.6 | 2.46 | mtp  | —     |
| merge-off-P-autoAR-cpu-r1   | 54.1   | 160.4   | 50.2 | 236.9 | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r2   | 52.6   | 159.8   | 48.9 | 237.8 | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r3   | 54.0   | 160.4   | 50.1 | 236.9 | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r4   | 52.9   | 159.8   | 49.2 | 237.9 | —    | off  | 14/14 |
| merge-off-P-autoAR-cpu-r5   | 52.7   | 161.1   | 49.0 | 235.9 | —    | off  | 14/14 |
| merge-off-P-autoAR-pcie-r1  | 64.6   | 158.7   | 59.1 | 239.5 | —    | off  | —     |
| merge-off-P-autoAR-pcie-r2  | 64.7   | 159.9   | 59.2 | 237.6 | —    | off  | —     |
| merge-off-P-autoAR-pcie-r3  | 64.7   | 161.0   | 59.2 | 236.0 | —    | off  | —     |
| merge-off-P-autoAR-pcie-r4  | 64.7   | 159.6   | 59.2 | 238.0 | —    | off  | —     |
| merge-off-P-autoAR-pcie-r5  | 64.7   | 159.9   | 59.2 | 237.6 | —    | off  | —     |
| merge-off-P-setAR-cpu-r1    | 49.1   | 146.5   | 45.6 | 259.4 | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r2    | 50.4   | 145.9   | 46.7 | 260.4 | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r3    | 50.8   | 148.4   | 47.1 | 256.1 | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r4    | 49.5   | 147.2   | 46.0 | 258.2 | —    | off  | 16/16 |
| merge-off-P-setAR-cpu-r5    | 50.2   | 147.3   | 46.6 | 258.0 | —    | off  | 16/16 |
| merge-off-P-setAR-pcie-r1   | 61.3   | 146.3   | 56.0 | 259.7 | —    | off  | —     |
| merge-off-P-setAR-pcie-r2   | 61.3   | 147.7   | 56.0 | 257.3 | —    | off  | —     |
| merge-off-P-setAR-pcie-r3   | 61.3   | 146.2   | 55.9 | 259.9 | —    | off  | —     |
| merge-off-P-setAR-pcie-r4   | 61.3   | 146.1   | 55.9 | 260.1 | —    | off  | —     |
| merge-off-P-setAR-pcie-r5   | 61.3   | 147.3   | 56.0 | 257.9 | —    | off  | —     |

Per-run prompt_md5 / `spec_route` / power objects live in the set dir's `*.bench.json`; placement lines in `*.log`; telemetry in `*.power.jsonl`; verifier summary in `VERIFY-FIXED-SUMMARY.txt`.
