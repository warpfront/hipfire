# MoE offload — manual placement vs hipfire's automatic scheduling

**Audience:** the agent (or human) who will run this. This is a *plan*, not a
result — no numbers here are measured. It is untracked by convention (working
draft); read it from the worktree.

**Companion docs:** [`perf-benchmarking.md`](perf-benchmarking.md) (protocol),
[`cpu-exec-offload-benchmark-handoff.md`](cpu-exec-offload-benchmark-handoff.md)
(hipfire's own `memory.offload_exec=cpu` runbook — fixture identity, the
`hipfire bench` arm loop, the five contamination traps, the per-step trace;
**this file adds the manual-vs-auto question, not a second measurement protocol**),
[`../plans/partial-gpu-offload-design.md`](../plans/partial-gpu-offload-design.md)
§ 6.2.1 (what the feature is). The stale cross-engine framing lives in
[`../perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md`](../perf-checkpoints/2026-09-26-llamacpp-offload-scaling-baseline.md).

**Non-goals.** Not an admission, not a `docs/BENCHMARKS.md` claim, not comparable
across host/model/quant/GPU/prompt/method. Do not tune code while measuring.

---

## Goal

Hipfire now schedules partial offload **automatically**: `memory.moe_expert_budget`
and `memory.gpu_layer_budget` accept `auto`, which resolves placement by fit
(`hipfire_runtime::offload::plan` — experts cold first, whole layers only as
fallback, against measured free VRAM minus the KV reservation, prefill floor,
draft bytes and slack).

The question this plan answers: **for MoE models, does `auto` land on the same
placement a careful manual sweep finds — and produce the same rate — or does it
mis-place?**

Two failure directions to detect:
- **over-spill** (auto spills more than it must → slower, VRAM left idle);
- **over-place** (auto spills too little → OOM / later allocation failure).

"Manual" here means an explicit count: `memory.moe_expert_budget = N` (or
`memory.gpu_layer_budget = N`), a **hard pin** — N layers keep their routed
experts on the GPU, the rest spill, regardless of whether they'd fit.
`auto` is the fit-driven search. The manual sweep over N *is* the baseline
frontier; `auto`'s chosen point is scored against it.

llama.cpp is the **mature implementation** of partial offload here — a yardstick
for what a well-optimised engine achieves, not a normative reference — and is
retained as a comparison anchor (§ F). The manual-vs-auto question itself is
entirely within hipfire and needs no other engine.

## What is measured

For each MoE fixture × `offload_exec` × KV mode × context:

1. **Placement parity** — the split `auto` picks vs the split the manual sweep
   finds for the same capacity. Compare the load lines:
   `partial offload: … routed experts host on K layers, device weights X MiB of Y
   MiB available` and the pinned-host bytes. `auto` should choose the **smallest
   spill that fits** (use the capacity, don't idle it) and never refuse where a
   manual split loads.
2. **Rate** — decode tok/s at `auto`'s point vs the manual point with the *same*
   spill; and vs the best manual point at the same context.
3. **Fit / refusal** — `auto` must load wherever any split loads, and where none
   does it must **refuse with numbers** (the loader names the bytes and the knob),
   never OOM at a layer.
4. **Zero-diff** — a fixture that fits resident stays fully resident under `auto`
   (byte-identical to no knobs). This is the regression guard, not a benchmark.
5. **Frontier** — the manual `(spilled layers, max_seq)` frontier (§ E); `auto`
   run at each context must land on it, or the gap is the finding.

## Fixture identity (verify before starting; record your own digests)

| artifact | pin | your digest |
|---|---|---|
| hipfire tree | `git rev-parse HEAD` | |
| `target/release/hipfire`, `target/release/daemon` | md5 (runbook § 1: a daemon md5 is not stable across comment-only edits — compare `.text` if adjudicating) | |
| MoE fixture (qt13) | `~/.hipfire/models/qwen3.5-35b-a3b.mq4` (40L, uniform MQ4G256) | |
| MoE fixture (qt44) | `~/.hipfire/models/ornith-1.5-35b-a3b.mq4` (uniform MQ4G256V2) | |
| dense control (fits) | `~/.hipfire/models/qwen3.5-9b.mq4` (32L) | |
| prompt | `benchmarks/prompts/gpu_offload_probe.txt`, md5 `5835c71e471849b4a72e1dc8e39695e7` (215 B) | |

Host: 1× RX 9070 XT `gfx1201` (16,304 MB), ROCm/HIP 7.2, Ryzen 7 7800X3D
(8c/16t), 28 GB RAM, THP `[always]`. **Both 35B MoE fixtures exceed the card, so
offload is the only route — there is no resident arm for them**; the 9B is the
resident/fit control.

## Method

### A · Host preparation

Quiesce (`uptime` idles ≈ 0.5; the runbook records a 6× error from a concurrent
build), one fresh process per arm, interleave arms, pin the prompt bytes.

### B · Manual arms (the baseline frontier)

```bash
PROMPT=benchmarks/prompts/gpu_offload_probe.txt
COMMON=(--spec off --runs 5 --warmups 2 --max-tokens 64 --backend noslots \
        --workload stateless --kv-mode q8 --prompt-file "$PROMPT" --json)

# sweep the expert pin: N = experts kept on GPU (max spill -> full resident)
for N in 40 34 28 22 16 10 4 0; do
  HIPFIRE_MOE_EXPERT_BUDGET=$N HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GRAPH=0 \
    ./target/release/hipfire bench qwen3.5:35b-a3b "${COMMON[@]}" \
    > "manual_moe_N$N.json" 2> "manual_moe_N$N.err"
done
# and the whole-layer pin, for the layer tier
for N in 40 30 20; do
  HIPFIRE_GPU_LAYER_BUDGET=$N HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GRAPH=0 \
    ./target/release/hipfire bench qwen3.5:35b-a3b "${COMMON[@]}" > "manual_layer_N$N.json" 2>&1
done
```

Requirements (the runbook enforces them): `--backend noslots --workload stateless
--spec off` explicit; `HIPFIRE_GRAPH=0` for `exec=cpu`; grep the stderr for
`partial offload:` **and** the `cpu exec: {spilled}/{spilled} … uncovered quants`
coverage line; record the **pinned-host bytes and device-weighted bytes** per
point, plus `vram_free_before_mb − vram_free_mb`.

### C · `auto` arms (the subject)

```bash
for seq in 2048 8192 32768; do
  HIPFIRE_MOE_EXPERT_BUDGET=auto HIPFIRE_OFFLOAD_EXEC=cpu HIPFIRE_GRAPH=0 \
    HIPFIRE_MAX_SEQ=$seq \
    ./target/release/hipfire bench qwen3.5:35b-a3b "${COMMON[@]}" \
    > "auto_moe_seq$seq.json" 2> "auto_moe_seq$seq.err"
done
```

`auto` is never silent: its load line states what it chose
(`partial offload: {host} of {n} layers host-placed, routed experts host on {k} layers, device weights {X} MiB of {Y} MiB available`, or
`partial offload: model fits - every layer resident`). **Capture that line** — it is the
placement being scored. Also run `moe_expert_budget=auto` with
`gpu_layer_budget` still `Full`, to check the case where the expert tier alone
cannot free enough (it must then refuse naming `memory.gpu_layer_budget`).

### D · The comparison

For each `(fixture, exec, kv-mode, context)`:

1. **Does `auto` load?** It must load wherever any manual N loads.
2. **Is `auto` on the frontier?** Compare its chosen split to the manual sweep's
   smallest-spill-that-fits for the same capacity. `auto` may pick a *slightly*
   larger spill than a fine manual grid (its units are whole expert-layers and
   whole layers); state the granularity gap if so — that is the expected
   difference, not a bug. A manual N that fits with less spill than `auto` chose
   **is** a finding.
3. **Rate at matched spill:** decode tok/s, `auto` vs the manual point with the
   same spill (should be equal — same placement, same kernels — any gap is a
   confound, re-measure).
4. **Idle capacity:** `auto`'s `device weights X of Y available` gap < one
   expert-layer (the granularity bound). A gap ≥ a whole layer means it stopped
   early.
5. **Refusals:** for a context that can't fit even fully spilled, `auto` must
   refuse with numbers (not OOM); record the text as the boundary.

### E · The spread (the surface `auto` is plotted on)

The manual sweep × context × KV mode is a multi-factor frontier. This host:
16 GiB VRAM + 28 GB RAM; the binding tradeoff is **spill ↔ context** (both spend
VRAM).

Factors: fixture (MoE 35B qt13/qt44; 9B control) × `exec` (pcie/cpu) × placement
(expert pin / layer pin) × `memory.max_seq` (2048/8192/32768) × KV mode
(`q8`/`fwht3`) × graph (`on` for pcie, `off` for cpu — not free).

Design: per fixture × exec, sweep the pin from max spill to resident; at each
point raise `max_seq` until the load refuses (record the refusal); repeat at the
two KV extremes. Present, per fixture, a matrix — **rows = spilled layers, cols =
max_seq, cell = decode tok/s** (or the refusal numbers) — and plot `auto`'s point
on it. The union of non-refused cells is "what runs on this machine"; the
boundary text is the capability answer for the rest.

### F · llama.cpp — the mature implementation (comparison anchor)

llama.cpp is the mature implementation of partial offload to measure against: its
fit scheduler and its CPU backend are the yardstick for what a well-optimised
engine achieves, not a spec we are conformant to. Flags, verified via Context7
(`common/arg.cpp`, `tools/completion/README.md`, `docs/multi-gpu.md`):

| flag | meaning |
|---|---|
| `-ncmoe, --n-cpu-moe N` | keep the MoE weights of the **first N layers** in the CPU — the inverse count of hipfire's `moe_expert_budget`, which keeps the **last** N on the GPU (`N ≈ n_layers − N_hipfire`) |
| `-cmoe, --cpu-moe` | keep **all** MoE weights in the CPU (the max-spill bound) |
| `-ncffn, --n-cpu-ffn N` | the **dense-FFN** analogue (first N layers' FFN to CPU) — use this for the dense control, not `--n-cpu-moe` |
| `-ngl, --n-gpu-layers N` | layers in VRAM; accepts `auto` (default) or `all` |
| `-ot, --override-tensor pattern=buffer_type` | generic per-tensor override; `--n-cpu-moe` is sugar over it |
| `-fit, --fit [on\|off]` | **default `on`** — "adjust unset arguments to fit in device memory". **This is llama.cpp's auto scheduler.** Tuned by `-fitt, --fit-target MiB` (per-device margin, default 1024) and `-fitc, --fit-ctx N` (minimum ctx it may set, default 4096) |
| `-ctk/-ctv, -fa, -c` | KV type / flash-attn / context — the KV + context factors |

**The manual arm must set `--fit off`**, or llama.cpp's fit runs and the point is
not manual. **The auto arm is `--fit on` (default) + `-ngl auto`** (plus
`--n-cpu-moe`/`--cpu-moe` if the expert tier is pinned):

```bash
# manual (reference): pin the split, disable the auto-fit
llama-bench -m model.gguf --fit off -ngl 24 --n-cpu-moe 20 -p 512 -n 64 -r 5 -o json
# auto (subject): let llama.cpp fit
llama-bench -m model.gguf -ngl auto -p 512 -n 64 -r 5 -o json
```

Record the llama.cpp binary identity, commit, and backends — the 2026-09-26
baseline used Vulkan/radv, not ROCm, which is a method difference to state.

## Reporting

- A new `docs/perf-checkpoints/<date>-moe-offload-manual-vs-auto.md`, lifecycle
  `historical`, fixture table filled, raw `--json` kept.
- One table per fixture/context: **manual-optimal (spill, VRAM used, rate) |
  `auto` (spill, VRAM used, rate) | delta | on-frontier?**
- If `auto` under-performs a manual split, name the cause from the `offload::plan`
  logic (whole-unit granularity; expert-tier-first order) and the refusals; do not
  propose a fix from one cell.
- Binary digests + prompt md5 in the commit message / PR description. Any product
  claim first needs a claim-scoped `scripts/serve_harness.py` pass.

## Limits

- The manual sweep is a **grid** — `auto` can legitimately beat a coarse grid, and
  a fine one is expensive. Record the grid resolution; a claim of "manual is
  better" needs a manual point *strictly* smaller-spill than `auto`.
- Auto's units are whole expert-layers/whole layers; llama.cpp's fit can spend
  partial layers. A residual idle-VRAM gap of that kind is expected, not a defect.
- Shared desktop / THP / swap — record them; a contaminated run is void.

## What counts as a finding

- `auto` refuses where a manual split loads, or OOMs where it should refuse.
- `auto` chooses a **larger** spill than a manual split that fits at the same
  capacity/context.
- `auto` leaves **≥ one whole unit** of capacity idle (stopped early) — as opposed
  to the sub-unit granularity gap.
- A rate gap at **matched** spill (would indicate the arms differ beyond placement).
- Conversely, `auto` matching the manual frontier across fixtures/contexts is the
  result that lets us trust the shipped default and stop hand-tuning.
