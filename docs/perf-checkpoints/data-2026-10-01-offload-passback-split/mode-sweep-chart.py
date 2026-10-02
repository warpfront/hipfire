#!/usr/bin/env python3
"""Figure for docs/perf-checkpoints/2026-10-01-offload-passback-split-gfx1201.md § 8.

Reads the raw `hipfire bench --json` outputs of the offload-amount sweep, one file
per (model, spilled, mode, round), and writes
`mode-sweep-vs-spilled-layers.{png,svg}` beside it.

Those JSONs are produced by (2 interleaved rounds, one fresh daemon per arm):

  for spilled in 4 8 12 16; do              # 9B: budget = 32 - spilled
    env HIPFIRE_GPU_LAYER_BUDGET=$((32-spilled)) HIPFIRE_OFFLOAD_EXEC=$mode \
      hipfire bench qwen3.5:9b --spec off --runs 1 --warmups 1 --max-tokens 128 \
        --backend noslots --workload stateless \
        --prompt-file benchmarks/prompts/gpu_offload_probe.txt --json \
        > sw_9b_${spilled}_${mode}_${round}.json
  done
  # 27B: qwen3.8-27b.mq3-xt with HIPFIRE_GPU_LAYER_BUDGET=56 (8 of 64 spilled)

and are keyed here by the `sw_<model>_<spilled>_<mode>_<round>.json` names; point
`SW_DIR` at wherever they live.
"""
import os
SW_DIR = os.environ.get("SW_DIR", "/tmp")

import json, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

MODES = ["cpu", "passback", "pcie"]
COLORS = {"cpu": "#c0392b", "passback": "#1f77b4", "pcie": "#7f8c8d"}

def load(prefix, spilled, mode, rounds=(1, 2)):
    vals, complete = [], True
    for r in rounds:
        try:
            j = json.load(open(f"{SW_DIR}/sw_{prefix}_{spilled}_{mode}_{r}.json"))
            s = j.get("samples", {}).get("decode") or []
        except Exception:
            complete = False; continue
        if not s:
            complete = False; continue
        vals.append(s[0])
    return (statistics.median(vals), vals, complete) if vals else None

def series(prefix, spills):
    return {(sp, m): r for sp in spills for m in MODES if (r := load(prefix, sp, m))}

SP9 = [4, 8, 12, 16]
b9, b27 = series("9b", SP9), series("27b", [8, 16])

fig = plt.figure(figsize=(14, 9.2))
gs = GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.42, wspace=0.22)
ax9, ax27, axg = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

# ── (a) 9B: decode tok/s vs layers spilled ───────────────────────────────────
for m in MODES:
    xs = [sp for sp in SP9 if (sp, m) in b9]
    ys = [b9[(sp, m)][0] for sp in xs]
    if not xs:
        continue
    ax9.plot(xs, ys, "o-", color=COLORS[m], label=m, linewidth=2.4, markersize=8,
             markeredgecolor="white", markeredgewidth=1.2)

# Every value lives in the legend, one column per spill amount; the plot area holds
# only the lines. Nothing can eclipse a number, and nothing can fall outside the
# frame, by construction (the earlier in-plot labels collided and the rightmost one
# spilled past the axis).
for m in MODES:
    xs = [sp for sp in SP9 if (sp, m) in b9]
    if not xs:
        continue
    vals = "  ".join(f"{b9[(sp, m)][0]:5.1f}" for sp in xs)
    ax9.plot(xs, [b9[(sp_, m)][0] for sp_ in xs], "o-", color=COLORS[m], linewidth=2.4,
             markersize=8, markeredgecolor="white", markeredgewidth=1.2,
             label=f"{m:<8}{vals}")
ax9.set_xlabel("layers spilled to host RAM (of 32)")
ax9.set_ylabel("decode tok/s")
ax9.set_title("(a) Qwen3.5-9B mq4 — decode tok/s by engine and spill",
              fontsize=11, loc="left")
ax9.set_xticks(SP9)
# Room for the outboard labels so they never spill into the neighbouring panel.
ax9.set_xlim(3.4, 16.6)
ax9.grid(alpha=0.3)
lo = min(v[0] for v in b9.values()); hi = max(v[0] for v in b9.values())
ax9.set_ylim(lo * 0.9, hi * 1.12)
leg9 = ax9.legend(title="memory.offload_exec        tok/s by layers spilled:  4    8   12   16",
                  loc="upper right", frameon=True, framealpha=0.96, fontsize=10,
                  title_fontsize=9, prop={"family": "monospace"},
                  handlelength=1.6, borderpad=0.7, labelspacing=0.5)
leg9.get_frame().set_edgecolor("#999999")

# ── (b) 27B comparison point ────────────────────────────────────────────────
sp27 = 8
vals = [b27.get((sp27, m), (float("nan"), [], False))[0] for m in MODES]
bars = ax27.bar(range(len(MODES)), vals, color=[COLORS[m] for m in MODES], width=0.62)
for b, v in zip(bars, vals):
    if v == v:
        ax27.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                      xytext=(0, 4), ha="center", va="bottom", fontsize=10, weight="bold")
ax27.set_xticks(range(len(MODES)))
ax27.set_xticklabels(MODES, fontsize=10)
ax27.set_ylabel("decode tok/s")
ax27.set_title("(b) Qwen3.8-27B mq3-xt (64 layers), 8 spilled — same engines",
               fontsize=11, loc="left")
ax27.set_ylim(0, max(vals) * 1.34)
ax27.grid(alpha=0.3, axis="y")
if vals[0] == vals[0] and vals[1] == vals[1]:
    ax27.axhline(vals[0], color=COLORS["cpu"], linestyle="--", linewidth=1.1, alpha=0.75, zorder=0)
    ax27.annotate(f"+{100*(vals[1]/vals[0]-1):.1f} % over cpu", (1, vals[1]), textcoords="offset points",
                  xytext=(0, 26), ha="center", va="bottom", fontsize=11, weight="bold",
                  color=COLORS["passback"],
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["passback"], lw=0.8))

# ── (c) the gains, on their own axis so nothing can eclipse them ────────────
width = 0.36
xs = list(range(len(SP9)))
gain_cpu = [100 * (b9[(sp, "passback")][0] / b9[(sp, "cpu")][0] - 1) for sp in SP9]
gain_pcie = [100 * (b9[(sp, "passback")][0] / b9[(sp, "pcie")][0] - 1) for sp in SP9]
b1 = axg.bar([x - width / 2 for x in xs], gain_cpu, width, label="vs `cpu`", color="#1f77b4")
b2 = axg.bar([x + width / 2 for x in xs], gain_pcie, width, label="vs `pcie`", color="#2ca02c")
for bars_ in (b1, b2):
    for b in bars_:
        axg.annotate(f"+{b.get_height():.1f} %", (b.get_x() + b.get_width() / 2, b.get_height()),
                     textcoords="offset points", xytext=(0, 4), ha="center", va="bottom",
                     fontsize=9.5, weight="bold")
axg.set_xticks(xs)
axg.set_xticklabels([f"{sp} of 32 spilled" for sp in SP9])
axg.set_ylabel("pass-back decode gain")
axg.set_title("(c) Pass-back's gain over the two single-engine routes — same 9B points as (a)",
              fontsize=11, loc="left")
axg.grid(alpha=0.3, axis="y")
axg.legend(loc="upper left", fontsize=10, title="pass-back compared with", title_fontsize=9.5)
axg.set_ylim(0, max(gain_pcie) * 1.24)

fig.suptitle(
    "Offload pass-back (`memory.offload_exec=passback`) — gfx1201, RX 9070 XT 16 GB, PCIe 4.0 x16, HIP 7.2\n"
    "128-token greedy decode, `--spec off`, one committed prompt, 2 interleaved fresh-process rounds per point "
    "(round spread ±0.5 tok/s); host load 3–8, so read the *ratios*, not the absolute rates",
    fontsize=11.5)
out = "docs/perf-checkpoints/data-2026-10-01-offload-passback-split"
os.makedirs(out, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(f"{out}/mode-sweep-vs-spilled-layers.{ext}", dpi=140, bbox_inches="tight")
print("rendered")
